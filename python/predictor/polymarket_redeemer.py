import asyncio
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import requests
from eth_abi import encode as eth_encode
from eth_utils import keccak

from py_builder_relayer_client.client import RelayClient
from py_builder_relayer_client.models import OperationType, RelayerTxType, SafeTransaction
from py_builder_signing_sdk.config import BuilderApiKeyCreds, BuilderConfig as BuilderSigningConfig

USDC_ADDRESS = "0xC011a7E12a19f7B1f670d46F03B03f3342E82DFB"  # pUSD
USDC_E_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"  # USDC.e (bridged)
CTF_ADDRESS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
NEG_RISK_ADAPTER = "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296"

POLYGON_RPC_ENDPOINTS = [
    "https://polygon.llamarpc.com",
    "https://polygon.drpc.org",
    "https://polygon.meowrpc.com",
    "https://rpc.ankr.com/polygon",
]

REDEEM_SELECTOR = keccak(text="redeemPositions(address,bytes32,bytes32,uint256[])")[:4]
NEG_RISK_REDEEM_SELECTOR = keccak(text="redeemPositions(bytes32,uint256[])")[:4]
PAYOUT_NUMERATOR_SELECTOR = keccak(text="payoutNumerators(bytes32,uint256)")[:4]
PAYOUT_DENOMINATOR_SELECTOR = keccak(text="payoutDenominator(bytes32)")[:4]
ERC1155_BALANCE_SELECTOR = bytes.fromhex("00fdd58e")
ERC20_BALANCE_SELECTOR = keccak(text="balanceOf(address)")[:4]

DATA_API_URL = "https://data-api.polymarket.com/positions"
DEFAULT_RELAYER_URL = "https://relayer-v2.polymarket.com"

RELAYER_RETRY_WAIT_SEC = int(os.getenv("POLY_RELAYER_RETRY_WAIT_SEC", "60"))
COOLDOWN_SECONDS = int(os.getenv("POLY_REDEEM_COOLDOWN_SEC", str(7 * 60)))

logger = logging.getLogger("poly_redeemer")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(name)s %(levelname)s %(asctime)s] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

_redeem_lock = asyncio.Lock()
_last_redeem_ts: float = 0.0


def _call_polygon_rpc(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Try multiple RPC endpoints until one returns a result."""
    for rpc_url in POLYGON_RPC_ENDPOINTS:
        try:
            resp = requests.post(rpc_url, json=payload, timeout=12)
            resp.raise_for_status()
            data = resp.json()
            if "error" in data and "result" not in data:
                print(f"[redeem] RPC {rpc_url} responded with error: {data['error']}")
                continue
            return data
        except Exception as exc:
            print(f"[redeem] RPC {rpc_url} failed: {exc}")
    print("[redeem] All Polygon RPC endpoints failed for payload")
    return None


@dataclass
class RedeemerSettings:
    private_key: str
    funder_address: str
    signature_type: int
    builder_api_key: str
    builder_api_secret: str
    builder_api_passphrase: str
    relayer_url: str
    chain_id: int


def _env_value(*names: str) -> Optional[str]:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def _load_settings() -> RedeemerSettings:
    private_key = _env_value("POLY_PRIVATE_KEY", "POLYMARKET_PRIVATE_KEY")
    funder_address = _env_value("POLY_FUNDER", "POLYMARKET_FUNDER_ADDRESS")
    sig_raw = _env_value("POLY_SIGNATURE_TYPE") or "0"
    builder_key = _env_value("POLY_BUILDER_API_KEY", "POLYMARKET_BUILDER_API_KEY")
    builder_secret = _env_value("POLY_BUILDER_API_SECRET", "POLYMARKET_BUILDER_SECRET")
    builder_passphrase = _env_value("POLY_BUILDER_API_PASSPHRASE", "POLYMARKET_BUILDER_PASSPHRASE")
    relayer_url = _env_value("POLY_RELAYER_HOST") or DEFAULT_RELAYER_URL
    chain_raw = _env_value("POLY_CHAIN_ID") or "137"

    missing = []
    if not private_key:
        missing.append("POLY_PRIVATE_KEY")
    if not funder_address:
        missing.append("POLY_FUNDER")
    if not builder_key:
        missing.append("POLY_BUILDER_API_KEY")
    if not builder_secret:
        missing.append("POLY_BUILDER_API_SECRET")
    if not builder_passphrase:
        missing.append("POLY_BUILDER_API_PASSPHRASE")
    if missing:
        raise RuntimeError(f"Missing required env vars for redemption: {', '.join(missing)}")

    try:
        signature_type = int(sig_raw)
    except (TypeError, ValueError):
        raise RuntimeError("POLY_SIGNATURE_TYPE must be an integer (0, 1, or 2)")

    try:
        chain_id = int(chain_raw)
    except (TypeError, ValueError):
        raise RuntimeError("POLY_CHAIN_ID must be an integer")

    return RedeemerSettings(
        private_key=private_key,
        funder_address=funder_address,
        signature_type=signature_type,
        builder_api_key=builder_key,
        builder_api_secret=builder_secret,
        builder_api_passphrase=builder_passphrase,
        relayer_url=relayer_url,
        chain_id=chain_id,
    )


def _normalize_condition_id(cid: str) -> str:
    cid = cid.strip()
    if not cid:
        return ""
    if not cid.startswith("0x"):
        cid = "0x" + cid
    return cid.lower()


def _parse_date(date_val) -> datetime:
    """Parse various date formats from API."""
    if not date_val:
        return datetime.min
    if isinstance(date_val, (int, float)):
        return datetime.fromtimestamp(date_val)
    if isinstance(date_val, str):
        # Try ISO format first
        try:
            return datetime.fromisoformat(date_val.replace('Z', '+00:00').replace('+00:00', ''))
        except ValueError:
            pass
        # Try timestamp string
        try:
            return datetime.fromtimestamp(float(date_val))
        except (ValueError, TypeError):
            pass
    return datetime.min


def _fetch_redeemable_positions(funder: str, max_positions: int = 30) -> List[Dict[str, Any]]:
    print(f'[redeem] fetching redeemable positions for {funder} ------------------------------------')
    print(f'[redeem] Will sort by date and return last {max_positions} positions')
    params = {"user": funder, "redeemable": "true", "sizeThreshold": 0,  'limit': 30, 'sortBy': 'TITLE', 'sortDirection' :'DESC'}
    for attempt in range(2):
        resp = requests.get(DATA_API_URL, params=params, timeout=15)
        if resp.status_code in (429, 1015):
            if attempt == 0:
                logger.warning("Data API rate limited (status=%s). Waiting %ss before retrying...", resp.status_code, RELAYER_RETRY_WAIT_SEC)
                time.sleep(RELAYER_RETRY_WAIT_SEC)
                continue
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, list):
            logger.warning("Unexpected positions payload: %s", data)
            return []

        print(f'[redeem] -----------------------------------------------------------')
        
        # Filter positions with size > 0 AND currentValue > 0 (has actual value to redeem)
        filtered = [
            p for p in data 
            if float(p.get("size", 0) or 0) > 0 
            and float(p.get("currentValue", 0) or 0) > 0
        ]
        print(f'[redeem] After filtering size>0 AND currentValue>0: {len(filtered)}/{len(data)} positions remain')
        
        # Log full JSON structure of filtered positions for debugging
        if filtered:
            print(f'[redeem] ===== FILTERED POSITIONS JSON =====')
            for p in filtered[:3]:  # Log first 3 positions
                pos_json = json.dumps(p, indent=2, default=str)
                print(f'[redeem] Position JSON:\n{pos_json}')
            if len(filtered) > 3:
                print(f'[redeem] ... and {len(filtered) - 3} more positions')
            print(f'[redeem] ===== END FILTERED POSITIONS JSON =====')

        # Sort by date (newest first) - try multiple date fields
        def get_date_key(p):
            for field in ['endDate', 'end_date', 'expiration', 'expirationDate', 'timestamp', 'updatedAt', 'createdAt']:
                if field in p and p[field]:
                    return _parse_date(p[field])
            return datetime.min
        
        sorted_positions = sorted(filtered, key=get_date_key, reverse=True)
        print(f'[redeem] Sorted by date (newest first)')
        
        # Limit to last N positions
        limited = sorted_positions[:max_positions]
        print(f'[redeem] Taking last {max_positions} positions: {len(limited)} positions selected')
        
        for p in limited:
            cid = p.get("conditionId") or p.get("condition_id", "N/A")
            title = p.get("title") or p.get("question", "Unknown")
            size = p.get("size", 0)
            current_value = p.get("currentValue", 0)
            end_date = p.get("endDate") or p.get("end_date", "N/A")
            print(f'  - {title[:35]:<35} | endDate={str(end_date)[:10]:<10} | size={size:>8} | currentValue={current_value}')
        print(f'[redeem] -----------------------------------------------------------')
        return limited
    raise RuntimeError("Data API rate limit exceeded while fetching positions")


def _build_client(settings: RedeemerSettings) -> RelayClient:
    wallet_type = RelayerTxType.PROXY if settings.signature_type == 1 else RelayerTxType.SAFE
    builder_config = BuilderSigningConfig(
        local_builder_creds=BuilderApiKeyCreds(
            key=settings.builder_api_key,
            secret=settings.builder_api_secret,
            passphrase=settings.builder_api_passphrase,
        )
    )
    return RelayClient(
        settings.relayer_url,
        chain_id=settings.chain_id,
        private_key=settings.private_key,
        builder_config=builder_config,
        relay_tx_type=wallet_type,
    )


def _neg_risk_amounts(pos: Dict[str, Any]) -> List[int]:
    size_raw = int(float(pos.get("size", 0) or 0) * 1e6)
    outcome_idx = int(float(pos.get("outcomeIndex", 0) or 0))
    outcomes = int(pos.get("outcomeCount") or 0)
    length = max(2, outcomes, outcome_idx + 1)
    amounts = [0] * length
    if 0 <= outcome_idx < length:
        amounts[outcome_idx] = size_raw
    return amounts


def _execute_with_retry(client: RelayClient, txn: SafeTransaction, label: str) -> Dict[str, Any]:
    """Execute transaction and return detailed response info."""
    for attempt in range(2):
        try:
            print(f'[redeem] Calling relayer execute for: {label}')
            resp = client.execute([txn], label)
            print(f'[redeem] Waiting for transaction confirmation...')
            resp.wait()
            
            # Extract response details
            result = {
                "status": "success",
                "label": label,
            }
            
            # Print ALL available attributes and their values for debugging
            print(f'[redeem] ===== Response Object Details =====')
            attrs = [attr for attr in dir(resp) if not attr.startswith('_') and not callable(getattr(resp, attr))]
            for attr in attrs:
                try:
                    val = getattr(resp, attr)
                    if attr in ('transaction_hash', 'tx_hash', 'hash', 'transaction_id', 'status', 'block_number', 'receipt'):
                        print(f'[redeem] {attr}: {val}')
                except Exception as e:
                    print(f'[redeem] {attr}: <error accessing: {e}>')
            print(f'[redeem] ====================================')
            
            # Try to get transaction hash
            tx_hash = None
            if hasattr(resp, 'transaction_hash') and resp.transaction_hash:
                tx_hash = resp.transaction_hash
            elif hasattr(resp, 'hash') and resp.hash:
                tx_hash = resp.hash
            elif hasattr(resp, 'tx_hash') and resp.tx_hash:
                tx_hash = resp.tx_hash
            
            if tx_hash:
                result["transaction_hash"] = tx_hash
                print(f'[redeem] ✅ Transaction confirmed! Hash: {tx_hash}')
            else:
                print(f'[redeem] ✅ Transaction confirmed (no hash found)')
            
            # Try to get transaction_id
            if hasattr(resp, 'transaction_id') and resp.transaction_id:
                result["transaction_id"] = resp.transaction_id
                print(f'[redeem] Transaction ID: {resp.transaction_id}')
            
            # Try to get status
            if hasattr(resp, 'status'):
                result["tx_status"] = resp.status
                print(f'[redeem] Status: {resp.status}')
            
            # Try to get receipt/block info directly from attributes
            if hasattr(resp, 'receipt') and resp.receipt:
                receipt = resp.receipt
                result["receipt"] = receipt
                print(f'[redeem] Receipt (from attr): {json.dumps(receipt, indent=2, default=str)[:800]}')
            
            if hasattr(resp, 'block_number') and resp.block_number:
                result["block_number"] = resp.block_number
                print(f'[redeem] Block number (from attr): {resp.block_number}')
            
            # Try to call get_transaction() if available (for fill details)
            if hasattr(resp, 'get_transaction') and callable(resp.get_transaction):
                print(f'[redeem] Calling get_transaction() for fill details...')
                try:
                    tx_details = resp.get_transaction()
                    print(f'[redeem] Transaction details: {json.dumps(tx_details, indent=2, default=str)[:1000]}')
                    result["transaction_details"] = tx_details
                    
                    # Try to extract fill/gas info from tx details
                    if isinstance(tx_details, dict):
                        if 'gasUsed' in tx_details:
                            print(f'[redeem] Gas used: {tx_details["gasUsed"]}')
                        if 'effectiveGasPrice' in tx_details:
                            print(f'[redeem] Gas price: {tx_details["effectiveGasPrice"]}')
                        if 'logs' in tx_details:
                            print(f'[redeem] Number of event logs: {len(tx_details["logs"])}')
                        
                        # Decode the actual transaction data to verify indexSets
                        tx_data = tx_details.get('data', '')
                        print(f'[redeem] Raw TX data length: {len(tx_data)}')
                        print(f'[redeem] Raw TX data (first 500 chars): {tx_data[:500]}')
                        print(f'[redeem] Raw TX data (last 200 chars): {tx_data[-200:] if len(tx_data) > 200 else tx_data}')
                        if tx_data and len(tx_data) > 200:
                            # Look for redeemPositions selector 0x01b7037c in the data
                            redeem_pos_idx = tx_data.find('01b7037c')
                            if redeem_pos_idx >= 0:
                                # redeemPositions: selector(8) + collateral(64) + parent(64) + condition(64) + offset(64) + len(64) + indexSets...
                                # offset starts after selector, so at byte 4 (8 hex chars) after selector start
                                args_start = redeem_pos_idx + 8  # after selector
                                if len(tx_data) >= args_start + 320 + 128:  # 5 * 64 chars minimum
                                    idx_sets_len_hex = tx_data[args_start + 256:args_start + 320]  # 5th word = array length
                                    idx_sets_0_hex = tx_data[args_start + 320:args_start + 384] if len(tx_data) >= args_start + 384 else ""
                                    try:
                                        idx_sets_len = int(idx_sets_len_hex, 16)
                                        idx_sets_0 = int(idx_sets_0_hex, 16) if idx_sets_0_hex else 0
                                        print(f'[redeem] DECODED FROM TX DATA: indexSets length={idx_sets_len}, indexSets[0]={idx_sets_0}')
                                        print(f'[redeem]    Hex: len=0x{idx_sets_len_hex}, val=0x{idx_sets_0_hex}')
                                    except Exception as decode_err:
                                        print(f'[redeem] Failed to decode indexSets: {decode_err}')
                            
                        # Check if transaction actually succeeded (status = 1)
                        status = tx_details.get('status')
                        if status == '0x0' or status == 0:
                            print(f'[redeem] ⚠️ Transaction REVERTED on-chain!')
                        elif status == '0x1' or status == 1:
                            print(f'[redeem] ✅ Transaction succeeded on-chain')
                        
                        # Look for TransferSingle/TransferBatch events (ERC1155 burns)
                        # TransferSingle topic0 = keccak256("TransferSingle(address,address,address,uint256,uint256)")
                        TRANSFER_SINGLE_TOPIC = "0xc3d58168c5ef739b506c5e4d7f6f0b1c3d5a7b9e1f3a5c7d9e1f3a5c7d9e1f3"[:66]  # Placeholder - will check actual
                        logs = tx_details.get('logs', [])
                        # Check for any transfer events (topic0 for TransferSingle)
                        erc1155_transfers = [l for l in logs if len(l.get('topics', [])) >= 4]
                        if erc1155_transfers:
                            print(f'[redeem] Found {len(erc1155_transfers)} ERC1155 Transfer events (tokens moved/burned)')
                        else:
                            print(f'[redeem] ⚠️ No ERC1155 Transfer events found - tokens may not have been burned')
                except Exception as get_tx_err:
                    print(f'[redeem] get_transaction() failed: {get_tx_err}')
            
            print(f'[redeem] ===== End Response Details =====')
            
            return result
            
        except Exception as exc:  # noqa: BLE001
            status = getattr(exc, "status_code", None)
            error_msg = str(exc)
            print(f'[redeem] ❌ Transaction failed (attempt {attempt+1}): {error_msg}')
            
            if status in (429, 1015) and attempt == 0:
                logger.warning("Relayer rate limited (status=%s). Waiting %ss before retrying...", status, RELAYER_RETRY_WAIT_SEC)
                time.sleep(RELAYER_RETRY_WAIT_SEC)
                continue
            
            # Print exception details
            print(f'[redeem] Exception type: {type(exc).__name__}')
            if hasattr(exc, 'response'):
                try:
                    print(f'[redeem] Error response: {exc.response.text if hasattr(exc.response, "text") else exc.response}')
                except Exception:
                    pass
            raise
    raise RuntimeError("Relayer rate limit exceeded")


def _check_position_token_balance(token_id: str, wallet: str) -> int:
    """Check ERC1155 balance of position token on-chain via Polygon RPC."""
    # ERC1155 balanceOf selector: 0x00fdd58e
    # balanceOf(address account, uint256 id)
    try:
        # Handle token_id as potentially hex or decimal string
        if token_id.startswith("0x"):
            token_id_int = int(token_id, 16)
        else:
            token_id_int = int(token_id)
        
        print(f'[redeem] Checking balance: wallet={wallet}, token_id={token_id} (int={token_id_int})')
        
        # Encode: function selector + address (padded to 32 bytes) + tokenId (32 bytes)
        # balanceOf(address,uint256) selector
        selector = "00fdd58e"
        addr_padded = wallet[2:].lower().zfill(64)  # Remove 0x, pad to 64 hex chars (32 bytes)
        token_padded = hex(token_id_int)[2:].lower().zfill(64)
        data = f"0x{selector}{addr_padded}{token_padded}"
        
        print(f'[redeem] Balance check calldata: {data[:100]}...')
        
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "eth_call",
            "params": [{"to": CTF_ADDRESS, "data": data}, "latest"]
        }
        resp_json = _call_polygon_rpc(payload)
        if resp_json is None:
            return -1
        if "error" in resp_json:
            print(f'[redeem] RPC error: {resp_json["error"]}')
            return -1
        result = resp_json.get("result", "0x0")
        balance = int(result, 16)
        print(f'[redeem] Raw RPC result: {result}, decoded balance: {balance}')
        return balance
    except Exception as e:
        print(f'[redeem] Failed to check token balance: {e}')
        return -1


def _check_token_balance_in_locations(token_id: str, proxy_wallet: str, funder: str) -> dict:
    """Check token balance in multiple locations to find where tokens actually are."""
    locations = {
        "proxy_wallet": proxy_wallet,
        "funder": funder,
        "ctf_contract": CTF_ADDRESS,
    }
    results = {}
    for name, addr in locations.items():
        bal = _check_position_token_balance(token_id, addr)
        results[name] = bal
        print(f'[redeem] Token balance in {name} ({addr[:20]}...): {bal}')
    return results


def _get_condition_payouts(condition_id: str, outcome_count: int = 2) -> Optional[List[int]]:
    """Fetch payoutNumerators for a condition; returns None if not resolved yet."""
    if not condition_id.startswith("0x"):
        condition_id = "0x" + condition_id
    cond_hex = condition_id[2:].lower().zfill(64)

    denom_payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "eth_call",
        "params": [{"to": CTF_ADDRESS, "data": "0x" + PAYOUT_DENOMINATOR_SELECTOR.hex() + cond_hex}, "latest"],
    }
    denom_resp = _call_polygon_rpc(denom_payload)
    if not denom_resp or "result" not in denom_resp:
        return None
    denominator = int(denom_resp["result"], 16)
    if denominator == 0:
        print(f"[redeem] Condition {condition_id} not resolved on-chain (payout denominator = 0)")
        return None

    slots = max(2, outcome_count)
    payouts: List[int] = []
    for slot in range(slots):
        slot_hex = format(slot, "064x")
        data = "0x" + PAYOUT_NUMERATOR_SELECTOR.hex() + cond_hex + slot_hex
        payload = {
            "jsonrpc": "2.0",
            "id": slot + 10,
            "method": "eth_call",
            "params": [{"to": CTF_ADDRESS, "data": data}, "latest"],
        }
        resp = _call_polygon_rpc(payload)
        if not resp or "result" not in resp:
            print(f"[redeem] Failed to fetch payoutNumerator for slot {slot} (condition {condition_id})")
            return None
        payouts.append(int(resp["result"], 16))

    print(f"[redeem] On-chain payout vector for {condition_id}: {payouts}, denominator={denominator}")
    return payouts


def _check_pusd_balance(wallet: str) -> int:
    try:
        selector = ERC20_BALANCE_SELECTOR.hex()
        addr = wallet[2:].lower().zfill(64)
        data = "0x" + selector + addr
        payload = {
            "jsonrpc": "2.0",
            "id": 99,
            "method": "eth_call",
            "params": [{"to": USDC_ADDRESS, "data": data}, "latest"],
        }
        resp = _call_polygon_rpc(payload)
        if not resp or "result" not in resp:
            return -1
        balance = int(resp["result"], 16)
        print(f"[redeem] pUSD balance check for {wallet[:12]}...: {balance}")
        return balance
    except Exception as exc:
        print(f"[redeem] Failed to read pUSD balance: {exc}")
        return -1


def _check_usdc_e_balance(wallet: str) -> int:
    try:
        selector = ERC20_BALANCE_SELECTOR.hex()
        addr = wallet[2:].lower().zfill(64)
        data = "0x" + selector + addr
        payload = {
            "jsonrpc": "2.0",
            "id": 99,
            "method": "eth_call",
            "params": [{"to": USDC_E_ADDRESS, "data": data}, "latest"],
        }
        resp = _call_polygon_rpc(payload)
        if not resp or "result" not in resp:
            return -1
        balance = int(resp["result"], 16)
        print(f"[redeem] USDC.e balance check for {wallet[:12]}...: {balance}")
        return balance
    except Exception as exc:
        print(f"[redeem] Failed to read USDC.e balance: {exc}")
        return -1


def _check_collateral_balances(wallet: str) -> dict:
    """Check both pUSD and USDC.e balances."""
    return {
        "pUSD": _check_pusd_balance(wallet),
        "USDC.e": _check_usdc_e_balance(wallet),
    }


def _redeem_positions_sync() -> Dict[str, Any]:
    settings = _load_settings()
    client = _build_client(settings)
    
    # Configurable max positions to redeem per run (sorted by date, newest first)
    max_positions = int(os.getenv("POLY_REDEEM_MAX_POSITIONS", "30"))

    start_ts = time.time()
    positions = _fetch_redeemable_positions(settings.funder_address, max_positions=max_positions)
    details: List[Dict[str, Any]] = []

    if not positions:
        duration = time.time() - start_ts
        logger.info("Redeem-all: no redeemable positions found")
        return {
            "success": True,
            "positions_found": 0,
            "redeemed": 0,
            "details": [],
            "duration_sec": round(duration, 3),
            "message": "No redeemable positions",
        }

    logger.info("Redeem-all: found %d redeemable positions", len(positions))

    redeemed = 0
    for pos in positions:
        cid = _normalize_condition_id(pos.get("conditionId") or pos.get("condition_id") or "")
        if not cid:
            details.append({"status": "skipped", "reason": "missing_condition_id", "raw": pos})
            continue
        market = pos.get("title") or pos.get("question") or cid[:12]
        neg_risk = pos.get("negativeRisk")
        try:
            condition_bytes = bytes.fromhex(cid[2:])
        except ValueError:
            details.append({"status": "skipped", "reason": "invalid_condition_id", "condition_id": cid})
            continue

        try:
            if neg_risk is True:
                amounts = _neg_risk_amounts(pos)
                args = eth_encode(["bytes32", "uint256[]"], [condition_bytes, amounts])
                txn = SafeTransaction(
                    to=NEG_RISK_ADAPTER,
                    operation=OperationType.Call,
                    data="0x" + (NEG_RISK_REDEEM_SELECTOR + args).hex(),
                    value="0",
                )
                print(f'NEGRISK=TRUE creating safe transaction REDEEM CTF: {txn}')
            elif neg_risk is False:
                # Determine outcome direction from position data
                outcome_idx = int(float(pos.get("outcomeIndex", -1) or -1))
                outcome_label = pos.get("outcome", "UNKNOWN")
                
                # Map to CTF indexSets: 1 = Yes/UP, 2 = No/DOWN
                # Note: Sometimes the mapping is flipped in the data API vs CTF contract
                if outcome_idx == 0 or outcome_label.upper() in ["YES", "UP", "TRUE", "0"]:
                    index_sets = [1]  # Standard: Yes/UP = slot 1
                    alt_index_sets = [2]  # Alternative: flipped mapping
                    direction = "UP"
                    alt_direction = "UP_ALT"
                elif outcome_idx == 1 or outcome_label.upper() in ["NO", "DOWN", "FALSE", "1"]:
                    index_sets = [2]  # Standard: No/DOWN = slot 2
                    alt_index_sets = [1]  # Alternative: flipped mapping  
                    direction = "DOWN"
                    alt_direction = "DOWN_ALT"
                else:
                    # Fallback: parse from title or default to [1, 2] for complete sets
                    title_lower = market.lower()
                    if "down" in title_lower and "up" in title_lower:
                        if "down" in title_lower.split("up")[-1]:
                            index_sets = [2]
                            alt_index_sets = [1]
                            direction = "DOWN"
                            alt_direction = "DOWN_ALT"
                        else:
                            index_sets = [1]
                            alt_index_sets = [2]
                            direction = "UP"
                            alt_direction = "UP_ALT"
                    else:
                        # Default to [1, 2] - requires both tokens for complete set redemption
                        index_sets = [1, 2]
                        alt_index_sets = None
                        direction = "COMPLETE_SET"
                        alt_direction = None
                
                if alt_index_sets:
                    print(f'[redeem] Position direction: outcome_idx={outcome_idx}, outcome="{outcome_label}"')
                    print(f'[redeem]   Standard mapping: {direction} -> indexSets={index_sets}')
                    print(f'[redeem]   Alternative mapping: {alt_direction} -> indexSets={alt_index_sets}')
                else:
                    print(f'[redeem] Position direction: outcome_idx={outcome_idx}, outcome="{outcome_label}" -> {direction} -> indexSets={index_sets}')

                payouts = _get_condition_payouts(cid, outcome_count=int(pos.get("outcomeCount") or 2))
                if payouts:
                    winning_sets = [1 << i for i, val in enumerate(payouts) if val > 0]
                    print(f'[redeem] Winning bitsets reported on-chain: {winning_sets}')
                    if winning_sets and not any(bit in winning_sets for bit in index_sets):
                        override_set = winning_sets[0]
                        print(f'[redeem] ⚠️ Overriding indexSets -> [{override_set}] based on payout vector {payouts}')
                        index_sets = [override_set]
                        direction = f"ONCHAIN_SLOT_{int(math.log2(override_set)) if override_set else 0}"
                        alt_index_sets = winning_sets[1:2] if len(winning_sets) > 1 else None
                        alt_direction = "SECONDARY_SLOT" if alt_index_sets else None

                # Check if market is actually resolved (one outcome should be at price 1.0)
                cur_price = pos.get("curPrice", 0)
                redeemable_flag = pos.get("redeemable", False)
                
                # A market is only redeemable if it's resolved: one outcome price = 1.0
                if cur_price != 1.0:
                    print(f'[redeem] ⚠️ Market NOT resolved! curPrice={cur_price}, expected 1.0 for winning outcome')
                    print(f'[redeem] Redemption will fail - market must be resolved before redeeming')
                    details.append({
                        "status": "skipped",
                        "reason": "market_not_resolved",
                        "condition_id": cid,
                        "market": market,
                        "curPrice": cur_price,
                        "message": "Market not resolved - winning outcome price must be 1.0",
                    })
                    continue
                
                # Log position token details for debugging
                asset_id = pos.get("asset", "N/A")
                proxy_wallet = pos.get("proxyWallet", settings.funder_address)
                print(f'[redeem] Position token ID (asset): {asset_id}')
                print(f'[redeem] Proxy wallet holding tokens: {proxy_wallet}')
                print(f'[redeem] Condition ID: {cid}')
                print(f'[redeem] CTF Contract: {CTF_ADDRESS}')
                print(f'[redeem] Collateral tokens: pUSD={USDC_ADDRESS}, USDC.e={USDC_E_ADDRESS}')
                
                # Build redeem transaction - no approval needed, we own the position tokens
                parent_collection = b"\x00" * 32  # Root collection (empty)
                
                # CRITICAL DEBUG: Log exact values being encoded
                print(f'[redeem] >>> FINAL INDEX_SETS BEFORE ENCODING: {index_sets}')
                print(f'[redeem] >>> Direction: {direction}, Winning sets: {winning_sets if payouts else "N/A"}')
                print(f'[redeem] >>> Condition: {cid}')
                
                args = eth_encode(
                    ["address", "bytes32", "bytes32", "uint256[]"],
                    [USDC_ADDRESS, parent_collection, condition_bytes, index_sets],
                )
                
                full_data = "0x" + (REDEEM_SELECTOR + args).hex()
                print(f'[redeem] Function selector: 0x{REDEEM_SELECTOR.hex()}')
                print(f'[redeem] Encoded args: 0x{args.hex()}')
                print(f'[redeem] Full calldata: {full_data}')
                
                # Extract and verify indexSets from encoded args
                # redeemPositions encoding: selector(4) + collateral(32) + parent(32) + condition(32) + offset(32) + len(32) + indexSets...
                args_hex = args.hex()
                # offset to array is at bytes 96-128 (after 3 address/bytes32 params)
                # offset value is 128 (0x80)
                # array length is at 128-160
                # array elements start at 160
                array_len_hex = args_hex[128*2:160*2]  # 64 chars = 32 bytes
                index_sets_0_hex = args_hex[160*2:192*2] if len(args_hex) >= 192*2 else ""
                print(f'[redeem] EXTRACTED FROM ENCODED: array_len=0x{array_len_hex}={int(array_len_hex, 16)}, indexSets[0]=0x{index_sets_0_hex}={int(index_sets_0_hex, 16) if index_sets_0_hex else "N/A"}')

                txn = SafeTransaction(
                    to=CTF_ADDRESS,
                    operation=OperationType.Call,
                    data=full_data,
                    value="0",
                )
                print(f'NEGRISK=FALSE creating safe transaction REDEEM CTF ({direction}): to={CTF_ADDRESS}')
            else:
                print(f'NEGRISK=UNKNPOWN creating safe transaction REDEEM')
                details.append({
                    "status": "skipped",
                    "reason": "unknown_market_type",
                    "condition_id": cid,
                    "market": market,
                    "negativeRisk": neg_risk,
                })
                continue

            # Check position token balance before redemption
            asset_id = pos.get("asset", "")
            proxy_wallet = pos.get("proxyWallet", settings.funder_address)
            # Pre-redemption checks - check both pUSD and USDC.e
            collateral_balances_before = _check_collateral_balances(proxy_wallet)
            pusd_before = collateral_balances_before["pUSD"]
            usdc_e_before = collateral_balances_before["USDC.e"]
            if asset_id and asset_id != "N/A":
                print(f'[redeem] ===== CHECKING TOKEN LOCATIONS =====')
                balances = _check_token_balance_in_locations(asset_id, proxy_wallet, settings.funder_address)
                balance_before = balances.get("proxy_wallet", 0)
                print(f'[redeem] ====================================')
            
            label = f"redeem {cid[:12]}"
            tx_result = _execute_with_retry(client, txn, label)
            
            # Check position token balance after redemption
            alt_tried = False
            usdc_e_tried = False
            if asset_id and asset_id != "N/A":
                balance_after = _check_position_token_balance(asset_id, proxy_wallet)
                print(f'[redeem] Position token balance AFTER: {balance_after}')
                collateral_balances_after = _check_collateral_balances(proxy_wallet)
                pusd_after = collateral_balances_after["pUSD"]
                usdc_e_after = collateral_balances_after["USDC.e"]
                
                if balance_after < balance_before:
                    print(f'[redeem] ✅ Position tokens burned! Amount: {balance_before - balance_after}')
                    # Determine which collateral was received
                    pusd_change = pusd_after - pusd_before
                    usdc_e_change = usdc_e_after - usdc_e_before
                    if pusd_change > 0:
                        print(f'[redeem] pUSD received: {pusd_change} (was {pusd_before}, now {pusd_after})')
                    if usdc_e_change > 0:
                        print(f'[redeem] USDC.e received: {usdc_e_change} (was {usdc_e_before}, now {usdc_e_after})')
                else:
                    print(f'[redeem] ❌ Token balance unchanged - redemption failed. Check payout vector vs indexSets.')
                    print(f'[redeem]    Tried indexSets={index_sets} with pUSD, but position still has {balance_after} tokens')
                    print(f'[redeem]    Collateral unchanged: pUSD={pusd_before}->{pusd_after}, USDC.e={usdc_e_before}->{usdc_e_after}')
                    
                    # Retry with alternative indexSets if available
                    if alt_index_sets and not alt_tried:
                        alt_tried = True
                        print(f'[redeem] 🔄 RETRYING with alternative indexSets={alt_index_sets} (was {index_sets})')
                        
                        # Build alternative redeem transaction with pUSD
                        alt_args = eth_encode(
                            ["address", "bytes32", "bytes32", "uint256[]"],
                            [USDC_ADDRESS, parent_collection, condition_bytes, alt_index_sets],
                        )
                        alt_txn = SafeTransaction(
                            to=CTF_ADDRESS,
                            operation=OperationType.Call,
                            data="0x" + (REDEEM_SELECTOR + alt_args).hex(),
                            value="0",
                        )
                        
                        alt_label = f"redeem_alt {cid[:12]}"
                        alt_result = _execute_with_retry(client, alt_txn, alt_label)
                        
                        # Check balance after retry
                        balance_after_alt = _check_position_token_balance(asset_id, proxy_wallet)
                        collateral_alt = _check_collateral_balances(proxy_wallet)
                        print(f'[redeem] Position token balance AFTER retry: {balance_after_alt}')
                        
                        if balance_after_alt < balance_after:
                            print(f'[redeem] ✅ Alternative indexSets worked! Tokens burned: {balance_after - balance_after_alt}')
                            tx_result = alt_result
                            balance_after = balance_after_alt
                        else:
                            print(f'[redeem] ❌ Alternative indexSets also failed. Position still has {balance_after_alt} tokens')
                    
                    # If still not redeemed, try with USDC.e as collateral
                    if balance_after >= balance_before and not usdc_e_tried:
                        usdc_e_tried = True
                        print(f'[redeem] 🔄 RETRYING with USDC.e collateral (was pUSD)')
                        
                        # Build redeem transaction with USDC.e
                        usdce_args = eth_encode(
                            ["address", "bytes32", "bytes32", "uint256[]"],
                            [USDC_E_ADDRESS, parent_collection, condition_bytes, index_sets],
                        )
                        usdce_txn = SafeTransaction(
                            to=CTF_ADDRESS,
                            operation=OperationType.Call,
                            data="0x" + (REDEEM_SELECTOR + usdce_args).hex(),
                            value="0",
                        )
                        
                        usdce_label = f"redeem_usdce {cid[:12]}"
                        usdce_result = _execute_with_retry(client, usdce_txn, usdce_label)
                        
                        # Check balance after USDC.e retry
                        balance_after_usdce = _check_position_token_balance(asset_id, proxy_wallet)
                        collateral_usdce = _check_collateral_balances(proxy_wallet)
                        print(f'[redeem] Position token balance AFTER USDC.e retry: {balance_after_usdce}')
                        
                        if balance_after_usdce < balance_after:
                            print(f'[redeem] ✅ USDC.e redemption worked! Tokens burned: {balance_after - balance_after_usdce}')
                            tx_result = usdce_result
                            balance_after = balance_after_usdce
                        else:
                            print(f'[redeem] ❌ USDC.e redemption also failed. Position still has {balance_after_usdce} tokens')
            
            redeemed += 1
            details.append({
                "status": "redeemed",
                "condition_id": cid,
                "market": market,
                "negativeRisk": neg_risk,
                "transaction_hash": tx_result.get("transaction_hash") if tx_result else None,
                "block_number": tx_result.get("block_number") if tx_result else None,
            })
            logger.info("Redeemed %s (tx_hash=%s)", market, tx_result.get("transaction_hash") if tx_result else "unknown")
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to redeem %s: %s", market, exc)
            details.append({
                "status": "error",
                "condition_id": cid,
                "market": market,
                "error": str(exc),
            })

    duration = time.time() - start_ts
    return {
        "success": True,
        "positions_found": len(positions),
        "redeemed": redeemed,
        "details": details,
        "duration_sec": round(duration, 3),
    }


def _cooldown_remaining(now: float) -> float:
    if _last_redeem_ts <= 0:
        return 0.0
    elapsed = now - _last_redeem_ts
    remaining = COOLDOWN_SECONDS - elapsed
    return max(0.0, remaining)


async def redeem_all_positions(force: bool = False) -> Dict[str, Any]:
    global _last_redeem_ts  # noqa: PLW0603
    async with _redeem_lock:
        now = time.time()
        remaining = _cooldown_remaining(now)
        if not force and remaining > 0:
            eta = datetime.utcnow() + timedelta(seconds=remaining)
            return {
                "success": False,
                "error": "Redeem cooldown active",
                "retry_after_sec": math.ceil(remaining),
                "next_window_at": eta.isoformat() + "Z",
                "cooldown": True,
            }

        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(None, _redeem_positions_sync)
            _last_redeem_ts = time.time()
            result.setdefault("cooldown_sec", COOLDOWN_SECONDS)
            return result
        except Exception as exc:  # noqa: BLE001
            logger.error("Redeem-all failed: %s", exc)
            return {"success": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Auto-redeem background task
# ---------------------------------------------------------------------------

_AUTO_REDEEM_TASK: Optional[asyncio.Task] = None
_AUTO_REDEEM_STOP_EVENT: Optional[asyncio.Event] = None


async def _auto_redeem_loop() -> None:
    """Background loop that runs redeem_all_positions periodically."""
    global _AUTO_REDEEM_STOP_EVENT
    if _AUTO_REDEEM_STOP_EVENT is None:
        _AUTO_REDEEM_STOP_EVENT = asyncio.Event()

    interval_sec = int(os.getenv("POLY_AUTO_REDEEM_INTERVAL_SEC", str(7 * 60)))
    logger.info("[auto_redeem] Starting auto-redeem loop (interval=%ds)", interval_sec)

    while not _AUTO_REDEEM_STOP_EVENT.is_set():
        try:
            logger.info("[auto_redeem] Running scheduled redeem-all...")
            result = await redeem_all_positions(force=False)
            redeemed = result.get("redeemed", 0)
            found = result.get("positions_found", 0)
            if result.get("success"):
                logger.info("[auto_redeem] Completed: %d/%d positions redeemed", redeemed, found)
            elif result.get("cooldown"):
                logger.info("[auto_redeem] Cooldown active, retry in %ds", result.get("retry_after_sec", interval_sec))
            else:
                logger.warning("[auto_redeem] Failed: %s", result.get("error"))
        except Exception as exc:
            logger.exception("[auto_redeem] Unexpected error in auto-redeem loop: %s", exc)

        # Wait for interval or until stop event is set
        try:
            await asyncio.wait_for(_AUTO_REDEEM_STOP_EVENT.wait(), timeout=interval_sec)
        except asyncio.TimeoutError:
            pass  # Normal interval timeout, continue loop

    logger.info("[auto_redeem] Auto-redeem loop stopped")


def start_auto_redeem() -> None:
    """Start the auto-redeem background task if not already running."""
    global _AUTO_REDEEM_TASK
    if _AUTO_REDEEM_TASK is not None and not _AUTO_REDEEM_TASK.done():
        logger.info("[auto_redeem] Already running")
        return
    _AUTO_REDEEM_TASK = asyncio.create_task(_auto_redeem_loop())
    logger.info("[auto_redeem] Started background task")


def stop_auto_redeem() -> None:
    """Stop the auto-redeem background task."""
    global _AUTO_REDEEM_TASK, _AUTO_REDEEM_STOP_EVENT
    if _AUTO_REDEEM_STOP_EVENT is not None:
        _AUTO_REDEEM_STOP_EVENT.set()
    if _AUTO_REDEEM_TASK is not None:
        _AUTO_REDEEM_TASK.cancel()
        logger.info("[auto_redeem] Stopping background task...")
