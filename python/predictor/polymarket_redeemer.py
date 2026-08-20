"""
Polymarket Auto-Redeem Module.

Redeems resolved Polymarket positions through the gasless relayer.
Supports POLY_PROXY (1), POLY_GNOSIS_SAFE (2), and DEPOSIT_WALLET (3) signature types.

Env vars (reads from .env):
  POLY_PRIVATE_KEY            - EOA private key for signing
  POLY_FUNDER                 - Funder / deposit wallet address
  POLY_SIGNATURE_TYPE         - 0=EOA, 1=POLY_PROXY, 2=GNOSIS_SAFE, 3=DEPOSIT_WALLET
  POLY_BUILDER_API_KEY        - Builder relayer API key
  POLY_BUILDER_API_SECRET     - Builder relayer API secret
  POLY_BUILDER_API_PASSPHRASE - Builder relayer API passphrase
  POLY_RELAYER_HOST           - Relayer URL (default: https://relayer-v2.polymarket.com)
  POLY_CHAIN_ID               - Chain ID (default: 137)
  POLY_REDEEM_COOLDOWN_SEC    - Cooldown between redeem runs (default: 300)
  POLY_AUTO_REDEEM_INTERVAL_SEC - Auto-redeem loop interval (default: 420)
  POLY_REDEEM_MAX_POSITIONS   - Max positions per run (default: 30)
"""

import asyncio
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
load_dotenv()

import requests
from eth_abi import encode as eth_encode
from eth_utils import keccak

from py_builder_relayer_client.client import RelayClient
from py_builder_relayer_client.models import OperationType, RelayerTxType, SafeTransaction
from py_builder_signing_sdk.config import BuilderApiKeyCreds, BuilderConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PUSD_ADDRESS = "0xC011a7E12a19f7B1f670d46F03B03f3342E82DFB"
USDC_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
USDC_E_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
CTF_ADDRESS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
NEG_RISK_ADAPTER = "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296"

# V2 collateral adapters: redeem positions AND wrap USDC.e -> pUSD in one tx
CTF_COLLATERAL_ADAPTER = "0xAdA100Db00Ca00073811820692005400218FcE1f"
NEG_RISK_COLLATERAL_ADAPTER = "0xadA2005600Dec949baf300f4C6120000bDB6eAab"

# Collateral Onramp: wraps USDC.e -> pUSD (1:1, no fee)
COLLATERAL_ONRAMP = "0x93070a847efEf7F70739046A929D47a521F5B8ee"

POLYGON_RPC_ENDPOINTS = [
    "https://polygon-bor.publicnode.com",
    "https://1rpc.io/matic",
    "https://polygon.drpc.org",
]

REDEEM_SELECTOR = keccak(text="redeemPositions(address,bytes32,bytes32,uint256[])")[:4]
NEG_RISK_REDEEM_SELECTOR = keccak(text="redeemPositions(bytes32,uint256[])")[:4]
PAYOUT_NUMERATOR_SELECTOR = keccak(text="payoutNumerators(bytes32,uint256)")[:4]
PAYOUT_DENOMINATOR_SELECTOR = keccak(text="payoutDenominator(bytes32)")[:4]
WRAP_SELECTOR = keccak(text="wrap(address,address,uint256)")[:4]
APPROVE_SELECTOR = keccak(text="approve(address,uint256)")[:4]

ERC1155_BALANCE_SELECTOR = bytes.fromhex("00fdd58e")  # balanceOf(address)
ERC20_BALANCE_SELECTOR = keccak(text="balanceOf(address)")[:4]

DATA_API_URL = "https://data-api.polymarket.com/positions"
DEFAULT_RELAYER_URL = "https://relayer-v2.polymarket.com"

RELAYER_RETRY_WAIT_SEC = int(os.getenv("POLY_RELAYER_RETRY_WAIT_SEC", "60"))
COOLDOWN_SECONDS = int(os.getenv("POLY_REDEEM_COOLDOWN_SEC", "300"))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("poly_redeemer")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[%(name)s %(levelname)s %(asctime)s] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

_redeem_lock = asyncio.Lock()
_last_redeem_ts: float = 0.0

_AUTO_REDEEM_TASK: Optional[asyncio.Task] = None
_AUTO_REDEEM_STOP_EVENT: Optional[asyncio.Event] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _call_polygon_rpc(payload: dict) -> Optional[dict]:
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

    missing: List[str] = []
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
        raise RuntimeError("Missing required env vars for redemption: " + ", ".join(missing))

    try:
        signature_type = int(sig_raw)
    except (TypeError, ValueError):
        raise ValueError("POLY_SIGNATURE_TYPE must be an integer (0, 1, 2, or 3)")

    try:
        chain_id = int(chain_raw)
    except (TypeError, ValueError):
        raise ValueError("POLY_CHAIN_ID must be an integer")

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
    if not cid:
        return ""
    if not cid.startswith("0x"):
        cid = "0x" + cid
    return cid


def _parse_date(date_val) -> Optional[datetime]:
    if not date_val:
        return None
    if isinstance(date_val, datetime):
        return date_val
    try:
        return datetime.fromisoformat(str(date_val).replace("Z", ""))
    except Exception:
        return None


def _fetch_redeemable_positions(funder: str, max_positions: int) -> List[dict]:
    print(f"[redeem] fetching redeemable positions for {funder} ------------------------------------")
    print(f"[redeem] Will sort by date and return last {max_positions} positions")

    params = {
        "user": funder,
        "redeemable": "true",
        "sizeThreshold": 0,
        "limit": max(50, max_positions),
        "sortBy": "TITLE",
        "sortDirection": "DESC",
    }

    for attempt in range(2):
        resp = requests.get(DATA_API_URL, params=params, timeout=15)
        if resp.status_code in (429, 1015):
            if attempt == 0:
                logger.warning("Data API rate limited (status=%s). Waiting %ss before retrying...",
                               resp.status_code, RELAYER_RETRY_WAIT_SEC)
                time.sleep(RELAYER_RETRY_WAIT_SEC)
                continue
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, list):
            raise RuntimeError(f"Unexpected positions payload: {data}")

        # Filter: size > 0 (all active non-zero positions)
        filtered = [p for p in data if float(p.get("size", 0) or 0) > 0]
        print(f"[redeem] After filtering size>0: {len(filtered)}/{len(data)} positions remain")

        # Debug: print filtered positions JSON
        print("[redeem] ===== FILTERED POSITIONS JSON =====")
        for i, p in enumerate(filtered):
            pos_json = json.dumps(p, default=str)
            if i < 5:
                print(f"[redeem] Position JSON:\n{pos_json}")
            elif i == 5:
                print(f"[redeem] ... and {len(filtered) - 5} more positions")
        print("[redeem] ===== END FILTERED POSITIONS JSON =====")

        # Sort by date (newest first)
        def get_date_key(p):
            d = _parse_date(p.get("endDate") or p.get("end_date"))
            return d or datetime.min

        sorted_positions = sorted(filtered, key=get_date_key, reverse=True)
        print(f"[redeem] Sorted by date (newest first)")

        # Take last N (oldest N among the sorted)
        limited = sorted_positions[:max_positions] if max_positions > 0 else sorted_positions
        print(f"[redeem] Taking last {max_positions} positions: {len(limited)} positions selected")

        for pos in limited:
            cid = pos.get("conditionId") or pos.get("condition_id") or "N/A"
            title = pos.get("title") or pos.get("question") or "Unknown"
            end_date = pos.get("endDate") or pos.get("end_date") or "?"
            size = pos.get("size", 0)
            current_value = pos.get("currentValue", 0)
            print(f"  - {title[:35]:<35} | endDate={str(end_date)[:10]} | size={size:>8} | currentValue={current_value}")

        return limited

    raise RuntimeError("Data API rate limit exceeded while fetching positions")


def _build_client(settings: RedeemerSettings) -> RelayClient:
    # sig_type 1 (POLY_PROXY) uses PROXY relayer mode
    # sig_type 0 (EOA) and 2 (GNOSIS_SAFE) use SAFE relayer mode
    # sig_type 3 (DEPOSIT_WALLET) uses execute_deposit_wallet_batch (no relay_tx_type needed)
    if settings.signature_type == 1:
        wallet_type = RelayerTxType.PROXY
    elif settings.signature_type == 3:
        wallet_type = None  # deposit wallet uses execute_deposit_wallet_batch
    else:
        wallet_type = RelayerTxType.SAFE

    builder_config = BuilderConfig(
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
        rpc_url=POLYGON_RPC_ENDPOINTS[0] if settings.signature_type == 3 else None,
    )


def _neg_risk_amounts(pos: dict) -> List[int]:
    size_raw = int(float(pos.get("size", 0) or 0) * 1_000_000)
    outcome_idx = int(float(pos.get("outcomeIndex", 0) or 0))
    outcomes = int(pos.get("outcomeCount", 0) or 0)
    amounts = [0] * max(outcomes, outcome_idx + 1)
    amounts[outcome_idx] = size_raw
    return amounts


def _execute_with_retry(client: RelayClient, txn, label: str, settings: RedeemerSettings = None):
    """Execute a relayer transaction, retrying once on rate limit."""
    use_deposit_wallet = settings is not None and settings.signature_type == 3

    if use_deposit_wallet:
        from py_builder_relayer_client.models import DepositWalletCall, TransactionType
        call = DepositWalletCall(
            target=txn.to,
            value="0",
            data=txn.data,
        )
        # Get nonce and deadline from relayer
        signer_addr = client.signer.address()
        payload = client.get_relay_payload(signer_addr, TransactionType.WALLET.value)
        nonce = str(payload.get("nonce", "0"))
        deadline = str(int(time.time()) + 3600)  # 1 hour deadline
        wallet_addr = settings.funder_address

        try:
            resp = client.execute_deposit_wallet_batch([call], wallet_addr, nonce, deadline)
            return resp
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            if status in (429, 1015):
                logger.warning("Relayer rate limited (HTTP %s), waiting %ss...", status, RELAYER_RETRY_WAIT_SEC)
                time.sleep(RELAYER_RETRY_WAIT_SEC)
                payload = client.get_relay_payload(signer_addr, TransactionType.WALLET.value)
                nonce = str(payload.get("nonce", "0"))
                resp = client.execute_deposit_wallet_batch([call], wallet_addr, nonce, deadline)
                return resp
            raise
    else:
        try:
            resp = client.execute([txn], label)
            resp.wait()
            return resp
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            if status in (429, 1015):
                logger.warning("Relayer rate limited (HTTP %s), waiting %ss...", status, RELAYER_RETRY_WAIT_SEC)
                time.sleep(RELAYER_RETRY_WAIT_SEC)
                resp = client.execute([txn], label)
                resp.wait()
                return resp
            raise


def _check_position_token_balance(token_id: str, wallet: str) -> int:
    """Check ERC1155 balanceOf for a position token."""
    args = eth_encode(["address"], [wallet])
    payload = {
        "jsonrpc": "2.0", "method": "eth_call", "id": 1,
        "params": [
            {"to": CTF_ADDRESS, "data": "0x" + (ERC1155_BALANCE_SELECTOR + eth_encode(["uint256"], [int(token_id)]) + args).hex()},
            "latest",
        ],
    }
    result = _call_polygon_rpc(payload)
    if result and "result" in result:
        return int(result["result"], 16)
    return 0


def _check_token_balance_in_locations(token_id: str, proxy_wallet: str, funder: str) -> Dict[str, int]:
    """Check token balance in both proxy wallet and funder address."""
    balances = {}
    for label, addr in [("proxy_wallet", proxy_wallet), ("funder", funder)]:
        if addr:
            balances[label] = _check_position_token_balance(token_id, addr)
    return balances


def _get_condition_payouts(condition_id: str, outcome_count: int) -> List[int]:
    """Read payoutNumerators from the CTF contract."""
    condition_bytes = bytes.fromhex(condition_id[2:])
    payouts = []
    for i in range(outcome_count):
        args = eth_encode(["bytes32", "uint256"], [condition_bytes, i])
        payload = {
            "jsonrpc": "2.0", "method": "eth_call", "id": 1,
            "params": [
                {"to": CTF_ADDRESS, "data": "0x" + (PAYOUT_NUMERATOR_SELECTOR + args).hex()},
                "latest",
            ],
        }
        result = _call_polygon_rpc(payload)
        if result and "result" in result:
            payouts.append(int(result["result"], 16))
        else:
            payouts.append(0)
    return payouts


def _check_pusd_balance(wallet: str) -> int:
    """Check pUSD (ERC20) balance."""
    args = eth_encode(["address"], [wallet])
    payload = {
        "jsonrpc": "2.0", "method": "eth_call", "id": 1,
        "params": [{"to": PUSD_ADDRESS, "data": "0x" + (ERC20_BALANCE_SELECTOR + args).hex()}, "latest"],
    }
    result = _call_polygon_rpc(payload)
    if result and "result" in result:
        return int(result["result"], 16)
    return 0


def _check_usdc_e_balance(wallet: str) -> int:
    """Check USDC.e (ERC20) balance."""
    args = eth_encode(["address"], [wallet])
    payload = {
        "jsonrpc": "2.0", "method": "eth_call", "id": 1,
        "params": [{"to": USDC_E_ADDRESS, "data": "0x" + (ERC20_BALANCE_SELECTOR + args).hex()}, "latest"],
    }
    result = _call_polygon_rpc(payload)
    if result and "result" in result:
        return int(result["result"], 16)
    return 0


def _check_collateral_balances(wallet: str) -> Dict[str, int]:
    """Check both pUSD and USDC.e balances."""
    return {
        "pUSD": _check_pusd_balance(wallet),
        "USDC.e": _check_usdc_e_balance(wallet),
    }


def _wrap_usdc_e_to_pusd(client: RelayClient, amount: int, wallet: str, settings: RedeemerSettings):
    """Wrap USDC.e to pUSD via the Collateral Onramp using the relayer.
    Bundles approve(ONRAMP, amount) + wrap(USDC.e, wallet, amount) as two calls."""
    try:
        from py_builder_relayer_client.models import DepositWalletCall, TransactionType

        # Call 1: approve ONRAMP to spend USDC.e
        approve_args = eth_encode(["address", "uint256"], [COLLATERAL_ONRAMP, amount])
        approve_call = DepositWalletCall(
            target=USDC_E_ADDRESS,
            value="0",
            data="0x" + (APPROVE_SELECTOR + approve_args).hex(),
        )

        # Call 2: wrap USDC.e -> pUSD via Onramp
        wrap_args = eth_encode(["address", "address", "uint256"], [USDC_E_ADDRESS, wallet, amount])
        wrap_call = DepositWalletCall(
            target=COLLATERAL_ONRAMP,
            value="0",
            data="0x" + (WRAP_SELECTOR + wrap_args).hex(),
        )

        # Get nonce from relayer
        signer_addr = client.signer.address()
        payload = client.get_relay_payload(signer_addr, TransactionType.WALLET.value)
        nonce = str(payload.get("nonce", "0"))
        deadline = str(int(time.time()) + 3600)

        resp = client.execute_deposit_wallet_batch(
            [approve_call, wrap_call], wallet, nonce, deadline
        )
        return {"status": "ok", "result": resp}
    except Exception as exc:
        logger.error("[redeem] Wrap failed: %s", exc)
        return {"status": "error", "error": str(exc)}


# ---------------------------------------------------------------------------
# Core redeem logic (sync — called from executor)
# ---------------------------------------------------------------------------

def _redeem_positions_sync() -> dict:
    """Synchronous redeem-all implementation."""
    settings = _load_settings()
    client = _build_client(settings)
    max_positions = int(os.getenv("POLY_REDEEM_MAX_POSITIONS", "30"))
    start_ts = time.time()

    positions = _fetch_redeemable_positions(settings.funder_address, max_positions)
    if not positions:
        logger.info("Redeem-all: no redeemable positions found")
        duration = round(time.time() - start_ts, 2)
        return {"success": True, "positions_found": 0, "redeemed": 0, "details": [], "duration_sec": duration, "message": "No redeemable positions"}

    logger.info("Redeem-all: found %d redeemable positions", len(positions))

    details: List[dict] = []
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
        except (ValueError, IndexError):
            details.append({"status": "error", "reason": "invalid_condition_id", "cid": cid})
            continue

        if neg_risk is True:
            # Neg-risk market: use NegRiskCollateralAdapter (redeems AND wraps to pUSD)
            # Same 4-arg signature as standard, only conditionId is used
            amounts = _neg_risk_amounts(pos)
            args = eth_encode(
                ["address", "bytes32", "bytes32", "uint256[]"],
                [USDC_ADDRESS, b"\x00" * 32, condition_bytes, [1, 2]],
            )
            txn = SafeTransaction(
                to=NEG_RISK_COLLATERAL_ADAPTER,
                operation=OperationType.Call,
                data="0x" + (REDEEM_SELECTOR + args).hex(),
                value="0",
            )
            print(f"NEGRISK=TRUE creating transaction REDEEM via collateral adapter: to={NEG_RISK_COLLATERAL_ADAPTER}")

        elif neg_risk is False:
            # Standard market: use CTF with 4-arg redeemPositions
            outcome_idx = int(float(pos.get("outcomeIndex", -1) or -1))
            outcome_label = pos.get("outcome", "UNKNOWN")

            # Determine index sets based on outcome
            if outcome_idx == 0:
                direction = "UP"
                index_sets = [1, 2]
                alt_index_sets = [2, 1]
            elif outcome_idx == 1:
                direction = "DOWN"
                index_sets = [1, 2]
                alt_index_sets = [2, 1]
            else:
                direction = "UNKNOWN"
                index_sets = [1, 2]
                alt_index_sets = [2, 1]

            title_lower = (pos.get("title") or "").lower()
            if "down" in title_lower:
                direction = "DOWN"
            elif "up" in title_lower:
                direction = "UP"

            print(f"[redeem] Position direction: outcome_idx={outcome_idx}, outcome=\"{outcome_label}\"")
            print(f"[redeem]   Standard mapping: {direction} -> indexSets={index_sets}")
            print(f"[redeem]   Alternative mapping: {direction} -> indexSets={alt_index_sets}")

            # Check on-chain payouts
            outcome_count = int(pos.get("outcomeCount", 2) or 2)
            payouts = _get_condition_payouts(cid, outcome_count)
            print(f"[redeem] Winning bitsets reported on-chain: {payouts}")

            # Override index sets based on payout vector if needed
            winning_sets = []
            for i, val in enumerate(payouts):
                if val > 0:
                    bitset = 1 << i
                    winning_sets.append(bitset)

            if winning_sets and winning_sets != index_sets:
                print(f"[redeem] WARNING: Overriding indexSets -> {winning_sets} based on payout vector {payouts}")
                index_sets = winning_sets

            # Check if market is actually resolved
            cur_price = float(pos.get("curPrice", 0) or 0)
            redeemable_flag = pos.get("redeemable", False)
            if cur_price != 1.0 and not redeemable_flag:
                print(f"[redeem] WARNING: Market NOT resolved! curPrice={cur_price}, expected 1.0 for winning outcome")
                print(f"[redeem] Redemption will fail - market must be resolved before redeeming")
                details.append({"status": "error", "reason": "market_not_resolved", "cid": cid, "market": market})
                continue

            asset_id = pos.get("asset", "N/A")
            proxy_wallet = pos.get("proxyWallet", settings.funder_address)

            print(f"[redeem] Position token ID (asset): {asset_id}")
            print(f"[redeem] Proxy wallet holding tokens: {proxy_wallet}")
            print(f"[redeem] Condition ID: {cid}")
            print(f"[redeem] Collateral Adapter: {CTF_COLLATERAL_ADAPTER}")

            # Check collateral balances before
            collateral_balances_before = _check_collateral_balances(proxy_wallet)
            pusd_before = collateral_balances_before.get("pUSD", 0)
            usdc_e_before = collateral_balances_before.get("USDC.e", 0)
            print(f"[redeem] Collateral tokens: pUSD={pusd_before}, USDC.e={usdc_e_before}")

            print(f"[redeem] >>> FINAL INDEX_SETS BEFORE ENCODING: {index_sets}")
            print(f"[redeem] >>> Direction: {direction}, Winning sets: {winning_sets}")
            print(f"[redeem] >>> Condition: {cid}")

            args = eth_encode(
                ["address", "bytes32", "bytes32", "uint256[]"],
                [USDC_ADDRESS, b"\x00" * 32, condition_bytes, index_sets],
            )
            args_hex = args.hex()
            array_len_hex = args_hex[-64:] if len(args_hex) >= 64 else "?"
            print(f"[redeem] Function selector: 0x{REDEEM_SELECTOR.hex()}")
            print(f"[redeem] Encoded args: 0x{args_hex}")
            print(f"[redeem] Full calldata: 0x{REDEEM_SELECTOR.hex()}{args_hex}")
            print(f"[redeem] EXTRACTED FROM ENCODED: array_len=0x{array_len_hex}")

            # Use CTF collateral adapter (redeems AND wraps USDC.e -> pUSD in one tx)
            txn = SafeTransaction(
                to=CTF_COLLATERAL_ADAPTER,
                operation=OperationType.Call,
                data="0x" + (REDEEM_SELECTOR + args).hex(),
                value="0",
            )
            print(f"NEGRISK=FALSE creating transaction REDEEM via collateral adapter ({direction}): to={CTF_COLLATERAL_ADAPTER}")

        else:
            print(f"NEGRISK=UNKNOWN creating safe transaction REDEEM")
            details.append({"status": "error", "reason": "unknown_market_type", "cid": cid, "neg_risk": neg_risk})
            continue

        # Execute redemption
        try:
            label = f"redeem {cid[:12]}"
            print(f"[redeem] ===== CHECKING TOKEN LOCATIONS =====")
            balances = _check_token_balance_in_locations(asset_id, proxy_wallet, settings.funder_address) if neg_risk is False else {}
            balance_before = balances.get("proxy_wallet", 0)
            print(f"[redeem] {'=' * 36}")

            tx_result = _execute_with_retry(client, txn, label, settings)

            # Check if tokens were burned
            if neg_risk is False and asset_id != "N/A":
                balance_after = _check_position_token_balance(asset_id, proxy_wallet)
                print(f"[redeem] Position token balance AFTER: {balance_after}")
                collateral_balances_after = _check_collateral_balances(proxy_wallet)
                pusd_after = collateral_balances_after.get("pUSD", 0)
                usdc_e_after = collateral_balances_after.get("USDC.e", 0)
                if balance_after < balance_before or balance_after == 0:
                    print(f"[redeem] OK: Position tokens burned! Amount: {balance_before - balance_after}")
                    print(f"[redeem] pUSD received: {pusd_after} (was {pusd_before}, now {pusd_after})")
                    print(f"[redeem] USDC.e received: {usdc_e_after}")
                else:
                    print(f"[redeem] FAIL: Token balance unchanged - redemption failed. Check payout vector vs indexSets.")
                    print(f"[redeem]    Tried indexSets={index_sets}, but position still has {balance_after} tokens")
                    print(f"[redeem]    Collateral: pUSD={pusd_before}->{pusd_after}, USDC.e={usdc_e_after}")

                    # Retry with alternative index sets
                    alt_tried = False
                    if neg_risk is False and alt_index_sets != index_sets:
                        print(f"[redeem] RETRY: Retrying with alternative indexSets={alt_index_sets}")
                        alt_args = eth_encode(
                            ["address", "bytes32", "bytes32", "uint256[]"],
                            [USDC_ADDRESS, b"\x00" * 32, condition_bytes, alt_index_sets],
                        )
                        alt_txn = SafeTransaction(
                            to=CTF_COLLATERAL_ADAPTER,
                            operation=OperationType.Call,
                            data="0x" + (REDEEM_SELECTOR + alt_args).hex(),
                            value="0",
                        )
                        alt_label = f"redeem_alt {cid[:12]}"
                        try:
                            alt_result = _execute_with_retry(client, alt_txn, alt_label, settings)
                            balance_after_alt = _check_position_token_balance(asset_id, proxy_wallet)
                            print(f"[redeem] Position token balance AFTER retry: {balance_after_alt}")
                            if balance_after_alt < balance_before:
                                print(f"[redeem] OK: Alternative indexSets worked! Tokens burned: {balance_before - balance_after_alt}")
                                tx_result = alt_result
                                alt_tried = True
                            else:
                                print(f"[redeem] FAIL: Alternative indexSets also failed. Position still has {balance_after_alt} tokens")
                        except Exception as alt_exc:
                            print(f"[redeem] Alternative redeem failed: {alt_exc}")

                    if not alt_tried:
                        details.append({"status": "error", "reason": "tokens_not_burned", "cid": cid, "market": market})
                        continue

            redeemed += 1
            tx_hash = getattr(tx_result, "transaction_hash", None) or getattr(tx_result, "tx_hash", None) or "unknown"
            block_number = getattr(tx_result, "block_number", None) or "unknown"
            logger.info("Redeemed %s (tx_hash=%s)", market, tx_hash)
            details.append({
                "status": "redeemed",
                "cid": cid,
                "market": market,
                "tx_hash": tx_hash,
                "block_number": block_number,
            })

        except Exception as exc:
            logger.error("Failed to redeem %s: %s", market, exc)
            details.append({"status": "error", "reason": str(exc), "cid": cid, "market": market})

    # Wrap USDC.e to pUSD if applicable
    if settings.signature_type in (1, 3) or hasattr(settings, "funder_address"):
        try:
            target_wallet = settings.funder_address
            print(f"[redeem] ===== WRAPPING USDC.e -> pUSD =====")
            print(f"[redeem] Target wallet for wrap: {target_wallet[:10]}...")
            usdc_e_balance = _check_usdc_e_balance(target_wallet)
            print(f"[redeem] USDC.e balance before wrap: {usdc_e_balance / 1e6:.6f}")
            if usdc_e_balance > 0:
                wrap_result = _wrap_usdc_e_to_pusd(client, usdc_e_balance, target_wallet, settings)
                if wrap_result.get("status") == "ok":
                    final_pusd = _check_pusd_balance(target_wallet)
                    final_usdc_e = _check_usdc_e_balance(target_wallet)
                    print(f"[redeem] pUSD after wrap: {final_pusd / 1e6:.6f}")
                    print(f"[redeem] USDC.e after wrap: {final_usdc_e / 1e6:.6f}")
                else:
                    print(f"[redeem] WARNING: Wrap failed: {wrap_result.get('error')}")
            else:
                print(f"[redeem] No USDC.e to wrap (balance: {usdc_e_balance / 1e6:.6f})")
            print(f"[redeem] ===== WRAP COMPLETE =====")
        except Exception as exc:
            logger.error("[redeem] Wrap error: %s", exc)
            details.append({"status": "wrap", "error": str(exc)})

    duration = round(time.time() - start_ts, 2)
    result = {
        "success": True,
        "positions_found": len(positions),
        "redeemed": redeemed,
        "details": details,
        "duration_sec": duration,
    }
    return result


def _cooldown_remaining(now: float) -> float:
    if _last_redeem_ts <= 0:
        return 0.0
    elapsed = now - _last_redeem_ts
    remaining = COOLDOWN_SECONDS - elapsed
    return max(0.0, remaining)


# ---------------------------------------------------------------------------
# Public async API
# ---------------------------------------------------------------------------

async def redeem_all_positions(force: bool = False) -> dict:
    """Redeem all redeemable positions. Respects cooldown unless force=True."""
    global _last_redeem_ts
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
            return result
        except Exception as exc:
            logger.error("Redeem-all failed: %s", exc)
            return {"success": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Auto-redeem background loop
# ---------------------------------------------------------------------------

async def _auto_redeem_loop():
    """Background loop that runs redeem_all_positions periodically."""
    global _AUTO_REDEEM_STOP_EVENT
    if _AUTO_REDEEM_STOP_EVENT is None:
        _AUTO_REDEEM_STOP_EVENT = asyncio.Event()

    interval_sec = int(os.getenv("POLY_AUTO_REDEEM_INTERVAL_SEC", "300"))
    logger.info("[auto_redeem] Starting auto-redeem loop (interval=%ds)", interval_sec)

    while not _AUTO_REDEEM_STOP_EVENT.is_set():
        logger.info("[auto_redeem] Running scheduled redeem-all...")
        try:
            result = await redeem_all_positions(force=False)
            redeemed = result.get("redeemed", 0)
            found = result.get("positions_found", 0)
            if result.get("success"):
                logger.info("[auto_redeem] Completed: %d/%d positions redeemed", redeemed, found)
            elif result.get("cooldown"):
                logger.warning("[auto_redeem] Cooldown active, retry in %ds", result.get("retry_after_sec", interval_sec))
            else:
                logger.warning("[auto_redeem] Failed: %s", result.get("error", "unknown"))
        except Exception as exc:
            logger.exception("[auto_redeem] Unexpected error in auto-redeem loop: %s", exc)

        try:
            await asyncio.wait_for(_AUTO_REDEEM_STOP_EVENT.wait(), timeout=interval_sec)
        except asyncio.TimeoutError:
            pass

    logger.info("[auto_redeem] Auto-redeem loop stopped")


def start_auto_redeem():
    """Start the auto-redeem background task if not already running."""
    global _AUTO_REDEEM_TASK, _AUTO_REDEEM_STOP_EVENT
    if _AUTO_REDEEM_TASK is not None and not _AUTO_REDEEM_TASK.done():
        logger.info("[auto_redeem] Already running")
        return
    _AUTO_REDEEM_STOP_EVENT = asyncio.Event()
    _AUTO_REDEEM_TASK = asyncio.create_task(_auto_redeem_loop())
    logger.info("[auto_redeem] Started background task")


def stop_auto_redeem():
    """Stop the auto-redeem background task."""
    global _AUTO_REDEEM_TASK, _AUTO_REDEEM_STOP_EVENT
    if _AUTO_REDEEM_STOP_EVENT is not None:
        _AUTO_REDEEM_STOP_EVENT.set()
    if _AUTO_REDEEM_TASK is not None:
        _AUTO_REDEEM_TASK.cancel()
    logger.info("[auto_redeem] Stopping background task...")
