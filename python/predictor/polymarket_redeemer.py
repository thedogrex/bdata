import asyncio
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

USDC_ADDRESS = "0xC011a7E12a19f7B1f670d46F03B03f3342E82DFB"
CTF_ADDRESS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
NEG_RISK_ADAPTER = "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296"

REDEEM_SELECTOR = keccak(text="redeemPositions(address,bytes32,bytes32,uint256[])")[:4]
NEG_RISK_REDEEM_SELECTOR = keccak(text="redeemPositions(bytes32,uint256[])")[:4]

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


def _fetch_redeemable_positions(funder: str) -> List[Dict[str, Any]]:
    params = {"user": funder, "redeemable": "true", "sizeThreshold": 1}
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
        filtered = [p for p in data if float(p.get("size", 0) or 0) > 0]
        return filtered
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


def _execute_with_retry(client: RelayClient, txn: SafeTransaction, label: str) -> None:
    for attempt in range(2):
        try:
            resp = client.execute([txn], label)
            resp.wait()
            return
        except Exception as exc:  # noqa: BLE001
            status = getattr(exc, "status_code", None)
            if status in (429, 1015) and attempt == 0:
                logger.warning("Relayer rate limited (status=%s). Waiting %ss before retrying...", status, RELAYER_RETRY_WAIT_SEC)
                time.sleep(RELAYER_RETRY_WAIT_SEC)
                continue
            raise


def _redeem_positions_sync() -> Dict[str, Any]:
    settings = _load_settings()
    client = _build_client(settings)

    start_ts = time.time()
    positions = _fetch_redeemable_positions(settings.funder_address)
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
            elif neg_risk is False:
                args = eth_encode(
                    ["address", "bytes32", "bytes32", "uint256[]"],
                    [USDC_ADDRESS, b"\x00" * 32, condition_bytes, [1, 2]],
                )
                txn = SafeTransaction(
                    to=CTF_ADDRESS,
                    operation=OperationType.Call,
                    data="0x" + (REDEEM_SELECTOR + args).hex(),
                    value="0",
                )
            else:
                details.append({
                    "status": "skipped",
                    "reason": "unknown_market_type",
                    "condition_id": cid,
                    "market": market,
                    "negativeRisk": neg_risk,
                })
                continue

            label = f"redeem {cid[:12]}"
            _execute_with_retry(client, txn, label)
            redeemed += 1
            details.append({
                "status": "redeemed",
                "condition_id": cid,
                "market": market,
                "negativeRisk": neg_risk,
            })
            logger.info("Redeemed %s", market)
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
