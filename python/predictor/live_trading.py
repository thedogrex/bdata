"""
Live trading service for Polymarket.

After a prediction is made, this module can:
 1. Buy the predicted outcome at the current snapshot price
 2. Record the order in poly_live_orders
 3. Track the position in poly_live_positions
 4. Report wallet summary for monitoring

All DB operations are async (aiomysql via DbProvider).
All CLOB operations are sync (py_clob_client) — run in executor.
"""

import asyncio
import json
import logging
from collections import defaultdict
from decimal import Decimal, ROUND_CEILING
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import app.config as config
from db import DbProvider
from predictor.poly_client import PolymarketClient

logger = logging.getLogger("live_trading")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(name)s %(levelname)s %(asctime)s] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(_h)

db = DbProvider()
trading_client = PolymarketClient()

MAX_LIMIT_PRICE_USD = 0.53
PRICE_RETRY_WINDOW_SEC = 180
PRICE_RETRY_INTERVAL_SEC = 10
MSK_UTC_OFFSET_HOURS = 3
MSK_UTC_OFFSET = timedelta(hours=MSK_UTC_OFFSET_HOURS)
MSK_UTC_OFFSET_SECONDS = int(MSK_UTC_OFFSET.total_seconds())
MSK_TZ_NAME = "+03:00"

_last_collateral_balance_usd: Optional[float] = None


def _set_cached_collateral_balance_usd(value: Optional[float]) -> None:
    global _last_collateral_balance_usd
    if value is None:
        return
    try:
        _last_collateral_balance_usd = float(value)
    except Exception:
        pass


def get_cached_collateral_balance_usd() -> Optional[float]:
    return _last_collateral_balance_usd


def _parse_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_fill_metrics(resp: Optional[Dict[str, Any]]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if not isinstance(resp, dict):
        return None, None, None
    taking = _parse_float(resp.get("takingAmount"))
    making = _parse_float(resp.get("makingAmount"))
    avg_cents: Optional[float] = None
    if taking is not None and taking > 0 and making is not None:
        net_spent = making
        avg_cents = (net_spent / taking) * 100.0
    return taking, making, avg_cents


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


async def refresh_collateral_balance(loop: Optional[asyncio.AbstractEventLoop] = None) -> Optional[float]:
    """Fetch collateral balance from Polymarket and update the cached value."""
    loop = loop or asyncio.get_event_loop()
    try:
        balance_allowance = await loop.run_in_executor(None, trading_client.get_balance_allowance)
    except Exception as exc:
        logger.warning("refresh_collateral_balance: failed to fetch balance_allowance: %s", exc)
        return None

    value = _extract_collateral_balance_usd(balance_allowance)
    if value is None:
        logger.debug("refresh_collateral_balance: unable to parse collateral balance from response")
    return value


def _extract_collateral_balance_usd(balance_allowance: Any) -> Optional[float]:
    """Best-effort parse collateral balance from CLOB get_balance_allowance response."""
    try:
        if not isinstance(balance_allowance, dict):
            return None
        coll = balance_allowance.get("collateral")
        if not coll:
            return None
        def _norm_usdc(x: Any) -> Optional[float]:
            try:
                if isinstance(x, str):
                    sx = x.strip()
                    if not sx:
                        return None
                    if sx.isdigit():
                        iv = int(sx)
                        return float(iv) / 1e6
                    fv = float(sx)
                    return float(fv)
                if isinstance(x, int):
                    return float(x) / 1e6
                if isinstance(x, float):
                    return float(x)
            except Exception:
                return None
            return None

        if isinstance(coll, dict):
            # Common patterns across SDK versions
            for k in (
                "balance",
                "availableBalance",
                "available_balance",
                "totalBalance",
                "total_balance",
                "amount",
                "value",
            ):
                if k in coll:
                    try:
                        v = _norm_usdc(coll[k])
                        if v is not None:
                            _set_cached_collateral_balance_usd(v)
                            return v
                    except Exception:
                        pass
            # sometimes nested
            for nest in ("balance", "available", "total"):
                v = coll.get(nest)
                if isinstance(v, (int, float, str)):
                    try:
                        nv = _norm_usdc(v)
                        if nv is not None:
                            _set_cached_collateral_balance_usd(nv)
                            return nv
                    except Exception:
                        pass
        # fallback: if collateral itself is numeric
        if isinstance(coll, (int, float, str)):
            nv = _norm_usdc(coll)
            if nv is not None:
                _set_cached_collateral_balance_usd(nv)
                return nv
    except Exception:
        return None
    return None


def compute_buy_amount_usd(
    bank_usd: float,
    bank_pct: float,
    min_usd: float,
    max_usd: float,
) -> float:
    """Compute buy amount as % of bank with min/max clamp."""
    try:
        bank_usd = float(bank_usd)
        bank_pct = float(bank_pct)
        min_usd = float(min_usd)
        max_usd = float(max_usd)
    except Exception:
        return float(min_usd)
    raw = bank_usd * bank_pct
    return _clamp(raw, min_usd, max_usd)


def _is_order_open(order: Any) -> bool:
    """Best-effort check if an order is still open / live."""
    if order is None:
        return False
    if isinstance(order, dict):
        st = str(order.get("status") or order.get("state") or "").lower()
        if st in ("matched", "filled", "canceled", "cancelled", "expired", "rejected"):
            return False
        if st in ("live", "open", "pending", "delayed", "partial", "partially_filled"):
            return True
        for k in ("remaining", "remaining_size", "remainingSize"):
            if k in order:
                try:
                    return float(order[k]) > 0
                except Exception:
                    pass
        # if we can't determine: be conservative and treat as open
        return True
    return True


async def _cancel_after_timeout(order_id: str, order_row_id: Any, timeout_sec: int = 60) -> None:
    """Cancel order after timeout if still open (best-effort)."""
    try:
        await asyncio.sleep(int(timeout_sec))
        loop = asyncio.get_event_loop()
        order = await loop.run_in_executor(None, lambda: trading_client.get_order(order_id))
        if not _is_order_open(order):
            return

        cancel_resp = await loop.run_in_executor(None, lambda: trading_client.cancel_order(order_id))
        try:
            await db.execute(
                "UPDATE poly_live_orders SET clob_status=%s, clob_response_json=%s, updated_at=NOW() WHERE id=%s",
                (
                    "canceled_timeout",
                    json.dumps({"order": order, "cancel": cancel_resp}, default=str),
                    order_row_id,
                ),
            )
        except Exception:
            pass
        logger.warning("Order %s canceled after timeout", order_id)
    except Exception as e:
        logger.error("cancel_after_timeout failed: %s", e)


async def _market_has_completed_buy(slug: str) -> bool:
    if not slug:
        return False
    try:
        row = await db.fetchone(
            "SELECT 1 FROM poly_live_orders WHERE slug=%s AND COALESCE(fill_shares,0) > 0 LIMIT 1",
            (slug,),
        )
        return bool(row)
    except Exception as exc:
        logger.error("market fill check failed for %s: %s", slug, exc)
        # Fail safe: avoid double-buy if unsure
        return True


async def _wait_for_price_within_threshold(
    slug: str,
    asset_id: str,
    price_threshold: float,
) -> Optional[float]:
    """Poll best ask every PRICE_RETRY_INTERVAL_SEC until it falls below threshold or timeout."""
    loop = asyncio.get_event_loop()
    deadline = loop.time() + PRICE_RETRY_WINDOW_SEC
    attempt = 0
    while loop.time() < deadline:
        await asyncio.sleep(PRICE_RETRY_INTERVAL_SEC)
        if await _market_has_completed_buy(slug):
            logger.info("Market %s already filled while waiting for better price", slug)
            return None
        try:
            refreshed = trading_client.get_best_ask(asset_id)
            if refreshed is None:
                continue
            attempt += 1
            current_price = float(refreshed)
            logger.info("  price retry #%d: %.4f", attempt, current_price)
            if current_price <= price_threshold:
                return current_price
        except Exception as exc:
            logger.debug("price retry fetch failed: %s", exc)
    return None

# ---------------------------------------------------------------------------
# Ensure tables exist (run once at startup)
# ---------------------------------------------------------------------------

_TABLES_SQL = [
    """
    CREATE TABLE IF NOT EXISTS `poly_live_orders` (
      `id` bigint NOT NULL AUTO_INCREMENT,
      `slug` varchar(255) NOT NULL,
      `asset_id` varchar(128) NOT NULL,
      `outcome_side` varchar(8) DEFAULT NULL,
      `side` varchar(8) NOT NULL,
      `order_type` varchar(16) NOT NULL DEFAULT 'FOK',
      `price` double NOT NULL,
      `amount` double NOT NULL,
      `clob_order_id` varchar(128) DEFAULT NULL,
      `clob_status` varchar(32) DEFAULT NULL,
      `clob_error_msg` text,
      `clob_response_json` json DEFAULT NULL,
      `prediction_batch_id` varchar(64) DEFAULT NULL,
      `template_id` int DEFAULT NULL,
      `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
      `updated_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
      PRIMARY KEY (`id`),
      KEY `idx_slug` (`slug`),
      KEY `idx_asset` (`asset_id`),
      KEY `idx_clob_order` (`clob_order_id`),
      KEY `idx_created` (`created_at`)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci
    """,
    """
    CREATE TABLE IF NOT EXISTS `poly_live_positions` (
      `id` bigint NOT NULL AUTO_INCREMENT,
      `slug` varchar(255) NOT NULL,
      `asset_id` varchar(128) NOT NULL,
      `outcome_side` varchar(8) DEFAULT NULL,
      `shares` double NOT NULL DEFAULT 0,
      `avg_price` double NOT NULL DEFAULT 0,
      `total_cost` double NOT NULL DEFAULT 0,
      `status` varchar(16) NOT NULL DEFAULT 'open',
      `resolved_outcome` varchar(16) DEFAULT NULL,
      `pnl` double DEFAULT NULL,
      `snapshot_price_cents` double DEFAULT NULL,
      `prediction_direction` varchar(8) DEFAULT NULL,
      `prediction_batch_id` varchar(64) DEFAULT NULL,
      `template_id` int DEFAULT NULL,
      `opened_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
      `closed_at` datetime DEFAULT NULL,
      `updated_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
      PRIMARY KEY (`id`),
      UNIQUE KEY `uq_position` (`slug`, `asset_id`, `status`),
      KEY `idx_slug` (`slug`),
      KEY `idx_status` (`status`),
      KEY `idx_opened` (`opened_at`)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci
    """,
]


async def ensure_tables():
    """Create trading tables if they don't exist."""
    for sql in _TABLES_SQL:
        try:
            await db.execute(sql)
        except Exception as e:
            logger.warning("ensure_tables: %s", e)


# ---------------------------------------------------------------------------
# Core: buy after prediction
# ---------------------------------------------------------------------------

async def buy_after_prediction(
    slug: str,
    asset_id: str,
    outcome_side: str,
    prediction_direction: str,
    amount_usd: float,
    snapshot_price: float,
    price_threshold: float = 0.52,
    bank_usd: Optional[float] = None,
    bank_pct: float = 0.05,
    min_buy_usd: float = 3.0,
    max_buy_usd: float = 20.0,
    batch_id: Optional[str] = None,
    template_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Execute a limit buy on Polymarket CLOB after a prediction.

    Args:
        slug:                 Market slug
        asset_id:             CLOB token ID for the outcome to buy
        outcome_side:         'UP' or 'DOWN' (the outcome we're buying)
        prediction_direction: 'UP' or 'DOWN' (what model predicted)
        amount_usd:           Dollar amount to spend
        snapshot_price:       Current best ask / snapshot price
        price_threshold:      Max acceptable price (default 0.52, hard cap 0.53)
        batch_id:             Prediction batch ID for linking
        template_id:          Template ID that triggered this

    Returns:
        Dict with order info and position info
    """
    # Compute amount from bank if caller passed 0/None
    computed_from_bank = False
    if amount_usd is None or float(amount_usd) <= 0:
        computed_from_bank = True
        fetched_bank = False
        if bank_usd is None:
            try:
                bal = trading_client.get_balance_allowance()
                bank_usd = _extract_collateral_balance_usd(bal) or 0.0
                fetched_bank = True
                logger.info("  fetched collateral balance from CLOB: %.6f", float(bank_usd))
            except Exception as e:
                logger.warning("  failed to fetch collateral balance: %s", e)
                bank_usd = 0.0
        bank_val = float(bank_usd or 0.0)
        bank_pct_val = float(bank_pct or 0.0)
        raw_amount = bank_val * bank_pct_val
        amount_usd = compute_buy_amount_usd(
            bank_usd=bank_val,
            bank_pct=bank_pct_val,
            min_usd=min_buy_usd,
            max_usd=max_buy_usd,
        )
        logger.info(
            "  sizing: bank_usd=%.6f%s pct=%.4f raw=%.6f min=%.2f max=%.2f -> amount=%.2f",
            bank_val,
            " (fetched)" if fetched_bank else "",
            bank_pct_val,
            raw_amount,
            float(min_buy_usd),
            float(max_buy_usd),
            float(amount_usd),
        )

    if float(price_threshold) > MAX_LIMIT_PRICE_USD:
        msg = f"Limit price {float(price_threshold):.4f} exceeds hard cap {MAX_LIMIT_PRICE_USD:.4f}"
        logger.error(msg)
        return {"success": False, "error": msg, "order_row_id": None, "position": None}

    logger.info("=== BUY AFTER PREDICTION ===")
    logger.info(
        "  slug=%s  asset=%s  side=%s  direction=%s  amount=$%.2f%s  snap_price=%.4f  threshold=%.4f",
        slug,
        asset_id[:16],
        outcome_side,
        prediction_direction,
        amount_usd,
        " (computed)" if computed_from_bank else "",
        snapshot_price,
        price_threshold,
    )
    if computed_from_bank and not fetched_bank:
        # Already logged detailed sizing above when fetched_bank True; keep legacy log for provided bank values.
        logger.info(
            "  sizing (provided bank): bank_usd=%.2f  pct=%.4f  min=%.2f  max=%.2f -> amount=%.2f",
            float(bank_usd or 0.0), float(bank_pct), float(min_buy_usd), float(max_buy_usd), float(amount_usd)
        )

    # Refresh market + best ask right before placing order (do not rely on UI cached snapshot).
    refreshed_best_ask = None
    try:
        try:
            # For debug parity with earlier logs (Gamma market prices)
            trading_client.fetch_market(slug)
        except Exception as e:
            logger.debug("fetch_market failed (non-fatal): %s", e)

        refreshed_best_ask = trading_client.get_best_ask(asset_id)
        if refreshed_best_ask is not None:
            logger.info("  refreshed_best_ask=%.4f (CLOB)", float(refreshed_best_ask))
    except Exception as e:
        logger.debug("refresh best ask failed (non-fatal): %s", e)

    if refreshed_best_ask is not None and float(refreshed_best_ask) > 0:
        snapshot_price = float(refreshed_best_ask)

    if await _market_has_completed_buy(slug):
        msg = f"Market {slug} already has a filled buy; skip duplicate order"
        logger.warning(msg)
        return {"success": False, "error": msg, "order_row_id": None, "position": None}

    if snapshot_price is None or float(snapshot_price) <= 0:
        msg = f"Invalid snapshot_price {snapshot_price}; skip buy"
        logger.warning(msg)
        return {"success": False, "error": msg, "order_row_id": None, "position": None}

    if snapshot_price > price_threshold:
        logger.info(
            "  snapshot price %.4f above threshold %.4f — waiting up to %ss for better price",
            snapshot_price,
            price_threshold,
            PRICE_RETRY_WINDOW_SEC,
        )
        awaited_price = await _wait_for_price_within_threshold(slug, asset_id, price_threshold)
        if awaited_price is None:
            msg = (
                f"Snapshot price {snapshot_price:.4f} exceeds threshold {price_threshold:.4f} "
                "and no better quote arrived within retry window"
            )
            logger.warning(msg)
            return {"success": False, "error": msg, "order_row_id": None, "position": None}
        snapshot_price = float(awaited_price)
        logger.info("  price recovered to %.4f — proceeding with buy", snapshot_price)

    if await _market_has_completed_buy(slug):
        msg = f"Market {slug} received a fill before execution; skip duplicate order"
        logger.warning(msg)
        return {"success": False, "error": msg, "order_row_id": None, "position": None}

    use_market = bool(getattr(config, "BUY_MARKET", True))
    if use_market:
        worst_price = float(min(float(price_threshold), float(MAX_LIMIT_PRICE_USD)))
        if worst_price <= 0:
            msg = f"Invalid worst_price {worst_price}; skip buy"
            logger.warning(msg)
            return {"success": False, "error": msg, "order_row_id": None, "position": None}

        size_shares = amount_usd / float(snapshot_price) if float(snapshot_price) > 0 else 0.0
        logger.info("  BUY_MARKET enabled: worst_price=%.4f", worst_price)
        logger.debug("Computed size_shares=%.6f from amount_usd=%.2f and snapshot_price=%.6f", size_shares, amount_usd, float(snapshot_price))
        loop = asyncio.get_event_loop()
        logger.debug("Calling trading_client.buy_market with token_id=%s, amount=%.6f, worst_price=%.6f", asset_id[:16], float(amount_usd), float(worst_price))
        try:
            clob_resp = await loop.run_in_executor(
                None,
                lambda: trading_client.buy_market(
                    token_id=asset_id,
                    amount=float(amount_usd),
                    worst_price=float(worst_price),
                )
            )
            logger.debug("Raw CLOB response: %s", clob_resp)
        except Exception as e:
            logger.error("Exception calling trading_client.buy_market: %s", e, exc_info=True)
            return {"success": False, "error": f"buy_market exception: {e}", "order_row_id": None, "position": None}

        success = clob_resp.get("success", False)
        order_id = clob_resp.get("orderID") or clob_resp.get("order_id") or None
        status = clob_resp.get("status", "error" if not success else "unknown")
        error_msg = clob_resp.get("errorMsg") or clob_resp.get("error_msg") or None

        logger.info("CLOB response: success=%s  order_id=%s  status=%s  error=%s",
                    success, order_id, status, error_msg)

        # Record order in DB
        fill_shares, fill_spent_usd, fill_avg_cents = _extract_fill_metrics(clob_resp)

        order_row_id = await db.execute(
            """
            INSERT INTO poly_live_orders
              (slug, asset_id, outcome_side, side, order_type, price, amount,
               fill_shares, fill_total_spent_usd, fill_avg_price_cents,
               clob_order_id, clob_status, clob_error_msg, clob_response_json,
               prediction_batch_id, template_id)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            (
                slug, asset_id, outcome_side, "BUY", "FOK",
                worst_price, amount_usd,
                fill_shares, fill_spent_usd, fill_avg_cents,
                order_id, status, error_msg,
                json.dumps(clob_resp, default=str),
                batch_id, template_id,
            ),
        )
        logger.info("Order recorded in DB: row_id=%s", order_row_id)

        position_info = None

        # If order was matched, update position
        if success and status in ("matched", "live"):
            actual_shares = float(fill_shares) if fill_shares is not None else size_shares
            actual_cost = float(fill_spent_usd) if fill_spent_usd is not None else float(amount_usd)
            actual_avg_price = float(fill_avg_cents) / 100.0 if fill_avg_cents is not None else float(worst_price)

            try:
                existing = await db.fetchone(
                    "SELECT id, shares, total_cost FROM poly_live_positions "
                    "WHERE slug=%s AND asset_id=%s AND closed=0 LIMIT 1",
                    (slug, asset_id),
                )
                if existing:
                    pos_id, shares0, cost0 = existing
                    shares_new = float(shares0 or 0) + float(actual_shares)
                    cost_new = float(cost0 or 0) + float(actual_cost)
                    avg_new = cost_new / shares_new if shares_new > 0 else 0
                    await db.execute(
                        "UPDATE poly_live_positions SET shares=%s,total_cost=%s,avg_price=%s,last_order_id=%s WHERE id=%s",
                        (shares_new, cost_new, avg_new, order_id, pos_id),
                    )
                    position_info = {"id": pos_id, "shares": shares_new, "total_cost": cost_new, "avg_price": avg_new}
                else:
                    pos_id = await db.execute(
                        "INSERT INTO poly_live_positions (slug, asset_id, outcome_side, shares, avg_price, total_cost, closed, last_order_id) VALUES (%s,%s,%s,%s,%s,%s,0,%s)",
                        (slug, asset_id, outcome_side, float(actual_shares), actual_avg_price, float(actual_cost), order_id),
                    )
                    position_info = {"id": pos_id, "shares": float(actual_shares), "total_cost": float(actual_cost), "avg_price": actual_avg_price}
            except Exception as e:
                logger.error("Position update failed (non-fatal): %s", e)

        return {
            "success": bool(success),
            "order_id": order_id,
            "status": status,
            "order_row_id": order_row_id,
            "position": position_info,
            "clob": clob_resp,
        }

    limit_price = float(snapshot_price)
    try:
        tick_size_s = trading_client.get_tick_size(asset_id)
        tick = Decimal(str(tick_size_s))
        if tick > 0:
            p = Decimal(str(limit_price))
            limit_price = float((p / tick).to_integral_value(rounding=ROUND_CEILING) * tick)
    except Exception:
        pass
    if limit_price > MAX_LIMIT_PRICE_USD:
        msg = f"Snapshot price {limit_price:.4f} exceeds hard cap {MAX_LIMIT_PRICE_USD:.4f}; skip buy"
        logger.warning(msg)
        return {"success": False, "error": msg, "order_row_id": None, "position": None}

    logger.info("  limit_price=%.4f (from snapshot)", limit_price)

    # Use limit buy at snapshot price. Shares derived from snapshot price.
    size_shares = amount_usd / limit_price if limit_price > 0 else 0
    logger.debug("Computed size_shares=%.6f from amount_usd=%.2f and limit_price=%.6f", size_shares, amount_usd, limit_price)
    loop = asyncio.get_event_loop()
    logger.debug("Calling trading_client.buy_limit with token_id=%s, price=%.6f, size=%.6f", asset_id[:16], limit_price, size_shares)
    try:
        clob_resp = await loop.run_in_executor(
            None,
            lambda: trading_client.buy_limit(
                token_id=asset_id,
                price=limit_price,
                size=size_shares,
            )
        )
        logger.debug("Raw CLOB response: %s", clob_resp)
    except Exception as e:
        logger.error("Exception calling trading_client.buy_limit: %s", e, exc_info=True)
        return {"success": False, "error": f"buy_limit exception: {e}", "order_row_id": None, "position": None}

    success = clob_resp.get("success", False)
    order_id = clob_resp.get("orderID") or clob_resp.get("order_id") or None
    status = clob_resp.get("status", "error" if not success else "unknown")
    error_msg = clob_resp.get("errorMsg") or clob_resp.get("error_msg") or None

    logger.info("CLOB response: success=%s  order_id=%s  status=%s  error=%s",
                success, order_id, status, error_msg)

    # Record order in DB
    fill_shares, fill_spent_usd, fill_avg_cents = _extract_fill_metrics(clob_resp)

    order_row_id = await db.execute(
        """
        INSERT INTO poly_live_orders
          (slug, asset_id, outcome_side, side, order_type, price, amount,
           fill_shares, fill_total_spent_usd, fill_avg_price_cents,
           clob_order_id, clob_status, clob_error_msg, clob_response_json,
           prediction_batch_id, template_id)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """,
        (
            slug, asset_id, outcome_side, "BUY", "GTC",
            price_threshold, amount_usd,
            fill_shares, fill_spent_usd, fill_avg_cents,
            order_id, status, error_msg,
            json.dumps(clob_resp, default=str),
            batch_id, template_id,
        ),
    )
    logger.info("Order recorded in DB: row_id=%s", order_row_id)

    if order_id and order_row_id:
        try:
            asyncio.create_task(_cancel_after_timeout(str(order_id), order_row_id, timeout_sec=60))
        except Exception:
            pass

    position_info = None

    # If order was matched, update position
    if success and status in ("matched", "live"):
        actual_shares = float(fill_shares) if fill_shares is not None else size_shares
        actual_cost = float(fill_spent_usd) if fill_spent_usd is not None else float(amount_usd)
        actual_avg_price = float(fill_avg_cents) / 100.0 if fill_avg_cents is not None else float(snapshot_price)

        try:
            # Try to update existing open position
            existing = await db.fetchone(
                "SELECT id, shares, total_cost FROM poly_live_positions "
                "WHERE slug=%s AND asset_id=%s AND status='open' LIMIT 1",
                (slug, asset_id),
            )

            if existing:
                pos_id, old_shares, old_cost = existing
                new_shares = old_shares + actual_shares
                new_cost = old_cost + actual_cost
                new_avg = new_cost / new_shares if new_shares > 0 else 0
                await db.execute(
                    "UPDATE poly_live_positions SET shares=%s, avg_price=%s, total_cost=%s, updated_at=NOW() "
                    "WHERE id=%s",
                    (new_shares, new_avg, new_cost, pos_id),
                )
                logger.info("Position updated: id=%s  shares=%.4f  avg_price=%.4f  total_cost=%.2f",
                            pos_id, new_shares, new_avg, new_cost)
                position_info = {"id": pos_id, "shares": new_shares, "avg_price": new_avg, "total_cost": new_cost}
            else:
                # Create new position
                avg_price = actual_avg_price
                pos_id = await db.execute(
                    """
                    INSERT INTO poly_live_positions
                      (slug, asset_id, outcome_side, shares, avg_price, total_cost,
                       status, snapshot_price_cents, prediction_direction,
                       prediction_batch_id, template_id)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (
                        slug, asset_id, outcome_side,
                        actual_shares, avg_price, actual_cost,
                        "open", (avg_price or snapshot_price) * 100,
                        prediction_direction, batch_id, template_id,
                    ),
                )
                logger.info("Position created: id=%s  shares=%.4f  avg=%.4f  cost=%.2f",
                            pos_id, actual_shares, avg_price, actual_cost)
                position_info = {"id": pos_id, "shares": actual_shares, "avg_price": avg_price, "total_cost": actual_cost}

        except Exception as e:
            logger.error("Failed to update position in DB: %s", e, exc_info=True)

    return {
        "success": success,
        "order_id": order_id,
        "clob_status": status,
        "error": error_msg,
        "order_row_id": order_row_id,
        "position": position_info,
    }


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

async def list_open_positions(slug: Optional[str] = None) -> List[Dict[str, Any]]:
    sql = (
        "SELECT id, slug, asset_id, outcome_side, shares, avg_price, total_cost, "
        "snapshot_price_cents, prediction_direction, prediction_batch_id, "
        "template_id, opened_at "
        "FROM poly_live_positions WHERE status='open'"
    )
    params: List[Any] = []
    if slug:
        sql += " AND slug=%s"
        params.append(slug)
    sql += " ORDER BY opened_at DESC"
    rows = await db.fetchall(sql, tuple(params) if params else None)
    out = []
    for r in rows:
        out.append({
            "id": r[0], "slug": r[1], "asset_id": r[2], "outcome_side": r[3],
            "shares": r[4], "avg_price": r[5], "total_cost": r[6],
            "snapshot_price_cents": r[7], "prediction_direction": r[8],
            "prediction_batch_id": r[9], "template_id": r[10],
            "opened_at": str(r[11]) if r[11] else None,
        })
    return out


async def wallet_summary(limit: int = 25) -> Dict[str, Any]:
    """Wallet summary for monitoring: balances, open positions, recent orders."""
    loop = asyncio.get_event_loop()
    balance = await loop.run_in_executor(None, trading_client.get_balance_allowance)
    positions = await list_open_positions()
    orders = await list_orders(limit=limit)
    return {
        "balance": balance,
        "positions": positions,
        "orders": orders,
    }


def _parse_ymd(date_str: Optional[str]):
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str.strip(), "%Y-%m-%d").date()
    except Exception:
        return None


async def order_flow_analytics(
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> Dict[str, Any]:
    """Aggregate order flow + prediction correctness per day."""

    today = datetime.utcnow().date()
    end_date = _parse_ymd(date_to) or today
    start_date = _parse_ymd(date_from) or (end_date - timedelta(days=6))
    if start_date > end_date:
        start_date, end_date = end_date, start_date

    day_cursor = start_date
    day_map: Dict[str, Dict[str, Any]] = {}
    while day_cursor <= end_date:
        iso = day_cursor.isoformat()
        day_map[iso] = {
            "date": iso,
            "total_orders": 0,
            "resolved_orders": 0,
            "pending_orders": 0,
            "win_count": 0,
            "loss_count": 0,
            "total_amount": 0.0,
            "winning_shares": 0.0,
            "winning_cost": 0.0,
            "winning_net": 0.0,
            "losing_amount": 0.0,
            "net_winning_amount": 0.0,
            "win_rate": None,
        }
        day_cursor += timedelta(days=1)

    sql = f"""
        SELECT
            DATE(CONVERT_TZ(o.created_at, '+00:00', '{MSK_TZ_NAME}')) AS day,
            COUNT(*) AS total_orders,
            SUM(CASE WHEN pm.resolved_outcome IN ('UP','DOWN') THEN 1 ELSE 0 END) AS resolved_orders,
            SUM(
                CASE
                    WHEN pm.resolved_outcome IN ('UP','DOWN')
                         AND UPPER(pm.resolved_outcome) = UPPER(o.outcome_side)
                    THEN 1 ELSE 0
                END
            ) AS win_count,
            SUM(
                CASE
                    WHEN pm.resolved_outcome IN ('UP','DOWN')
                         AND UPPER(pm.resolved_outcome) <> UPPER(o.outcome_side)
                    THEN 1 ELSE 0
                END
            ) AS loss_count,
            SUM(COALESCE(o.amount, 0)) AS total_amount,
            SUM(
                CASE
                    WHEN pm.resolved_outcome IN ('UP','DOWN')
                         AND UPPER(pm.resolved_outcome) = UPPER(o.outcome_side)
                    THEN COALESCE(o.fill_shares, 0)
                    ELSE 0
                END
            ) AS winning_shares,
            SUM(
                CASE
                    WHEN pm.resolved_outcome IN ('UP','DOWN')
                         AND UPPER(pm.resolved_outcome) = UPPER(o.outcome_side)
                    THEN COALESCE(o.amount, 0)
                    ELSE 0
                END
            ) AS winning_cost,
            SUM(
                CASE
                    WHEN pm.resolved_outcome IN ('UP','DOWN')
                         AND UPPER(pm.resolved_outcome) <> UPPER(o.outcome_side)
                    THEN COALESCE(o.amount, 0)
                    ELSE 0
                END
            ) AS losing_amount
        FROM poly_live_orders o
        LEFT JOIN poly_markets pm ON pm.slug = o.slug
        WHERE DATE(CONVERT_TZ(o.created_at, '+00:00', '{MSK_TZ_NAME}')) BETWEEN %s AND %s
          AND COALESCE(o.fill_shares, 0) > 0
        GROUP BY day
        ORDER BY day ASC
    """

    rows = await db.fetchall(sql, (start_date.isoformat(), end_date.isoformat()))

    for row in rows:
        day_val = row[0]
        if hasattr(day_val, "isoformat"):
            day_key = day_val.isoformat()
        else:
            day_key = str(day_val)
        entry = day_map.get(day_key)
        if not entry:
            continue
        total_orders = int(row[1] or 0)
        resolved_orders = int(row[2] or 0)
        win_count = int(row[3] or 0)
        loss_count = int(row[4] or 0)
        total_amount = float(row[5] or 0.0)
        winning_shares = float(row[6] or 0.0)
        winning_cost = float(row[7] or 0.0)
        losing_amount = float(row[8] or 0.0)
        winning_payout = winning_shares
        pending = max(total_orders - resolved_orders, 0)

        entry.update(
            {
                "total_orders": total_orders,
                "resolved_orders": resolved_orders,
                "pending_orders": pending,
                "win_count": win_count,
                "loss_count": loss_count,
                "total_amount": total_amount,
                "winning_shares": winning_shares,
                "winning_cost": winning_cost,
                "winning_net": winning_payout - winning_cost,
                "losing_amount": losing_amount,
                "net_winning_amount": winning_payout - winning_cost - losing_amount,
                "win_rate": (win_count / resolved_orders) if resolved_orders > 0 else None,
            }
        )

    daily_rows = list(day_map.values())

    totals = defaultdict(float)
    totals_counts = defaultdict(int)

    for entry in daily_rows:
        for key in ("total_orders", "resolved_orders", "pending_orders", "win_count", "loss_count"):
            totals_counts[key] += int(entry.get(key) or 0)
        for key in ("total_amount", "losing_amount", "net_winning_amount"):
            totals[key] += float(entry.get(key) or 0.0)
        totals["winning_shares"] += float(entry.get("winning_shares") or 0.0)
        totals["winning_cost"] += float(entry.get("winning_cost") or 0.0)
        totals["winning_net"] += float(entry.get("winning_net") or 0.0)

    resolved_total = totals_counts["resolved_orders"]
    win_total = totals_counts["win_count"]
    totals_summary = {
        "total_orders": totals_counts["total_orders"],
        "resolved_orders": resolved_total,
        "pending_orders": totals_counts["pending_orders"],
        "win_count": win_total,
        "loss_count": totals_counts["loss_count"],
        "win_rate": (win_total / resolved_total) if resolved_total > 0 else None,
        "total_amount": totals["total_amount"],
        "winning_shares": totals["winning_shares"],
        "winning_cost": totals["winning_cost"],
        "winning_net": totals["winning_net"],
        "losing_amount": totals["losing_amount"],
        "net_winning_amount": totals["net_winning_amount"],
    }

    return {
        "range": {
            "start": start_date.isoformat(),
            "end": end_date.isoformat(),
        },
        "daily": daily_rows,
        "totals": totals_summary,
    }


async def list_all_positions(limit: int = 100, slug: Optional[str] = None) -> List[Dict[str, Any]]:
    sql = (
        "SELECT id, slug, asset_id, outcome_side, shares, avg_price, total_cost, "
        "status, pnl, prediction_direction, opened_at, closed_at "
        "FROM poly_live_positions"
    )
    params: List[Any] = []
    if slug:
        sql += " WHERE slug=%s"
        params.append(slug)
    sql += " ORDER BY opened_at DESC LIMIT %s"
    params.append(limit)
    rows = await db.fetchall(sql, tuple(params))
    out = []
    for r in rows:
        out.append({
            "id": r[0], "slug": r[1], "asset_id": r[2], "outcome_side": r[3],
            "shares": r[4], "avg_price": r[5], "total_cost": r[6],
            "status": r[7], "pnl": r[8], "prediction_direction": r[9],
            "opened_at": str(r[10]) if r[10] else None,
            "closed_at": str(r[11]) if r[11] else None,
        })
    return out


async def list_orders(limit: int = 100, slug: Optional[str] = None) -> List[Dict[str, Any]]:
    sql = (
        "SELECT id, slug, asset_id, outcome_side, side, order_type, price, amount, "
        "fill_shares, fill_total_spent_usd, fill_avg_price_cents, "
        "clob_order_id, clob_status, clob_error_msg, created_at "
        "FROM poly_live_orders"
    )
    params: List[Any] = []
    clauses = ["COALESCE(fill_shares, 0) > 0"]
    if slug:
        clauses.append("slug=%s")
        params.append(slug)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY created_at DESC LIMIT %s"
    params.append(limit)
    rows = await db.fetchall(sql, tuple(params))
    out = []
    for r in rows:
        out.append({
            "id": r[0], "slug": r[1], "asset_id": r[2], "outcome_side": r[3],
            "side": r[4], "order_type": r[5], "price": r[6], "amount": r[7],
            "fill_shares": r[8], "fill_total_spent_usd": r[9], "fill_avg_price_cents": r[10],
            "clob_order_id": r[11], "clob_status": r[12], "clob_error_msg": r[13],
            "created_at": str(r[14]) if r[14] else None,
        })
    return out
