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
from decimal import Decimal, ROUND_CEILING
from datetime import datetime
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

MAX_LIMIT_PRICE_USD = 0.52


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


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
                # Strings like '37926149' or ints >= 1e6 are base units (USDC 6 decimals)
                if isinstance(x, str):
                    sx = x.strip()
                    if sx.isdigit():
                        iv = int(sx)
                        return float(iv) / 1e6 if abs(iv) >= 1_000_000 else float(iv)
                    fv = float(sx)
                    return fv
                if isinstance(x, int):
                    return float(x) / 1e6 if abs(x) >= 1_000_000 else float(x)
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
                            return nv
                    except Exception:
                        pass
        # fallback: if collateral itself is numeric
        if isinstance(coll, (int, float, str)):
            nv = _norm_usdc(coll)
            if nv is not None:
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
        price_threshold:      Max acceptable price (default 0.52)
        batch_id:             Prediction batch ID for linking
        template_id:          Template ID that triggered this

    Returns:
        Dict with order info and position info
    """
    # Compute amount from bank if caller passed 0/None
    computed_from_bank = False
    if amount_usd is None or float(amount_usd) <= 0:
        computed_from_bank = True
        if bank_usd is None:
            try:
                bal = trading_client.get_balance_allowance()
                bank_usd = _extract_collateral_balance_usd(bal) or 0.0
            except Exception:
                bank_usd = 0.0
        amount_usd = compute_buy_amount_usd(
            bank_usd=bank_usd,
            bank_pct=bank_pct,
            min_usd=min_buy_usd,
            max_usd=max_buy_usd,
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
    if computed_from_bank:
        logger.info(
            "  sizing: bank_usd=%.2f  pct=%.4f  min=%.2f  max=%.2f",
            float(bank_usd or 0.0), float(bank_pct), float(min_buy_usd), float(max_buy_usd)
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

    if snapshot_price is None or float(snapshot_price) <= 0:
        msg = f"Invalid snapshot_price {snapshot_price}; skip buy"
        logger.warning(msg)
        return {"success": False, "error": msg, "order_row_id": None, "position": None}

    if snapshot_price > price_threshold:
        msg = f"Snapshot price {snapshot_price:.4f} exceeds threshold {price_threshold:.4f}; skip buy"
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
        order_row_id = await db.execute(
            """
            INSERT INTO poly_live_orders
              (slug, asset_id, outcome_side, side, order_type, price, amount,
               clob_order_id, clob_status, clob_error_msg, clob_response_json,
               prediction_batch_id, template_id)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            (
                slug, asset_id, outcome_side, "BUY", "FOK",
                worst_price, amount_usd,
                order_id, status, error_msg,
                json.dumps(clob_resp, default=str),
                batch_id, template_id,
            ),
        )
        logger.info("Order recorded in DB: row_id=%s", order_row_id)

        position_info = None

        # If order was matched, update position
        if success and status in ("matched", "live"):
            estimated_shares = size_shares
            try:
                existing = await db.fetchone(
                    "SELECT id, shares, total_cost FROM poly_live_positions "
                    "WHERE slug=%s AND asset_id=%s AND closed=0 LIMIT 1",
                    (slug, asset_id),
                )
                if existing:
                    pos_id, shares0, cost0 = existing
                    shares_new = float(shares0 or 0) + float(estimated_shares)
                    cost_new = float(cost0 or 0) + float(amount_usd)
                    avg_new = cost_new / shares_new if shares_new > 0 else 0
                    await db.execute(
                        "UPDATE poly_live_positions SET shares=%s,total_cost=%s,avg_price=%s,last_order_id=%s WHERE id=%s",
                        (shares_new, cost_new, avg_new, order_id, pos_id),
                    )
                    position_info = {"id": pos_id, "shares": shares_new, "total_cost": cost_new, "avg_price": avg_new}
                else:
                    pos_id = await db.execute(
                        "INSERT INTO poly_live_positions (slug, asset_id, outcome_side, shares, avg_price, total_cost, closed, last_order_id) VALUES (%s,%s,%s,%s,%s,%s,0,%s)",
                        (slug, asset_id, outcome_side, float(estimated_shares), worst_price, float(amount_usd), order_id),
                    )
                    position_info = {"id": pos_id, "shares": float(estimated_shares), "total_cost": float(amount_usd), "avg_price": worst_price}
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
    order_row_id = await db.execute(
        """
        INSERT INTO poly_live_orders
          (slug, asset_id, outcome_side, side, order_type, price, amount,
           clob_order_id, clob_status, clob_error_msg, clob_response_json,
           prediction_batch_id, template_id)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """,
        (
            slug, asset_id, outcome_side, "BUY", "GTC",
            price_threshold, amount_usd,
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
        # Estimate shares bought: amount / price
        estimated_shares = size_shares

        try:
            # Try to update existing open position
            existing = await db.fetchone(
                "SELECT id, shares, total_cost FROM poly_live_positions "
                "WHERE slug=%s AND asset_id=%s AND status='open' LIMIT 1",
                (slug, asset_id),
            )

            if existing:
                pos_id, old_shares, old_cost = existing
                new_shares = old_shares + estimated_shares
                new_cost = old_cost + amount_usd
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
                avg_price = snapshot_price
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
                        estimated_shares, avg_price, amount_usd,
                        "open", snapshot_price * 100,
                        prediction_direction, batch_id, template_id,
                    ),
                )
                logger.info("Position created: id=%s  shares=%.4f  avg=%.4f  cost=%.2f",
                            pos_id, estimated_shares, avg_price, amount_usd)
                position_info = {"id": pos_id, "shares": estimated_shares, "avg_price": avg_price, "total_cost": amount_usd}

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
        "clob_order_id, clob_status, clob_error_msg, created_at "
        "FROM poly_live_orders"
    )
    params: List[Any] = []
    if slug:
        sql += " WHERE slug=%s"
        params.append(slug)
    sql += " ORDER BY created_at DESC LIMIT %s"
    params.append(limit)
    rows = await db.fetchall(sql, tuple(params))
    out = []
    for r in rows:
        out.append({
            "id": r[0], "slug": r[1], "asset_id": r[2], "outcome_side": r[3],
            "side": r[4], "order_type": r[5], "price": r[6], "amount": r[7],
            "clob_order_id": r[8], "clob_status": r[9], "clob_error_msg": r[10],
            "created_at": str(r[11]) if r[11] else None,
        })
    return out
