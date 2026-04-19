"""5-second-early prediction service.

Uses the `c_5m_5s` offset candle table (snapshot taken ~5s before close) to
run the same RSI mean-reversion prediction that the normal autopredict uses,
but 5 seconds earlier.

Flow (mirrors poly_service.predict_for_market / batch_predict_for_market):
  1. Load all active prediction templates.
  2. For a given slug load window_size candles from c_5m, replace the last
     candle with the snapshot from c_5m_5s for the same open_time.
  3. Run strategy.fit + strategy.predict exactly as the normal flow does.
  4. Persist result to poly_predictions_4s (upsert by slug).
  5. If auto_place is enabled, place order into poly_live_orders_4s.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

import app.config as config
from db import DbProvider
from predictor.data_loader import add_direction
from predictor.features import add_technical_features
from predictor.strategies import get_strategy, STRATEGY_REGISTRY
from predictor.utils.async_utils import resolve_awaitable
from predictor.utils.prediction_thresholds import (
    classify_probability,
    label_from_prediction,
    resolve_probability_threshold,
)

logger = logging.getLogger("predict_4s")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(name)s %(levelname)s %(asctime)s] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(_h)
logger.propagate = False

db = DbProvider()

_OFFSET_TABLE = "c_5m_5s"
_CANDLE_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quota_volume", "trades", "taker_base_volume", "taker_quota_volume",
]

# ── ensure tables ──────────────────────────────────────────────────────────────

_CREATE_PREDICTIONS_4S = """
CREATE TABLE IF NOT EXISTS `poly_predictions_4s` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `slug` varchar(255) NOT NULL,
  `prediction_ts` bigint NOT NULL,
  `payload_json` json DEFAULT NULL,
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uq_slug` (`slug`),
  KEY `idx_prediction_ts` (`prediction_ts`),
  KEY `idx_created` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci
"""

_ADD_IS_4S_EARLY_COL = """
ALTER TABLE `poly_live_orders`
  ADD COLUMN IF NOT EXISTS `is_4s_early` tinyint(1) NOT NULL DEFAULT 0
  AFTER `template_id`
"""


async def ensure_tables() -> None:
    try:
        await db.execute(_CREATE_PREDICTIONS_4S)
    except Exception as exc:
        logger.warning("ensure_tables 4s (predictions): %s", exc)
    try:
        await db.execute(_ADD_IS_4S_EARLY_COL)
    except Exception as exc:
        logger.debug("is_4s_early column already exists or error: %s", exc)


# ── helpers ────────────────────────────────────────────────────────────────────

async def _load_base_window(market_ts_us: int, window_size: int) -> Optional[List]:
    rows = await db.fetchall(
        """
        SELECT open_time, open, high, low, close, volume,
               close_time, quota_volume, trades, taker_base_volume, taker_quota_volume
        FROM c_5m
        WHERE open_time <= %s
        ORDER BY open_time DESC
        LIMIT %s
        """,
        (market_ts_us, window_size),
    )
    if not rows or len(rows) < window_size:
        return None
    return list(reversed(rows))


def _live_fields_to_row(fields: dict) -> tuple:
    """Convert a binance_snapshot _map_fields dict to the same tuple shape as a DB row."""
    return (
        int(fields["open_time"]),
        float(fields["open"]),
        float(fields["high"]),
        float(fields["low"]),
        float(fields["close"]),
        float(fields["volume"]),
        int(fields["close_time"]),
        float(fields["quota_volume"]),
        int(fields["trades"]),
        float(fields["taker_base_volume"]),
        float(fields["taker_quota_volume"]),
    )


async def _load_offset_candle(open_time_us: int) -> Optional[tuple]:
    return await db.fetchone(
        f"""
        SELECT open_time, open, high, low, close, volume,
               close_time, quota_volume, trades, taker_base_volume, taker_quota_volume
        FROM {_OFFSET_TABLE}
        WHERE open_time = %s
        LIMIT 1
        """,
        (int(open_time_us),),
    )


def _build_df(rows: List) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=_CANDLE_COLS)
    for c in ["open", "high", "low", "close", "volume", "quota_volume",
              "taker_base_volume", "taker_quota_volume"]:
        df[c] = df[c].astype(float)
    df["trades"] = df["trades"].astype(int)
    return df


async def _run_prediction(
    df: pd.DataFrame,
    strategy_name: str,
    strategy_params: dict,
    horizon: int,
) -> Optional[Dict[str, Any]]:
    if len(df) < 90:
        return None
    df = add_direction(df).reset_index(drop=True)
    df_feat = add_technical_features(df)
    df_train = df_feat.iloc[:-1].reset_index(drop=True)
    df_predict = df_feat.iloc[[-1]].reset_index(drop=True)
    if len(df_train) < 90:
        return None

    logger.debug(
        "[predict_4s] running strategy=%s horizon=%s window=%s", strategy_name, horizon, len(df_train)
    )
    strategy = get_strategy(strategy_name, strategy_params)
    strategy.fit(df_train, horizon=horizon)
    prob_arr = await resolve_awaitable(strategy.predict_proba(df_predict, horizon=horizon))
    prob = float(prob_arr[0])
    threshold = resolve_probability_threshold(strategy.params)
    pred = classify_probability(prob, threshold)
    label = label_from_prediction(pred)

    period = strategy.params.get("rsi_period", 14)
    rsi_col = f"rsi_{period}" if f"rsi_{period}" in df_predict.columns else "rsi_14"
    rsi_val = float(np.nan_to_num(df_predict.iloc[0][rsi_col], nan=50.0))

    last_open_us = int(df.iloc[-1]["open_time"])
    return {
        "prediction": label,
        "probability": round(prob, 4),
        "rsi": round(rsi_val, 1),
        "last_candle_ts": last_open_us,
        "last_candle_dt": pd.Timestamp(last_open_us, unit="us").strftime("%Y-%m-%d %H:%M:%S"),
        "source_table": _OFFSET_TABLE,
        "is_4s_early": True,
    }


# ── core: predict for one market ──────────────────────────────────────────────

async def predict_for_market_4s(
    slug: str,
    live_candle_fields: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run 4s-early prediction for a single market slug.

    Returns the prediction dict (same shape as poly_service.predict_for_market)
    and saves it to poly_predictions_4s.
    """
    from predictor.poly_service import list_pred_templates

    # Dedup guard: if a 4s prediction already exists for this market slug, skip
    try:
        exist = await db.fetchone(
            "SELECT prediction_ts FROM poly_predictions_4s WHERE slug=%s LIMIT 1",
            (slug,),
        )
        if exist:
            return {"error": "4s prediction already exists for this market", "prediction_ts": int(exist[0]) if exist[0] else None}
    except Exception:
        # best-effort; continue if lookup fails
        pass

    logger.info("[predict_4s] start slug=%s", slug)
    templates = await list_pred_templates()
    active = [t for t in templates if t["active"]]
    if not active:
        return {"error": "No active prediction templates."}

    m_row = await db.fetchone("SELECT ts FROM poly_markets WHERE slug=%s", (slug,))
    if not m_row:
        return {"error": f"Market not found: {slug}"}
    market_ts = int(m_row[0])
    market_ts_us = market_ts * 1_000_000
    interval_us = 5 * 60 * 1_000_000

    tpl = active[0]
    strategy_name = tpl["strategy"]
    strategy_params = tpl["params"] or {}
    horizon = max(1, min(3, int(tpl["horizon"])))
    window_size = int(tpl["window_size"])

    if strategy_name not in STRATEGY_REGISTRY:
        return {"error": f"Unknown strategy: {strategy_name}"}

    # 1. Signal candle = the candle about to close (4s early snapshot).
    #    With horizon=H, it predicts H candles ahead, so:
    #      signal_open_us = market_ts_us - H * interval   (not H-1 like regular)
    #    Example: market_ts=1773726900, horizon=2
    #      signal = 1773726900 - 2*300 = 1773726300  (the candle closing now)
    signal_open_us = market_ts_us - horizon * interval_us

    # 2. Get the 4s-early snapshot — from live memory first, DB fallback
    if live_candle_fields is not None and int(live_candle_fields.get("open_time", 0)) == signal_open_us:
        offset_row = _live_fields_to_row(live_candle_fields)
        logger.debug("[predict_4s] %s using live in-memory candle for signal %s", slug, signal_open_us)
    else:
        offset_row = await _load_offset_candle(signal_open_us)
    if offset_row is None:
        return {
            "error": f"4s-early candle not found for open_time={signal_open_us}",
            "signal_open_time": signal_open_us,
        }

    # 3. Load base candles from c_5m strictly BEFORE the signal candle
    #    (the signal candle itself is not in c_5m yet — it hasn't closed)
    base_rows = await db.fetchall(
        """
        SELECT open_time, open, high, low, close, volume,
               close_time, quota_volume, trades, taker_base_volume, taker_quota_volume
        FROM c_5m
        WHERE open_time < %s
        ORDER BY open_time DESC
        LIMIT %s
        """,
        (signal_open_us, window_size - 1),
    )
    need = window_size - 1
    if not base_rows or len(base_rows) < need:
        logger.warning(
            "[predict_4s] insufficient base candles slug=%s need=%s have=%s",
            slug,
            need,
            len(base_rows) if base_rows else 0,
        )
        return {"error": f"Not enough candles in c_5m for {slug} (need {need}, got {len(base_rows) if base_rows else 0})"}
    base_rows = list(reversed(base_rows))

    # 4. Build window: base candles + 4s snapshot appended as last (signal) row
    rows_with_offset = base_rows + [offset_row]

    df = _build_df(rows_with_offset)
    result = await _run_prediction(df, strategy_name, strategy_params, horizon)
    if result is None:
        logger.warning("[predict_4s] prediction returned no result slug=%s", slug)
        return {"error": "Prediction failed (insufficient data after feature engineering)"}

    prediction_ts = int(time.time())
    result.update({
        "market_slug": slug,
        "market_ts": market_ts,
        "prediction_ts": prediction_ts,
        "template_id": tpl["id"],
        "template_name": tpl["name"],
        "strategy": strategy_name,
        "horizon": horizon,
        "window_size": window_size,
        "signal_open_time": signal_open_us,
    })

    # 5. Write prediction debug file (same as regular predictions)
    try:
        from predictor.poly_service import _maybe_log_prediction_window
        _maybe_log_prediction_window(
            slug=slug,
            market_ts=market_ts,
            rows=rows_with_offset,
            window_size=window_size,
            table=_OFFSET_TABLE,
            prediction_label=result["prediction"],
        )
    except Exception as exc:
        logger.warning("[predict_4s] debug log failed for %s: %s", slug, exc)

    # 6. Persist to poly_predictions_4s
    try:
        payload_json = json.dumps(result, ensure_ascii=False)
        await db.execute(
            """
            INSERT INTO poly_predictions_4s (slug, prediction_ts, payload_json)
            VALUES (%s, %s, %s) AS new
            ON DUPLICATE KEY UPDATE
                prediction_ts = new.prediction_ts,
                payload_json  = new.payload_json
            """,
            (slug, prediction_ts, payload_json),
        )
    except Exception as exc:
        logger.warning("Failed to persist poly_predictions_4s for %s: %s", slug, exc)

    logger.info(
        "[predict_4s] %s → %s (%.4f) candle_ts=%s",
        slug, result["prediction"], result["probability"], result["last_candle_dt"],
    )
    return result


# ── order placement ────────────────────────────────────────────────────────────

async def _place_order_4s(slug: str, prediction: str) -> None:
    """Place a live order into poly_live_orders with is_4s_early=1."""
    try:
        pred = str(prediction or "").upper()
        emulate_down = bool(getattr(config, "EMULATE_DOWN", False))
        if pred == "UNDEFINED" and emulate_down:
            pred = "DOWN"
        if pred not in ("UP", "DOWN"):
            return

        from predictor.poly_service import get_live_trade_settings
        live_settings = await get_live_trade_settings()
        if not live_settings.get("auto_place"):
            logger.info("[predict_4s] auto_place disabled, skipping order for %s", slug)
            return

        m_row = await db.fetchone("SELECT ts, closed FROM poly_markets WHERE slug=%s", (slug,))
        if not m_row:
            return
        market_ts = int(m_row[0]) if m_row[0] is not None else 0
        closed = int(m_row[1]) if len(m_row) > 1 and m_row[1] is not None else 0
        if closed:
            return
        now_utc = int(time.time())
        if not (market_ts and now_utc < market_ts):
            logger.info("[predict_4s] market not in future, skip order %s", slug)
            return

        o_rows = await db.fetchall("SELECT asset_id, name FROM poly_outcomes WHERE slug=%s", (slug,))
        if not o_rows:
            return
        up_id = down_id = None
        for asset_id, name in o_rows:
            n = str(name or "").upper()
            if "UP" in n and not up_id:
                up_id = str(asset_id)
            if "DOWN" in n and not down_id:
                down_id = str(asset_id)
        if (not up_id or not down_id) and len(o_rows) >= 2:
            up_id = up_id or str(o_rows[0][0])
            down_id = down_id or str(o_rows[1][0])

        asset_id = down_id if pred == "DOWN" else up_id
        if not asset_id:
            return

        from predictor import live_trading
        from predictor.poly_service import DEFAULT_LIVE_TRADE_SETTINGS, MAX_PRICE_CAP_CENTS

        bet_size_usd = float(
            live_settings.get("bet_size_usd", DEFAULT_LIVE_TRADE_SETTINGS["bet_size_usd"])
            or DEFAULT_LIVE_TRADE_SETTINGS["bet_size_usd"]
        )
        bet_size_usd = max(0.0, bet_size_usd)
        price_cap_cents = int(live_settings.get("price_cap_cents", 52) or 52)
        price_cap_cents = max(1, min(MAX_PRICE_CAP_CENTS, price_cap_cents))
        price_threshold = price_cap_cents / 100.0

        # Dedup guard — poly_live_orders already checked inside buy_after_prediction
        from predictor.live_trading import _market_has_completed_buy
        if await _market_has_completed_buy(slug):
            logger.info("[predict_4s] %s already has a filled order, skip 4s buy", slug)
            return

        logger.info("[predict_4s] placing order slug=%s pred=%s bet=$%.2f", slug, pred, bet_size_usd)

        result = await live_trading.buy_after_prediction(
            slug=slug,
            asset_id=str(asset_id),
            outcome_side=pred,
            prediction_direction=pred,
            snapshot_price=0.0,
            price_threshold=price_threshold,
            batch_id=None,
            template_id=None,
            enable_price_wait=True,
            is_4s_early=True,
        )

        logger.info("[predict_4s] order result for %s: %s", slug, result.get("status"))
    except Exception as exc:
        logger.exception("[predict_4s] _place_order_4s error for %s", slug, exc_info=exc)


# ── autopredict entry point ────────────────────────────────────────────────────

async def try_autopredict_4s(
    slug: str,
    live_candle_fields: Optional[Dict[str, Any]] = None,
) -> None:
    """Run 4s-early prediction + optional order placement for a market.

    Called from the snapshot collector when a 4s-early candle arrives.
    Only runs if autopredict is enabled in settings.
    """
    try:
        from predictor.poly_service import get_settings
        settings = await get_settings()
        if not settings.get("autopredict"):
            return

        m_row = await db.fetchone(
            "SELECT ts, closed FROM poly_markets WHERE slug=%s", (slug,)
        )
        if not m_row:
            return
        market_ts = int(m_row[0]) if m_row[0] else 0
        closed = int(m_row[1]) if len(m_row) > 1 and m_row[1] is not None else 0
        if closed:
            return
        now_utc = int(time.time())
        if not (market_ts and now_utc < market_ts):
            return

        # Skip if we already have a saved 4s prediction for this market slug
        try:
            exists = await db.fetchone(
                "SELECT 1 FROM poly_predictions_4s WHERE slug=%s LIMIT 1",
                (slug,),
            )
            if exists:
                logger.info("[predict_4s] skip %s: 4s prediction already exists", slug)
                return
        except Exception:
            # On DB error, proceed (best-effort dedup)
            pass

        result = await predict_for_market_4s(slug, live_candle_fields=live_candle_fields)
        if result.get("error"):
            logger.info("[predict_4s] skip %s: %s", slug, result["error"])
            return

        prediction = result.get("prediction", "UNDEFINED")
        asyncio.create_task(_place_order_4s(slug=slug, prediction=prediction))

    except Exception as exc:
        logger.exception("[predict_4s] try_autopredict_4s error for %s", slug, exc_info=exc)


# ── query helpers ──────────────────────────────────────────────────────────────

async def _resolve_prediction_4s_lookup_slug(slug: str) -> str:
    """Resolve which market slug should be used for 4s prediction lookup.

    If the requested slug is the currently active 5m market, the relevant 4s
    prediction belongs to the next market (+300s). For future/ended markets,
    keep the slug unchanged.
    """
    if not slug:
        return slug
    try:
        row = await db.fetchone(
            "SELECT slug, ts, closed FROM poly_markets WHERE slug=%s LIMIT 1",
            (slug,),
        )
        if not row:
            return slug
        base_slug = str(row[0] or slug)
        market_ts = int(row[1]) if row[1] is not None else 0
        closed = int(row[2]) if len(row) > 2 and row[2] is not None else 0
        now_utc = int(time.time())
        is_active = bool(not closed and market_ts and market_ts <= now_utc < (market_ts + 300))
        if not is_active:
            return base_slug

        next_row = await db.fetchone(
            "SELECT slug FROM poly_markets WHERE ts=%s AND closed=0 LIMIT 1",
            (market_ts + 300,),
        )
        if next_row and next_row[0]:
            return str(next_row[0])
        return base_slug
    except Exception:
        return slug


async def get_saved_prediction_4s(slug: str) -> Optional[Dict[str, Any]]:
    """Return the stored 4s-early prediction payload for a slug, or None."""
    lookup_slug = await _resolve_prediction_4s_lookup_slug(slug)
    row = await db.fetchone(
        "SELECT payload_json, prediction_ts FROM poly_predictions_4s WHERE slug=%s LIMIT 1",
        (lookup_slug,),
    )
    if not row or not row[0]:
        return None
    try:
        data = json.loads(row[0])
        data["_prediction_ts"] = int(row[1]) if row[1] else None
        data["_lookup_slug"] = lookup_slug
        data["_requested_slug"] = slug
        return data
    except Exception:
        return None


async def list_predictions_4s(limit: int = 200) -> List[Dict[str, Any]]:
    """Return recent 4s-early predictions, newest first."""
    rows = await db.fetchall(
        """
        SELECT slug, prediction_ts, payload_json, created_at
        FROM poly_predictions_4s
        ORDER BY prediction_ts DESC
        LIMIT %s
        """,
        (int(limit),),
    )
    out = []
    for r in rows:
        payload = {}
        try:
            if r[2]:
                payload = json.loads(r[2])
        except Exception:
            pass
        out.append({
            "slug": r[0],
            "prediction_ts": int(r[1]) if r[1] else None,
            "created_at": str(r[3]) if r[3] else None,
            **payload,
        })
    return out


async def list_orders_4s(slug: Optional[str] = None, limit: int = 200) -> List[Dict[str, Any]]:
    """Return recent poly_live_orders rows where is_4s_early=1."""
    sql = """
        SELECT id, slug, asset_id, outcome_side, side, order_type, price, amount,
               fill_shares, fill_total_spent_usd, fill_avg_price_cents,
               clob_order_id, clob_status, clob_error_msg, prediction_direction, created_at
        FROM poly_live_orders
        WHERE is_4s_early=1
    """
    params: list = []
    if slug:
        sql += " AND slug=%s"
        params.append(slug)
    sql += " ORDER BY created_at DESC LIMIT %s"
    params.append(int(limit))
    rows = await db.fetchall(sql, tuple(params))
    out = []
    for r in rows:
        out.append({
            "id": r[0], "slug": r[1], "asset_id": r[2], "outcome_side": r[3],
            "side": r[4], "order_type": r[5], "price": r[6], "amount": r[7],
            "fill_shares": r[8], "fill_total_spent_usd": r[9], "fill_avg_price_cents": r[10],
            "clob_order_id": r[11], "clob_status": r[12], "clob_error_msg": r[13],
            "prediction_direction": r[14],
            "created_at": str(r[15]) if r[15] else None,
        })
    return out
