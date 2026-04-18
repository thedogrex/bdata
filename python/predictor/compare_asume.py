"""Compare Asume.

Run exact signal logic from the first active template in `poly_pred_templates`
for markets in a date range. For each market candle:
1) Base prediction on `c_5m`
2) Emulated prediction by replacing only the market candle with snapshot from
   `c_5m_3s`, `c_5m_5s`, `c_5m_7s`, `c_5m_8s`.
"""

import time
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional

from db import DbProvider
from predictor.data_loader import add_direction
from predictor.features import add_technical_features
from predictor.strategies import get_strategy, STRATEGY_REGISTRY
from predictor.utils.async_utils import resolve_awaitable
from predictor.poly_service import list_pred_templates

db = DbProvider()

OFFSET_TABLES = {
    "c_5m_8s": "8 s early",
    "c_5m_7s": "7 s early",
    "c_5m_5s": "5 s early",
    "c_5m_4s": "4 s early",
    "c_5m_3s": "3 s early",
}

_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quota_volume", "trades", "taker_base_volume", "taker_quota_volume",
]


def _date_to_epoch_sec(date_str: str, end_of_day: bool = False) -> int:
    t = "23:59:59" if end_of_day else "00:00:00"
    return int(pd.Timestamp(f"{date_str} {t}").timestamp())


async def _load_market_rows(date_from: str, date_to: str, limit: int) -> List[Dict[str, Any]]:
    from_ts = _date_to_epoch_sec(date_from, False)
    to_ts = _date_to_epoch_sec(date_to, True)
    rows = await db.fetchall(
        """
        SELECT slug, ts
        FROM poly_markets
        WHERE ts >= %s AND ts <= %s
        ORDER BY ts DESC
        LIMIT %s
        """,
        (int(from_ts), int(to_ts), int(limit)),
    )
    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append({"slug": str(r[0]), "ts": int(r[1])})
    return out


async def _load_window_for_market(market_ts: int, window_size: int) -> pd.DataFrame:
    market_ts_us = int(market_ts) * 1_000_000
    rows = await db.fetchall(
        """
        SELECT open_time, open, high, low, close, volume,
               close_time, quota_volume, trades, taker_base_volume, taker_quota_volume
        FROM c_5m
        WHERE open_time <= %s
        ORDER BY open_time DESC
        LIMIT %s
        """,
        (market_ts_us, int(window_size)),
    )
    if not rows:
        return pd.DataFrame(columns=_COLS)
    rows = list(reversed(rows))
    df = pd.DataFrame(rows, columns=_COLS)
    for c in ["open", "high", "low", "close", "volume", "quota_volume",
              "taker_base_volume", "taker_quota_volume"]:
        df[c] = df[c].astype(float)
    df["trades"] = df["trades"].astype(int)
    return df


async def _load_offset_candle(table: str, open_time_us: int) -> Optional[Dict[str, Any]]:
    open_time_us = int(open_time_us)
    row = await db.fetchone(
        f"""
        SELECT open_time, open, high, low, close, volume,
               close_time, quota_volume, trades, taker_base_volume, taker_quota_volume
        FROM {table}
        WHERE open_time = %s
        LIMIT 1
        """,
        (open_time_us,),
    )
    if not row:
        return None
    return {
        "open_time": int(row[0]),
        "open": float(row[1]),
        "high": float(row[2]),
        "low": float(row[3]),
        "close": float(row[4]),
        "volume": float(row[5]),
        "close_time": int(row[6]),
        "quota_volume": float(row[7]),
        "trades": int(row[8]),
        "taker_base_volume": float(row[9]),
        "taker_quota_volume": float(row[10]),
    }


async def _load_cached_offset(
    template_id: int,
    signal_open_time: int,
    source_table: str,
) -> Optional[Dict[str, Any]]:
    row = await db.fetchone(
        """
        SELECT has_data, delayed_signal, delayed_prob, delayed_close, delayed_rsi,
               is_diff, message, ref_signal, ref_prob, ref_close, ref_rsi
        FROM compare_asume_signal_cache
        WHERE template_id = %s AND signal_open_time = %s AND source_table = %s
        ORDER BY updated_at DESC, id DESC
        LIMIT 1
        """,
        (int(template_id), int(signal_open_time), source_table),
    )
    if not row:
        return None
    return {
        "has_data": bool(row[0]),
        "delayed_signal": row[1],
        "delayed_prob": float(row[2]) if row[2] is not None else None,
        "delayed_close": float(row[3]) if row[3] is not None else None,
        "delayed_rsi": float(row[4]) if row[4] is not None else None,
        "is_diff": bool(row[5]),
        "message": row[6],
        "ref_signal": row[7],
        "ref_prob": float(row[8]) if row[8] is not None else None,
        "ref_close": float(row[9]) if row[9] is not None else None,
        "ref_rsi": float(row[10]) if row[10] is not None else None,
    }


async def _upsert_cached_offset(
    template_id: int,
    strategy_name: str,
    strategy_params: dict,
    window_size: int,
    horizon: int,
    market_slug: str,
    market_ts: int,
    market_open_time: int,
    signal_open_time: int,
    source_table: str,
    ref_signal: str,
    ref_prob: float,
    ref_close: float,
    ref_rsi: float,
    has_data: bool,
    delayed_signal: Optional[str],
    delayed_prob: Optional[float],
    delayed_close: Optional[float],
    delayed_rsi: Optional[float],
    is_diff: bool,
    message: Optional[str],
) -> None:
    params_json = json.dumps(strategy_params or {}, separators=(",", ":"))
    await db.execute(
        """
        INSERT INTO compare_asume_signal_cache (
            template_id, strategy_name, strategy_params_json, window_size, horizon,
            market_slug, market_ts, market_open_time, signal_open_time, source_table,
            ref_signal, ref_prob, ref_close, ref_rsi,
            has_data, delayed_signal, delayed_prob, delayed_close, delayed_rsi, is_diff, message
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) AS new
        ON DUPLICATE KEY UPDATE
            strategy_name = new.strategy_name,
            strategy_params_json = new.strategy_params_json,
            window_size = new.window_size,
            horizon = new.horizon,
            market_slug = new.market_slug,
            market_open_time = new.market_open_time,
            ref_signal = new.ref_signal,
            ref_prob = new.ref_prob,
            ref_close = new.ref_close,
            ref_rsi = new.ref_rsi,
            has_data = new.has_data,
            delayed_signal = new.delayed_signal,
            delayed_prob = new.delayed_prob,
            delayed_close = new.delayed_close,
            delayed_rsi = new.delayed_rsi,
            is_diff = new.is_diff,
            message = new.message
        """,
        (
            int(template_id),
            strategy_name,
            params_json,
            int(window_size),
            int(horizon),
            market_slug,
            int(market_ts),
            int(market_open_time),
            int(signal_open_time),
            source_table,
            ref_signal,
            float(ref_prob),
            float(ref_close),
            float(ref_rsi),
            int(has_data),
            delayed_signal,
            delayed_prob,
            delayed_close,
            delayed_rsi,
            int(is_diff),
            message,
        ),
    )


async def _predict_last_candle(
    df_window: pd.DataFrame,
    strategy_name: str,
    strategy_params: dict,
    horizon: int,
) -> Optional[Dict[str, Any]]:
    if len(df_window) < 120:
        return None

    df = add_direction(df_window).reset_index(drop=True)
    df_feat = add_technical_features(df)
    df_train = df_feat.iloc[:-1].reset_index(drop=True)
    df_predict = df_feat.iloc[[-1]].reset_index(drop=True)
    if len(df_train) < 100:
        return None

    strategy = get_strategy(strategy_name, strategy_params)
    strategy.fit(df_train, horizon=horizon)

    pred_arr = await resolve_awaitable(strategy.predict(df_predict, horizon=horizon))
    prob_arr = await resolve_awaitable(strategy.predict_proba(df_predict, horizon=horizon))
    pred = int(pred_arr[0])
    prob = float(prob_arr[0])
    label = "UP" if pred == 1 else ("DOWN" if pred == 0 else "UNDEFINED")

    period = strategy.params.get("rsi_period", 14)
    rsi_col = f"rsi_{period}" if f"rsi_{period}" in df_predict.columns else "rsi_14"
    rsi_val = float(np.nan_to_num(df_predict.iloc[0][rsi_col], nan=50.0))

    return {
        "signal": label,
        "prob": round(prob, 4),
        "close": round(float(df_predict.iloc[0]["close"]), 2),
        "rsi": round(rsi_val, 1),
    }


async def _predict_at_row(
    df_window: pd.DataFrame,
    strategy_name: str,
    strategy_params: dict,
    horizon: int,
    row_index: int,
) -> Optional[Dict[str, Any]]:
    if len(df_window) < 120:
        return None
    if row_index < 0 or row_index >= len(df_window):
        return None

    df = add_direction(df_window).reset_index(drop=True)
    df_feat = add_technical_features(df)
    df_train = df_feat.iloc[:row_index].reset_index(drop=True)
    df_predict = df_feat.iloc[[row_index]].reset_index(drop=True)
    if len(df_train) < 100:
        return None

    strategy = get_strategy(strategy_name, strategy_params)
    strategy.fit(df_train, horizon=horizon)

    pred_arr = await resolve_awaitable(strategy.predict(df_predict, horizon=horizon))
    prob_arr = await resolve_awaitable(strategy.predict_proba(df_predict, horizon=horizon))
    pred = int(pred_arr[0])
    prob = float(prob_arr[0])
    label = "UP" if pred == 1 else ("DOWN" if pred == 0 else "UNDEFINED")

    period = strategy.params.get("rsi_period", 14)
    rsi_col = f"rsi_{period}" if f"rsi_{period}" in df_predict.columns else "rsi_14"
    rsi_val = float(np.nan_to_num(df_predict.iloc[0][rsi_col], nan=50.0))

    return {
        "signal": label,
        "prob": round(prob, 4),
        "close": round(float(df_predict.iloc[0]["close"]), 2),
        "rsi": round(rsi_val, 1),
    }


async def run_compare(
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    limit: int = 500,
) -> Dict[str, Any]:
    t0 = time.time()
    today = datetime.utcnow().date().isoformat()
    date_from = date_from or today
    date_to = date_to or date_from

    templates = await list_pred_templates()
    active = [t for t in templates if t["active"]]
    if not active:
        return {"error": "No active prediction templates. Create and enable at least one."}
    tpl = active[0]
    strategy_name = tpl["strategy"]
    strategy_params = tpl["params"] or {}
    horizon = max(1, min(3, int(tpl["horizon"])))
    window_size = int(tpl["window_size"])

    markets = await _load_market_rows(date_from=date_from, date_to=date_to, limit=limit)
    if not markets:
        return {"error": f"No markets found in range {date_from}..{date_to}."}

    comparison: List[Dict[str, Any]] = []
    summary = {
        tbl: {
            "label": lbl,
            "found_offset_candle": 0,
            "total_compared": 0,
            "same_signal": 0,
            "diff_signal": 0,
            "missing_offset_candle": 0,
            "false_positive": 0,
            "match_pct": 0.0,
        }
        for tbl, lbl in OFFSET_TABLES.items()
    }

    processed = 0
    skipped = 0

    for m in markets:
        slug = m["slug"]
        ts = int(m["ts"])
        market_ot = ts * 1_000_000
        interval_us = 5 * 60 * 1_000_000
        target_open_us = market_ot - max(0, horizon - 1) * interval_us
        dt = pd.Timestamp(market_ot, unit="us").strftime("%Y-%m-%d %H:%M")

        # Try to satisfy from cache entirely
        cached_offsets: Dict[str, Optional[Dict[str, Any]]] = {}
        fully_cached = True
        cached_ref = None
        for tbl in OFFSET_TABLES:
            cached_val = await _load_cached_offset(
                template_id=tpl["id"],
                signal_open_time=target_open_us,
                source_table=tbl,
            )
            cached_offsets[tbl] = cached_val
            if cached_val is None:
                fully_cached = False
            elif cached_ref is None and cached_val.get("ref_signal") is not None:
                cached_ref = cached_val

        if fully_cached and cached_ref is not None:
            row = {
                "slug": slug,
                "market_ts": ts,
                "market_open_time": market_ot,
                "dt": dt,
                "ref_signal": cached_ref.get("ref_signal"),
                "ref_prob": cached_ref.get("ref_prob"),
                "ref_close": cached_ref.get("ref_close"),
                "ref_rsi": cached_ref.get("ref_rsi"),
                "signal_open_time": target_open_us,
                "offsets": {},
                "any_diff": False,
            }
            for tbl, cached in cached_offsets.items():
                if cached is None:
                    continue
                if not cached["has_data"]:
                    row["offsets"][tbl] = {
                        "has_data": False,
                        "message": cached.get("message")
                        or f"candle not found (open_time={target_open_us})",
                    }
                    summary[tbl]["missing_offset_candle"] += 1
                else:
                    diff = bool(cached.get("is_diff"))
                    if diff:
                        row["any_diff"] = True
                    row["offsets"][tbl] = {
                        "has_data": True,
                        "source_open_time": target_open_us,
                        "signal": cached.get("delayed_signal"),
                        "prob": cached.get("delayed_prob"),
                        "close": cached.get("delayed_close"),
                        "rsi": cached.get("delayed_rsi"),
                        "diff": diff,
                    }
                    summary[tbl]["total_compared"] += 1
                    if diff:
                        summary[tbl]["diff_signal"] += 1
                    else:
                        summary[tbl]["same_signal"] += 1
                    if cached_ref.get("ref_signal") == "UNDEFINED" and cached.get("delayed_signal") not in (
                        None,
                        "UNDEFINED",
                    ):
                        summary[tbl]["false_positive"] += 1
            comparison.append(row)
            processed += 1
            continue

        # Need to compute base and remaining offsets
        base_window = await _load_window_for_market(market_ts=ts, window_size=window_size)
        if len(base_window) < window_size:
            skipped += 1
            continue

        ridx_arr = base_window.index[base_window["open_time"] == int(target_open_us)]
        if len(ridx_arr) == 0:
            skipped += 1
            continue
        signal_row_idx = ridx_arr[-1]
        signal_open_us = int(base_window.at[signal_row_idx, "open_time"])

        base_pred = await _predict_at_row(
            df_window=base_window,
            strategy_name=strategy_name,
            strategy_params=strategy_params,
            horizon=horizon,
            row_index=int(signal_row_idx),
        )
        if not base_pred:
            skipped += 1
            continue

        row = {
            "slug": slug,
            "market_ts": ts,
            "market_open_time": market_ot,
            "dt": dt,
            "ref_signal": base_pred["signal"],
            "ref_prob": base_pred["prob"],
            "ref_close": base_pred["close"],
            "ref_rsi": base_pred["rsi"],
            "signal_open_time": signal_open_us,
            "offsets": {},
            "any_diff": False,
        }

        for tbl in OFFSET_TABLES:
            # reuse cached if available
            if cached_offsets.get(tbl) is not None:
                cached = cached_offsets[tbl]
                if not cached["has_data"]:
                    row["offsets"][tbl] = {
                        "has_data": False,
                        "message": cached.get("message")
                        or f"candle not found (open_time={signal_open_us})",
                    }
                    summary[tbl]["missing_offset_candle"] += 1
                else:
                    diff = bool(cached.get("is_diff"))
                    if diff:
                        row["any_diff"] = True
                    row["offsets"][tbl] = {
                        "has_data": True,
                        "source_open_time": signal_open_us,
                        "signal": cached.get("delayed_signal"),
                        "prob": cached.get("delayed_prob"),
                        "close": cached.get("delayed_close"),
                        "rsi": cached.get("delayed_rsi"),
                        "diff": diff,
                    }
                    summary[tbl]["total_compared"] += 1
                    if diff:
                        summary[tbl]["diff_signal"] += 1
                    else:
                        summary[tbl]["same_signal"] += 1
                    if base_pred["signal"] == "UNDEFINED" and cached.get("delayed_signal") not in (
                        None,
                        "UNDEFINED",
                    ):
                        summary[tbl]["false_positive"] += 1
                continue

            off = await _load_offset_candle(table=tbl, open_time_us=signal_open_us)
            if not off:
                row["offsets"][tbl] = {
                    "has_data": False,
                    "message": f"candle not found (open_time={signal_open_us})",
                }
                summary[tbl]["missing_offset_candle"] += 1
                await _upsert_cached_offset(
                    template_id=tpl["id"],
                    strategy_name=strategy_name,
                    strategy_params=strategy_params,
                    window_size=window_size,
                    horizon=horizon,
                    market_slug=slug,
                    market_ts=ts,
                    market_open_time=market_ot,
                    signal_open_time=signal_open_us,
                    source_table=tbl,
                    ref_signal=base_pred["signal"],
                    ref_prob=base_pred["prob"],
                    ref_close=base_pred["close"],
                    ref_rsi=base_pred["rsi"],
                    has_data=False,
                    delayed_signal=None,
                    delayed_prob=None,
                    delayed_close=None,
                    delayed_rsi=None,
                    is_diff=False,
                    message=f"candle not found (open_time={signal_open_us})",
                )
                continue
            summary[tbl]["found_offset_candle"] += 1

            hybrid = base_window.copy()
            signal_ridx_arr = hybrid.index[hybrid["open_time"] == int(signal_open_us)]
            if len(signal_ridx_arr) == 0:
                row["offsets"][tbl] = {
                    "has_data": False,
                    "message": f"candle not found in c_5m window (open_time={signal_open_us})",
                }
                summary[tbl]["missing_offset_candle"] += 1
                await _upsert_cached_offset(
                    template_id=tpl["id"],
                    strategy_name=strategy_name,
                    strategy_params=strategy_params,
                    window_size=window_size,
                    horizon=horizon,
                    market_slug=slug,
                    market_ts=ts,
                    market_open_time=market_ot,
                    signal_open_time=signal_open_us,
                    source_table=tbl,
                    ref_signal=base_pred["signal"],
                    ref_prob=base_pred["prob"],
                    ref_close=base_pred["close"],
                    ref_rsi=base_pred["rsi"],
                    has_data=False,
                    delayed_signal=None,
                    delayed_prob=None,
                    delayed_close=None,
                    delayed_rsi=None,
                    is_diff=False,
                    message=f"candle not found in c_5m window (open_time={signal_open_us})",
                )
                continue
            ridx = signal_ridx_arr[-1]
            for col in _COLS:
                if col == "open_time":
                    continue
                hybrid.at[ridx, col] = off[col]

            off_pred = await _predict_at_row(
                df_window=hybrid,
                strategy_name=strategy_name,
                strategy_params=strategy_params,
                horizon=horizon,
                row_index=int(signal_ridx_arr[-1]),
            )
            if not off_pred:
                row["offsets"][tbl] = {
                    "has_data": False,
                    "message": "prediction unavailable after substitution",
                }
                summary[tbl]["missing_offset_candle"] += 1
                await _upsert_cached_offset(
                    template_id=tpl["id"],
                    strategy_name=strategy_name,
                    strategy_params=strategy_params,
                    window_size=window_size,
                    horizon=horizon,
                    market_slug=slug,
                    market_ts=ts,
                    market_open_time=market_ot,
                    signal_open_time=signal_open_us,
                    source_table=tbl,
                    ref_signal=base_pred["signal"],
                    ref_prob=base_pred["prob"],
                    ref_close=base_pred["close"],
                    ref_rsi=base_pred["rsi"],
                    has_data=False,
                    delayed_signal=None,
                    delayed_prob=None,
                    delayed_close=None,
                    delayed_rsi=None,
                    is_diff=False,
                    message="prediction unavailable after substitution",
                )
                continue

            diff = bool(off_pred["signal"] != base_pred["signal"])
            if diff:
                row["any_diff"] = True
            row["offsets"][tbl] = {
                "has_data": True,
                "source_open_time": int(off["open_time"]),
                "signal": off_pred["signal"],
                "prob": off_pred["prob"],
                "close": off_pred["close"],
                "rsi": off_pred["rsi"],
                "diff": diff,
            }
            summary[tbl]["total_compared"] += 1
            if diff:
                summary[tbl]["diff_signal"] += 1
            else:
                summary[tbl]["same_signal"] += 1
            if base_pred["signal"] == "UNDEFINED" and off_pred["signal"] != "UNDEFINED":
                summary[tbl]["false_positive"] += 1

            await _upsert_cached_offset(
                template_id=tpl["id"],
                strategy_name=strategy_name,
                strategy_params=strategy_params,
                window_size=window_size,
                horizon=horizon,
                market_slug=slug,
                market_ts=ts,
                market_open_time=market_ot,
                signal_open_time=signal_open_us,
                source_table=tbl,
                ref_signal=base_pred["signal"],
                ref_prob=base_pred["prob"],
                ref_close=base_pred["close"],
                ref_rsi=base_pred["rsi"],
                has_data=True,
                delayed_signal=off_pred["signal"],
                delayed_prob=off_pred["prob"],
                delayed_close=off_pred["close"],
                delayed_rsi=off_pred["rsi"],
                is_diff=diff,
                message=None,
            )

        comparison.append(row)
        processed += 1

    for tbl in OFFSET_TABLES:
        s = summary[tbl]
        total = int(s["total_compared"])
        s["match_pct"] = round((s["same_signal"] / total) * 100, 1) if total > 0 else 0.0

    elapsed = round(time.time() - t0, 2)
    return {
        "template": {
            "id": tpl["id"],
            "name": tpl["name"],
            "strategy": strategy_name,
            "params": strategy_params,
            "horizon": horizon,
            "window_size": window_size,
        },
        "date_from": date_from,
        "date_to": date_to,
        "requested_markets": len(markets),
        "processed_markets": processed,
        "skipped_markets": skipped,
        "comparison": comparison,
        "summary": summary,
        "elapsed_sec": elapsed,
    }
