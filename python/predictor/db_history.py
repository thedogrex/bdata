import json
import datetime
import time
import logging
from typing import Optional
from db import DbProvider
import app.config as config

db = DbProvider()
logger = logging.getLogger(__name__)


def _safe_json_loads(val, default):
    if not val:
        return default
    try:
        return json.loads(val)
    except Exception:
        return default


async def save_backtest_run(result: dict) -> int:
    """Save a backtest run + horizon results to MySQL. Returns run_id."""
    params_json = json.dumps(result.get("params", {}), ensure_ascii=False)
    log_timings = bool(getattr(config, "LOG_TIME_RSI", False) and result.get("strategy") == "rsi_mean_reversion")
    run_insert_start = time.time() if log_timings else None

    query = """
        INSERT INTO backtest_runs
            (strategy, params_json, train_start, train_end, test_start, test_end,
             tbl, window_size, train_candles, test_candles, total_time_sec,
             is_bruteforce, bruteforce_id)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    run_id = await db.execute(query, (
        result["strategy"],
        params_json,
        result.get("train_start", ""),
        result.get("train_end", ""),
        result.get("test_start", ""),
        result.get("test_end", ""),
        result.get("table", "c_5m"),
        result.get("window_size", 0),
        result.get("train_candles", 0),
        result.get("test_candles", 0),
        result.get("total_time_sec", 0),
        1 if result.get("is_bruteforce") else 0,
        result.get("bruteforce_id"),
    ))

    if log_timings and run_insert_start is not None:
        logger.info(
            "[RSI_TIMING] DB_RUN_INSERT run_id=%s %.4fs",
            run_id,
            time.time() - run_insert_start,
        )

    horizons = result.get("horizons", {})
    horizon_rows: list[tuple] = []
    for horizon_str, h in horizons.items():
        err_msg = h.get("error") if isinstance(h, dict) else None
        conf_json = h.get("confidence_distribution", {}) if isinstance(h, dict) else {}
        if err_msg:
            # Persist the horizon anyway (otherwise history shows "no results")
            # Store the reason inside confidence_json for visibility.
            conf_json = {"error": str(err_msg)}
        horizon_rows.append((
            run_id,
            int(horizon_str),
            h.get("accuracy", 0),
            h.get("accuracy_pct", 0),
            h.get("total_candles", 0),
            h.get("signals", 0),
            h.get("skipped", 0),
            h.get("volatility_skips", 0),
            h.get("correct", 0),
            h.get("wrong", 0),
            h.get("up_predictions", 0),
            h.get("up_correct", 0),
            h.get("up_accuracy", 0),
            h.get("down_predictions", 0),
            h.get("down_correct", 0),
            h.get("down_accuracy", 0),
            h.get("streaks", {}).get("max_win_streak", 0),
            h.get("streaks", {}).get("max_lose_streak", 0),
            h.get("fit_time_sec", 0),
            h.get("predict_time_sec", 0),
            json.dumps(h.get("monthly", []), ensure_ascii=False),
            json.dumps(h.get("daily", []), ensure_ascii=False),
            json.dumps(conf_json, ensure_ascii=False),
        ))

    if horizon_rows:
        horizon_insert_start = time.time() if log_timings else None
        horizon_insert = """
            INSERT INTO backtest_horizons
                (run_id, horizon, accuracy, accuracy_pct, total_candles, signals,
                 skipped, volatility_skips, correct, wrong, up_predictions, up_correct, up_accuracy,
                 down_predictions, down_correct, down_accuracy,
                 max_win_streak, max_lose_streak, fit_time_sec, predict_time_sec,
                 monthly_json, daily_json, confidence_json)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """
        await db.executemany(horizon_insert, horizon_rows)
        if log_timings and horizon_insert_start is not None:
            logger.info(
                "[RSI_TIMING] DB_HORIZON_INSERT run_id=%s rows=%d %.4fs",
                run_id,
                len(horizon_rows),
                time.time() - horizon_insert_start,
            )

    return run_id


async def get_history(limit: int = 100, strategy: Optional[str] = None,
                      min_accuracy: Optional[float] = None,
                      bruteforce_id: Optional[int] = None,
                      exclude_bruteforce: bool = False) -> list[dict]:
    """Load backtest history from MySQL.

    Uses a single JOIN query instead of N+1 to fetch run + horizon summary.
    Excludes heavy blobs (monthly/daily/confidence) — use get_history_detail for those.
    """
    where = ["1=1"]
    params: list = []
    if strategy:
        where.append("r.strategy = %s")
        params.append(strategy)
    if bruteforce_id is not None:
        where.append("r.bruteforce_id = %s")
        params.append(bruteforce_id)
    if exclude_bruteforce:
        where.append("(r.is_bruteforce = 0 OR r.is_bruteforce IS NULL)")
    if min_accuracy is not None:
        where.append("h.accuracy_pct >= %s")
        params.append(float(min_accuracy))

    where_sql = " AND ".join(where)

    query = f"""
        SELECT r.id, r.strategy, r.params_json, r.train_start, r.train_end,
               r.test_start, r.test_end, r.tbl, r.window_size,
               r.train_candles, r.test_candles, r.total_time_sec,
               r.is_bruteforce, r.bruteforce_id, r.created_at,
               h.horizon, h.accuracy_pct, h.signals, h.correct, h.wrong,
               h.volatility_skips, h.max_win_streak, h.max_lose_streak
        FROM backtest_runs r
        LEFT JOIN backtest_horizons h ON h.run_id = r.id
        WHERE {where_sql}
        ORDER BY r.created_at DESC, r.id DESC, h.horizon ASC
        LIMIT %s
    """
    params.append(int(limit) * 5)  # up to 5 horizon rows per run
    rows = await db.fetchall(query, params)

    runs_map: dict = {}
    run_order: list = []
    for row in rows:
        run_id = row[0]
        if run_id not in runs_map:
            runs_map[run_id] = {
                "id": run_id,
                "strategy": row[1],
                "params": _safe_json_loads(row[2], {}),
                "train_start": row[3],
                "train_end": row[4],
                "test_start": row[5],
                "test_end": row[6],
                "train_period": f"{row[3]} -> {row[4]}",
                "test_period": f"{row[5]} -> {row[6]}",
                "table": row[7],
                "window_size": row[8],
                "train_candles": row[9],
                "test_candles": row[10],
                "total_time_sec": row[11],
                "is_bruteforce": bool(row[12]),
                "bruteforce_id": row[13],
                "created_at": str(row[14]) if row[14] else "",
                "horizons": {},
            }
            run_order.append(run_id)
        if row[15] is not None:  # horizon column
            runs_map[run_id]["horizons"][str(row[15])] = {
                "accuracy_pct": row[16],
                "signals": row[17],
                "correct": row[18],
                "wrong": row[19],
                "streaks": {
                    "max_win_streak": row[21],
                    "max_lose_streak": row[22],
                },
                "volatility_skips": row[20],
            }

    results = []
    for run_id in run_order:
        if len(results) >= limit:
            break
        results.append(runs_map[run_id])
    return results


async def get_bf_runs_paginated(
    bruteforce_id: int,
    offset: int = 0,
    limit: int = 20,
    min_accuracy: Optional[float] = None,
    window_size: Optional[int] = None,
) -> dict:
    """Server-side paginated BF runs with total count. Returns {runs, total}."""
    where = ["r.bruteforce_id = %s"]
    params: list = [bruteforce_id]
    if min_accuracy is not None:
        where.append("h.accuracy_pct >= %s")
        params.append(float(min_accuracy))
    if window_size is not None:
        where.append("r.window_size = %s")
        params.append(int(window_size))
    where_sql = " AND ".join(where)

    # Count distinct runs matching filters
    count_q = f"""
        SELECT COUNT(DISTINCT r.id)
        FROM backtest_runs r
        LEFT JOIN backtest_horizons h ON h.run_id = r.id
        WHERE {where_sql}
    """
    count_row = await db.fetchone(count_q, params)
    total = int(count_row[0]) if count_row else 0

    # Fetch paginated run IDs first
    ids_q = f"""
        SELECT DISTINCT r.id
        FROM backtest_runs r
        LEFT JOIN backtest_horizons h ON h.run_id = r.id
        WHERE {where_sql}
        ORDER BY r.id DESC
        LIMIT %s OFFSET %s
    """
    id_params = list(params) + [int(limit), int(offset)]
    id_rows = await db.fetchall(ids_q, id_params)
    run_ids = [r[0] for r in id_rows]

    if not run_ids:
        return {"runs": [], "total": total}

    # Fetch full data for those run IDs (with horizon summary, no heavy blobs)
    placeholders = ",".join(["%s"] * len(run_ids))
    data_q = f"""
        SELECT r.id, r.strategy, r.params_json, r.train_start, r.train_end,
               r.test_start, r.test_end, r.tbl, r.window_size,
               r.train_candles, r.test_candles, r.total_time_sec,
               r.is_bruteforce, r.bruteforce_id, r.created_at,
               h.horizon, h.accuracy_pct, h.signals, h.correct, h.wrong,
               h.volatility_skips, h.max_win_streak, h.max_lose_streak
        FROM backtest_runs r
        LEFT JOIN backtest_horizons h ON h.run_id = r.id
        WHERE r.id IN ({placeholders})
        ORDER BY r.id DESC, h.horizon ASC
    """
    rows = await db.fetchall(data_q, tuple(run_ids))

    runs_map: dict = {}
    run_order: list = []
    for row in rows:
        run_id = row[0]
        if run_id not in runs_map:
            runs_map[run_id] = {
                "id": run_id,
                "strategy": row[1],
                "params": _safe_json_loads(row[2], {}),
                "train_start": row[3],
                "train_end": row[4],
                "test_start": row[5],
                "test_end": row[6],
                "train_period": f"{row[3]} -> {row[4]}",
                "test_period": f"{row[5]} -> {row[6]}",
                "table": row[7],
                "window_size": row[8],
                "train_candles": row[9],
                "test_candles": row[10],
                "total_time_sec": row[11],
                "is_bruteforce": bool(row[12]),
                "bruteforce_id": row[13],
                "created_at": str(row[14]) if row[14] else "",
                "horizons": {},
            }
            run_order.append(run_id)
        if row[15] is not None:
            runs_map[run_id]["horizons"][str(row[15])] = {
                "accuracy_pct": row[16],
                "signals": row[17],
                "correct": row[18],
                "wrong": row[19],
                "volatility_skips": row[20],
                "streaks": {
                    "max_win_streak": row[21],
                    "max_lose_streak": row[22],
                },
            }

    runs = [runs_map[rid] for rid in run_order]
    return {"runs": runs, "total": total}


async def get_history_detail(run_id: int) -> Optional[dict]:
    """Load a single backtest run with its horizon data by run_id."""
    row = await db.fetchone(
        "SELECT id, strategy, params_json, train_start, train_end, "
        "test_start, test_end, tbl, window_size, train_candles, test_candles, "
        "total_time_sec, is_bruteforce, bruteforce_id, created_at "
        "FROM backtest_runs WHERE id = %s", (run_id,)
    )
    if not row:
        return None

    run = {
        "id": row[0],
        "strategy": row[1],
        "params": _safe_json_loads(row[2], {}),
        "train_start": row[3],
        "train_end": row[4],
        "test_start": row[5],
        "test_end": row[6],
        "train_period": f"{row[3]} -> {row[4]}",
        "test_period": f"{row[5]} -> {row[6]}",
        "table": row[7],
        "window_size": row[8],
        "train_candles": row[9],
        "test_candles": row[10],
        "total_time_sec": row[11],
        "is_bruteforce": bool(row[12]),
        "bruteforce_id": row[13],
        "created_at": str(row[14]) if row[14] else "",
        "horizons": {},
    }

    h_rows = await db.fetchall(
        "SELECT * FROM backtest_horizons WHERE run_id = %s ORDER BY horizon", (run_id,)
    )
    for hr in h_rows:
        run["horizons"][str(hr[2])] = {
            "accuracy": hr[3],
            "accuracy_pct": hr[4],
            "total_candles": hr[5],
            "signals": hr[6],
            "skipped": hr[7],
            "volatility_skips": hr[8],
            "correct": hr[9],
            "wrong": hr[10],
            "up_predictions": hr[11],
            "up_correct": hr[12],
            "up_accuracy": hr[13],
            "down_predictions": hr[14],
            "down_correct": hr[15],
            "down_accuracy": hr[16],
            "streaks": {
                "max_win_streak": hr[17],
                "max_lose_streak": hr[18],
            },
            "fit_time_sec": hr[19],
            "predict_time_sec": hr[20],
            "monthly": _safe_json_loads(hr[21], []),
            "daily": _safe_json_loads(hr[22], []),
            "confidence_distribution": _safe_json_loads(hr[23], {}),
        }

    return run


async def delete_run(run_id: int) -> bool:
    await db.execute("DELETE FROM backtest_horizons WHERE run_id = %s", (run_id,))
    await db.execute("DELETE FROM backtest_runs WHERE id = %s", (run_id,))
    return True


async def delete_bruteforce_group(bruteforce_id: int) -> dict:
    run_rows = await db.fetchall(
        "SELECT id FROM backtest_runs WHERE bruteforce_id = %s",
        (bruteforce_id,)
    )
    run_ids = [r[0] for r in run_rows if r and r[0] is not None]
    deleted_runs = len(run_ids)

    if run_ids:
        placeholders = ",".join(["%s"] * len(run_ids))
        await db.execute(
            f"DELETE FROM backtest_horizons WHERE run_id IN ({placeholders})",
            tuple(run_ids),
        )
    await db.execute(
        "DELETE FROM backtest_runs WHERE bruteforce_id = %s",
        (bruteforce_id,),
    )

    await db.execute(
        "DELETE FROM bruteforce_sessions WHERE id = %s",
        (bruteforce_id,),
    )

    return {
        "status": "deleted",
        "bruteforce_id": bruteforce_id,
        "deleted_runs": deleted_runs,
    }


async def clear_history() -> int:
    await db.execute("DELETE FROM backtest_horizons", None)
    await db.execute("DELETE FROM backtest_runs", None)
    return 1


async def save_bruteforce_session(session: dict) -> int:
    query = """
        INSERT INTO bruteforce_sessions
            (strategy, param_grid_json, train_start, train_end, test_start, test_end,
             tbl, horizon, window_size, retrain_every, total_combos, combos_json, status)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    return await db.execute(query, (
        session["strategy"],
        json.dumps(session["param_grid"], ensure_ascii=False),
        session["train_start"],
        session["train_end"],
        session["test_start"],
        session["test_end"],
        session.get("table", "c_5m"),
        session.get("horizon", 1),
        session.get("window_size", 5000),
        session.get("retrain_every", 500),
        session.get("total_combos", 0),
        json.dumps(session.get("combos", []), ensure_ascii=False),
        "running",
    ))


async def update_bruteforce_session(bf_id: int, updates: dict) -> None:
    sets = []
    params = []
    for k, v in updates.items():
        sets.append(f"{k} = %s")
        if isinstance(v, (dict, list)):
            params.append(json.dumps(v, ensure_ascii=False))
        else:
            params.append(v)
    params.append(bf_id)
    query = f"UPDATE bruteforce_sessions SET {', '.join(sets)} WHERE id = %s"
    await db.execute(query, tuple(params))


async def get_bruteforce_sessions(limit: int = 50) -> list[dict]:
    rows = await db.fetchall(
        f"SELECT s.id, s.strategy, s.param_grid_json, s.train_start, s.train_end, "
        f"s.test_start, s.test_end, s.tbl, s.horizon, s.window_size, s.retrain_every, "
        f"s.total_combos, s.completed, s.best_accuracy, s.best_params_json, "
        f"s.status, s.total_time_sec, s.elapsed_before_pause, s.created_at, "
        f"COALESCE(h.signals, 0) as total_signals, "
        f"(SELECT COUNT(*) FROM backtest_runs br WHERE br.bruteforce_id = s.id) AS run_count "
        f"FROM bruteforce_sessions s "
        f"LEFT JOIN backtest_runs r ON r.id = ( "
        f"  SELECT r2.id "
        f"  FROM backtest_runs r2 "
        f"  LEFT JOIN backtest_horizons h2 ON h2.run_id = r2.id AND h2.horizon = s.horizon "
        f"  WHERE r2.bruteforce_id = s.id AND r2.is_bruteforce = 1 "
        f"  ORDER BY h2.accuracy_pct DESC, r2.id DESC LIMIT 1 "
        f") "
        f"LEFT JOIN backtest_horizons h ON h.run_id = r.id AND h.horizon = s.horizon "
        f"ORDER BY s.created_at DESC LIMIT {limit}"
    )
    results = []
    for r in rows:
        # Compute avg signals per month if we have signals and test dates
        avg_signals_per_month = None
        if r[19] and r[5] and r[6]:  # total_signals, test_start, test_end
            try:
                start = datetime.datetime.strptime(str(r[5]), "%Y-%m-%d")
                end = datetime.datetime.strptime(str(r[6]), "%Y-%m-%d")
                if end > start:
                    months = (end.year - start.year) * 12 + (end.month - start.month) + 1
                    if months > 0:
                        avg_signals_per_month = round(r[19] / months)
            except Exception:
                pass
        results.append({
            "id": r[0],
            "strategy": r[1],
            "param_grid": json.loads(r[2]) if r[2] else {},
            "train_start": r[3],
            "train_end": r[4],
            "test_start": r[5],
            "test_end": r[6],
            "table": r[7],
            "horizon": r[8],
            "window_size": r[9],
            "retrain_every": r[10],
            "total_combos": r[11],
            "completed": r[12],
            "best_accuracy": r[13],
            "best_params": json.loads(r[14]) if r[14] else {},
            "status": r[15],
            "total_time_sec": r[16],
            "elapsed_before_pause": r[17],
            "created_at": str(r[18]) if r[18] else "",
            "signals": r[19] or None,
            "avg_signals_per_month": avg_signals_per_month,
            "run_count": r[20] or 0,
        })
    return results


async def get_bruteforce_session_by_id(bf_id: int) -> Optional[dict]:
    """Load a single BF session with full data including combos_json for resume."""
    row = await db.fetchone(
        "SELECT id, strategy, param_grid_json, train_start, train_end, "
        "test_start, test_end, tbl, horizon, window_size, retrain_every, "
        "total_combos, combos_json, completed, best_accuracy, best_params_json, "
        "status, total_time_sec, elapsed_before_pause, created_at "
        "FROM bruteforce_sessions WHERE id = %s", (bf_id,)
    )
    if not row:
        return None
    return {
        "id": row[0],
        "strategy": row[1],
        "param_grid": json.loads(row[2]) if row[2] else {},
        "train_start": row[3],
        "train_end": row[4],
        "test_start": row[5],
        "test_end": row[6],
        "table": row[7],
        "horizon": row[8],
        "window_size": row[9],
        "retrain_every": row[10],
        "total_combos": row[11],
        **_parse_combos_payload(row[12]),
        "completed": row[13],
        "best_accuracy": row[14],
        "best_params": json.loads(row[15]) if row[15] else {},
        "status": row[16],
        "total_time_sec": row[17],
        "elapsed_before_pause": row[18],
        "created_at": str(row[19]) if row[19] else "",
    }


def _parse_combos_payload(raw_json: str | None) -> dict:
    if not raw_json:
        return {"combos": [], "processes": 1}
    try:
        decoded = json.loads(raw_json)
    except json.JSONDecodeError:
        return {"combos": [], "processes": 1}
    if isinstance(decoded, dict):
        combos = decoded.get("items")
        if not isinstance(combos, list):
            combos = []
        processes = decoded.get("processes", 1)
        return {"combos": combos, "processes": max(1, int(processes or 1))}
    if isinstance(decoded, list):
        return {"combos": decoded, "processes": 1}
    return {"combos": [], "processes": 1}


async def get_completed_combo_indices(
    bf_id: int,
    ignore_keys: set[str] | None = None,
    required_thresholds: list[float] | None = None,
) -> set[str]:
    """Get set of normalized params JSON strings for completed combos in a BF session."""
    rows = await db.fetchall(
        "SELECT params_json FROM backtest_runs WHERE bruteforce_id = %s",
        (bf_id,)
    )
    result = set()
    threshold_seen: dict[str, set[str]] = {}
    required_thr_keys: set[str] | None = None
    if required_thresholds:
        required_thr_keys = {f"{float(t):.6f}" for t in required_thresholds}
    for r in rows:
        if r[0]:
            try:
                # Normalize: parse and re-dump with sort_keys for consistent matching
                parsed = json.loads(r[0])
                filtered = parsed
                if ignore_keys:
                    filtered = {k: v for k, v in parsed.items() if k not in ignore_keys}
                normalized = json.dumps(filtered, sort_keys=True, ensure_ascii=False)
                if required_thr_keys:
                    thr_val = parsed.get("threshold")
                    try:
                        thr_key = f"{float(thr_val):.6f}"
                    except (TypeError, ValueError):
                        continue
                    threshold_seen.setdefault(normalized, set()).add(thr_key)
                else:
                    result.add(normalized)
            except (json.JSONDecodeError, TypeError):
                result.add(r[0])
    if required_thr_keys:
        for combo_key, seen in threshold_seen.items():
            if seen.issuperset(required_thr_keys):
                result.add(combo_key)
    return result


async def get_runs_by_ids(run_ids: list[int], horizon: int = 1) -> list[dict]:
    """Fetch multiple runs by their IDs with full horizon data (including monthly)."""
    if not run_ids:
        return []
    placeholders = ",".join(["%s"] * len(run_ids))
    rows = await db.fetchall(f"""
        SELECT r.id, r.strategy, r.params_json, r.train_start, r.train_end,
               r.test_start, r.test_end, r.window_size,
               h.accuracy_pct, h.signals, h.correct, h.wrong, h.skipped, h.volatility_skips,
               h.max_win_streak, h.max_lose_streak,
               r.total_time_sec, r.created_at,
               h.monthly_json, h.up_predictions, h.up_correct, h.up_accuracy,
               h.down_predictions, h.down_correct, h.down_accuracy, h.horizon
        FROM backtest_runs r
        JOIN backtest_horizons h ON h.run_id = r.id
        WHERE r.id IN ({placeholders}) AND h.horizon = %s
        ORDER BY h.accuracy_pct DESC
    """, tuple(run_ids) + (horizon,))

    results = []
    for r in rows:
        monthly = _safe_json_loads(r[18], [])
        results.append({
            "id": r[0],
            "strategy": r[1],
            "params": json.loads(r[2]) if r[2] else {},
            "train_start": r[3],
            "train_end": r[4],
            "test_start": r[5],
            "test_end": r[6],
            "window_size": r[7],
            "accuracy_pct": r[8],
            "signals": r[9],
            "correct": r[10],
            "wrong": r[11],
            "skipped": r[12],
            "volatility_skips": r[13],
            "max_win_streak": r[14],
            "max_lose_streak": r[15],
            "total_time_sec": r[16],
            "created_at": str(r[17]) if r[17] else "",
            "monthly": monthly,
            "up_predictions": r[19],
            "up_correct": r[20],
            "up_accuracy": r[21],
            "down_predictions": r[22],
            "down_correct": r[23],
            "down_accuracy": r[24],
            "horizon": r[25],
        })
    return results


async def get_best_runs(
    limit: int = 20,
    horizon: int = 1,
    signals_min: int | None = None,
    signals_max: int | None = None,
    bruteforce_id: int | None = None,
) -> list[dict]:
    """Get top N runs sorted by accuracy for a given horizon."""
    where = ["h.horizon = %s", "h.signals > 0"]
    params: list = [horizon]
    if signals_min is not None:
        where.append("h.signals >= %s")
        params.append(int(signals_min))
    if signals_max is not None:
        where.append("h.signals <= %s")
        params.append(int(signals_max))
    if bruteforce_id is not None:
        where.append("r.bruteforce_id = %s")
        params.append(int(bruteforce_id))

    rows = await db.fetchall(f"""
        SELECT r.id, r.strategy, r.params_json, r.train_start, r.train_end,
               r.test_start, r.test_end, r.window_size,
               h.accuracy_pct, h.signals, h.correct, h.wrong, h.skipped, h.volatility_skips,
               h.max_win_streak, h.max_lose_streak,
               r.total_time_sec, r.created_at
        FROM backtest_runs r
        JOIN backtest_horizons h ON h.run_id = r.id
        WHERE {" AND ".join(where)}
        ORDER BY h.accuracy_pct DESC
        LIMIT %s
    """, tuple(params + [limit]))

    results = []
    for r in rows:
        results.append({
            "id": r[0],
            "strategy": r[1],
            "params": json.loads(r[2]) if r[2] else {},
            "train_start": r[3],
            "train_end": r[4],
            "test_start": r[5],
            "test_end": r[6],
            "window_size": r[7],
            "accuracy_pct": r[8],
            "signals": r[9],
            "correct": r[10],
            "wrong": r[11],
            "skipped": r[12],
            "volatility_skips": r[13],
            "max_win_streak": r[14],
            "max_lose_streak": r[15],
            "total_time_sec": r[16],
            "created_at": str(r[17]) if r[17] else "",
        })
    return results
