import json
from typing import Optional
from db import DbProvider

db = DbProvider()


async def save_backtest_run(result: dict) -> int:
    """Save a backtest run + horizon results to MySQL. Returns run_id."""
    params_json = json.dumps(result.get("params", {}), ensure_ascii=False)

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

    horizons = result.get("horizons", {})
    for horizon_str, h in horizons.items():
        if "error" in h:
            continue
        await db.execute("""
            INSERT INTO backtest_horizons
                (run_id, horizon, accuracy, accuracy_pct, total_candles, signals,
                 skipped, correct, wrong, up_predictions, up_correct, up_accuracy,
                 down_predictions, down_correct, down_accuracy,
                 max_win_streak, max_lose_streak, fit_time_sec, predict_time_sec,
                 monthly_json, daily_json, confidence_json)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (
            run_id,
            int(horizon_str),
            h.get("accuracy", 0),
            h.get("accuracy_pct", 0),
            h.get("total_candles", 0),
            h.get("signals", 0),
            h.get("skipped", 0),
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
            json.dumps(h.get("confidence_distribution", {}), ensure_ascii=False),
        ))

    return run_id


async def get_history(limit: int = 100, strategy: Optional[str] = None,
                      min_accuracy: Optional[float] = None,
                      bruteforce_id: Optional[int] = None) -> list[dict]:
    """Load backtest history from MySQL."""
    where = "1=1"
    params = []
    if strategy:
        where += " AND r.strategy = %s"
        params.append(strategy)
    if bruteforce_id is not None:
        where += " AND r.bruteforce_id = %s"
        params.append(bruteforce_id)

    query = f"""
        SELECT r.id, r.strategy, r.params_json, r.train_start, r.train_end,
               r.test_start, r.test_end, r.tbl, r.window_size,
               r.train_candles, r.test_candles, r.total_time_sec,
               r.is_bruteforce, r.bruteforce_id, r.created_at
        FROM backtest_runs r
        WHERE {where}
        ORDER BY r.created_at DESC
        LIMIT {limit}
    """
    rows = await db.fetchall(query, params if params else None)
    results = []
    for row in rows:
        run_id = row[0]
        run = {
            "id": run_id,
            "strategy": row[1],
            "params": json.loads(row[2]) if row[2] else {},
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
                "correct": hr[8],
                "wrong": hr[9],
                "up_predictions": hr[10],
                "up_correct": hr[11],
                "up_accuracy": hr[12],
                "down_predictions": hr[13],
                "down_correct": hr[14],
                "down_accuracy": hr[15],
                "streaks": {
                    "max_win_streak": hr[16],
                    "max_lose_streak": hr[17],
                },
                "fit_time_sec": hr[18],
                "predict_time_sec": hr[19],
                "monthly": json.loads(hr[20]) if hr[20] else [],
                "daily": json.loads(hr[21]) if hr[21] else [],
                "confidence_distribution": json.loads(hr[22]) if hr[22] else {},
            }

        # Apply min_accuracy filter after loading horizons
        if min_accuracy is not None:
            dominated = True
            for h_data in run["horizons"].values():
                if h_data.get("accuracy_pct", 0) >= min_accuracy:
                    dominated = False
                    break
            if dominated and run["horizons"]:
                continue

        results.append(run)
    return results


async def get_history_detail(run_id: int) -> Optional[dict]:
    rows = await get_history(limit=99999)
    for r in rows:
        if r["id"] == run_id:
            return r
    return None


async def delete_run(run_id: int) -> bool:
    await db.execute("DELETE FROM backtest_horizons WHERE run_id = %s", (run_id,))
    await db.execute("DELETE FROM backtest_runs WHERE id = %s", (run_id,))
    return True


async def clear_history() -> int:
    await db.execute("DELETE FROM backtest_horizons", None)
    await db.execute("DELETE FROM backtest_runs", None)
    return 1


async def save_bruteforce_session(session: dict) -> int:
    query = """
        INSERT INTO bruteforce_sessions
            (strategy, param_grid_json, train_start, train_end, test_start, test_end,
             tbl, horizon, window_size, total_combos, status)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
        session.get("total_combos", 0),
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
        f"SELECT * FROM bruteforce_sessions ORDER BY created_at DESC LIMIT {limit}"
    )
    results = []
    for r in rows:
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
            "total_combos": r[10],
            "completed": r[11],
            "best_accuracy": r[12],
            "best_params": json.loads(r[13]) if r[13] else {},
            "status": r[14],
            "total_time_sec": r[15],
            "created_at": str(r[16]) if r[16] else "",
        })
    return results


async def get_best_runs(limit: int = 20, horizon: int = 1) -> list[dict]:
    """Get top N runs sorted by accuracy for a given horizon."""
    rows = await db.fetchall("""
        SELECT r.id, r.strategy, r.params_json, r.train_start, r.train_end,
               r.test_start, r.test_end, r.window_size,
               h.accuracy_pct, h.signals, h.correct, h.wrong, h.skipped,
               h.max_win_streak, h.max_lose_streak,
               r.total_time_sec, r.created_at
        FROM backtest_runs r
        JOIN backtest_horizons h ON h.run_id = r.id
        WHERE h.horizon = %s AND h.signals > 0
        ORDER BY h.accuracy_pct DESC
        LIMIT %s
    """, (horizon, limit))

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
            "max_win_streak": r[13],
            "max_lose_streak": r[14],
            "total_time_sec": r[15],
            "created_at": str(r[16]) if r[16] else "",
        })
    return results
