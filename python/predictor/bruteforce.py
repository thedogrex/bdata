import time
import asyncio
import itertools
import json
from typing import Any, TYPE_CHECKING

from predictor.backtester import run_backtest
from predictor.db_history import (
    save_backtest_run,
    save_bruteforce_session,
    update_bruteforce_session,
)

if TYPE_CHECKING:
    from predictor.task_manager import TaskProgress


# Default grids per strategy
PARAM_GRIDS: dict[str, dict[str, list]] = {
    "xgboost": {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.03, 0.05, 0.08, 0.1],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
        "threshold": [0.50, 0.52, 0.53, 0.55],
    },
    "rsi_mean_reversion": {
        "rsi_period": [6, 14],
        "rsi_oversold": [20, 25, 30, 35],
        "rsi_overbought": [65, 70, 75, 80],
        "bb_low": [0.15, 0.2, 0.25],
        "bb_high": [0.75, 0.8, 0.85],
    },
    "momentum": {
        "ema_fast": [3, 5, 8],
        "ema_slow": [15, 20, 30],
        "macd_weight": [0.25, 0.35, 0.45],
        "ema_weight": [0.2, 0.3, 0.4],
        "volume_weight": [0.1, 0.2, 0.3],
        "momentum_weight": [0.1, 0.15, 0.2],
        "volume_surge_threshold": [1.3, 1.5, 2.0],
    },
    "pattern_sequence": {
        "lookback_lengths": [[3, 4, 5], [4, 5, 6, 7], [3, 4, 5, 6, 7], [5, 6, 7, 8]],
        "min_occurrences": [3, 5, 10, 20],
    },
}


def get_default_grid(strategy: str) -> dict[str, list]:
    return PARAM_GRIDS.get(strategy, {})


def build_combos(param_grid: dict[str, list]) -> list[dict]:
    """Generate all combinations from a param grid."""
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combos = []
    for combo in itertools.product(*values):
        combos.append(dict(zip(keys, combo)))
    return combos


async def run_bruteforce(
    strategy: str,
    param_grid: dict[str, list],
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    horizon: int = 1,
    table: str = "c_5m",
    window_size: int = 5000,
    retrain_every: int = 500,
    max_combos: int = 200,
    progress: "TaskProgress | None" = None,
) -> dict:
    """
    Brute-force search over hyperparameter grid with pause/cancel support.
    """
    combos = build_combos(param_grid)
    if len(combos) > max_combos:
        step = len(combos) // max_combos
        combos = combos[::step][:max_combos]

    total = len(combos)

    # Save session
    bf_id = await save_bruteforce_session({
        "strategy": strategy,
        "param_grid": param_grid,
        "train_start": train_start,
        "train_end": train_end,
        "test_start": test_start,
        "test_end": test_end,
        "table": table,
        "horizon": horizon,
        "window_size": window_size,
        "total_combos": total,
    })

    if progress:
        progress.total = total
        progress.extra["bruteforce_id"] = bf_id

    t0 = time.time()
    best_accuracy = 0.0
    best_params = {}
    best_result = None
    completed = 0

    for idx, params in enumerate(combos):
        # Check pause/cancel before each combo
        if progress:
            await progress.check_pause_cancel()
            progress.update(
                completed, total,
                f"Combo {completed+1}/{total} | best: {best_accuracy}%"
            )
            progress.extra["best_accuracy"] = best_accuracy
            progress.extra["best_params"] = best_params
            progress.extra["completed"] = completed

        try:
            result = await run_backtest(
                strategy_name=strategy,
                strategy_params=params,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                horizons=[horizon],
                table=table,
                window_size=window_size,
                retrain_every=retrain_every,
            )

            result["is_bruteforce"] = True
            result["bruteforce_id"] = bf_id
            await save_backtest_run(result)

            h_data = result.get("horizons", {}).get(str(horizon), {})
            acc = h_data.get("accuracy_pct", 0)

            if acc > best_accuracy:
                best_accuracy = acc
                best_params = params
                best_result = result

            completed += 1

            # Update session progress in DB
            if completed % 5 == 0 or completed == total:
                await update_bruteforce_session(bf_id, {
                    "completed": completed,
                    "best_accuracy": best_accuracy,
                    "best_params_json": best_params,
                })

            print(f"[BF {bf_id}] {completed}/{total} | acc={acc}% | best={best_accuracy}%", flush=True)

        except Exception as e:
            from predictor.task_manager import CancelledError
            if isinstance(e, CancelledError):
                raise
            completed += 1
            print(f"[BF {bf_id}] {completed}/{total} ERROR: {e}", flush=True)

    total_time = time.time() - t0

    await update_bruteforce_session(bf_id, {
        "completed": completed,
        "best_accuracy": best_accuracy,
        "best_params_json": best_params,
        "status": "done",
        "total_time_sec": round(total_time, 2),
    })

    if progress:
        progress.update(total, total, "Done")
        progress.extra["best_accuracy"] = best_accuracy
        progress.extra["best_params"] = best_params

    return {
        "bruteforce_id": bf_id,
        "strategy": strategy,
        "total_combos": total,
        "completed": completed,
        "best_accuracy": best_accuracy,
        "best_params": best_params,
        "best_result": best_result,
        "total_time_sec": round(total_time, 2),
    }
