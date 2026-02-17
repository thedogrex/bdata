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
    get_bruteforce_session_by_id,
    get_completed_combo_indices,
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
        "window_size": [3000, 5000, 8000],
    },
    "rsi_mean_reversion": {
        "rsi_period": [6, 14],
        "rsi_oversold": [20, 25, 30, 35],
        "rsi_overbought": [65, 70, 75, 80],
        "bb_low": [0.15, 0.2, 0.25],
        "bb_high": [0.75, 0.8, 0.85],
        "window_size": [2000, 3000, 5000],
    },
    "momentum": {
        "ema_fast": [3, 5, 8],
        "ema_slow": [15, 20, 30],
        "macd_weight": [0.25, 0.35, 0.45],
        "ema_weight": [0.2, 0.3, 0.4],
        "volume_weight": [0.1, 0.2, 0.3],
        "momentum_weight": [0.1, 0.15, 0.2],
        "volume_surge_threshold": [1.3, 1.5, 2.0],
        "window_size": [2000, 5000, 8000],
    },
    "pattern_sequence": {
        "lookback_lengths": [[3, 4, 5], [4, 5, 6, 7], [3, 4, 5, 6, 7], [5, 6, 7, 8]],
        "min_occurrences": [3, 5, 10, 20],
        "window_size": [2000, 5000],
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


async def _run_bf_loop(
    bf_id: int,
    strategy: str,
    combos: list[dict],
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    horizon: int,
    table: str,
    default_window_size: int,
    retrain_every: int,
    skip_params_set: set[str],
    initial_completed: int,
    initial_best_accuracy: float,
    initial_best_params: dict,
    elapsed_before: float,
    progress: "TaskProgress | None",
) -> dict:
    """
    Core brute-force loop shared by run_bruteforce and resume_bruteforce.
    Checkpoints to DB after every combo. Supports pause/cancel with DB persistence.
    """
    total = len(combos)
    best_accuracy = initial_best_accuracy
    best_params = initial_best_params.copy() if initial_best_params else {}
    best_result = None
    completed = initial_completed

    if progress:
        progress.total = total
        progress.extra["bruteforce_id"] = bf_id

    t0 = time.time()

    for idx, params in enumerate(combos):
        # Skip already-completed combos (for resume)
        params_key = json.dumps(params, sort_keys=True, ensure_ascii=False)
        if params_key in skip_params_set:
            continue

        # Check pause/cancel before each combo
        if progress:
            try:
                await progress.check_pause_cancel()
            except Exception:
                # On cancel/pause, save state to DB before re-raising
                elapsed_now = time.time() - t0
                await update_bruteforce_session(bf_id, {
                    "completed": completed,
                    "best_accuracy": best_accuracy,
                    "best_params_json": best_params,
                    "status": "paused",
                    "elapsed_before_pause": elapsed_before + elapsed_now,
                    "total_time_sec": round(elapsed_before + elapsed_now, 2),
                })
                raise

            progress.update(
                completed, total,
                (
                    f"BF {round((completed/max(total,1))*100,1)}% | "
                    f"Combo {completed+1}/{total} | "
                    f"best: {best_accuracy}% | "
                    f"params: {json.dumps(params, ensure_ascii=False)[:160]}"
                )
            )
            progress.extra["best_accuracy"] = best_accuracy
            progress.extra["best_params"] = best_params
            progress.extra["completed"] = completed
            progress.extra["current_params"] = params

        # Extract window_size from combo params if present
        combo_window = params.pop("window_size", None) if "window_size" in params else None
        ws = combo_window if combo_window is not None else default_window_size
        # Put it back for saving
        if combo_window is not None:
            params["window_size"] = combo_window

        # Build strategy params (without window_size)
        strat_params = {k: v for k, v in params.items() if k != "window_size"}

        try:
            combo_t0 = time.time()
            result = await run_backtest(
                strategy_name=strategy,
                strategy_params=strat_params,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                horizons=[horizon],
                table=table,
                window_size=ws,
                retrain_every=retrain_every,
            )

            result["is_bruteforce"] = True
            result["bruteforce_id"] = bf_id
            result["params"] = params  # Save full combo params (incl. window_size) for resume matching
            await save_backtest_run(result)

            h_data = result.get("horizons", {}).get(str(horizon), {})
            acc = h_data.get("accuracy_pct", 0)
            combo_elapsed = time.time() - combo_t0
            progress_last_phase = (
                f"BF {round(((completed+1)/max(total,1))*100,1)}% | "
                f"Combo {completed+1}/{total} done | "
                f"acc: {acc}% | best: {best_accuracy}% | "
                f"combo: {round(combo_elapsed,1)}s"
            )

            if acc > best_accuracy:
                best_accuracy = acc
                best_params = params
                best_result = result

            completed += 1

            # Checkpoint to DB after EVERY combo
            elapsed_now = time.time() - t0
            await update_bruteforce_session(bf_id, {
                "completed": completed,
                "best_accuracy": best_accuracy,
                "best_params_json": best_params,
                "total_time_sec": round(elapsed_before + elapsed_now, 2),
            })

            if progress:
                progress.extra["last_accuracy"] = acc
                progress.extra["last_combo_time_sec"] = round(combo_elapsed, 2)
                progress.phase = progress_last_phase

            print(f"[BF {bf_id}] {completed}/{total} | acc={acc}% | best={best_accuracy}%", flush=True)

        except Exception as e:
            from predictor.task_manager import CancelledError
            if isinstance(e, CancelledError):
                # Save paused state to DB
                elapsed_now = time.time() - t0
                await update_bruteforce_session(bf_id, {
                    "completed": completed,
                    "best_accuracy": best_accuracy,
                    "best_params_json": best_params,
                    "status": "paused",
                    "elapsed_before_pause": elapsed_before + elapsed_now,
                    "total_time_sec": round(elapsed_before + elapsed_now, 2),
                })
                raise
            completed += 1
            print(f"[BF {bf_id}] {completed}/{total} ERROR: {e}", flush=True)

    total_time = elapsed_before + (time.time() - t0)

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
    Checkpoints every combo to MySQL for resume after server restart.
    """
    combos = build_combos(param_grid)
    if len(combos) > max_combos:
        step = len(combos) // max_combos
        combos = combos[::step][:max_combos]

    total = len(combos)

    # Save session with full combo list for resume
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
        "retrain_every": retrain_every,
        "total_combos": total,
        "combos": combos,
    })

    return await _run_bf_loop(
        bf_id=bf_id,
        strategy=strategy,
        combos=combos,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        horizon=horizon,
        table=table,
        default_window_size=window_size,
        retrain_every=retrain_every,
        skip_params_set=set(),
        initial_completed=0,
        initial_best_accuracy=0.0,
        initial_best_params={},
        elapsed_before=0.0,
        progress=progress,
    )


async def resume_bruteforce(
    bf_id: int,
    progress: "TaskProgress | None" = None,
) -> dict:
    """
    Resume a paused/interrupted brute-force session from its DB checkpoint.
    Loads the session, finds completed combos, and continues from where it stopped.
    """
    session = await get_bruteforce_session_by_id(bf_id)
    if not session:
        raise ValueError(f"Brute-force session {bf_id} not found")

    if session["status"] == "done":
        raise ValueError(f"Session {bf_id} is already completed")

    combos = session["combos"]
    if not combos:
        raise ValueError(f"Session {bf_id} has no saved combo list — cannot resume")

    # Find which combos are already done
    completed_params_set = await get_completed_combo_indices(bf_id)
    remaining = sum(
        1 for c in combos
        if json.dumps(c, sort_keys=True, ensure_ascii=False) not in completed_params_set
    )

    print(f"[BF {bf_id}] Resuming: {session['completed']}/{session['total_combos']} done, "
          f"{remaining} remaining", flush=True)

    # Mark as running again
    await update_bruteforce_session(bf_id, {"status": "running"})

    return await _run_bf_loop(
        bf_id=bf_id,
        strategy=session["strategy"],
        combos=combos,
        train_start=session["train_start"],
        train_end=session["train_end"],
        test_start=session["test_start"],
        test_end=session["test_end"],
        horizon=session["horizon"],
        table=session["table"],
        default_window_size=session["window_size"],
        retrain_every=session["retrain_every"],
        skip_params_set=completed_params_set,
        initial_completed=session["completed"],
        initial_best_accuracy=session["best_accuracy"],
        initial_best_params=session["best_params"],
        elapsed_before=session.get("elapsed_before_pause", 0.0),
        progress=progress,
    )
