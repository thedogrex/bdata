import time
import asyncio
import itertools
import json
import math
from concurrent.futures import ProcessPoolExecutor
from typing import Any, TYPE_CHECKING

from db import close_all_db_providers
from predictor.backtester import (
    run_backtest, preload_backtest_data, run_backtest_vectorized, RULE_BASED_STRATEGIES,
)
from predictor.db_history import (
    save_backtest_run,
    save_bruteforce_session,
    update_bruteforce_session,
    get_bruteforce_session_by_id,
    get_completed_combo_indices,
)

BF_COMBO_DELAY_SEC = 0.2
THRESHOLD_SWEEP_STRATEGIES = {"lightgbm", "xgboost", "rsi_mean_reversion"}
DEFAULT_BRUTEFORCE_PROCESSES = 1
PARALLEL_MIN_CHUNKS_PER_WORKER = 4
PARALLEL_MAX_COMBOS_PER_CHUNK = 1

if TYPE_CHECKING:
    from predictor.task_manager import TaskProgress


# Default grids per strategy
PARAM_GRIDS: dict[str, dict[str, list]] = {
    "xgboost": {
        "n_estimators": [80, 120, 180, 240],
        "max_depth": [2, 3, 4],
        "learning_rate": [0.05, 0.08, 0.12],
        "subsample": [0.8, 0.9],
        "colsample_bytree": [0.8, 0.9],
        "min_child_weight": [1, 5, 10],
        "reg_lambda": [1.0, 3.0],
        "gamma": [0.0, 0.5],
        "tree_method": ["hist"],
        "max_bin": [128, 256],
        "n_jobs": [2, 4],
        "threshold": [0.51, 0.53, 0.55],
        "window_size": [2000, 3000, 5000],
    },
    "rsi_mean_reversion": {
        "rsi_period": [6, 14],
        "rsi_oversold": [20, 25, 30, 35],
        "rsi_overbought": [65, 70, 75, 80],
        "bb_low": [0.15, 0.2, 0.25],
        "bb_high": [0.75, 0.8, 0.85],
        "bb_period": [10, 14, 20, 30, 50],
        "bb_std": [1.0, 1.5, 2.0, 2.5, 3.0],
        "min_vol": [0.0005, 0.001, 0.002],
        "max_vol": [0.005, 0.01, 0.02],
        "vol_ratio_max": [1.2, 1.5, 2.0],
        "window_size": [2000, 3000, 5000, 8000],
    },
    "momentum": {
        "ema_fast": [3, 5, 8],
        "ema_slow": [15, 20, 30],
        "macd_weight": [0.25, 0.35, 0.45],
        "ema_weight": [0.2, 0.3, 0.4],
        "volume_weight": [0.1, 0.2, 0.3],
        "momentum_weight": [0.1, 0.15, 0.2],
        "volume_surge_threshold": [1.3, 1.5, 2.0],
        "threshold": [0.50, 0.52, 0.53, 0.55],
        "window_size": [2000, 3000, 5000, 8000],
    },
    "pattern_sequence": {
        "lookback_lengths": [[3, 4, 5], [4, 5, 6, 7], [3, 4, 5, 6, 7], [5, 6, 7, 8]],
        "min_occurrences": [3, 5, 10, 20],
    },
    "lightgbm": {
        "n_estimators": [150, 250, 400, 600],
        "max_depth": [-1, 3, 5, 7, 9],
        "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08],
        "num_leaves": [15, 31, 63, 127],
        "subsample": [0.6, 0.7, 0.8, 0.9],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
        "min_child_samples": [5, 10, 20, 50, 100],
        "lambda_l1": [0, 0.1, 0.5, 1.0],
        "lambda_l2": [0, 0.1, 0.5, 1.0],
        "threshold": [0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.60],
        "window_size": [2000, 3000, 5000, 8000, 12000],
    },
    "random_forest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 8, 12],
        "min_samples_leaf": [10, 20, 50],
        "max_features": ["sqrt", 0.5, 0.8],
        "threshold": [0.50, 0.52, 0.53],
        "window_size": [3000, 5000, 8000],
    },
    "lstm": {
        "seq_len": [5, 10, 20],
        "hidden_size": [32, 64, 128],
        "num_layers": [1, 2],
        "dropout": [0.1, 0.2, 0.3],
        "epochs": [5, 10, 15],
        "batch_size": [128, 256],
        "learning_rate": [0.0005, 0.001, 0.002],
        "threshold": [0.50, 0.52],
        "window_size": [3000, 5000, 8000],
    },
    "stochastic_adx": {
        "stoch_oversold": [15, 20, 25, 30],
        "stoch_overbought": [70, 75, 80, 85],
        "adx_strong": [20, 25, 30],
        "adx_weak": [10, 15, 20],
        "trend_weight": [0.3, 0.4, 0.5],
        "stoch_weight": [0.3, 0.4, 0.5],
        "crossover_weight": [0.1, 0.2, 0.3],
        "window_size": [2000, 3000, 5000, 8000],
    },
    "candlestick_pattern": {
        "pattern_weight": [0.3, 0.4, 0.5, 0.6],
        "volume_confirm_weight": [0.1, 0.2, 0.3],
        "trend_context_weight": [0.2, 0.3, 0.4],
        "volume_surge_threshold": [1.2, 1.3, 1.5, 2.0],
        "ema_trend_period": [10, 20, 50],
        "min_pattern_score": [0.05, 0.1, 0.15],
        "window_size": [2000, 3000, 5000, 8000],
    },
}


def get_default_grid(strategy: str) -> dict[str, list]:
    return PARAM_GRIDS.get(strategy, {})


def build_combos(param_grid: dict[str, list]) -> list[dict]:
    """Generate all combinations from a param grid."""
    keys = list(param_grid.keys())
    values = []
    for k in keys:
        v = param_grid.get(k)
        # Allow scalars for convenience (treat as a single-option list)
        if isinstance(v, (list, tuple, set, range)):
            values.append(list(v))
        else:
            values.append([v])
    combos = []
    for combo in itertools.product(*values):
        combos.append(dict(zip(keys, combo)))
    return combos


def _extract_threshold_sweep(raw_values) -> list[float] | None:
    if raw_values is None:
        return None
    values: list[float] = []
    source = raw_values
    if isinstance(raw_values, (list, tuple, set, range)):
        source = list(raw_values)
    else:
        source = [raw_values]
    for val in source:
        try:
            values.append(float(val))
        except (TypeError, ValueError):
            continue
    return values or None


def _normalize_threshold_list(thresholds: list[float] | None) -> list[float] | None:
    if not thresholds:
        return None
    seen_keys: set[str] = set()
    normalized: list[float] = []
    for raw in thresholds:
        try:
            val = float(raw)
        except (TypeError, ValueError):
            continue
        key = f"{val:.6f}"
        if key in seen_keys:
            continue
        seen_keys.add(key)
        normalized.append(val)
    return normalized or None


def _prepare_combo_payload(
    params: dict,
    default_window_size: int,
    normalized_thresholds: list[float] | None,
) -> tuple[dict, int, dict]:
    combo_params = dict(params)
    combo_window = combo_params.pop("window_size", None)
    ws = combo_window if combo_window is not None else default_window_size
    if combo_window is not None:
        combo_params["window_size"] = combo_window

    strat_params = {k: v for k, v in combo_params.items() if k != "window_size"}
    if normalized_thresholds and "threshold" not in strat_params:
        strat_params["threshold"] = normalized_thresholds[0]

    return combo_params, ws, strat_params


async def _persist_combo_result(
    result: dict,
    params: dict,
    horizon: int,
    bf_id: int,
    best_accuracy: float,
    best_params: dict,
    best_result: dict | None,
) -> tuple[float, dict, dict | None, float, float | None]:
    threshold_variants = result.pop("threshold_variants", None)
    result["is_bruteforce"] = True
    result["bruteforce_id"] = bf_id

    combo_best_acc = -1.0
    combo_best_thr = None

    if threshold_variants:
        base_common = {
            k: v for k, v in result.items()
            if k not in {"threshold_variants", "horizons", "params"}
        }
        for thr_key, horizon_map in threshold_variants.items():
            try:
                thr_val = float(thr_key)
            except (TypeError, ValueError):
                continue
            variant_params = dict(params)
            variant_params["threshold"] = thr_val
            variant_result = dict(base_common)
            variant_result["params"] = variant_params
            variant_result["horizons"] = horizon_map
            variant_result["is_bruteforce"] = True
            variant_result["bruteforce_id"] = bf_id
            await save_backtest_run(variant_result)

            h_data = horizon_map.get(str(horizon), {})
            acc_candidate = h_data.get("accuracy_pct", 0)
            if acc_candidate > combo_best_acc:
                combo_best_acc = acc_candidate
                combo_best_thr = thr_val
            if acc_candidate > best_accuracy:
                best_accuracy = acc_candidate
                best_params = variant_params
                best_result = variant_result

        acc = combo_best_acc if combo_best_acc >= 0 else 0
    else:
        result["params"] = params
        await save_backtest_run(result)
        h_data = result.get("horizons", {}).get(str(horizon), {})
        acc = h_data.get("accuracy_pct", 0)
        if acc > best_accuracy:
            best_accuracy = acc
            best_params = params
            best_result = result

    return best_accuracy, best_params, best_result, acc, combo_best_thr


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
    threshold_sweep: list[float] | None,
    processes: int = DEFAULT_BRUTEFORCE_PROCESSES,
) -> dict:
    """
    Core brute-force loop shared by run_bruteforce and resume_bruteforce.
    Checkpoints to DB after every combo. Supports pause/cancel with DB persistence.
    """
    processes = max(1, int(processes or 1))
    if processes > 1:
        return await _run_bf_loop_parallel(
            bf_id=bf_id,
            strategy=strategy,
            combos=combos,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            horizon=horizon,
            table=table,
            default_window_size=default_window_size,
            retrain_every=retrain_every,
            skip_params_set=skip_params_set,
            initial_completed=initial_completed,
            initial_best_accuracy=initial_best_accuracy,
            initial_best_params=initial_best_params,
            elapsed_before=elapsed_before,
            progress=progress,
            threshold_sweep=threshold_sweep,
            processes=processes,
        )

    total = len(combos)
    best_accuracy = initial_best_accuracy
    best_params = initial_best_params.copy() if initial_best_params else {}
    best_result = None
    completed = initial_completed

    normalized_thresholds = _normalize_threshold_list(threshold_sweep)

    if progress:
        progress.total = total
        progress.extra["bruteforce_id"] = bf_id
        progress.extra["state"] = "preloading"
        progress.extra["total_combos"] = total
        progress.phase = "Preloading data (candles + features)..."

    # ── Preload data once for ALL combos ──
    preloaded = await preload_backtest_data(
        train_start, train_end, test_start, test_end, [horizon], table, progress,
        strategy_name=strategy,
    )
    if not preloaded["test_indices"]:
        raise RuntimeError("No test data found in the given range")

    is_fast = strategy in RULE_BASED_STRATEGIES
    mode = "FAST vectorized" if is_fast else "moving-window (preloaded)"
    print(f"[BF {bf_id}] Data preloaded: {len(preloaded['df_feat'])} candles, "
          f"{len(preloaded['test_indices'])} test | {mode} mode", flush=True)

    t0 = time.time()

    for idx, params in enumerate(combos):
        # Skip already-completed combos (for resume)
        params_key = json.dumps(params, sort_keys=True, ensure_ascii=False)
        if params_key in skip_params_set:
            continue

        combo_params, ws, strat_params = _prepare_combo_payload(
            params,
            default_window_size,
            normalized_thresholds,
        )

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
                    f"params: {json.dumps(combo_params, ensure_ascii=False)[:160]}"
                )
            )
            progress.extra["best_accuracy"] = best_accuracy
            progress.extra["best_params"] = best_params
            progress.extra["completed"] = completed
            progress.extra["current_params"] = combo_params

        try:
            combo_t0 = time.time()

            if is_fast:
                # Vectorized: single predict call on full test set (~0.01s)
                result = run_backtest_vectorized(
                    strategy_name=strategy,
                    strategy_params=strat_params,
                    preloaded=preloaded,
                    horizons=[horizon],
                    window_size=ws,
                    retrain_every=retrain_every,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    table=table,
                    threshold_sweep=normalized_thresholds,
                )
            else:
                # Moving-window with preloaded data (skip data reload)
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
                    preloaded=preloaded,
                    threshold_sweep=normalized_thresholds,
                )

            if isinstance(result, dict) and result.get("error"):
                raise RuntimeError(str(result.get("error")))

            best_accuracy, best_params, best_result, acc, combo_best_thr = await _persist_combo_result(
                result,
                combo_params,
                horizon,
                bf_id,
                best_accuracy,
                best_params,
                best_result,
            )

            if combo_best_thr is not None and progress:
                progress.extra["last_threshold"] = combo_best_thr

            combo_elapsed = time.time() - combo_t0
            progress_last_phase = (
                f"BF {round(((completed+1)/max(total,1))*100,1)}% | "
                f"Combo {completed+1}/{total} done | "
                f"acc: {acc}% | best: {best_accuracy}% | "
                f"combo: {round(combo_elapsed,1)}s"
            )

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

            print(f"[BF {bf_id}] {completed}/{total} | ws={ws} | acc={acc}% | best={best_accuracy}% | {round(combo_elapsed,1)}s", flush=True)

            # Yield to event loop periodically in fast mode for UI updates
            if is_fast and completed % 20 == 0:
                await asyncio.sleep(0)

            # Small delay between combos to keep the event loop responsive for
            # other tasks (e.g., live market/prediction requests).
            if BF_COMBO_DELAY_SEC and BF_COMBO_DELAY_SEC > 0:
                await asyncio.sleep(BF_COMBO_DELAY_SEC)

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

            elapsed_now = time.time() - t0
            await update_bruteforce_session(bf_id, {
                "completed": completed,
                "best_accuracy": best_accuracy,
                "best_params_json": best_params,
                "total_time_sec": round(elapsed_before + elapsed_now, 2),
            })

            print(f"[BF {bf_id}] {completed}/{total} | ws={ws} ERROR: {e}", flush=True)

            if BF_COMBO_DELAY_SEC and BF_COMBO_DELAY_SEC > 0:
                await asyncio.sleep(BF_COMBO_DELAY_SEC)

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


async def _run_bf_loop_parallel(
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
    threshold_sweep: list[float] | None,
    processes: int,
) -> dict:
    total = len(combos)
    best_accuracy = initial_best_accuracy
    best_params = initial_best_params.copy() if initial_best_params else {}
    best_result = None
    completed = initial_completed
    normalized_thresholds = _normalize_threshold_list(threshold_sweep)

    pending_combos: list[dict] = []
    skip_keys = skip_params_set or set()
    for combo in combos:
        combo_key = json.dumps(combo, sort_keys=True, ensure_ascii=False)
        if combo_key in skip_keys:
            continue
        pending_combos.append(combo)

    worker_count = max(1, min(processes, len(pending_combos) or 1))
    t0 = time.time()

    if not pending_combos:
        total_time = elapsed_before + (time.time() - t0)
        await update_bruteforce_session(bf_id, {
            "completed": completed,
            "best_accuracy": best_accuracy,
            "best_params_json": best_params,
            "status": "done",
            "total_time_sec": round(total_time, 2),
        })
        if progress:
            progress.update(total, total, "Done (nothing to process)")
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

    chunk_target = max(worker_count * PARALLEL_MIN_CHUNKS_PER_WORKER, 1)
    chunk_size = max(1, math.ceil(len(pending_combos) / chunk_target))
    if PARALLEL_MAX_COMBOS_PER_CHUNK:
        chunk_size = min(chunk_size, PARALLEL_MAX_COMBOS_PER_CHUNK)
    chunks = [
        pending_combos[i:i + chunk_size]
        for i in range(0, len(pending_combos), chunk_size)
    ]
    chunks_total = len(chunks)
    chunks_done = 0

    if progress:
        progress.total = total
        progress.extra["bruteforce_id"] = bf_id
        progress.extra["state"] = "parallel"
        progress.extra["completed"] = completed
        progress.extra["total_combos"] = total
        progress.extra["parallel_workers"] = worker_count
        progress.extra["parallel_chunk_size"] = chunk_size
        progress.extra["parallel_chunks_total"] = chunks_total
        progress.extra["parallel_chunks_done"] = chunks_done
        progress.phase = (
            f"Parallel mode: {worker_count} worker(s), {chunks_total} chunk(s) (~{chunk_size} combos each)"
        )

    loop = asyncio.get_running_loop()
    futures = []
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        for chunk in chunks:
            payload = {
                "bf_id": bf_id,
                "strategy": strategy,
                "combos": chunk,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "horizon": horizon,
                "table": table,
                "default_window_size": default_window_size,
                "retrain_every": retrain_every,
                "threshold_sweep": normalized_thresholds,
            }
            futures.append(loop.run_in_executor(executor, _bf_worker_entry, payload))

        for fut in asyncio.as_completed(futures):
            try:
                worker_result = await fut
            except Exception:
                # Propagate after marking session paused
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

            processed = worker_result.get("processed", 0)
            completed += processed

            worker_best_acc = worker_result.get("best_accuracy", -1)
            if worker_best_acc is not None and worker_best_acc > best_accuracy:
                best_accuracy = worker_best_acc
                candidate_params = worker_result.get("best_params")
                if candidate_params:
                    best_params = candidate_params
                candidate_result = worker_result.get("best_result")
                if candidate_result:
                    best_result = candidate_result

            elapsed_now = time.time() - t0
            await update_bruteforce_session(bf_id, {
                "completed": completed,
                "best_accuracy": best_accuracy,
                "best_params_json": best_params,
                "total_time_sec": round(elapsed_before + elapsed_now, 2),
            })

            last_accuracy = worker_result.get("last_accuracy", 0)
            print(
                f"[BF {bf_id}] chunk finished +{processed} | "
                f"{completed}/{total} combos | last {round(last_accuracy,2)}% | "
                f"best {round(best_accuracy,2)}%",
                flush=True,
            )

            chunks_done += 1

            if progress:
                last_params = worker_result.get("last_params")
                last_accuracy = worker_result.get("last_accuracy", 0)
                if last_params:
                    progress.extra["current_params"] = last_params
                if worker_result.get("last_threshold") is not None:
                    progress.extra["last_threshold"] = worker_result["last_threshold"]
                progress.extra["last_accuracy"] = last_accuracy
                progress.extra["last_combo_time_sec"] = round(worker_result.get("elapsed", 0), 2)
                progress.extra["completed"] = completed
                progress.extra["remaining"] = max(total - completed, 0)
                progress.extra["best_accuracy"] = best_accuracy
                progress.extra["parallel_chunks_done"] = chunks_done
                progress.extra["parallel_chunks_total"] = chunks_total
                progress.phase = (
                    f"BF {round((completed/max(total,1))*100,1)}% | "
                    f"Combos {completed}/{total} | "
                    f"last acc: {round(last_accuracy, 2)}% | best: {round(best_accuracy, 2)}%"
                )
                progress.update(completed, total, progress.phase)

            for err in worker_result.get("errors", []):
                print(f"[BF {bf_id}] Worker error: {err}", flush=True)

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


def _bf_worker_entry(payload: dict) -> dict:
    return asyncio.run(_bf_worker_async(payload))


async def _bf_worker_async(payload: dict) -> dict:
    try:
        combos = payload.get("combos") or []
        start = time.time()
        processed = 0
        best_accuracy = -1.0
        best_params: dict[str, Any] = {}
        best_result: dict | None = None
        last_params: dict | None = None
        last_accuracy = 0.0
        last_threshold = None
        errors: list[str] = []

        normalized_thresholds = _normalize_threshold_list(payload.get("threshold_sweep"))
        is_fast = payload["strategy"] in RULE_BASED_STRATEGIES

        preloaded = await preload_backtest_data(
            payload["train_start"],
            payload["train_end"],
            payload["test_start"],
            payload["test_end"],
            [payload["horizon"]],
            payload["table"],
            strategy_name=payload["strategy"],
        )
        if not preloaded["test_indices"]:
            raise RuntimeError("No test data found in the given range")

        for combo in combos:
            combo_params = None
            try:
                combo_params, ws, strat_params = _prepare_combo_payload(
                    combo,
                    payload["default_window_size"],
                    normalized_thresholds,
                )

                if is_fast:
                    result = run_backtest_vectorized(
                        strategy_name=payload["strategy"],
                        strategy_params=strat_params,
                        preloaded=preloaded,
                        horizons=[payload["horizon"]],
                        window_size=ws,
                        retrain_every=payload["retrain_every"],
                        train_start=payload["train_start"],
                        train_end=payload["train_end"],
                        test_start=payload["test_start"],
                        test_end=payload["test_end"],
                        table=payload["table"],
                        threshold_sweep=normalized_thresholds,
                    )
                else:
                    result = await run_backtest(
                        strategy_name=payload["strategy"],
                        strategy_params=strat_params,
                        train_start=payload["train_start"],
                        train_end=payload["train_end"],
                        test_start=payload["test_start"],
                        test_end=payload["test_end"],
                        horizons=[payload["horizon"]],
                        table=payload["table"],
                        window_size=ws,
                        retrain_every=payload["retrain_every"],
                        preloaded=preloaded,
                        threshold_sweep=normalized_thresholds,
                    )

                if isinstance(result, dict) and result.get("error"):
                    raise RuntimeError(str(result.get("error")))

                best_accuracy, best_params, best_result, acc, combo_best_thr = await _persist_combo_result(
                    result,
                    combo_params,
                    payload["horizon"],
                    payload["bf_id"],
                    best_accuracy,
                    best_params,
                    best_result,
                )

                last_params = combo_params
                last_accuracy = acc
                last_threshold = combo_best_thr

            except Exception as exc:  # noqa: BLE001
                errors.append(f"{combo_params or combo} -> {exc}")
                print(f"[BF {payload['bf_id']}] Worker combo error: {exc}", flush=True)
            finally:
                processed += 1

        elapsed = time.time() - start
        return {
            "processed": processed,
            "best_accuracy": best_accuracy,
            "best_params": best_params,
            "best_result": best_result,
            "elapsed": elapsed,
            "last_params": last_params,
            "last_accuracy": last_accuracy,
            "last_threshold": last_threshold,
            "errors": errors,
        }
    finally:
        await close_all_db_providers()


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
    processes: int = DEFAULT_BRUTEFORCE_PROCESSES,
    progress: "TaskProgress | None" = None,
) -> dict:
    """
    Brute-force search over hyperparameter grid with pause/cancel support.
    Checkpoints every combo to MySQL for resume after server restart.
    """
    grid_for_combos = dict(param_grid)
    threshold_sweep = None
    if strategy in THRESHOLD_SWEEP_STRATEGIES and "threshold" in grid_for_combos:
        threshold_sweep = _extract_threshold_sweep(grid_for_combos.pop("threshold"))

    combos = build_combos(grid_for_combos)
    if len(combos) > max_combos:
        step = len(combos) // max_combos
        combos = combos[::step][:max_combos]

    total = len(combos)

    # Save session with full combo list for resume
    combo_payload = {
        "items": combos,
        "processes": max(1, int(processes or 1)),
    }

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
        "combos": combo_payload,
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
        threshold_sweep=threshold_sweep,
        processes=processes,
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
    processes = session["processes"]
    if not combos:
        raise ValueError(f"Session {bf_id} has no saved combo list — cannot resume")

    # Find which combos are already done
    threshold_sweep = None
    if session["strategy"] in THRESHOLD_SWEEP_STRATEGIES and "threshold" in (session.get("param_grid") or {}):
        threshold_sweep = _extract_threshold_sweep(session["param_grid"].get("threshold"))

    ignore_keys = {"threshold"} if threshold_sweep else None
    completed_params_set = await get_completed_combo_indices(
        bf_id,
        ignore_keys=ignore_keys,
        required_thresholds=threshold_sweep,
    )
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
        threshold_sweep=threshold_sweep,
        processes=processes,
    )
