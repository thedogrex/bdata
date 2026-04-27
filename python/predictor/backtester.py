import time
import asyncio
import math
import json
import re
import numpy as np
import pandas as pd
from typing import Any, Optional, TYPE_CHECKING
import logging

from predictor.data_loader import load_candles, add_direction, add_future_directions, date_to_us
from predictor.features import add_technical_features, set_feature_usage, FEATURE_USAGE
from predictor.strategies import get_strategy, BaseStrategy
from predictor.utils.async_utils import resolve_awaitable

logger = logging.getLogger(__name__)

# Minimal feature blocks needed per strategy for preloading.
# Strategies not listed here get ALL features.
_STRATEGY_MINIMAL_FEATURES: dict[str, set[str]] = {
    "rsi_mean_reversion": {"rsi", "bollinger", "moving_averages", "volatility"},
    "momentum": {"macd", "momentum", "volume", "moving_averages"},
    "stochastic_adx": {"stochastic", "adx_dmi"},
    "candlestick_pattern": {"candlestick_patterns", "volume", "moving_averages"},
}

if TYPE_CHECKING:
    from predictor.task_manager import TaskProgress


RULE_BASED_STRATEGIES = {"rsi_mean_reversion", "momentum", "stochastic_adx", "candlestick_pattern"}
_RSI_VARIANT_CACHE_MAX = 128
_RSI_VARIANT_CACHE: dict[tuple[str, str], list[tuple[str, tuple[tuple[str, Any], ...]]]] = {}
_RSI_FIT_STATE_ATTRS = ("_rsi_median", "_rsi_p10", "_rsi_p90", "_bb_median")


def _is_cpu_heavy_strategy(strategy_name: str) -> bool:
    return strategy_name in {"xgboost", "lightgbm", "lstm"}


def _snapshot_rsi_fit_state(strategy: BaseStrategy | None) -> dict[str, Any] | None:
    if strategy is None:
        return None
    snapshot: dict[str, Any] = {}
    for attr in _RSI_FIT_STATE_ATTRS:
        if hasattr(strategy, attr):
            snapshot[attr] = getattr(strategy, attr)
    return snapshot or None


def _apply_rsi_fit_state(strategy: BaseStrategy | None, snapshot: dict[str, Any] | None) -> None:
    if not strategy or not snapshot:
        return
    for attr, value in snapshot.items():
        setattr(strategy, attr, value)


def _table_interval_minutes(table: str | None) -> int:
    default_minutes = 5
    if not table:
        return default_minutes
    suffix = table.split("_", 1)[1] if "_" in table else table
    match = re.fullmatch(r"(\d+)([mhd])", suffix)
    if not match:
        return default_minutes
    value = int(match.group(1))
    unit = match.group(2)
    multiplier = {"m": 1, "h": 60, "d": 1440}.get(unit, 1)
    return max(1, value * multiplier)


def _resolve_train_window_start(
    train_start: str | None,
    test_start: str,
    min_candles: int | None,
    table: str,
) -> str:
    if train_start:
        return train_start
    if not test_start:
        raise ValueError("test_start is required when train_start is omitted")
    if not min_candles or min_candles <= 0:
        return test_start
    candle_minutes = _table_interval_minutes(table)
    lookback_minutes = max(1, min_candles) * candle_minutes
    test_dt = pd.to_datetime(test_start)
    derived_dt = test_dt - pd.to_timedelta(lookback_minutes, unit="m")
    return derived_dt.strftime("%Y-%m-%d")


async def run_backtest(
    strategy_name: str,
    strategy_params: dict | None,
    train_start: str | None,
    train_end: str | None,
    test_start: str,
    test_end: str,
    horizons: list[int] | None = None,
    table: str = "c_5m",
    window_size: int = 5000,
    retrain_every: int = 500,
    progress: "TaskProgress | None" = None,
    preloaded: dict | None = None,
    threshold_sweep: list[float] | None = None,
) -> dict:
    """
    Moving-window backtest with progress tracking and pause/cancel support.
    Optimized: features are pre-computed ONCE on the full dataset.
    """
    if horizons is None:
        horizons = [1]

    t0 = time.time()

    default_strategy = None
    if not strategy_params or "threshold" not in (strategy_params or {}):
        default_strategy = get_strategy(strategy_name)
    if strategy_params and "threshold" in strategy_params:
        base_threshold = float(strategy_params["threshold"])
    else:
        fallback_params = default_strategy.params if default_strategy else get_strategy(strategy_name).params
        base_threshold = float(fallback_params.get("threshold", 0.5))

    sweep_thresholds = _normalize_threshold_sweep(threshold_sweep, base_threshold) if threshold_sweep else []
    threshold_variant_results: dict[str, dict[str, dict]] = {}

    effective_train_start = train_start
    effective_train_end = train_end

    if preloaded:
        # Reuse preloaded data (skip expensive data loading + feature computation)
        df_raw = preloaded["df_raw"]
        df_feat = preloaded["df_feat"]
        test_indices = preloaded["test_indices"]
        load_time = preloaded["load_time"]
        feat_time = preloaded["feat_time"]
        effective_train_start = preloaded.get("train_start", train_start)
        effective_train_end = preloaded.get("train_end", train_end)
    else:
        # ── Phase 1: Load candle data ──
        if progress:
            progress.phase = "Phase 1/3: Loading candle data from DB..."
            await asyncio.sleep(0)

        effective_train_start = _resolve_train_window_start(train_start, test_start, window_size, table)
        effective_train_end = train_end or test_start

        df_raw = await load_candles(table, effective_train_start, test_end)
        df_raw = add_direction(df_raw)
        max_horizon = max(horizons)
        df_raw = add_future_directions(df_raw, horizons)
        df_raw = df_raw.reset_index(drop=True)

        load_time = time.time() - t0

        if progress:
            progress.phase = f"Phase 2/3: Computing technical features for {len(df_raw)} candles..."
            await asyncio.sleep(0)

        # ── Phase 2: Pre-compute ALL technical features ONCE ──
        t_feat = time.time()
        df_feat = add_technical_features(df_raw)
        feat_time = time.time() - t_feat

        # Find test start index
        test_start_us = date_to_us(test_start)
        test_end_us = date_to_us(test_end, True)
        test_indices = df_feat.index[
            (df_feat["open_time"] >= test_start_us) & (df_feat["open_time"] <= test_end_us)
        ].tolist()

    if not test_indices:
        return {"error": "No test data found in the given range"}

    test_start_idx = test_indices[0]
    test_end_idx = test_indices[-1]

    # We need at least window_size candles before the first test candle
    actual_window = min(window_size, test_start_idx)

    # Total work units = sum of test_indices per horizon
    total_work = len(test_indices) * len(horizons)
    work_done = 0

    if progress:
        progress.total = total_work
        progress.phase = f"Phase 3/3: Testing {len(test_indices)} candles x {len(horizons)} horizon(s)"

    results_by_horizon = {}
    threshold_variant_results: dict[str, dict[str, dict]] = {}

    offload_cpu = _is_cpu_heavy_strategy(strategy_name)

    if progress:
        # Extra fields for UI visibility (bruteforce): how many candles are processed in current run
        progress.extra["candles_total"] = int(len(test_indices))
        progress.extra["candles_done"] = 0
        progress.extra["horizon"] = int(horizons[0]) if horizons else 1
        progress.extra["train_count"] = 0
        progress.extra["train_total"] = max(1, math.ceil(len(test_indices) / max(retrain_every, 1)))
        progress.extra["state"] = "init"
        progress.extra["train_elapsed_sec"] = 0.0

    for h_idx, horizon in enumerate(horizons):
        t1 = time.time()
        target_col = f"future_dir_{horizon}"

        all_preds: list[tuple[int, int, float, int, int]] = []
        strategy: BaseStrategy | None = None
        train_count = 0
        total_train_time = 0.0
        last_train_idx = -retrain_every
        train_total_est = math.ceil(len(test_indices) / retrain_every)
        work_done = 0

        if progress:
            progress.extra["train_total"] = train_total_est
            progress.extra["train_count"] = 0
            progress.extra["horizon"] = int(horizon)

        for step, i in enumerate(test_indices):
            # Pause/cancel check every 50 candles
            if progress and step % 50 == 0:
                await progress.check_pause_cancel()
                progress.extra["state"] = "walking"
                progress.extra["train_elapsed_sec"] = 0.0
                progress.extra["candles_done"] = int(step)
                progress.extra["candles_total"] = int(len(test_indices))
                progress.extra["horizon"] = int(horizon)
                progress.extra["train_count"] = int(train_count)
                progress.extra["train_total"] = train_total_est
                if step % 200 == 0:
                    try:
                        print(
                            f"[backtest] {strategy_name} H{horizon} walking: {step}/{len(test_indices)} candles"
                            f" | train {train_count}/{train_total_est}",
                            flush=True,
                        )
                    except Exception:
                        pass
                correct_so_far = sum(1 for p in all_preds if p[1] == p[3] and p[1] != -1)
                signals_so_far = sum(1 for p in all_preds if p[1] != -1)
                acc_so_far = round(correct_so_far / signals_so_far * 100, 1) if signals_so_far > 0 else 0
                phase_info = f"H{horizon}: {step}/{len(test_indices)} candles | acc: {acc_so_far}% ({correct_so_far}/{signals_so_far})"
                if train_count > 0:
                    phase_info += f" | {train_count} trains ({round(total_train_time, 1)}s)"
                progress.update(work_done + step, total_work, phase_info)
                await asyncio.sleep(0)

            # Skip if we can't compute the actual future direction
            if i + horizon >= len(df_feat):
                break

            actual_val = df_feat.at[i, target_col]
            if pd.isna(actual_val):
                break

            # Retrain if needed
            if i - last_train_idx >= retrain_every or strategy is None:
                train_lo = max(0, i - actual_window)
                train_hi = i  # exclusive: everything BEFORE current candle
                df_train = df_feat.iloc[train_lo:train_hi].reset_index(drop=True)

                if len(df_train) < 100:
                    continue

                if progress:
                    progress.extra["train_count"] = int(train_count)
                    progress.extra["train_total"] = train_total_est
                    progress.phase = f"H{horizon}: Training model ({train_count + 1}/{train_total_est}) on {len(df_train)} candles..."

                t_train = time.time()
                strategy = get_strategy(strategy_name, strategy_params)
                if offload_cpu:
                    if progress:
                        progress.extra["state"] = "training"
                        progress.extra["train_elapsed_sec"] = 0.0
                    train_started_at = time.time()
                    last_print_bucket = -1

                    async def _train_ticker():
                        nonlocal last_print_bucket
                        while True:
                            if progress:
                                progress.extra["state"] = "training"
                                progress.extra["train_count"] = int(train_count)
                                progress.extra["train_total"] = train_total_est
                                progress.extra["train_elapsed_sec"] = round(time.time() - train_started_at, 1)
                            try:
                                bucket = int((time.time() - train_started_at) // 10)
                                if bucket != last_print_bucket:
                                    last_print_bucket = bucket
                                    print(
                                        f"[backtest] {strategy_name} H{horizon}"
                                        f" TRAINING {train_count + 1}/{train_total_est}"
                                        f" | {int(time.time() - train_started_at)}s elapsed"
                                        f" | candles so far: {len(all_preds)}",
                                        flush=True,
                                    )
                            except Exception:
                                pass
                            await asyncio.sleep(1)

                    ticker_task = asyncio.create_task(_train_ticker())
                    try:
                        await asyncio.to_thread(strategy.fit, df_train, horizon)
                    finally:
                        ticker_task.cancel()
                        if progress:
                            progress.extra["train_elapsed_sec"] = round(time.time() - train_started_at, 1)
                else:
                    strategy.fit(df_train, horizon)
                train_elapsed = time.time() - t_train
                total_train_time += train_elapsed
                train_count += 1
                last_train_idx = i
                if progress:
                    progress.extra["train_count"] = int(train_count)
                    progress.extra["train_total"] = train_total_est
                    progress.extra["state"] = "walking"
                    if not offload_cpu:
                        print(
                            f"[backtest] {strategy_name} H{horizon}"
                            f" train {train_count}/{train_total_est} done"
                            f" in {round(train_elapsed, 1)}s",
                            flush=True,
                        )

                # Yield after training (can be slow for XGBoost)
                await asyncio.sleep(0)

            # Predict single candle (optimized)
            vol_flag = 0
            if strategy_name == "random_forest" and hasattr(strategy, "feature_cols") and hasattr(strategy, "predict_row"):
                cols = strategy.feature_cols
                x_row = df_feat.loc[i, cols].to_numpy()
                pred = int(strategy.predict_row(x_row))
                prob = float(strategy.predict_proba_row(x_row))
                vol_flag = int(getattr(strategy, "_last_vol_skip_count", 0) or 0)
            else:
                df_single = df_feat.iloc[[i]]
                if offload_cpu:
                    pred_arr = await asyncio.to_thread(strategy.predict, df_single, horizon)
                    prob_arr = await asyncio.to_thread(strategy.predict_proba, df_single, horizon)
                else:
                    pred_arr = await resolve_awaitable(strategy.predict(df_single, horizon))
                    prob_arr = await resolve_awaitable(strategy.predict_proba(df_single, horizon))
                pred = int(pred_arr[0])
                prob = float(prob_arr[0])
                vol_arr = _extract_vol_skip_flags(getattr(strategy, "_last_vol_skip_flags", None), len(df_single))
                vol_flag = int(vol_arr.sum())

            all_preds.append((i, pred, prob, int(actual_val), vol_flag))

        work_done += len(test_indices)

        if progress:
            progress.extra["candles_done"] = int(len(test_indices))
            progress.extra["candles_total"] = int(len(test_indices))
            progress.extra["horizon"] = int(horizon)
            progress.extra["train_count"] = int(train_count)
            progress.extra["train_total"] = train_total_est
            progress.extra["state"] = "done"
        fit_time = time.time() - t1

        # Evaluate
        if not all_preds:
            results_by_horizon[str(horizon)] = {
                "error": "No predictions generated",
                "total_candles": len(test_indices),
                "signals": 0,
            }
            continue

        arr = np.array(all_preds)  # columns: idx, pred, prob, actual, vol_flag
        preds = arr[:, 1].astype(int)
        probas = arr[:, 2].astype(float)
        actuals = arr[:, 3].astype(int)
        vol_flags = arr[:, 4].astype(int)
        idxs = arr[:, 0].astype(int)

        total_candles_tested = len(arr)
        total_vol_skips = int(vol_flags.sum())

        base_metrics = _evaluate_predictions_metrics(
            preds,
            probas,
            actuals,
            idxs,
            df_raw,
            horizon,
            total_candles_tested,
            fit_time,
            train_count,
            total_train_time,
            threshold_value=base_threshold,
            vol_skip_flags=vol_flags,
        )

        if not base_metrics.get("signals") and not base_metrics.get("error"):  # no valid signals
            base_metrics["error"] = "All predictions were SKIP (confidence too low)"

        base_metrics["volatility_skips"] = total_vol_skips
        skip_breakdown = getattr(strategy, "_last_skip_breakdown", None)
        if skip_breakdown:
            base_metrics["ema_skip_breakdown"] = {
                "ema_distance": int(skip_breakdown.get("ema_distance", 0)),
                "ema_trend_strength": int(skip_breakdown.get("ema_trend_strength", 0)),
                "ema_direction": int(skip_breakdown.get("ema_direction", 0)),
            }
        logger.info(
            "[backtest] %s H%d summary: signals=%s correct=%s wrong=%s skipped=%s vol_skips=%s ema_skips=%s",
            strategy_name,
            horizon,
            base_metrics.get("signals"),
            base_metrics.get("correct"),
            base_metrics.get("wrong"),
            base_metrics.get("skipped"),
            total_vol_skips,
            {
                "ema_distance": int(skip_breakdown.get("ema_distance", 0)) if skip_breakdown else 0,
                "ema_trend_strength": int(skip_breakdown.get("ema_trend_strength", 0)) if skip_breakdown else 0,
                "ema_direction": int(skip_breakdown.get("ema_direction", 0)) if skip_breakdown else 0,
            },
        )
        results_by_horizon[str(horizon)] = base_metrics

        if sweep_thresholds:
            base_key = _format_threshold_key(base_threshold)
            for thr in sweep_thresholds:
                thr_preds = _apply_threshold_to_probs(probas, thr)
                thr_metrics = _evaluate_predictions_metrics(
                    thr_preds,
                    probas,
                    actuals,
                    idxs,
                    df_raw,
                    horizon,
                    total_candles_tested,
                    fit_time,
                    train_count,
                    total_train_time,
                    threshold_value=thr,
                    vol_skip_flags=vol_flags,
                )
                thr_metrics["volatility_skips"] = total_vol_skips
                if skip_breakdown:
                    thr_metrics["ema_skip_breakdown"] = {
                        "ema_distance": int(skip_breakdown.get("ema_distance", 0)),
                        "ema_trend_strength": int(skip_breakdown.get("ema_trend_strength", 0)),
                        "ema_direction": int(skip_breakdown.get("ema_direction", 0)),
                    }
                logger.info(
                    "[backtest] %s H%d threshold=%s summary: signals=%s correct=%s wrong=%s skipped=%s vol_skips=%s ema_skips=%s",
                    strategy_name,
                    horizon,
                    thr,
                    thr_metrics.get("signals"),
                    thr_metrics.get("correct"),
                    thr_metrics.get("wrong"),
                    thr_metrics.get("skipped"),
                    total_vol_skips,
                    thr_metrics.get("ema_skip_breakdown") or {
                        "ema_distance": 0,
                        "ema_trend_strength": 0,
                        "ema_direction": 0,
                    },
                )
                thr_key = _format_threshold_key(thr)
                threshold_variant_results.setdefault(thr_key, {})[str(horizon)] = thr_metrics

            if base_key in threshold_variant_results:
                results_by_horizon[str(horizon)] = threshold_variant_results[base_key][str(horizon)]

    total_time = time.time() - t0

    if strategy_params:
        resolved_params = strategy_params
    else:
        resolved_params = default_strategy.params if default_strategy else get_strategy(strategy_name).params

    if progress:
        progress.update(total_work, total_work, "Done")

    result = {
        "strategy": strategy_name,
        "params": resolved_params,
        "train_start": effective_train_start,
        "train_end": effective_train_end,
        "test_start": test_start,
        "test_end": test_end,
        "train_period": f"{effective_train_start} -> {effective_train_end}",
        "test_period": f"{test_start} -> {test_end}",
        "train_candles": actual_window,
        "test_candles": len(test_indices),
        "table": table,
        "window_size": window_size,
        "retrain_every": retrain_every,
        "horizons": results_by_horizon,
        "total_time_sec": round(total_time, 2),
        "load_time_sec": round(load_time, 2),
        "feature_time_sec": round(feat_time, 2),
    }

    if threshold_variant_results:
        result["threshold_variants"] = threshold_variant_results

    return result


def _build_rsi_filter_variants(base_params: dict, sweep_params: dict[str, list]) -> dict[str, dict]:
    if not sweep_params:
        return {}
    cache_key = _make_rsi_variant_cache_key(base_params, sweep_params)
    cached = _RSI_VARIANT_CACHE.get(cache_key)
    if cached:
        return {
            key: {k: v for k, v in items}
            for key, items in cached
        }
    keys = list(sweep_params.keys())
    values = []
    for key in keys:
        raw = sweep_params.get(key) or []
        values.append(list(raw) if isinstance(raw, (list, tuple, set, range)) else [raw])
    variants: dict[str, dict] = {}
    if not keys:
        return variants
    for combo in np.array(np.meshgrid(*values, indexing="ij"), dtype=object).T.reshape(-1, len(keys)):
        variant_params = dict(base_params)
        labels: list[str] = []
        for idx, key in enumerate(keys):
            value = combo[idx].item() if hasattr(combo[idx], "item") else combo[idx]
            variant_params[key] = value
            labels.append(f"{key}={value}")
        variants["|".join(labels)] = variant_params
    if variants:
        _store_rsi_variant_cache(cache_key, variants)
    return variants


def _store_rsi_variant_cache(cache_key: tuple[str, str], variants: dict[str, dict]) -> None:
    _RSI_VARIANT_CACHE[cache_key] = [
        (variant_key, tuple(sorted(params.items())))
        for variant_key, params in variants.items()
    ]
    if len(_RSI_VARIANT_CACHE) > _RSI_VARIANT_CACHE_MAX:
        oldest_key = next(iter(_RSI_VARIANT_CACHE))
        _RSI_VARIANT_CACHE.pop(oldest_key, None)


def _make_rsi_variant_cache_key(base_params: dict, sweep_params: dict[str, list]) -> tuple[str, str]:
    base_serialized = json.dumps(base_params, sort_keys=True, default=_json_default)
    sweep_serialized = json.dumps(
        {k: list(v) if isinstance(v, (list, tuple, set, range)) else [v] for k, v in sweep_params.items()},
        sort_keys=True,
        default=_json_default,
    )
    return base_serialized, sweep_serialized


def _json_default(obj: Any):  # pragma: no cover - helper for serialization
    if isinstance(obj, (np.generic,)):
        return obj.item()
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


async def preload_backtest_data(
    train_start: str | None,
    train_end: str | None,
    test_start: str,
    test_end: str,
    horizons: list[int],
    table: str = "c_5m",
    progress: "TaskProgress | None" = None,
    strategy_name: str | None = None,
    min_train_candles: int | None = None,
) -> dict:
    """Load candle data and compute features once, for reuse across brute-force combos."""
    t0 = time.time()

    if progress:
        progress.phase = "Preloading candle data from DB..."
        await asyncio.sleep(0)

    effective_train_start = _resolve_train_window_start(train_start, test_start, min_train_candles, table)
    effective_train_end = train_end or test_start

    df_raw = await load_candles(table, effective_train_start, test_end)
    df_raw = add_direction(df_raw)
    df_raw = add_future_directions(df_raw, horizons)
    df_raw = df_raw.reset_index(drop=True)
    load_time = time.time() - t0

    if progress:
        progress.phase = f"Computing technical features for {len(df_raw)} candles..."
        await asyncio.sleep(0)

    # Optimise: only compute features the strategy actually needs
    saved_usage = dict(FEATURE_USAGE)  # snapshot
    minimal = _STRATEGY_MINIMAL_FEATURES.get(strategy_name) if strategy_name else None
    if minimal:
        for key in FEATURE_USAGE:
            set_feature_usage(**{key: key in minimal})

    t_feat = time.time()
    try:
        df_feat = add_technical_features(df_raw)
    finally:
        # restore global feature flags
        for key, val in saved_usage.items():
            FEATURE_USAGE[key] = val
    feat_time = time.time() - t_feat

    test_start_us = date_to_us(test_start)
    test_end_us = date_to_us(test_end, True)
    test_indices = df_feat.index[
        (df_feat["open_time"] >= test_start_us) & (df_feat["open_time"] <= test_end_us)
    ].tolist()

    return {
        "df_raw": df_raw,
        "df_feat": df_feat,
        "test_indices": test_indices,
        "load_time": round(load_time, 2),
        "feat_time": round(feat_time, 2),
        "train_start": effective_train_start,
        "train_end": effective_train_end,
    }


async def run_backtest_vectorized(
    strategy_name: str,
    strategy_params: dict | None,
    preloaded: dict,
    horizons: list[int],
    window_size: int,
    retrain_every: int,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    table: str,
    threshold_sweep: list[float] | None = None,
    rsi_filter_sweep: dict[str, list] | None = None,
    log_timings: bool = False,
) -> dict:
    """Ultra-fast backtest for rule-based strategies (RSI, Momentum)."""

    df_feat = preloaded["df_feat"]
    df_raw = preloaded["df_raw"]
    test_indices = preloaded["test_indices"]

    if not test_indices:
        return {"error": "No test data found in the given range"}

    t0 = time.time()
    test_start_idx = test_indices[0]
    actual_window = min(window_size, test_start_idx)

    results_by_horizon: dict[str, dict] = {}
    threshold_variant_results: dict[str, dict[str, dict]] = {}
    rsi_filter_variant_results: dict[str, dict] = {}
    timing_enabled = bool(log_timings and strategy_name == "rsi_mean_reversion")
    timing_rows: list[dict[str, Any]] = []

    for horizon in horizons:
        t1 = time.time()
        horizon_start = t1
        target_col = f"future_dir_{horizon}"

        valid_test = np.array([
            i for i in test_indices
            if i + horizon < len(df_feat) and not pd.isna(df_feat.at[i, target_col])
        ])

        if len(valid_test) == 0:
            results_by_horizon[str(horizon)] = {
                "error": "No predictions generated",
                "total_candles": len(test_indices),
                "signals": 0,
            }
            continue

        df_test = df_feat.iloc[valid_test].reset_index(drop=True)
        train_lo = max(0, test_start_idx - actual_window)
        df_train = df_feat.iloc[train_lo:test_start_idx].reset_index(drop=True)

        strategy = get_strategy(strategy_name, strategy_params)
        fit_timer = time.time()
        strategy.fit(df_train, horizon)
        fit_sec = time.time() - fit_timer
        base_fit_state = _snapshot_rsi_fit_state(strategy) if strategy_name == "rsi_mean_reversion" else None

        predict_timer = time.time()
        prob_arr = await resolve_awaitable(strategy.predict_proba(df_test, horizon))
        predict_sec = time.time() - predict_timer
        vol_skip_flags = _extract_vol_skip_flags(getattr(strategy, "_last_vol_skip_flags", None), len(df_test))
        volatility_skips = int(vol_skip_flags.sum())

        base_threshold = float(strategy.params.get("threshold", 0.55))
        sweep_thresholds = (
            _normalize_threshold_sweep(threshold_sweep, base_threshold)
            if threshold_sweep
            else []
        )

        pred_arr = _apply_threshold_to_probs(prob_arr, base_threshold)

        actuals = df_feat[target_col].iloc[valid_test].values.astype(int)
        probas = prob_arr.astype(float)
        total_candles_tested = len(probas)
        core_elapsed = time.time() - horizon_start
        fit_time = core_elapsed

        base_metrics = _evaluate_predictions_metrics(
            y_pred=pred_arr,
            probas=probas,
            actuals=actuals,
            idxs=valid_test,
            df_raw=df_raw,
            horizon=horizon,
            total_candles=total_candles_tested,
            fit_time=fit_time,
            train_count=0,
            total_train_time=0.0,
            threshold_value=base_threshold,
            vol_skip_flags=vol_skip_flags,
        )
        base_metrics["volatility_skips"] = volatility_skips
        results_by_horizon[str(horizon)] = base_metrics

        threshold_sec = 0.0
        threshold_variants_count = len(sweep_thresholds) if sweep_thresholds else 0
        if sweep_thresholds:
            threshold_start = time.time()
            base_key = _format_threshold_key(base_threshold)
            for thr in sweep_thresholds:
                thr_preds = _apply_threshold_to_probs(probas, thr)
                thr_metrics = _evaluate_predictions_metrics(
                    y_pred=thr_preds,
                    probas=probas,
                    actuals=actuals,
                    idxs=valid_test,
                    df_raw=df_raw,
                    horizon=horizon,
                    total_candles=total_candles_tested,
                    fit_time=fit_time,
                    train_count=0,
                    total_train_time=0.0,
                    threshold_value=thr,
                    vol_skip_flags=vol_skip_flags,
                )
                thr_metrics["volatility_skips"] = volatility_skips
                thr_key = _format_threshold_key(thr)
                threshold_variant_results.setdefault(thr_key, {})[str(horizon)] = thr_metrics

            if base_key in threshold_variant_results:
                results_by_horizon[str(horizon)] = threshold_variant_results[base_key][str(horizon)]
            threshold_sec = time.time() - threshold_start

        rsi_variant_sec = 0.0
        rsi_variant_count = 0
        if strategy_name == "rsi_mean_reversion" and rsi_filter_sweep:
            rsi_variant_start = time.time()
            variant_payloads = _build_rsi_filter_variants(dict(strategy.params), rsi_filter_sweep)
            rsi_variant_count = len(variant_payloads)
            for variant_key, variant_params in variant_payloads.items():
                variant_strategy = get_strategy(strategy_name, variant_params)
                if base_fit_state:
                    _apply_rsi_fit_state(variant_strategy, base_fit_state)
                else:
                    variant_strategy.fit(df_train, horizon)
                variant_prob_arr = await resolve_awaitable(variant_strategy.predict_proba(df_test, horizon))
                variant_vol_flags = _extract_vol_skip_flags(
                    getattr(variant_strategy, "_last_vol_skip_flags", None),
                    len(df_test),
                )
                variant_vol_skips = int(variant_vol_flags.sum())
                variant_threshold = float(variant_strategy.params.get("threshold", base_threshold))
                variant_preds = _apply_threshold_to_probs(variant_prob_arr.astype(float), variant_threshold)
                variant_metrics = _evaluate_predictions_metrics(
                    y_pred=variant_preds,
                    probas=variant_prob_arr.astype(float),
                    actuals=actuals,
                    idxs=valid_test,
                    df_raw=df_raw,
                    horizon=horizon,
                    total_candles=total_candles_tested,
                    fit_time=fit_time,
                    train_count=0,
                    total_train_time=0.0,
                    threshold_value=variant_threshold,
                    vol_skip_flags=variant_vol_flags,
                )
                variant_metrics["volatility_skips"] = variant_vol_skips
                rsi_filter_variant_results.setdefault(
                    variant_key,
                    {"params": variant_params, "horizons": {}},
                )["horizons"][str(horizon)] = variant_metrics
            rsi_variant_sec = time.time() - rsi_variant_start

        total_elapsed = time.time() - horizon_start
        if timing_enabled:
            timing_rows.append({
                "horizon": horizon,
                "fit_sec": fit_sec,
                "predict_sec": predict_sec,
                "threshold_sec": threshold_sec,
                "threshold_variants": threshold_variants_count,
                "rsi_sec": rsi_variant_sec,
                "rsi_variants": rsi_variant_count,
                "total_sec": total_elapsed,
                "candles": total_candles_tested,
            })

    total_time = time.time() - t0
    resolved_params = strategy_params if strategy_params else get_strategy(strategy_name).params
    resolved_train_start = preloaded.get("train_start", train_start)
    resolved_train_end = preloaded.get("train_end", train_end)

    result = {
        "strategy": strategy_name,
        "params": resolved_params,
        "train_start": resolved_train_start,
        "train_end": resolved_train_end,
        "test_start": test_start,
        "test_end": test_end,
        "train_period": f"{resolved_train_start} -> {resolved_train_end}",
        "test_period": f"{test_start} -> {test_end}",
        "train_candles": actual_window,
        "test_candles": len(test_indices),
        "table": table,
        "window_size": window_size,
        "retrain_every": retrain_every,
        "horizons": results_by_horizon,
        "total_time_sec": round(total_time, 4),
        "load_time_sec": preloaded["load_time"],
        "feature_time_sec": preloaded["feat_time"],
    }

    if threshold_variant_results:
        result["threshold_variants"] = threshold_variant_results

    if rsi_filter_variant_results:
        result["rsi_filter_variants"] = rsi_filter_variant_results

    if timing_enabled and timing_rows:
        for row in timing_rows:
            horizon_label = row["horizon"]
            logger.info("[RSI_TIMING] H%s FIT %.4fs", horizon_label, row["fit_sec"])
            logger.info("[RSI_TIMING] H%s PREDICT %.4fs", horizon_label, row["predict_sec"])
            logger.info(
                "[RSI_TIMING] H%s THRESHOLDS %.4fs variants=%d",
                horizon_label,
                row["threshold_sec"],
                row["threshold_variants"],
            )
            logger.info(
                "[RSI_TIMING] H%s RSI_VARIANTS %.4fs variants=%d",
                horizon_label,
                row["rsi_sec"],
                row["rsi_variants"],
            )
            logger.info(
                "[RSI_TIMING] H%s TOTAL %.4fs candles=%d",
                horizon_label,
                row["total_sec"],
                row["candles"],
            )

    return result


def _extract_vol_skip_flags(raw_flags: Any, expected_len: int) -> np.ndarray:
    if expected_len <= 0:
        return np.zeros(0, dtype=int)
    if raw_flags is None:
        return np.zeros(expected_len, dtype=int)
    arr = np.asarray(raw_flags)
    if arr.ndim == 0:
        arr = np.full(expected_len, int(arr), dtype=int)
    else:
        arr = arr.astype(int, copy=False)
    if arr.size != expected_len:
        return np.zeros(expected_len, dtype=int)
    return arr


def _monthly_breakdown(signal_dates, y_true, y_pred, vol_skip_dates=None) -> list[dict]:
    if len(signal_dates) == 0 and (vol_skip_dates is None or len(vol_skip_dates) == 0):
        return []

    signal_months = pd.to_datetime(signal_dates).to_period("M") if len(signal_dates) else pd.PeriodIndex([], freq="M")
    vol_months = pd.to_datetime(vol_skip_dates).to_period("M") if vol_skip_dates is not None and len(vol_skip_dates) else pd.PeriodIndex([], freq="M")

    signal_labels = np.array(signal_months.astype(str)) if len(signal_months) else np.array([], dtype=str)
    vol_labels = np.array(vol_months.astype(str)) if len(vol_months) else np.array([], dtype=str)

    unique_months = sorted(set(signal_labels.tolist()) | set(vol_labels.tolist()))
    result = []
    for month in unique_months:
        mask = signal_labels == month
        total = int(mask.sum())
        correct = int((y_true[mask] == y_pred[mask]).sum()) if total > 0 else 0
        accuracy = round(correct / total * 100, 2) if total > 0 else 0
        vol_skips = int((vol_labels == month).sum()) if len(vol_labels) else 0
        result.append({
            "month": month,
            "total": total,
            "correct": correct,
            "accuracy": accuracy,
            "volatility_skips": vol_skips,
        })

    return result


def _daily_breakdown(dates, y_true, y_pred) -> list[dict]:
    days = pd.to_datetime(dates).date
    unique_days = sorted(set(days))
    result = []
    for d in unique_days:
        mask = np.array([dd == d for dd in days])
        correct = (y_true[mask] == y_pred[mask]).sum()
        total = mask.sum()
        result.append({
            "date": str(d),
            "total": int(total),
            "correct": int(correct),
            "accuracy": round(correct / total * 100, 2) if total > 0 else 0,
        })
    return result


def _confidence_distribution(probas) -> dict:
    bins = {
        "50-55%": 0, "55-60%": 0, "60-65%": 0, "65-70%": 0,
        "70-75%": 0, "75-80%": 0, "80-100%": 0,
    }
    conf = np.abs(probas - 0.5) * 2  # 0..1 scale
    actual_conf = 0.5 + conf * 0.5    # back to 50..100%

    for c in actual_conf * 100:
        if c < 55:
            bins["50-55%"] += 1
        elif c < 60:
            bins["55-60%"] += 1
        elif c < 65:
            bins["60-65%"] += 1
        elif c < 70:
            bins["65-70%"] += 1
        elif c < 75:
            bins["70-75%"] += 1
        elif c < 80:
            bins["75-80%"] += 1
        else:
            bins["80-100%"] += 1

    return {k: int(v) for k, v in bins.items()}


def _streak_analysis(y_true, y_pred) -> dict:
    correct_seq = (y_true == y_pred).astype(int)
    max_win = _max_streak(correct_seq, 1)
    max_lose = _max_streak(correct_seq, 0)
    return {
        "max_win_streak": int(max_win),
        "max_lose_streak": int(max_lose),
    }


def _max_streak(arr, value) -> int:
    max_s = 0
    current = 0
    for v in arr:
        if v == value:
            current += 1
            max_s = max(max_s, current)
        else:
            current = 0
    return max_s


def _format_threshold_key(value: float) -> str:
    formatted = f"{value:.6f}".rstrip("0").rstrip(".")
    return formatted if formatted else "0"


def _normalize_threshold_sweep(thresholds: list[float], base_threshold: float) -> list[float]:
    values: list[float] = []
    for t in thresholds:
        if t is None:
            continue
        try:
            values.append(float(t))
        except (TypeError, ValueError):
            continue
    values.append(float(base_threshold))
    seen: set[str] = set()
    normalized: list[float] = []
    for val in sorted(values):
        key = _format_threshold_key(val)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(val)
    return normalized


def _apply_threshold_to_probs(probas: np.ndarray, threshold: float) -> np.ndarray:
    preds = np.full(len(probas), -1, dtype=np.int8)
    up_thr = float(threshold)
    down_thr = 1.0 - up_thr
    preds[probas > up_thr] = 1
    preds[probas < down_thr] = 0
    return preds


def _evaluate_predictions_metrics(
    y_pred: np.ndarray,
    probas: np.ndarray,
    actuals: np.ndarray,
    idxs: np.ndarray,
    df_raw: pd.DataFrame,
    horizon: int,
    total_candles: int,
    fit_time: float,
    train_count: int,
    total_train_time: float,
    threshold_value: float | None = None,
    vol_skip_flags: np.ndarray | None = None,
) -> dict:
    valid_mask = y_pred != -1
    if valid_mask.sum() == 0:
        result = {
            "error": "All predictions were SKIP (confidence too low)",
            "total_candles": total_candles,
            "signals": 0,
        }
        if threshold_value is not None:
            result["threshold"] = float(threshold_value)
        return result

    y_true = actuals[valid_mask]
    y_probs = probas[valid_mask]
    y_idxs = idxs[valid_mask]
    y_pred_valid = y_pred[valid_mask]

    correct = int((y_true == y_pred_valid).sum())
    total_signals = int(valid_mask.sum())
    accuracy = correct / total_signals if total_signals else 0.0

    up_pred_mask = y_pred_valid == 1
    down_pred_mask = y_pred_valid == 0
    up_correct = int((y_true[up_pred_mask] == 1).sum()) if up_pred_mask.sum() > 0 else 0
    down_correct = int((y_true[down_pred_mask] == 0).sum()) if down_pred_mask.sum() > 0 else 0
    up_total = int(up_pred_mask.sum())
    down_total = int(down_pred_mask.sum())

    test_times = pd.to_datetime(df_raw["open_time"].iloc[y_idxs].values, unit="us")
    if vol_skip_flags is not None and len(vol_skip_flags) == len(idxs):
        vol_mask = np.asarray(vol_skip_flags).astype(bool)
        all_times = pd.to_datetime(df_raw["open_time"].iloc[idxs].values, unit="us")
        vol_skip_dates = all_times[vol_mask]
        vol_skip_values = vol_skip_dates.values if len(vol_skip_dates) else None
    else:
        vol_skip_values = None

    monthly = _monthly_breakdown(test_times.values, y_true, y_pred_valid, vol_skip_dates=vol_skip_values)
    daily = _daily_breakdown(test_times.values, y_true, y_pred_valid)
    conf_dist = _confidence_distribution(y_probs)
    streaks = _streak_analysis(y_true, y_pred_valid)

    result = {
        "accuracy": round(accuracy, 6),
        "accuracy_pct": round(accuracy * 100, 2),
        "total_candles": total_candles,
        "signals": total_signals,
        "skipped": total_candles - total_signals,
        "correct": correct,
        "wrong": total_signals - correct,
        "up_predictions": up_total,
        "up_correct": up_correct,
        "up_accuracy": round(up_correct / up_total * 100, 2) if up_total > 0 else 0,
        "down_predictions": down_total,
        "down_correct": down_correct,
        "down_accuracy": round(down_correct / down_total * 100, 2) if down_total > 0 else 0,
        "monthly": monthly,
        "daily": daily,
        "confidence_distribution": conf_dist,
        "streaks": streaks,
        "fit_time_sec": round(fit_time, 2),
        "train_count": train_count,
        "total_train_time_sec": round(total_train_time, 2),
        "predict_time_sec": round(max(fit_time - total_train_time, 0.0), 2),
    }

    if threshold_value is not None:
        result["threshold"] = float(threshold_value)

    return result
