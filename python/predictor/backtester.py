import time
import asyncio
import numpy as np
import pandas as pd
from typing import Optional, TYPE_CHECKING

from predictor.data_loader import load_candles, add_direction, add_future_directions, date_to_us
from predictor.features import add_technical_features
from predictor.strategies import get_strategy, BaseStrategy

if TYPE_CHECKING:
    from predictor.task_manager import TaskProgress


RULE_BASED_STRATEGIES = {"rsi_mean_reversion", "momentum", "stochastic_adx", "candlestick_pattern"}


async def run_backtest(
    strategy_name: str,
    strategy_params: dict | None,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    horizons: list[int] | None = None,
    table: str = "c_5m",
    window_size: int = 5000,
    retrain_every: int = 500,
    progress: "TaskProgress | None" = None,
    preloaded: dict | None = None,
) -> dict:
    """
    Moving-window backtest with progress tracking and pause/cancel support.
    Optimized: features are pre-computed ONCE on the full dataset.
    """
    if horizons is None:
        horizons = [1]

    t0 = time.time()

    if preloaded:
        # Reuse preloaded data (skip expensive data loading + feature computation)
        df_raw = preloaded["df_raw"]
        df_feat = preloaded["df_feat"]
        test_indices = preloaded["test_indices"]
        load_time = preloaded["load_time"]
        feat_time = preloaded["feat_time"]
    else:
        # ── Phase 1: Load candle data ──
        if progress:
            progress.phase = "Phase 1/3: Loading candle data from DB..."
            await asyncio.sleep(0)

        df_raw = await load_candles(table, train_start, test_end)
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

    for h_idx, horizon in enumerate(horizons):
        t1 = time.time()
        target_col = f"future_dir_{horizon}"

        all_preds = []       # (index_in_df, prediction, proba, actual)
        strategy = None
        last_train_idx = -retrain_every  # force first train
        train_count = 0
        total_train_time = 0.0

        for step, i in enumerate(test_indices):
            # Pause/cancel check every 50 candles
            if progress and step % 50 == 0:
                await progress.check_pause_cancel()
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
                    progress.phase = f"H{horizon}: Training model (retrain #{train_count + 1}) on {len(df_train)} candles..."

                t_train = time.time()
                strategy = get_strategy(strategy_name, strategy_params)
                strategy.fit(df_train, horizon)
                train_elapsed = time.time() - t_train
                total_train_time += train_elapsed
                train_count += 1
                last_train_idx = i

                # Yield after training (can be slow for XGBoost)
                await asyncio.sleep(0)

            # Predict single candle (optimized)
            if strategy_name == "random_forest" and hasattr(strategy, "feature_cols") and hasattr(strategy, "predict_row"):
                cols = strategy.feature_cols
                x_row = df_feat.loc[i, cols].to_numpy()
                pred = int(strategy.predict_row(x_row))
                prob = float(strategy.predict_proba_row(x_row))
            else:
                df_single = df_feat.iloc[[i]]
                pred_arr = strategy.predict(df_single, horizon)
                prob_arr = strategy.predict_proba(df_single, horizon)
                pred = int(pred_arr[0])
                prob = float(prob_arr[0])

            all_preds.append((i, pred, prob, int(actual_val)))

        work_done += len(test_indices)
        fit_time = time.time() - t1

        # Evaluate
        if not all_preds:
            results_by_horizon[str(horizon)] = {
                "error": "No predictions generated",
                "total_candles": len(test_indices),
                "signals": 0,
            }
            continue

        arr = np.array(all_preds)  # columns: idx, pred, prob, actual
        preds = arr[:, 1].astype(int)
        probas = arr[:, 2].astype(float)
        actuals = arr[:, 3].astype(int)
        idxs = arr[:, 0].astype(int)

        valid_mask = preds != -1
        total_candles_tested = len(arr)

        if valid_mask.sum() == 0:
            results_by_horizon[str(horizon)] = {
                "error": "All predictions were SKIP (confidence too low)",
                "total_candles": total_candles_tested,
                "signals": 0,
            }
            continue

        y_true = actuals[valid_mask]
        y_pred = preds[valid_mask]
        y_prob = probas[valid_mask]
        y_idxs = idxs[valid_mask]

        correct = int((y_true == y_pred).sum())
        total_signals = int(valid_mask.sum())
        accuracy = correct / total_signals

        up_pred_mask = y_pred == 1
        down_pred_mask = y_pred == 0
        up_correct = int((y_true[up_pred_mask] == 1).sum()) if up_pred_mask.sum() > 0 else 0
        down_correct = int((y_true[down_pred_mask] == 0).sum()) if down_pred_mask.sum() > 0 else 0
        up_total = int(up_pred_mask.sum())
        down_total = int(down_pred_mask.sum())

        # Date-based breakdowns
        test_times = pd.to_datetime(df_raw["open_time"].iloc[y_idxs].values, unit="us")
        monthly = _monthly_breakdown(test_times.values, y_true, y_pred)
        daily = _daily_breakdown(test_times.values, y_true, y_pred)
        conf_dist = _confidence_distribution(y_prob)
        streaks = _streak_analysis(y_true, y_pred)

        results_by_horizon[str(horizon)] = {
            "accuracy": round(accuracy, 6),
            "accuracy_pct": round(accuracy * 100, 2),
            "total_candles": total_candles_tested,
            "signals": total_signals,
            "skipped": total_candles_tested - total_signals,
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
            "predict_time_sec": round(fit_time - total_train_time, 2),
        }

    total_time = time.time() - t0

    resolved_params = strategy_params if strategy_params else get_strategy(strategy_name).params

    if progress:
        progress.update(total_work, total_work, "Done")

    return {
        "strategy": strategy_name,
        "params": resolved_params,
        "train_start": train_start,
        "train_end": train_end,
        "test_start": test_start,
        "test_end": test_end,
        "train_period": f"{train_start} -> {train_end}",
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


async def preload_backtest_data(
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    horizons: list[int],
    table: str = "c_5m",
    progress: "TaskProgress | None" = None,
) -> dict:
    """Load candle data and compute features once, for reuse across brute-force combos."""
    t0 = time.time()

    if progress:
        progress.phase = "Preloading candle data from DB..."
        await asyncio.sleep(0)

    df_raw = await load_candles(table, train_start, test_end)
    df_raw = add_direction(df_raw)
    df_raw = add_future_directions(df_raw, horizons)
    df_raw = df_raw.reset_index(drop=True)
    load_time = time.time() - t0

    if progress:
        progress.phase = f"Computing technical features for {len(df_raw)} candles..."
        await asyncio.sleep(0)

    t_feat = time.time()
    df_feat = add_technical_features(df_raw)
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
    }


def run_backtest_vectorized(
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
) -> dict:
    """
    Ultra-fast backtest for rule-based strategies (RSI, Momentum).
    Calls predict/predict_proba ONCE on the full test set — no moving window.
    ~100-1000x faster per combo than run_backtest.
    """
    df_feat = preloaded["df_feat"]
    df_raw = preloaded["df_raw"]
    test_indices = preloaded["test_indices"]

    if not test_indices:
        return {"error": "No test data found in the given range"}

    t0 = time.time()
    test_start_idx = test_indices[0]
    actual_window = min(window_size, test_start_idx)

    results_by_horizon = {}

    for horizon in horizons:
        t1 = time.time()
        target_col = f"future_dir_{horizon}"

        # Filter valid test indices (must have future direction available)
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

        # Get test data as contiguous DataFrame
        df_test = df_feat.iloc[valid_test].reset_index(drop=True)

        # Training window: last `window_size` candles before test start
        train_lo = max(0, test_start_idx - actual_window)
        df_train = df_feat.iloc[train_lo:test_start_idx].reset_index(drop=True)

        # Create strategy, fit on training window, predict on full test set
        strategy = get_strategy(strategy_name, strategy_params)
        strategy.fit(df_train, horizon)  # learns adaptive stats from training window
        pred_arr = strategy.predict(df_test, horizon)
        prob_arr = strategy.predict_proba(df_test, horizon)

        actuals = df_feat[target_col].iloc[valid_test].values.astype(int)
        preds = pred_arr.astype(int)
        probas = prob_arr.astype(float)

        valid_mask = preds != -1
        total_candles_tested = len(preds)

        if valid_mask.sum() == 0:
            results_by_horizon[str(horizon)] = {
                "error": "All predictions were SKIP (confidence too low)",
                "total_candles": total_candles_tested,
                "signals": 0,
            }
            continue

        y_true = actuals[valid_mask]
        y_pred = preds[valid_mask]
        y_prob = probas[valid_mask]
        y_idxs = valid_test[valid_mask]

        correct = int((y_true == y_pred).sum())
        total_signals = int(valid_mask.sum())
        accuracy = correct / total_signals

        up_pred_mask = y_pred == 1
        down_pred_mask = y_pred == 0
        up_correct = int((y_true[up_pred_mask] == 1).sum()) if up_pred_mask.sum() > 0 else 0
        down_correct = int((y_true[down_pred_mask] == 0).sum()) if down_pred_mask.sum() > 0 else 0
        up_total = int(up_pred_mask.sum())
        down_total = int(down_pred_mask.sum())

        test_times = pd.to_datetime(df_raw["open_time"].iloc[y_idxs].values, unit="us")
        monthly = _monthly_breakdown(test_times.values, y_true, y_pred)
        daily = _daily_breakdown(test_times.values, y_true, y_pred)
        conf_dist = _confidence_distribution(y_prob)
        streaks = _streak_analysis(y_true, y_pred)

        fit_time = time.time() - t1

        results_by_horizon[str(horizon)] = {
            "accuracy": round(accuracy, 6),
            "accuracy_pct": round(accuracy * 100, 2),
            "total_candles": total_candles_tested,
            "signals": total_signals,
            "skipped": total_candles_tested - total_signals,
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
            "fit_time_sec": round(fit_time, 4),
            "train_count": 0,
            "total_train_time_sec": 0,
            "predict_time_sec": round(fit_time, 4),
        }

    total_time = time.time() - t0
    resolved_params = strategy_params if strategy_params else get_strategy(strategy_name).params

    return {
        "strategy": strategy_name,
        "params": resolved_params,
        "train_start": train_start,
        "train_end": train_end,
        "test_start": test_start,
        "test_end": test_end,
        "train_period": f"{train_start} -> {train_end}",
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


def _monthly_breakdown(dates, y_true, y_pred) -> list[dict]:
    months = pd.to_datetime(dates).to_period("M")
    result = []
    for m in months.unique():
        mask = months == m
        correct = (y_true[mask] == y_pred[mask]).sum()
        total = mask.sum()
        result.append({
            "month": str(m),
            "total": int(total),
            "correct": int(correct),
            "accuracy": round(correct / total * 100, 2) if total > 0 else 0,
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
