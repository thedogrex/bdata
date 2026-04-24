"""
Super Backtest Module with Hidden Markov Model for Market Regime Detection.

This module provides detailed per-prediction backtesting capabilities
for analyzing winning/losing streak patterns and detecting market regimes
where trading should be paused.
"""

import json
import time
import math
import numpy as np
import pandas as pd
from typing import Any, Optional, Dict, List, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict

from db import DbProvider
from predictor.data_loader import load_candles, add_direction, add_future_directions, date_to_us
from predictor.features import add_technical_features
from predictor.strategies import get_strategy

# Optional HMM - install via: pip install hmmlearn
try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    print("[super_backtest] hmmlearn not available, HMM analysis disabled")

db = DbProvider()


@dataclass
class PredictionRecord:
    """Single prediction record with features for analysis."""
    candle_idx: int
    open_time: int  # microseconds
    prediction: int  # -1=skip, 0=down, 1=up
    probability: float
    actual: int  # 0=down, 1=up
    is_correct: bool
    is_signal: bool
    vol_skip: bool
    
    # Technical features
    rsi: Optional[float] = None
    rsi_percentile: Optional[float] = None
    bb_position: Optional[float] = None
    volatility_short: Optional[float] = None
    volatility_long: Optional[float] = None
    trend_3c: Optional[float] = None
    trend_10c: Optional[float] = None
    volume_ratio: Optional[float] = None
    
    # Streak features (computed post-run)
    prev_streak_type: Optional[int] = None  # 0=lose, 1=win
    prev_streak_len: Optional[int] = None
    
    # HMM state (computed post-run)
    hmm_state: Optional[int] = None
    hmm_state_prob: Optional[float] = None
    
    def to_db_row(self, super_run_id: int) -> tuple:
        """Convert to database row tuple."""
        return (
            super_run_id,
            self.candle_idx,
            self.open_time,
            self.prediction,
            self.probability,
            self.actual,
            1 if self.is_correct else 0,
            1 if self.is_signal else 0,
            1 if self.vol_skip else 0,
            self.rsi,
            self.rsi_percentile,
            self.bb_position,
            self.volatility_short,
            self.volatility_long,
            self.trend_3c,
            self.trend_10c,
            self.volume_ratio,
            self.prev_streak_type,
            self.prev_streak_len,
            self.hmm_state,
            self.hmm_state_prob,
        )


async def run_super_backtest(
    strategy_name: str,
    strategy_params: Optional[dict],
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    horizon: int = 1,
    table: str = "c_5m",
    window_size: int = 5000,
    retrain_every: int = 500,
    hmm_states: int = 2,
) -> Dict[str, Any]:
    """
    Run a super backtest that saves every prediction with features.
    
    This is slower than regular backtest but provides data for HMM analysis
    of winning/losing streak patterns.
    """
    t0 = time.time()
    
    # Load data
    df_raw = await load_candles(table, train_start, test_end)
    df_raw = add_direction(df_raw)
    df_raw = add_future_directions(df_raw, [horizon])
    df_raw = df_raw.reset_index(drop=True)
    
    df_feat = add_technical_features(df_raw)
    
    # Find test indices
    test_start_us = date_to_us(test_start)
    test_end_us = date_to_us(test_end, True)
    test_indices = df_feat.index[
        (df_feat["open_time"] >= test_start_us) & (df_feat["open_time"] <= test_end_us)
    ].tolist()
    
    if not test_indices:
        return {"error": "No test data found in the given range"}
    
    test_start_idx = test_indices[0]
    actual_window = min(window_size, test_start_idx)
    target_col = f"future_dir_{horizon}"
    
    # Initialize strategy
    strategy = get_strategy(strategy_name, strategy_params)
    
    predictions: List[PredictionRecord] = []
    last_train_idx = -retrain_every
    train_count = 0
    total_steps = len(test_indices)
    
    for step, i in enumerate(test_indices):
        # Progress print every 100 steps
        if step % 100 == 0 and step > 0:
            pct = (step / total_steps) * 100
            remaining = total_steps - step
            print(f"[super_backtest] Progress: {step}/{total_steps} ({pct:.1f}%) - {remaining} remaining")
        
        # Skip if no future data
        if i + horizon >= len(df_feat):
            break
            
        actual_val = df_feat.at[i, target_col]
        if pd.isna(actual_val):
            break
        
        # Retrain if needed
        if i - last_train_idx >= retrain_every:
            train_lo = max(0, i - actual_window)
            train_hi = i
            df_train = df_feat.iloc[train_lo:train_hi].reset_index(drop=True)
            
            if len(df_train) >= 100:
                strategy.fit(df_train, horizon)
                train_count += 1
                last_train_idx = i
        
        # Predict single candle
        df_single = df_feat.iloc[[i]]
        pred_arr = await _resolve(strategy.predict(df_single, horizon))
        prob_arr = await _resolve(strategy.predict_proba(df_single, horizon))
        
        pred = int(pred_arr[0])
        prob = float(prob_arr[0])
        actual = int(actual_val)
        
        # Extract features
        row = df_feat.iloc[i]
        
        record = PredictionRecord(
            candle_idx=i,
            open_time=int(row["open_time"]),
            prediction=pred,
            probability=prob,
            actual=actual,
            is_correct=(pred == actual) if pred in [0, 1] else False,
            is_signal=pred in [0, 1],
            vol_skip=pred == -1 and getattr(strategy, '_last_vol_skip_count', 0) > 0,
            rsi=_safe_float(row.get("rsi_14")),
            rsi_percentile=_extract_rsi_percentile(strategy, row),
            bb_position=_safe_float(row.get("bb_position")),
            volatility_short=_safe_float(row.get("volatility_5")),
            volatility_long=_safe_float(row.get("volatility_20")),
            trend_3c=_safe_float(row.get("trend_3")),
            trend_10c=_safe_float(row.get("trend_10")),
            volume_ratio=_safe_float(row.get("volume_ratio")),
        )
        
        predictions.append(record)
    
    # Compute streak features
    predictions = _compute_streak_features(predictions)
    
    # Save to database
    super_run_id = await _save_super_backtest_run(
        strategy_name=strategy_name,
        params=strategy.params if strategy else (strategy_params or {}),
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        table=table,
        window_size=window_size,
        horizon=horizon,
        predictions=predictions,
        hmm_states=hmm_states,
    )
    
    total_time = time.time() - t0
    
    return {
        "super_run_id": super_run_id,
        "strategy": strategy_name,
        "predictions_count": len(predictions),
        "train_count": train_count,
        "time_sec": round(total_time, 2),
    }


async def analyze_with_hmm(
    super_run_id: int, 
    n_states: int = 2,
    use_prev_result: bool = True,
    good_threshold: float = 55.0,  # Accuracy >= this is "good" regime
    bad_threshold: float = 45.0,    # Accuracy <= this is "bad" regime
    filter_threshold: float = 0.6,  # P(bad_state) > this → skip trade
) -> Dict[str, Any]:
    """
    Apply Hidden Markov Model to identify market regimes.
    
    Features used (as recommended):
    - rsi_14: mean reversion indicator
    - delta_rsi_14: momentum (change in RSI)
    - ema_diff_20: trend (price vs EMA20)
    - atr_14: volatility regime
    - prev_result (optional): behavioral feature - previous prediction result
    
    Thresholds:
    - good_threshold: Accuracy % to classify state as "good" regime
    - bad_threshold: Accuracy % to classify state as "bad" regime  
    - filter_threshold: Skip trade if P(bad_state) > threshold
    
    Returns regimes classified as 'good' (high accuracy), 'bad' (low accuracy), 
    or 'neutral' based on prediction success rates in each hidden state.
    """
    if not HMM_AVAILABLE:
        return {"error": "hmmlearn not installed. Run: pip install hmmlearn"}
    
    # Load predictions with raw candle data for computing features
    rows = await db.fetchall(
        """
        SELECT p.candle_idx, p.open_time, p.is_correct, p.is_signal, p.probability,
               p.rsi, p.prev_streak_type, p.prev_streak_len,
               c.open, c.high, c.low, c.close, c.volume
        FROM super_backtest_predictions p
        JOIN c_5m c ON c.open_time = p.open_time
        WHERE p.super_run_id = %s AND p.is_signal = 1
        ORDER BY p.candle_idx
        """,
        (super_run_id,)
    )
    
    if not rows or len(rows) < 100:
        return {"error": "Not enough signals for HMM analysis (need 100+)"}
    
    # Build feature matrix for HMM
    # Features: rsi_14, delta_rsi_14, ema_diff_20, atr_14, [prev_result]
    features = []
    
    for i, row in enumerate(rows):
        rsi_val = float(row[5] or 50)  # rsi_14
        
        # delta_rsi_14 (change from previous)
        delta_rsi = 0.0
        if i >= 1:
            prev_rsi = float(rows[i-1][5] or 50)
            delta_rsi = rsi_val - prev_rsi
        
        # ema_diff_20 - price vs EMA20 (computed from closes)
        # Approximate: use current close vs moving average of last 20
        ema_diff = 0.0
        if i >= 19:
            closes = [float(rows[j][11]) for j in range(i-19, i+1)]  # row[11] = close
            ema20 = pd.Series(closes).ewm(span=20, adjust=False).mean().iloc[-1]
            current_close = float(row[11])
            ema_diff = (current_close - ema20) / ema20  # normalized diff
        
        # atr_14 - approximate using recent volatility
        atr = 0.0
        if i >= 13:
            highs = [float(rows[j][9]) for j in range(i-13, i+1)]  # row[9] = high
            lows = [float(rows[j][10]) for j in range(i-13, i+1)]  # row[10] = low
            closes = [float(rows[j][11]) for j in range(i-13, i+1)]
            
            tr_list = []
            for j in range(1, len(highs)):
                tr = max(
                    highs[j] - lows[j],
                    abs(highs[j] - closes[j-1]),
                    abs(lows[j] - closes[j-1])
                )
                tr_list.append(tr)
            
            if tr_list:
                atr = pd.Series(tr_list).ewm(span=14, adjust=False).mean().iloc[-1]
                # Normalize by price
                atr = atr / closes[-1] if closes[-1] > 0 else 0
        
        # prev_result - behavioral feature (1=win, 0=loss, 0.5=first)
        prev_result = 0.5
        if use_prev_result and i >= 1:
            prev_result = 1.0 if rows[i-1][2] == 1 else 0.0  # is_correct from t-1
        
        feat_vec = [
            rsi_val / 100.0,  # rsi_14 normalized to 0-1
            delta_rsi / 20.0,  # delta_rsi normalized (typical range -20 to +20)
            ema_diff * 100.0,  # ema_diff as percentage
            min(atr * 100.0, 5.0),  # atr normalized, capped at 5%
        ]
        
        if use_prev_result:
            feat_vec.append(prev_result)  # already 0, 0.5, or 1
        
        features.append(feat_vec)
    
    X = np.array(features)
    
    # Fit Gaussian HMM
    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=100,
        random_state=42,
    )
    
    try:
        model.fit(X)
        hidden_states = model.predict(X)
        state_probs = model.predict_proba(X)  # Get probabilities for each state
        
        # Determine which state is "bad" based on accuracy
        # We'll label states after analyzing performance below
        
        # Update predictions with HMM state AND probability of being in that state
        for i, row in enumerate(rows):
            state = int(hidden_states[i])
            state_prob = float(state_probs[i][state])
            await db.execute(
                """
                UPDATE super_backtest_predictions
                SET hmm_state = %s, hmm_state_prob = %s
                WHERE super_run_id = %s AND candle_idx = %s
                """,
                (state, state_prob, super_run_id, row[0])
            )
        
        # Analyze each state's performance
        state_stats = []
        for state in range(n_states):
            state_mask = hidden_states == state
            state_indices = np.where(state_mask)[0]
            
            if len(state_indices) == 0:
                continue
            
            # Get predictions in this state
            correct_count = sum(1 for i in state_indices if rows[i][2] == 1)
            total_count = len(state_indices)
            accuracy = correct_count / total_count if total_count > 0 else 0
            
            # Get feature means for this state
            state_features = X[state_mask]
            feature_means = state_features.mean(axis=0).tolist()
            
            feature_names = ["rsi_14", "delta_rsi", "ema_diff_20", "atr_14"]
            if use_prev_result:
                feature_names.append("prev_result")
            
            feature_means_dict = {}
            for idx, name in enumerate(feature_names):
                if name == "rsi_14":
                    feature_means_dict[name] = round(feature_means[idx] * 100, 1)
                elif name == "delta_rsi":
                    feature_means_dict[name] = round(feature_means[idx] * 20, 2)
                elif name == "ema_diff_20":
                    feature_means_dict[name] = round(feature_means[idx] / 100.0, 4)
                elif name == "atr_14":
                    feature_means_dict[name] = round(feature_means[idx] / 100.0, 4)
                elif name == "prev_result":
                    feature_means_dict[name] = round(feature_means[idx], 2)
            
            state_stats.append({
                "state": state,
                "predictions_count": total_count,
                "correct": correct_count,
                "wrong": total_count - correct_count,
                "accuracy_pct": round(accuracy * 100, 2),
                "feature_means": feature_means_dict
            })
        
        # Classify states by accuracy using configurable thresholds
        bad_state_id = None
        good_state_id = None
        
        for s in state_stats:
            acc = s["accuracy_pct"]
            if acc >= good_threshold:
                s["label"] = "good"
                good_state_id = s["state"]
            elif acc <= bad_threshold:
                s["label"] = "bad"
                if bad_state_id is None:
                    bad_state_id = s["state"]  # Use first bad state for filtering
            else:
                s["label"] = "neutral"
        
        # Fallback: if no good state found, pick highest accuracy as good
        if good_state_id is None and len(state_stats) >= 1:
            sorted_by_acc = sorted(state_stats, key=lambda x: x["accuracy_pct"], reverse=True)
            sorted_by_acc[0]["label"] = "good"
            good_state_id = sorted_by_acc[0]["state"]
        
        # If no explicit bad state found, use lowest accuracy as bad
        if bad_state_id is None and len(state_stats) >= 2:
            sorted_by_acc = sorted(state_stats, key=lambda x: x["accuracy_pct"])
            # Don't mark as bad if it's already good
            if sorted_by_acc[0]["state"] != good_state_id:
                sorted_by_acc[0]["label"] = "bad"
                bad_state_id = sorted_by_acc[0]["state"]
        
        # Calculate how many trades would be filtered by P(bad) > filter_threshold rule
        trades_skipped = 0
        bad_skips_correct = 0
        bad_skips_wrong = 0
        if bad_state_id is not None:
            for i, probs in enumerate(state_probs):
                if probs[bad_state_id] > filter_threshold:
                    trades_skipped += 1
                    # Check if skipping would have been correct (actual was loss)
                    if rows[i][2] == 0:  # is_correct = 0 (wrong prediction)
                        bad_skips_correct += 1
                    else:
                        bad_skips_wrong += 1
        
        # Add filtering effectiveness to stats
        for s in state_stats:
            if s["label"] == "bad":
                s["filter_threshold"] = filter_threshold
                s["trades_would_skip"] = trades_skipped
                if trades_skipped > 0:
                    s["skip_accuracy"] = round(bad_skips_correct / trades_skipped * 100, 2)
        
        # Calculate total winrate with regime filter applied
        # (only count trades when P(good_state) is high or P(bad_state) is low)
        trades_taken_with_filter = 0
        trades_correct_with_filter = 0
        
        if good_state_id is not None:
            for i, probs in enumerate(state_probs):
                # Trade when probability of good state is high enough
                # or probability of bad state is below threshold
                should_trade = False
                if bad_state_id is not None:
                    # Trade if P(bad) <= filter_threshold
                    should_trade = probs[bad_state_id] <= filter_threshold
                else:
                    # Only trade in good state if no bad state identified
                    should_trade = hidden_states[i] == good_state_id
                
                if should_trade:
                    trades_taken_with_filter += 1
                    if rows[i][2] == 1:  # is_correct = 1
                        trades_correct_with_filter += 1
        
        filtered_winrate = None
        if trades_taken_with_filter > 0:
            filtered_winrate = round(trades_correct_with_filter / trades_taken_with_filter * 100, 2)
        
        # Calculate baseline (original) winrate for comparison
        total_signals = len(rows)
        total_correct = sum(1 for r in rows if r[2] == 1)
        baseline_winrate = round(total_correct / total_signals * 100, 2) if total_signals > 0 else 0
        
        # Save regimes to database
        await _save_regimes(super_run_id, rows, hidden_states, state_probs, bad_state_id, state_stats)
        
        # Update run with HMM model
        hmm_model_json = json.dumps({
            "n_states": n_states,
            "means": model.means_.tolist(),
            "covars": model.covars_.tolist(),
            "transmat": model.transmat_.tolist(),
        })
        
        await db.execute(
            "UPDATE super_backtest_runs SET hmm_model_json = %s WHERE id = %s",
            (hmm_model_json, super_run_id)
        )
        
        # Build month-by-month analysis
        monthly_stats = _compute_monthly_analysis(rows, hidden_states, state_stats, bad_state_id, good_threshold, bad_threshold)
        
        return {
            "super_run_id": super_run_id,
            "n_states": n_states,
            "states": state_stats,
            "transition_matrix": model.transmat_.tolist(),
            "thresholds": {
                "good_threshold": good_threshold,
                "bad_threshold": bad_threshold,
                "filter_threshold": filter_threshold,
            },
            "monthly_analysis": monthly_stats,
            "regime_strategy": {
                "baseline_winrate": baseline_winrate,
                "baseline_trades": total_signals,
                "filtered_winrate": filtered_winrate,
                "filtered_trades": trades_taken_with_filter,
                "skipped_trades": trades_skipped,
                "improvement": round(filtered_winrate - baseline_winrate, 2) if filtered_winrate else None,
            }
        }
        
    except Exception as e:
        return {"error": f"HMM fitting failed: {str(e)}"}


def _compute_monthly_analysis(
    rows: List[Tuple],
    hidden_states: np.ndarray,
    state_stats: List[Dict],
    bad_state_id: Optional[int],
    good_threshold: float,
    bad_threshold: float,
) -> List[Dict]:
    """
    Compute month-by-month statistics:
    1. Standard RSI prediction rate
    2. Number of RSI signals
    3. Signals count during good regime
    4. Winning rate during good regime
    """
    from collections import defaultdict
    
    # Identify which states are "good"
    good_state_ids = {s["state"] for s in state_stats if s.get("label") == "good"}
    
    # Group predictions by month
    monthly_data = defaultdict(lambda: {
        "total_signals": 0,
        "total_correct": 0,
        "good_regime_signals": 0,
        "good_regime_correct": 0,
    })
    
    for i, row in enumerate(rows):
        # row[1] = open_time (microseconds)
        open_time_us = int(row[1])
        dt = datetime.utcfromtimestamp(open_time_us / 1_000_000)
        month_key = dt.strftime("%Y-%m")
        
        is_correct = row[2] == 1  # is_correct
        
        monthly_data[month_key]["total_signals"] += 1
        if is_correct:
            monthly_data[month_key]["total_correct"] += 1
        
        # Check if this prediction was in a good regime
        if hidden_states[i] in good_state_ids:
            monthly_data[month_key]["good_regime_signals"] += 1
            if is_correct:
                monthly_data[month_key]["good_regime_correct"] += 1
    
    # Build result list sorted by month
    result = []
    for month in sorted(monthly_data.keys()):
        data = monthly_data[month]
        total_signals = data["total_signals"]
        total_correct = data["total_correct"]
        good_signals = data["good_regime_signals"]
        good_correct = data["good_regime_correct"]
        
        result.append({
            "month": month,
            "rsi_signals_count": total_signals,
            "rsi_win_rate": round(total_correct / total_signals * 100, 2) if total_signals > 0 else 0,
            "good_regime_signals": good_signals,
            "good_regime_win_rate": round(good_correct / good_signals * 100, 2) if good_signals > 0 else 0,
            "good_regime_pct": round(good_signals / total_signals * 100, 1) if total_signals > 0 else 0,
        })
    
    return result


async def get_super_backtest_results(super_run_id: int) -> Optional[Dict[str, Any]]:
    """Get complete results for a super backtest run."""
    row = await db.fetchone(
        """
        SELECT id, strategy, params_json, train_start, train_end, test_start, test_end,
               table_name, window_size, horizon, total_candles, signals, correct, wrong,
               accuracy_pct, hmm_states, hmm_model_json, created_at
        FROM super_backtest_runs WHERE id = %s
        """,
        (super_run_id,)
    )
    
    if not row:
        return None
    
    return {
        "id": row[0],
        "strategy": row[1],
        "params": json.loads(row[2]) if row[2] else {},
        "train_start": row[3],
        "train_end": row[4],
        "test_start": row[5],
        "test_end": row[6],
        "table": row[7],
        "window_size": row[8],
        "horizon": row[9],
        "total_candles": row[10],
        "signals": row[11],
        "correct": row[12],
        "wrong": row[13],
        "accuracy_pct": row[14],
        "hmm_states": row[15],
        "hmm_model": json.loads(row[16]) if row[16] else None,
        "created_at": str(row[17]) if row[17] else None,
    }


async def get_super_predictions(super_run_id: int, limit: int = 1000) -> List[Dict]:
    """Get detailed predictions for a super backtest."""
    rows = await db.fetchall(
        """
        SELECT candle_idx, open_time, prediction, probability, actual, is_correct,
               rsi, rsi_percentile, bb_position, volatility_short, prev_streak_len,
               hmm_state
        FROM super_backtest_predictions
        WHERE super_run_id = %s AND is_signal = 1
        ORDER BY candle_idx
        LIMIT %s
        """,
        (super_run_id, limit)
    )
    
    results = []
    for row in rows:
        results.append({
            "candle_idx": row[0],
            "open_time": row[1],
            "prediction": row[2],
            "probability": row[3],
            "actual": row[4],
            "is_correct": bool(row[5]),
            "rsi": row[6],
            "rsi_percentile": row[7],
            "bb_position": row[8],
            "volatility_short": row[9],
            "prev_streak_len": row[10],
            "hmm_state": row[11],
        })
    
    return results


async def get_regimes(super_run_id: int) -> List[Dict]:
    """Get detected regimes for a super backtest."""
    rows = await db.fetchall(
        """
        SELECT state, state_label, start_idx, end_idx, start_time, end_time,
               predictions_count, signals_count, correct_count, wrong_count,
               accuracy_pct, avg_probability
        FROM super_backtest_regimes
        WHERE super_run_id = %s
        ORDER BY start_idx
        """,
        (super_run_id,)
    )
    
    results = []
    for row in rows:
        results.append({
            "state": row[0],
            "label": row[1],
            "start_idx": row[2],
            "end_idx": row[3],
            "start_time": row[4],
            "end_time": row[5],
            "predictions_count": row[6],
            "signals_count": row[7],
            "correct_count": row[8],
            "wrong_count": row[9],
            "accuracy_pct": row[10],
            "avg_probability": row[11],
        })
    
    return results


# -----------------------------------------------------------------------------
# Private helpers
# -----------------------------------------------------------------------------

async def _resolve(awaitable):
    """Resolve potentially async result."""
    if hasattr(awaitable, '__await__'):
        return await awaitable
    return awaitable


def _safe_float(val) -> Optional[float]:
    """Safely convert to float, return None if invalid."""
    try:
        if pd.isna(val):
            return None
        return float(val)
    except (TypeError, ValueError):
        return None


def _extract_rsi_percentile(strategy, row) -> Optional[float]:
    """Extract RSI percentile from strategy if available."""
    try:
        if hasattr(strategy, '_rsi_percentiles') and hasattr(strategy, '_rsi_period'):
            rsi_col = f"rsi_{strategy._rsi_period}"
            if rsi_col in row:
                rsi_val = float(row[rsi_col])
                if hasattr(strategy, '_train_rsi'):
                    pct = (strategy._train_rsi < rsi_val).mean()
                    return float(pct * 100)
    except Exception:
        pass
    return None


def _compute_streak_features(predictions: List[PredictionRecord]) -> List[PredictionRecord]:
    """Compute previous streak features for each prediction."""
    # Track streak as we iterate
    current_streak_type = None  # None, 0 (lose), 1 (win)
    current_streak_len = 0
    
    for pred in predictions:
        # Set streak from previous predictions
        if current_streak_type is not None:
            pred.prev_streak_type = current_streak_type
            pred.prev_streak_len = current_streak_len
        
        # Update streak for next iteration (only if this was a signal)
        if pred.is_signal:
            if pred.is_correct:
                if current_streak_type == 1:
                    current_streak_len += 1
                else:
                    current_streak_type = 1
                    current_streak_len = 1
            else:
                if current_streak_type == 0:
                    current_streak_len += 1
                else:
                    current_streak_type = 0
                    current_streak_len = 1
    
    return predictions


async def _save_super_backtest_run(
    strategy_name: str,
    params: dict,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    table: str,
    window_size: int,
    horizon: int,
    predictions: List[PredictionRecord],
    hmm_states: int,
) -> int:
    """Save super backtest run and predictions to database."""
    
    # Calculate metrics
    total = len(predictions)
    signals = [p for p in predictions if p.is_signal]
    correct = [p for p in predictions if p.is_correct]
    
    accuracy_pct = (len(correct) / len(signals) * 100) if signals else 0
    
    # Insert run record
    run_id = await db.execute(
        """
        INSERT INTO super_backtest_runs
            (strategy, params_json, train_start, train_end, test_start, test_end,
             table_name, window_size, horizon, total_candles, signals, correct, wrong,
             accuracy_pct, hmm_states)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            strategy_name,
            json.dumps(params, ensure_ascii=False),
            train_start,
            train_end,
            test_start,
            test_end,
            table,
            window_size,
            horizon,
            total,
            len(signals),
            len(correct),
            len(signals) - len(correct),
            round(accuracy_pct, 2),
            hmm_states,
        )
    )
    
    # Insert predictions in batches
    BATCH_SIZE = 500
    for i in range(0, len(predictions), BATCH_SIZE):
        batch = predictions[i:i+BATCH_SIZE]
        rows = [p.to_db_row(run_id) for p in batch]
        
        await db.executemany(
            """
            INSERT INTO super_backtest_predictions
                (super_run_id, candle_idx, open_time, prediction, probability, actual,
                 is_correct, is_signal, vol_skip, rsi, rsi_percentile, bb_position,
                 volatility_short, volatility_long, trend_3c, trend_10c, volume_ratio,
                 prev_streak_type, prev_streak_len, hmm_state, hmm_state_prob)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            rows
        )
    
    return run_id


async def _save_regimes(
    super_run_id: int,
    rows: List[tuple],
    hidden_states: np.ndarray,
    state_probs: np.ndarray,
    bad_state_id: Optional[int],
    state_stats: List[dict]
):
    """Save detected regimes to database with P(bad_state) filtering info."""
    # Find contiguous segments of same state
    if len(hidden_states) == 0:
        return
    
    current_state = hidden_states[0]
    start_idx = 0
    
    segments = []
    for i in range(1, len(hidden_states)):
        if hidden_states[i] != current_state:
            segments.append((current_state, start_idx, i - 1))
            current_state = hidden_states[i]
            start_idx = i
    
    # Add final segment
    segments.append((current_state, start_idx, len(hidden_states) - 1))
    
    # Create label lookup
    label_map = {s["state"]: s.get("label", "unknown") for s in state_stats}
    
    # Calculate average state probability for each segment
    for state, seg_start, seg_end in segments:
        segment_correct = sum(1 for i in range(seg_start, seg_end + 1) if rows[i][2] == 1)
        segment_total = seg_end - seg_start + 1
        
        # Average probability of being in this state during the segment
        avg_state_prob = float(state_probs[seg_start:seg_end+1, state].mean())
        
        # Count how many predictions in this segment would be filtered
        # (if P(bad_state) > 0.6 for bad state)
        filtered_count = 0
        if bad_state_id is not None and state == bad_state_id:
            filtered_count = sum(
                1 for i in range(seg_start, seg_end + 1)
                if state_probs[i, bad_state_id] > 0.6
            )
        
        start_row = rows[seg_start]
        end_row = rows[seg_end]
        
        await db.execute(
            """
            INSERT INTO super_backtest_regimes
                (super_run_id, state, state_label, start_idx, end_idx, start_time, end_time,
                 predictions_count, signals_count, correct_count, wrong_count, accuracy_pct,
                 avg_probability)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                super_run_id,
                int(state),
                label_map.get(int(state), "unknown"),
                int(start_row[0]),  # candle_idx
                int(end_row[0]),
                int(start_row[1]),  # open_time
                int(end_row[1]),
                segment_total,
                segment_total,
                segment_correct,
                segment_total - segment_correct,
                round(segment_correct / segment_total * 100, 2) if segment_total > 0 else 0,
                round(avg_state_prob, 4),
            )
        )
