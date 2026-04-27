"""
HMM Analysis v2 for Super Backtest.

Multiple HMM analyses per super_run, with:
- Flexible feature selection
- Fit modes: all_candles | signals_only | walk_forward
- Regime smoothing (min_regime_len)
- Per-candle state labels stored separately (can compare analyses)
- Timeline data for visual chart

Requires `hmmlearn` package.
"""

from __future__ import annotations

import json
import time
import itertools
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd

from db import DbProvider
from predictor.data_loader import load_candles, add_direction, add_future_directions, date_to_us
from predictor.features import add_technical_features, FEATURE_USAGE

try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False


db = DbProvider()


# Whitelist of features user can pick in UI. Must exist in add_technical_features output.
AVAILABLE_FEATURES = [
    "rsi_14", "rsi_6",
    "bb_pos", "bb_width",
    "volatility_5", "volatility_20", "atr_14",
    "ema_diff_5", "ema_diff_20", "ema_diff_50",
    "macd", "macd_hist",
    "returns_1", "returns_3", "returns_5",
    "volume_ratio", "volume_zscore",
    "stoch_k", "adx",
    "hour", "dow",
    "delta_rsi_14", "delta_bb_pos", "delta_close",
    "rolling_skew_20", "rolling_kurtosis_20",
    "orderflow_proxy",
]

MAX_SWEEP_COMBOS = 75


# ======================================================================
# Helpers (shared)
# ======================================================================

def _clean_threshold_list(values: Optional[list[float]]) -> list[float]:
    if not values:
        return []
    clean: list[float] = []
    seen: set[str] = set()
    for raw in values:
        try:
            val = float(raw)
        except (TypeError, ValueError):
            continue
        key = f"{val:.6f}"
        if key in seen:
            continue
        seen.add(key)
        clean.append(val)
    return clean


# ======================================================================
# Public API
# ======================================================================

async def create_hmm_analysis(
    super_run_id: int,
    name: Optional[str],
    n_states: int,
    features: list[str],
    fit_mode: str = "all_candles",
    walk_train_len: Optional[int] = None,
    walk_step: Optional[int] = None,
    good_threshold: float = 55.0,
    bad_threshold: float = 45.0,
    filter_threshold: float = 0.6,
    min_regime_len: int = 1,
) -> dict[str, Any]:
    """
    Fit HMM with the given config, persist analysis + per-candle state labels.
    Returns the created analysis dict (same shape as get_hmm_analysis).
    """
    if not HMM_AVAILABLE:
        return {"error": "hmmlearn not installed. Run: pip install hmmlearn"}

    if n_states < 2 or n_states > 8:
        return {"error": "n_states must be 2-8"}

    if fit_mode not in ("all_candles", "signals_only", "walk_forward"):
        return {"error": f"Unknown fit_mode: {fit_mode}"}

    # Validate features
    clean_features = [f for f in features if f in AVAILABLE_FEATURES]
    if not clean_features:
        return {"error": "No valid features selected"}

    t0 = time.time()

    core = await _fit_hmm_core(
        super_run_id=super_run_id,
        clean_features=clean_features,
        n_states=n_states,
        fit_mode=fit_mode,
        walk_train_len=walk_train_len,
        walk_step=walk_step,
        min_regime_len=min_regime_len,
    )
    if "error" in core:
        return core

    merged_valid = core["merged_valid"]
    states = core["states"]
    state_probs = core["state_probs"]
    state_stats = core["state_stats"]
    walk_train_len = core.get("walk_train_len")
    walk_step = core.get("walk_step")
    model_summary = core["model_summary"]
    transition_matrix = core["transition_matrix"]
    candles_analyzed = core["candles_analyzed"]

    state_labels = _assign_state_labels(state_stats, good_threshold, bad_threshold)
    for stat in state_stats:
        stat["label"] = state_labels.get(stat["state"], "neutral")

    baseline_metrics, filtered_metrics, skip_mask = _compute_filter_metrics(
        states=states,
        state_probs=state_probs,
        merged=merged_valid,
        state_stats=state_stats,
        filter_threshold=filter_threshold,
        state_labels=state_labels,
    )

    analysis_id = await db.execute(
        """INSERT INTO super_backtest_hmm_analyses
           (super_run_id, name, n_states, feature_set, fit_mode,
            walk_train_len, walk_step,
            good_threshold, bad_threshold, filter_threshold, min_regime_len,
            states_json, transition_matrix_json, model_json,
            baseline_winrate, filtered_winrate, improvement,
            trades_total, trades_taken, trades_skipped, candles_analyzed)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                   %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
        (
            super_run_id,
            name or f"{n_states}st-{fit_mode}-{len(clean_features)}f",
            n_states,
            json.dumps(clean_features),
            fit_mode,
            walk_train_len,
            walk_step,
            good_threshold,
            bad_threshold,
            filter_threshold,
            min_regime_len,
            json.dumps(state_stats),
            json.dumps(transition_matrix),
            json.dumps(model_summary),
            baseline_metrics["winrate"],
            filtered_metrics["winrate"],
            (filtered_metrics["winrate"] - baseline_metrics["winrate"])
                if filtered_metrics["winrate"] is not None else None,
            baseline_metrics["trades"],
            filtered_metrics["trades"],
            filtered_metrics["skipped"],
            candles_analyzed,
        ),
    )

    # Save per-candle states
    await _save_prediction_states(analysis_id, merged_valid, states, state_probs, skip_mask)

    elapsed = time.time() - t0

    return {
        "id": analysis_id,
        "super_run_id": super_run_id,
        "n_states": n_states,
        "features": clean_features,
        "fit_mode": fit_mode,
        "states": state_stats,
        "transition_matrix": transition_matrix,
        "baseline": baseline_metrics,
        "filtered": filtered_metrics,
        "candles_analyzed": candles_analyzed,
        "time_sec": round(elapsed, 2),
    }


async def list_hmm_analyses(super_run_id: int) -> list[dict]:
    rows = await db.fetchall(
        """SELECT id, name, n_states, fit_mode, feature_set,
                  good_threshold, bad_threshold, filter_threshold,
                  baseline_winrate, filtered_winrate, improvement,
                  trades_total, trades_taken, trades_skipped,
                  candles_analyzed, created_at
           FROM super_backtest_hmm_analyses
           WHERE super_run_id = %s
           ORDER BY id DESC""",
        (super_run_id,),
    )
    out = []
    for r in rows:
        out.append({
            "id": r[0],
            "name": r[1],
            "n_states": r[2],
            "fit_mode": r[3],
            "features": json.loads(r[4]) if r[4] else [],
            "good_threshold": r[5],
            "bad_threshold": r[6],
            "filter_threshold": r[7],
            "baseline_winrate": r[8],
            "filtered_winrate": r[9],
            "improvement": r[10],
            "trades_total": r[11],
            "trades_taken": r[12],
            "trades_skipped": r[13],
            "candles_analyzed": r[14],
            "created_at": str(r[15]) if r[15] else None,
        })
    return out


async def get_hmm_analysis(analysis_id: int) -> Optional[dict]:
    row = await db.fetchone(
        """SELECT id, super_run_id, name, n_states, fit_mode, feature_set,
                  walk_train_len, walk_step,
                  good_threshold, bad_threshold, filter_threshold, min_regime_len,
                  states_json, transition_matrix_json, model_json,
                  baseline_winrate, filtered_winrate, improvement,
                  trades_total, trades_taken, trades_skipped,
                  candles_analyzed, created_at
           FROM super_backtest_hmm_analyses WHERE id = %s""",
        (analysis_id,),
    )
    if not row:
        return None
    return {
        "id": row[0],
        "super_run_id": row[1],
        "name": row[2],
        "n_states": row[3],
        "fit_mode": row[4],
        "features": json.loads(row[5]) if row[5] else [],
        "walk_train_len": row[6],
        "walk_step": row[7],
        "good_threshold": row[8],
        "bad_threshold": row[9],
        "filter_threshold": row[10],
        "min_regime_len": row[11],
        "states": json.loads(row[12]) if row[12] else [],
        "transition_matrix": json.loads(row[13]) if row[13] else None,
        "model": json.loads(row[14]) if row[14] else None,
        "baseline_winrate": row[15],
        "filtered_winrate": row[16],
        "improvement": row[17],
        "trades_total": row[18],
        "trades_taken": row[19],
        "trades_skipped": row[20],
        "candles_analyzed": row[21],
        "created_at": str(row[22]) if row[22] else None,
    }


async def delete_hmm_analysis(analysis_id: int) -> dict:
    await db.execute(
        "DELETE FROM super_backtest_prediction_states WHERE hmm_analysis_id = %s",
        (analysis_id,),
    )
    await db.execute(
        "DELETE FROM super_backtest_hmm_analyses WHERE id = %s",
        (analysis_id,),
    )
    return {"deleted": analysis_id}


async def create_hmm_sweep(
    super_run_id: int,
    name: Optional[str],
    n_states: int,
    features: list[str],
    fit_mode: str,
    walk_train_len: Optional[int],
    walk_step: Optional[int],
    min_regime_len: int,
    good_thresholds: list[float],
    bad_thresholds: list[float],
    filter_thresholds: list[float],
) -> dict:
    if not HMM_AVAILABLE:
        return {"error": "hmmlearn not installed. Run: pip install hmmlearn"}

    if n_states < 2 or n_states > 8:
        return {"error": "n_states must be 2-8"}

    if fit_mode not in ("all_candles", "signals_only", "walk_forward"):
        return {"error": f"Unknown fit_mode: {fit_mode}"}

    clean_features = [f for f in features if f in AVAILABLE_FEATURES]
    if len(clean_features) < 2:
        return {"error": "Select at least 2 valid features"}

    good_vals = _clean_threshold_list(good_thresholds)
    bad_vals = _clean_threshold_list(bad_thresholds)
    filter_vals = _clean_threshold_list(filter_thresholds)
    if not good_vals or not bad_vals or not filter_vals:
        return {"error": "Provide at least one value for good, bad, and filter thresholds"}

    combos = list(itertools.product(good_vals, bad_vals, filter_vals))
    if not combos:
        return {"error": "No threshold combinations to evaluate"}
    if len(combos) > MAX_SWEEP_COMBOS:
        return {"error": f"Too many combinations ({len(combos)}). Max {MAX_SWEEP_COMBOS}."}

    t0 = time.time()
    core = await _fit_hmm_core(
        super_run_id=super_run_id,
        clean_features=clean_features,
        n_states=n_states,
        fit_mode=fit_mode,
        walk_train_len=walk_train_len,
        walk_step=walk_step,
        min_regime_len=min_regime_len,
    )
    if "error" in core:
        return core

    merged_valid = core["merged_valid"]
    states = core["states"]
    state_probs = core["state_probs"]
    state_stats = core["state_stats"]
    walk_train_len = core.get("walk_train_len")
    walk_step = core.get("walk_step")
    transition_matrix = core["transition_matrix"]
    model_summary = core["model_summary"]
    candles_analyzed = core["candles_analyzed"]

    is_signal = merged_valid["is_signal"].astype(bool).values
    is_correct = merged_valid["is_correct"].astype(bool).values
    baseline_trades = int(is_signal.sum())
    baseline_correct = int((is_signal & is_correct).sum())
    baseline_winrate = round(baseline_correct / baseline_trades * 100, 2) if baseline_trades else 0.0

    sweep_id = await db.execute(
        """INSERT INTO super_backtest_hmm_sweeps
               (super_run_id, name, n_states, feature_set, fit_mode,
                walk_train_len, walk_step, min_regime_len,
                combos_total, baseline_trades, baseline_correct, baseline_winrate,
                candles_analyzed, states_json, transition_matrix_json, model_json)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
        (
            super_run_id,
            name or f"sweep-{n_states}st-{fit_mode}-{len(clean_features)}f",
            n_states,
            json.dumps(clean_features),
            fit_mode,
            walk_train_len,
            walk_step,
            min_regime_len,
            len(combos),
            baseline_trades,
            baseline_correct,
            baseline_winrate,
            candles_analyzed,
            json.dumps(state_stats),
            json.dumps(transition_matrix),
            json.dumps(model_summary),
        ),
    )

    combo_rows = []
    combo_payload = []
    for idx, (good_thr, bad_thr, filter_thr) in enumerate(combos, start=1):
        state_labels = _assign_state_labels(state_stats, good_thr, bad_thr)
        baseline_metrics, filtered_metrics, skip_mask = _compute_filter_metrics(
            states=states,
            state_probs=state_probs,
            merged=merged_valid,
            state_stats=state_stats,
            filter_threshold=filter_thr,
            state_labels=state_labels,
        )
        monthly_summary = _build_sweep_monthly_summary(merged_valid, skip_mask)
        filtered_winrate = filtered_metrics["winrate"]
        improvement = None
        if filtered_winrate is not None and baseline_metrics["winrate"] is not None:
            improvement = round(filtered_winrate - baseline_metrics["winrate"], 2)

        combo_rows.append(
            (
                sweep_id,
                super_run_id,
                idx,
                good_thr,
                bad_thr,
                filter_thr,
                baseline_metrics["winrate"],
                baseline_metrics["trades"],
                baseline_metrics["correct"],
                filtered_metrics["winrate"],
                filtered_metrics["trades"],
                filtered_metrics["correct"],
                filtered_metrics.get("skipped"),
                improvement,
                json.dumps(state_labels),
                json.dumps(monthly_summary),
            )
        )
        combo_payload.append({
            "combo_index": idx,
            "good_threshold": good_thr,
            "bad_threshold": bad_thr,
            "filter_threshold": filter_thr,
            "baseline_winrate": baseline_metrics["winrate"],
            "baseline_trades": baseline_metrics["trades"],
            "filtered_winrate": filtered_winrate,
            "filtered_trades": filtered_metrics["trades"],
            "trades_skipped": filtered_metrics.get("skipped"),
            "improvement": improvement,
            "state_labels": state_labels,
            "monthly": monthly_summary,
        })

    if combo_rows:
        await db.executemany(
            """INSERT INTO super_backtest_hmm_sweep_results
                   (sweep_id, super_run_id, combo_index,
                    good_threshold, bad_threshold, filter_threshold,
                    baseline_winrate, baseline_trades, baseline_correct,
                    filtered_winrate, filtered_trades, filtered_correct,
                    trades_skipped, improvement, state_labels_json, monthly_json)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            combo_rows,
        )

    elapsed = time.time() - t0
    return {
        "id": sweep_id,
        "super_run_id": super_run_id,
        "name": name or f"sweep-{n_states}st-{fit_mode}-{len(clean_features)}f",
        "n_states": n_states,
        "features": clean_features,
        "fit_mode": fit_mode,
        "walk_train_len": walk_train_len,
        "walk_step": walk_step,
        "min_regime_len": min_regime_len,
        "candles_analyzed": candles_analyzed,
        "combos_total": len(combos),
        "baseline": {
            "winrate": baseline_winrate,
            "trades": baseline_trades,
            "correct": baseline_correct,
        },
        "states": state_stats,
        "transition_matrix": transition_matrix,
        "model": model_summary,
        "combos": combo_payload,
        "threshold_inputs": {
            "good_thresholds": good_vals,
            "bad_thresholds": bad_vals,
            "filter_thresholds": filter_vals,
        },
        "time_sec": round(elapsed, 2),
    }


async def list_hmm_sweeps(super_run_id: int) -> list[dict]:
    rows = await db.fetchall(
        """SELECT id, name, n_states, feature_set, fit_mode, walk_train_len,
                      walk_step, min_regime_len, combos_total, baseline_trades,
                      baseline_correct, baseline_winrate, created_at
               FROM super_backtest_hmm_sweeps
               WHERE super_run_id = %s
               ORDER BY id DESC""",
        (super_run_id,),
    )
    sweeps: list[dict] = []
    for row in rows:
        sweep_id = row[0]
        best = await db.fetchone(
            """SELECT id, good_threshold, bad_threshold, filter_threshold,
                         filtered_winrate, filtered_trades, improvement, monthly_json
                   FROM super_backtest_hmm_sweep_results
                   WHERE sweep_id = %s
                   ORDER BY (improvement IS NULL), improvement DESC, filtered_winrate DESC
                   LIMIT 1""",
            (sweep_id,),
        )
        sweeps.append({
            "id": sweep_id,
            "name": row[1],
            "n_states": row[2],
            "features": json.loads(row[3]) if row[3] else [],
            "fit_mode": row[4],
            "walk_train_len": row[5],
            "walk_step": row[6],
            "min_regime_len": row[7],
            "combos_total": row[8],
            "baseline_trades": row[9],
            "baseline_correct": row[10],
            "baseline_winrate": row[11],
            "created_at": str(row[12]) if row[12] else None,
            "best_result": None if not best else {
                "id": best[0],
                "good_threshold": best[1],
                "bad_threshold": best[2],
                "filter_threshold": best[3],
                "filtered_winrate": best[4],
                "filtered_trades": best[5],
                "improvement": best[6],
            },
        })
    return sweeps


async def get_hmm_sweep(sweep_id: int) -> Optional[dict]:
    row = await db.fetchone(
        """SELECT id, super_run_id, name, n_states, feature_set, fit_mode,
                      walk_train_len, walk_step, min_regime_len, combos_total,
                      baseline_trades, baseline_correct, baseline_winrate,
                      candles_analyzed, states_json, transition_matrix_json,
                      model_json, created_at
               FROM super_backtest_hmm_sweeps
               WHERE id = %s""",
        (sweep_id,),
    )
    if not row:
        return None

    combos = await db.fetchall(
        """SELECT id, combo_index, good_threshold, bad_threshold, filter_threshold,
                      baseline_winrate, baseline_trades, filtered_winrate,
                      filtered_trades, trades_skipped, improvement, state_labels_json, monthly_json
               FROM super_backtest_hmm_sweep_results
               WHERE sweep_id = %s
               ORDER BY (improvement IS NULL), improvement DESC, filtered_winrate DESC, combo_index ASC""",
        (sweep_id,),
    )

    combo_dicts = []
    for c in combos:
        combo_dicts.append({
            "id": c[0],
            "combo_index": c[1],
            "good_threshold": c[2],
            "bad_threshold": c[3],
            "filter_threshold": c[4],
            "baseline_winrate": c[5],
            "baseline_trades": c[6],
            "filtered_winrate": c[7],
            "filtered_trades": c[8],
            "trades_skipped": c[9],
            "improvement": c[10],
            "state_labels": json.loads(c[11]) if c[11] else {},
            "monthly": json.loads(c[12]) if c[12] else {},
        })

    return {
        "id": row[0],
        "super_run_id": row[1],
        "name": row[2],
        "n_states": row[3],
        "features": json.loads(row[4]) if row[4] else [],
        "fit_mode": row[5],
        "walk_train_len": row[6],
        "walk_step": row[7],
        "min_regime_len": row[8],
        "combos_total": row[9],
        "baseline_trades": row[10],
        "baseline_correct": row[11],
        "baseline_winrate": row[12],
        "candles_analyzed": row[13],
        "states": json.loads(row[14]) if row[14] else [],
        "transition_matrix": json.loads(row[15]) if row[15] else None,
        "model": json.loads(row[16]) if row[16] else None,
        "created_at": str(row[17]) if row[17] else None,
        "results": combo_dicts,
    }


async def delete_hmm_sweep(sweep_id: int) -> dict:
    await db.execute(
        "DELETE FROM super_backtest_hmm_sweep_results WHERE sweep_id = %s",
        (sweep_id,),
    )
    await db.execute(
        "DELETE FROM super_backtest_hmm_sweeps WHERE id = %s",
        (sweep_id,),
    )
    return {"deleted": sweep_id}


async def list_hmm_sweep_results(
    super_run_id: Optional[int] = None,
    min_total_signals: Optional[int] = None,
    min_taken_signals: Optional[int] = None,
    min_winrate: Optional[float] = None,
    limit: int = 200,
) -> list[dict]:
    limit = max(1, min(limit, 500))
    where_clauses = []
    params: list[Any] = []

    if super_run_id is not None:
        where_clauses.append("res.super_run_id = %s")
        params.append(super_run_id)
    if min_total_signals is not None:
        where_clauses.append("res.baseline_trades >= %s")
        params.append(min_total_signals)
    if min_taken_signals is not None:
        where_clauses.append("res.filtered_trades >= %s")
        params.append(min_taken_signals)
    if min_winrate is not None:
        where_clauses.append("res.filtered_winrate >= %s")
        params.append(min_winrate)

    where_sql = ""
    if where_clauses:
        where_sql = "WHERE " + " AND ".join(where_clauses)

    rows = await db.fetchall(
        f"""SELECT res.id, res.sweep_id, res.super_run_id, res.combo_index,
                       res.good_threshold, res.bad_threshold, res.filter_threshold,
                       res.baseline_winrate, res.baseline_trades,
                       res.filtered_winrate, res.filtered_trades,
                       res.trades_skipped, res.improvement, res.monthly_json,
                       sweeps.name, sweeps.n_states, sweeps.feature_set, sweeps.fit_mode,
                       sweeps.created_at,
                       runs.strategy, runs.horizon, runs.train_start, runs.test_start, runs.test_end
                FROM super_backtest_hmm_sweep_results res
                JOIN super_backtest_hmm_sweeps sweeps ON sweeps.id = res.sweep_id
                JOIN super_backtest_runs runs ON runs.id = res.super_run_id
                {where_sql}
                ORDER BY (res.improvement IS NULL), res.improvement DESC,
                         res.filtered_winrate DESC, res.filtered_trades DESC
                LIMIT %s""",
        (*params, limit),
    )

    results = []
    for r in rows:
        results.append({
            "id": r[0],
            "sweep_id": r[1],
            "super_run_id": r[2],
            "combo_index": r[3],
            "good_threshold": r[4],
            "bad_threshold": r[5],
            "filter_threshold": r[6],
            "baseline_winrate": r[7],
            "baseline_trades": r[8],
            "filtered_winrate": r[9],
            "filtered_trades": r[10],
            "trades_skipped": r[11],
            "improvement": r[12],
            "monthly": json.loads(r[13]) if r[13] else {},
            "sweep_name": r[14],
            "n_states": r[15],
            "features": json.loads(r[16]) if r[16] else [],
            "fit_mode": r[17],
            "sweep_created_at": str(r[18]) if r[18] else None,
            "strategy": r[19],
            "horizon": r[20],
            "train_start": r[21],
            "test_start": r[22],
            "test_end": r[23],
        })
    return results


async def get_hmm_timeline(analysis_id: int, max_points: int = 3000) -> dict:
    """
    Returns candles + state assignments + signals for plotting.
    Down-samples to `max_points` for chart performance.
    """
    analysis = await get_hmm_analysis(analysis_id)
    if not analysis:
        return {"error": "Analysis not found"}

    super_run_id = analysis["super_run_id"]

    # Load state labels
    state_rows = await db.fetchall(
        """SELECT candle_idx, open_time, hmm_state, hmm_state_prob, is_skipped
           FROM super_backtest_prediction_states
           WHERE hmm_analysis_id = %s
           ORDER BY candle_idx""",
        (analysis_id,),
    )
    if not state_rows:
        return {"error": "No state data for this analysis"}

    # Load candle closes and signals via a single join
    pred_rows = await db.fetchall(
        """SELECT p.candle_idx, p.open_time, p.is_signal, p.is_correct, p.prediction,
                  p.probability, p.actual, c.close
           FROM super_backtest_predictions p
           JOIN c_5m c ON c.open_time = p.open_time
           WHERE p.super_run_id = %s
           ORDER BY p.candle_idx""",
        (super_run_id,),
    )

    pred_map = {r[0]: r for r in pred_rows}
    state_map = {r[0]: r for r in state_rows}

    all_idxs = sorted(set(pred_map.keys()) & set(state_map.keys()))
    total = len(all_idxs)
    if total == 0:
        return {"error": "No overlap between predictions and state labels"}

    # Down-sample candles (keep every Nth) but always keep signal candles
    step = max(1, total // max_points)
    sampled_idxs = []
    for pos, idx in enumerate(all_idxs):
        p = pred_map[idx]
        is_signal = bool(p[2])
        if is_signal or (pos % step == 0):
            sampled_idxs.append(idx)
    # Cap hard limit in case too many signals
    if len(sampled_idxs) > max_points * 3:
        sampled_idxs = sampled_idxs[:: max(1, len(sampled_idxs) // max_points)]

    # Monthly summary uses full dataset (no downsampling)
    monthly = _build_monthly_summary(pred_map, state_map)

    candles = []
    signals = []
    for idx in sampled_idxs:
        p = pred_map[idx]
        s = state_map[idx]
        open_time_us = int(p[1])
        open_time_ms = open_time_us // 1000
        candles.append({
            "t": open_time_ms,
            "close": float(p[7]),
            "state": int(s[2]),
            "state_prob": float(s[3]),
        })
        if p[2]:  # is_signal
            skipped = bool(s[4])
            signals.append({
                "t": open_time_ms,
                "close": float(p[7]),
                "prediction": int(p[4]),
                "actual": int(p[6]),
                "correct": bool(p[3]),
                "state": int(s[2]),
                "skipped": skipped,
            })

    # State label lookup (from state_stats)
    state_labels = {s["state"]: s.get("label", "neutral") for s in analysis["states"]}

    return {
        "analysis_id": analysis_id,
        "super_run_id": super_run_id,
        "n_states": analysis["n_states"],
        "state_labels": state_labels,
        "candles": candles,
        "signals": signals,
        "monthly": monthly,
        "total_candles": total,
        "sampled": len(candles),
    }


# ======================================================================
# Internal helpers
# ======================================================================

async def _prepare_feature_df(
    table: str,
    train_start: str,
    test_end: str,
    horizon: int,
    needed_features: list[str],
) -> pd.DataFrame:
    df_raw = await load_candles(table, train_start, test_end)
    df_raw = add_direction(df_raw)
    df_raw = add_future_directions(df_raw, [horizon])
    df_raw = df_raw.reset_index(drop=True)

    # Ensure the feature blocks for requested features are enabled
    saved = dict(FEATURE_USAGE)
    need_blocks = _feature_blocks_for(needed_features)
    for k in FEATURE_USAGE:
        FEATURE_USAGE[k] = k in need_blocks or saved.get(k, False)

    try:
        df_feat = add_technical_features(df_raw)
    finally:
        for k, v in saved.items():
            FEATURE_USAGE[k] = v
    return df_feat


async def _fit_hmm_core(
    super_run_id: int,
    clean_features: list[str],
    n_states: int,
    fit_mode: str,
    walk_train_len: Optional[int],
    walk_step: Optional[int],
    min_regime_len: int,
) -> dict:
    """Shared pipeline: load data, fit HMM, compute base stats."""
    run = await db.fetchone(
        """SELECT train_start, train_end, test_start, test_end, table_name, horizon
           FROM super_backtest_runs WHERE id = %s""",
        (super_run_id,),
    )
    if not run:
        return {"error": f"super_run_id {super_run_id} not found"}
    train_start, train_end, test_start, test_end, table_name, horizon = run

    df_feat = await _prepare_feature_df(
        table=table_name,
        train_start=train_start,
        test_end=test_end,
        horizon=int(horizon),
        needed_features=clean_features,
    )

    test_start_us = date_to_us(test_start)
    test_end_us = date_to_us(test_end, True)
    mask = (df_feat["open_time"] >= test_start_us) & (df_feat["open_time"] <= test_end_us)
    df_test = df_feat[mask].reset_index(drop=True)

    pred_rows = await db.fetchall(
        """SELECT open_time, is_signal, is_correct, prediction, probability, actual, candle_idx
           FROM super_backtest_predictions WHERE super_run_id = %s
           ORDER BY candle_idx""",
        (super_run_id,),
    )
    if not pred_rows:
        return {"error": "No predictions stored for this run"}

    pred_df = pd.DataFrame(pred_rows, columns=[
        "open_time", "is_signal", "is_correct", "prediction", "probability", "actual", "candle_idx",
    ])
    merged = df_test.merge(pred_df, on="open_time", how="inner")
    if merged.empty:
        return {"error": "No overlap between predictions and candles"}

    X_all = merged[clean_features].astype(float)
    valid_mask = X_all.notna().all(axis=1).values
    if valid_mask.sum() < 100:
        return {"error": f"Only {int(valid_mask.sum())} valid rows — need at least 100"}

    X_raw = X_all[valid_mask].values
    mu = X_raw.mean(axis=0)
    sigma = X_raw.std(axis=0)
    sigma[sigma == 0] = 1.0
    X_std = (X_raw - mu) / sigma

    is_signal_mask = merged.loc[valid_mask, "is_signal"].values.astype(bool)

    try:
        if fit_mode == "all_candles":
            states, state_probs = _fit_predict_all(X_std, n_states)
        elif fit_mode == "signals_only":
            fit_X = X_std[is_signal_mask]
            if len(fit_X) < 50:
                return {"error": f"Too few signal rows to fit: {len(fit_X)}"}
            states, state_probs, _ = _fit_on_subset_predict_all(fit_X, X_std, n_states)
        elif fit_mode == "walk_forward":
            if not walk_train_len or walk_train_len < 200:
                walk_train_len = max(500, len(X_std) // 5)
            if not walk_step:
                walk_step = max(100, walk_train_len // 5)
            states, state_probs = _fit_predict_walk_forward(X_std, n_states, walk_train_len, walk_step)
        else:
            return {"error": f"Unknown fit_mode {fit_mode}"}
    except Exception as e:
        return {"error": f"HMM fit failed: {e}"}

    if min_regime_len > 1:
        states = _smooth_states(states, min_regime_len)

    merged_valid = merged[valid_mask].reset_index(drop=True)
    state_stats = _compute_state_stats(
        states=states,
        state_probs=state_probs,
        merged=merged_valid,
        features=clean_features,
        X_raw=X_raw,
    )

    model_summary = _model_summary(states, state_probs, n_states)
    transition_matrix = _compute_transition_matrix(states, n_states)

    return {
        "merged_valid": merged_valid,
        "states": states,
        "state_probs": state_probs,
        "state_stats": state_stats,
        "model_summary": model_summary,
        "transition_matrix": transition_matrix,
        "walk_train_len": walk_train_len,
        "walk_step": walk_step,
        "candles_analyzed": int(valid_mask.sum()),
    }


def _feature_blocks_for(features: list[str]) -> set[str]:
    """Map requested feature columns to FEATURE_USAGE blocks."""
    mapping = {
        "rsi": {"rsi_14", "rsi_6"},
        "bollinger": {"bb_pos", "bb_width"},
        "volatility": {"volatility_5", "volatility_20", "atr_14"},
        "macd": {"macd", "macd_hist"},
        "moving_averages": set(),  # required for ema_diff
        "ema_diff": {"ema_diff_5", "ema_diff_20", "ema_diff_50"},
        "returns": {"returns_1", "returns_3", "returns_5"},
        "volume": {"volume_ratio"},
        "volume_zscore": {"volume_zscore"},
        "stochastic": {"stoch_k"},
        "adx_dmi": {"adx"},
        "time_features": {"hour", "dow"},
        "lag_deltas": {"delta_rsi_14", "delta_bb_pos", "delta_close"},
        "higher_moments": {"rolling_skew_20", "rolling_kurtosis_20"},
        "orderflow": {"orderflow_proxy"},
    }
    wanted = set(features)
    blocks: set[str] = set()
    for blk, cols in mapping.items():
        if cols & wanted:
            blocks.add(blk)
    # ema_diff depends on moving_averages
    if "ema_diff" in blocks:
        blocks.add("moving_averages")
    return blocks


def _fit_predict_all(X: np.ndarray, n_states: int) -> tuple[np.ndarray, np.ndarray]:
    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=100,
        random_state=42,
        tol=0.01,
    )
    model.fit(X)
    states = model.predict(X)
    probs = model.predict_proba(X)
    return states, probs


def _fit_on_subset_predict_all(
    fit_X: np.ndarray,
    all_X: np.ndarray,
    n_states: int,
) -> tuple[np.ndarray, np.ndarray, Any]:
    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=100,
        random_state=42,
        tol=0.01,
    )
    model.fit(fit_X)
    states = model.predict(all_X)
    probs = model.predict_proba(all_X)
    return states, probs, model


def _fit_predict_walk_forward(
    X: np.ndarray,
    n_states: int,
    train_len: int,
    step: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Walk-forward: fit on [i-train_len, i), predict [i, i+step).
    Align state IDs across refits by matching means to previous fit.
    """
    n = len(X)
    states = np.full(n, -1, dtype=int)
    probs = np.zeros((n, n_states), dtype=float)

    # Warm-up: first train_len rows use the first model's prediction on itself
    if train_len > n:
        return _fit_predict_all(X, n_states)

    prev_means = None
    mapping = None  # current_state -> global_state

    i = train_len
    while i < n:
        j = min(i + step, n)
        train_slice = X[max(0, i - train_len):i]
        print(
            f"[HMM walk] fitting window {max(0, i - train_len)}:{i} -> predicting {i}:{j} "
            f"({j}/{n} candles)",
            flush=True,
        )
        try:
            model = hmm.GaussianHMM(
                n_components=n_states,
                covariance_type="diag",
                n_iter=50,
                random_state=42,
                tol=0.01,
            )
            model.fit(train_slice)
        except Exception:
            # Failed fit → reuse previous mapping as zeros
            states[i:j] = 0
            probs[i:j, 0] = 1.0
            i = j
            continue

        # Align new state IDs to global reference
        if prev_means is None:
            mapping = {s: s for s in range(n_states)}
            prev_means = model.means_.copy()
        else:
            mapping = _align_states(model.means_, prev_means)
            # Smoothly update prev_means as exponential moving average
            new_means = model.means_.copy()
            for cur, glob in mapping.items():
                prev_means[glob] = 0.5 * prev_means[glob] + 0.5 * new_means[cur]

        # Also backfill warm-up window with first model's predictions
        if i == train_len:
            warm_states = model.predict(X[:i])
            warm_probs = model.predict_proba(X[:i])
            for k in range(i):
                cur_s = int(warm_states[k])
                glob_s = mapping[cur_s]
                states[k] = glob_s
                probs[k, :] = _remap_prob_row(warm_probs[k], mapping, n_states)

        # Predict next chunk
        chunk_states = model.predict(X[i:j])
        chunk_probs = model.predict_proba(X[i:j])
        for k in range(j - i):
            cur_s = int(chunk_states[k])
            glob_s = mapping[cur_s]
            states[i + k] = glob_s
            probs[i + k, :] = _remap_prob_row(chunk_probs[k], mapping, n_states)

        remaining = n - j
        print(
            f"[HMM walk] processed {j}/{n} candles | remaining {remaining}",
            flush=True,
        )
        i = j

    return states, probs


def _align_states(new_means: np.ndarray, prev_means: np.ndarray) -> dict[int, int]:
    """Greedy assignment: for each new state, pick closest previous state (unused)."""
    n = len(new_means)
    mapping: dict[int, int] = {}
    used: set[int] = set()
    # Compute distance matrix
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dists[i, j] = np.linalg.norm(new_means[i] - prev_means[j])
    # Greedy nearest-first
    order = sorted(
        [(i, j, dists[i, j]) for i in range(n) for j in range(n)],
        key=lambda x: x[2],
    )
    for i, j, _ in order:
        if i in mapping or j in used:
            continue
        mapping[i] = j
        used.add(j)
    # Fill any unmapped
    for i in range(n):
        if i not in mapping:
            remaining = [j for j in range(n) if j not in used]
            mapping[i] = remaining[0] if remaining else i
            used.add(mapping[i])
    return mapping


def _remap_prob_row(row: np.ndarray, mapping: dict[int, int], n_states: int) -> np.ndarray:
    out = np.zeros(n_states, dtype=float)
    for cur, glob in mapping.items():
        out[glob] = row[cur]
    return out


def _smooth_states(states: np.ndarray, min_len: int) -> np.ndarray:
    """Merge contiguous state segments shorter than min_len into the preceding segment."""
    if len(states) == 0 or min_len <= 1:
        return states
    out = states.copy()
    n = len(out)
    i = 0
    while i < n:
        j = i
        while j < n and out[j] == out[i]:
            j += 1
        seg_len = j - i
        if seg_len < min_len and i > 0:
            out[i:j] = out[i - 1]
        i = j
    return out


def _compute_state_stats(
    states: np.ndarray,
    state_probs: np.ndarray,
    merged: pd.DataFrame,
    features: list[str],
    X_raw: np.ndarray,
) -> list[dict]:
    """Per-state accuracy on signal rows + feature means (raw scale)."""
    n_states = state_probs.shape[1]
    is_signal = merged["is_signal"].astype(bool).values
    is_correct = merged["is_correct"].astype(bool).values

    stats = []
    for s in range(n_states):
        mask = states == s
        total_candles = int(mask.sum())
        signal_mask = mask & is_signal
        sig_count = int(signal_mask.sum())
        correct_count = int((signal_mask & is_correct).sum())
        acc = (correct_count / sig_count * 100) if sig_count > 0 else 0.0
        # Feature means on raw scale
        means = X_raw[mask].mean(axis=0) if total_candles else np.zeros(len(features))
        feature_means = {f: float(round(means[i], 4)) for i, f in enumerate(features)}

        stats.append({
            "state": int(s),
            "candles": total_candles,
            "signals": sig_count,
            "correct": correct_count,
            "wrong": sig_count - correct_count,
            "accuracy_pct": round(acc, 2),
            "signal_rate_pct": round(sig_count / total_candles * 100, 2) if total_candles else 0.0,
            "feature_means": feature_means,
        })

    return stats


def _assign_state_labels(
    state_stats: list[dict],
    good_threshold: float,
    bad_threshold: float,
) -> dict[int, str]:
    """Derive good/bad/neutral labels per state based on accuracy thresholds."""
    labels: dict[int, str] = {}
    MIN_SIG = 10

    for stat in state_stats:
        state_id = int(stat["state"])
        sigs = stat.get("signals", 0)
        acc = stat.get("accuracy_pct", 0.0)
        if sigs >= MIN_SIG and acc >= good_threshold:
            labels[state_id] = "good"
        elif sigs >= MIN_SIG and acc <= bad_threshold:
            labels[state_id] = "bad"
        else:
            labels[state_id] = "neutral"

    # Fallbacks: ensure at least one good/bad when possible
    has_good = any(lbl == "good" for lbl in labels.values())
    has_bad = any(lbl == "bad" for lbl in labels.values())

    stats_with_signals = [s for s in state_stats if s.get("signals", 0) >= MIN_SIG]

    if not has_good and stats_with_signals:
        best = max(stats_with_signals, key=lambda x: x.get("accuracy_pct", 0))
        labels[best["state"]] = "good"

    if not has_bad and len(stats_with_signals) >= 2:
        worst = min(stats_with_signals, key=lambda x: x.get("accuracy_pct", 999))
        if labels.get(worst["state"]) != "good":
            labels[worst["state"]] = "bad"

    return labels


def _compute_filter_metrics(
    states: np.ndarray,
    state_probs: np.ndarray,
    merged: pd.DataFrame,
    state_stats: list[dict],
    filter_threshold: float,
    state_labels: Optional[dict[int, str]] = None,
) -> tuple[dict, dict, np.ndarray]:
    """Compare baseline signal performance vs regime-filtered."""
    is_signal = merged["is_signal"].astype(bool).values
    is_correct = merged["is_correct"].astype(bool).values

    if state_labels is not None:
        bad_state_ids = [state for state, label in state_labels.items() if label == "bad"]
    else:
        bad_state_ids = [s["state"] for s in state_stats if s.get("label") == "bad"]

    total_signals = int(is_signal.sum())
    total_correct = int((is_signal & is_correct).sum())
    baseline_winrate = round(total_correct / total_signals * 100, 2) if total_signals else 0.0

    # Filter: skip signals where P(any bad state) > threshold
    if bad_state_ids:
        bad_prob = state_probs[:, bad_state_ids].sum(axis=1)
        skip_mask = is_signal & (bad_prob > filter_threshold)
        kept_mask = is_signal & ~skip_mask
        taken = int(kept_mask.sum())
        correct = int((kept_mask & is_correct).sum())
        skipped = int(skip_mask.sum())
        filtered_winrate = round(correct / taken * 100, 2) if taken else None
    else:
        skip_mask = np.zeros_like(is_signal, dtype=bool)
        taken = total_signals
        correct = total_correct
        skipped = 0
        filtered_winrate = baseline_winrate

    return (
        {"winrate": baseline_winrate, "trades": total_signals, "correct": total_correct},
        {"winrate": filtered_winrate, "trades": taken, "correct": correct, "skipped": skipped},
        skip_mask,
    )


def _compute_transition_matrix(states: np.ndarray, n_states: int) -> list[list[float]]:
    mat = np.zeros((n_states, n_states), dtype=float)
    for i in range(len(states) - 1):
        a = int(states[i])
        b = int(states[i + 1])
        if 0 <= a < n_states and 0 <= b < n_states:
            mat[a, b] += 1
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    norm = (mat / row_sums).round(4)
    return norm.tolist()


def _model_summary(states: np.ndarray, state_probs: np.ndarray, n_states: int) -> dict:
    counts = [int((states == s).sum()) for s in range(n_states)]
    return {"state_counts": counts, "total": int(len(states))}


def _build_monthly_summary(pred_map: dict, state_map: dict) -> dict:
    monthly: dict[str, dict[str, float]] = {}
    for idx, pred in pred_map.items():
        if not bool(pred[2]):  # is_signal
            continue
        state_row = state_map.get(idx)
        if not state_row:
            continue
        skipped = bool(state_row[4])
        open_time_ms = int(pred[1]) // 1000
        key = datetime.utcfromtimestamp(open_time_ms / 1000).strftime("%Y-%m")
        bucket = monthly.setdefault(key, {
            "total": 0,
            "taken": 0,
            "skipped": 0,
            "baseline_correct": 0,
            "filtered_correct": 0,
        })
        bucket["total"] += 1
        if skipped:
            bucket["skipped"] += 1
        else:
            bucket["taken"] += 1
            if pred[3]:
                bucket["filtered_correct"] += 1
        if pred[3]:
            bucket["baseline_correct"] += 1
    return monthly


def _build_sweep_monthly_summary(merged: pd.DataFrame, skip_mask: np.ndarray) -> dict[str, dict[str, int]]:
    monthly: dict[str, dict[str, int]] = {}
    is_signal = merged["is_signal"].astype(bool).values
    is_correct = merged["is_correct"].astype(bool).values
    open_times = merged["open_time"].astype(int).values
    for idx in range(len(merged)):
        if not is_signal[idx]:
            continue
        key = datetime.utcfromtimestamp((int(open_times[idx]) // 1000) / 1000).strftime("%Y-%m")
        bucket = monthly.setdefault(key, {
            "total": 0,
            "taken": 0,
            "skipped": 0,
            "baseline_correct": 0,
            "filtered_correct": 0,
        })
        bucket["total"] += 1
        if is_correct[idx]:
            bucket["baseline_correct"] += 1
        if bool(skip_mask[idx]):
            bucket["skipped"] += 1
        else:
            bucket["taken"] += 1
            if is_correct[idx]:
                bucket["filtered_correct"] += 1
    return monthly


async def _save_prediction_states(
    analysis_id: int,
    merged: pd.DataFrame,
    states: np.ndarray,
    state_probs: np.ndarray,
    skip_mask: np.ndarray,
) -> None:
    rows = []
    for i, row in merged.iterrows():
        s = int(states[i])
        prob = float(state_probs[i][s]) if state_probs.shape[1] > s else 0.0
        skipped = bool(skip_mask[i]) if i < len(skip_mask) else False
        rows.append((
            int(analysis_id),
            int(row["candle_idx"]),
            int(row["open_time"]),
            s,
            prob,
            int(skipped),
        ))
    if not rows:
        return
    # Batch insert
    BATCH = 500
    for i in range(0, len(rows), BATCH):
        await db.executemany(
            """INSERT INTO super_backtest_prediction_states
               (hmm_analysis_id, candle_idx, open_time, hmm_state, hmm_state_prob, is_skipped)
               VALUES (%s, %s, %s, %s, %s, %s)""",
            rows[i:i + BATCH],
        )
