from __future__ import annotations

import copy
import json
import time
from typing import Any, Optional

from db import DbProvider

from predictor.bruteforce import build_combos, get_default_grid
from predictor.hmm_analysis import create_hmm_sweep, AVAILABLE_FEATURES
from predictor.super_backtest import run_super_backtest


db = DbProvider()


_BASE_RSIB_GRID = copy.deepcopy(get_default_grid("rsi_mean_reversion"))
_BASE_RSIB_GRID["threshold"] = [0.51, 0.53, 0.55, 0.57]


DEFAULT_RSIB_CONFIG = {
    "strategy": "rsi_mean_reversion",
    "train_start": "2022-01-01",
    "train_end": "2025-06-30",
    "test_start": "2025-07-01",
    "test_end": "2026-01-31",
    "horizon": 1,
    "table": "c_5m",
    "window_size": 5000,
    "retrain_every": 500,
    "max_combos": 50,
    "param_grid": _BASE_RSIB_GRID,
    "profit": {
        "start_bank": 1000.0,
        "buy_price_cents": 52.0,
        "max_bet": 500.0,
        "half_kelly_pct": 1.70,
        "full_kelly_pct": 3.34,
        "fee_pct": 1.56,
    },
    "hmm": {
        "n_states": 3,
        "features": ["rsi_14", "bb_pos", "volatility_5", "atr_14", "ema_diff_20"],
        "fit_mode": "all_candles",
        "walk_train_len": None,
        "walk_step": None,
        "min_regime_len": 1,
        "good_thresholds": [53.0, 55.0, 57.0],
        "bad_thresholds": [45.0, 47.0],
        "filter_thresholds": [0.45, 0.55, 0.65],
    },
}


def get_default_rsib_config() -> dict[str, Any]:
    return json.loads(json.dumps(DEFAULT_RSIB_CONFIG))


async def create_rsib_session(
    config: dict[str, Any],
    progress: "TaskProgress | None" = None,
) -> dict[str, Any]:
    strategy = str(config.get("strategy") or "rsi_mean_reversion")
    if strategy != "rsi_mean_reversion":
        return {"error": "RSIB supports only rsi_mean_reversion"}

    param_grid = config.get("param_grid") or {}
    combos = build_combos(param_grid)
    max_combos = max(1, int(config.get("max_combos") or 50))
    combos = combos[:max_combos]
    if not combos:
        return {"error": "No parameter combinations to run"}

    hmm_cfg = config.get("hmm") or {}
    features = [f for f in (hmm_cfg.get("features") or []) if f in AVAILABLE_FEATURES]
    if len(features) < 2:
        return {"error": "Select at least 2 valid HMM features"}

    profit_cfg = config.get("profit") or {}
    session_id = await db.execute(
        """INSERT INTO rsib_sessions
           (config_json, param_grid_json, hmm_json, profit_json, total_combos, completed, status)
           VALUES (%s, %s, %s, %s, %s, %s, %s)""",
        (
            json.dumps(config, ensure_ascii=False),
            json.dumps(param_grid, ensure_ascii=False),
            json.dumps(hmm_cfg, ensure_ascii=False),
            json.dumps(profit_cfg, ensure_ascii=False),
            len(combos),
            0,
            "running",
        ),
    )

    started = time.time()
    best_profit = None
    best_final_bank = None
    total_variants = 0

    for idx, params in enumerate(combos, start=1):
        if progress:
            progress.update(idx - 1, len(combos), f"RSIB combo {idx}/{len(combos)}")

        result = await run_super_backtest(
            strategy_name="rsi_mean_reversion",
            strategy_params=params,
            train_start=str(config.get("train_start") or ""),
            train_end=str(config.get("train_end") or ""),
            test_start=str(config.get("test_start") or ""),
            test_end=str(config.get("test_end") or ""),
            horizon=int(config.get("horizon") or 1),
            table=str(config.get("table") or "c_5m"),
            window_size=int(config.get("window_size") or params.get("window_size") or 5000),
            retrain_every=int(config.get("retrain_every") or 500),
            hmm_states=int(hmm_cfg.get("n_states") or 3),
        )
        if result.get("error"):
            await _save_rsib_result(
                session_id=session_id,
                super_run_id=None,
                combo_index=idx,
                variant_type="error",
                variant_label="error",
                strategy="rsi_mean_reversion",
                strategy_params=params,
                monthly=[],
                metrics={"error": result.get("error")},
                bankroll=profit_cfg,
                sweep_id=None,
                sweep_combo_index=None,
            )
            await _update_rsib_session_progress(session_id, idx, started)
            continue

        super_run_id = int(result["super_run_id"])
        baseline_monthly = await _load_super_run_monthly(super_run_id)
        baseline_summary = await _load_super_run_summary(super_run_id)
        base_sim_half = _simulate_monthly_bankroll(
            baseline_monthly,
            float(profit_cfg.get("start_bank") or 1000),
            float(profit_cfg.get("buy_price_cents") or 52),
            float(profit_cfg.get("max_bet") or 500),
            float(profit_cfg.get("half_kelly_pct") or 1.70) / 100.0,
            float(profit_cfg.get("fee_pct") or 1.56) / 100.0,
        )
        base_sim_full = _simulate_monthly_bankroll(
            baseline_monthly,
            float(profit_cfg.get("start_bank") or 1000),
            float(profit_cfg.get("buy_price_cents") or 52),
            float(profit_cfg.get("max_bet") or 500),
            float(profit_cfg.get("full_kelly_pct") or 3.34) / 100.0,
            float(profit_cfg.get("fee_pct") or 1.56) / 100.0,
        )
        base_row_id = await _save_rsib_result(
            session_id=session_id,
            super_run_id=super_run_id,
            combo_index=idx,
            variant_type="baseline",
            variant_label="Baseline",
            strategy="rsi_mean_reversion",
            strategy_params=params,
            monthly=baseline_monthly,
            metrics={
                "accuracy_pct": baseline_summary.get("accuracy_pct", 0),
                "signals": baseline_summary.get("signals", 0),
                "correct": baseline_summary.get("correct", 0),
                "wrong": baseline_summary.get("wrong", 0),
                "trades_taken": baseline_summary.get("signals", 0),
                "trades_skipped": 0,
                "horizon": config.get("horizon"),
            },
            bankroll=profit_cfg,
            sweep_id=None,
            sweep_combo_index=None,
            sim_half=base_sim_half,
            sim_full=base_sim_full,
        )
        total_variants += 1
        best_profit, best_final_bank = _update_best(best_profit, best_final_bank, base_sim_full)

        sweep = await create_hmm_sweep(
            super_run_id=super_run_id,
            name=f"RSIB #{session_id} combo {idx}",
            n_states=int(hmm_cfg.get("n_states") or 3),
            features=features,
            fit_mode=str(hmm_cfg.get("fit_mode") or "all_candles"),
            walk_train_len=_nullable_int(hmm_cfg.get("walk_train_len")),
            walk_step=_nullable_int(hmm_cfg.get("walk_step")),
            min_regime_len=int(hmm_cfg.get("min_regime_len") or 1),
            good_thresholds=list(hmm_cfg.get("good_thresholds") or [55.0]),
            bad_thresholds=list(hmm_cfg.get("bad_thresholds") or [45.0]),
            filter_thresholds=list(hmm_cfg.get("filter_thresholds") or [0.55]),
        )
        if not sweep.get("error"):
            sweep_id = int(sweep["id"])
            for sweep_result in sweep.get("combos") or []:
                sweep_monthly = _normalize_sweep_monthly(sweep_result.get("monthly") or {})
                sweep_sim_half = _simulate_monthly_bankroll(
                    sweep_monthly,
                    float(profit_cfg.get("start_bank") or 1000),
                    float(profit_cfg.get("buy_price_cents") or 52),
                    float(profit_cfg.get("max_bet") or 500),
                    float(profit_cfg.get("half_kelly_pct") or 1.70) / 100.0,
                    float(profit_cfg.get("fee_pct") or 1.56) / 100.0,
                )
                sweep_sim_full = _simulate_monthly_bankroll(
                    sweep_monthly,
                    float(profit_cfg.get("start_bank") or 1000),
                    float(profit_cfg.get("buy_price_cents") or 52),
                    float(profit_cfg.get("max_bet") or 500),
                    float(profit_cfg.get("full_kelly_pct") or 3.34) / 100.0,
                    float(profit_cfg.get("fee_pct") or 1.56) / 100.0,
                )
                await _save_rsib_result(
                    session_id=session_id,
                    super_run_id=super_run_id,
                    combo_index=idx,
                    variant_type="hmm_sweep",
                    variant_label=_format_sweep_label(sweep_result),
                    strategy="rsi_mean_reversion",
                    strategy_params=params,
                    monthly=sweep_monthly,
                    metrics={
                        "accuracy_pct": sweep_result.get("filtered_winrate"),
                        "signals": sweep_result.get("filtered_trades"),
                        "correct": None,
                        "wrong": None,
                        "trades_taken": sweep_result.get("filtered_trades"),
                        "trades_skipped": sweep_result.get("trades_skipped"),
                        "baseline_winrate": sweep_result.get("baseline_winrate"),
                        "improvement": sweep_result.get("improvement"),
                        "good_threshold": sweep_result.get("good_threshold"),
                        "bad_threshold": sweep_result.get("bad_threshold"),
                        "filter_threshold": sweep_result.get("filter_threshold"),
                        "horizon": config.get("horizon"),
                    },
                    bankroll=profit_cfg,
                    sweep_id=sweep_id,
                    sweep_combo_index=sweep_result.get("combo_index"),
                    sim_half=sweep_sim_half,
                    sim_full=sweep_sim_full,
                )
                total_variants += 1
                best_profit, best_final_bank = _update_best(best_profit, best_final_bank, sweep_sim_full)

        await _update_rsib_session_progress(session_id, idx, started)

    elapsed = round(time.time() - started, 2)
    await db.execute(
        """UPDATE rsib_sessions
           SET completed = %s, total_variants = %s, status = %s,
               best_profit = %s, best_final_bank = %s, total_time_sec = %s
           WHERE id = %s""",
        (len(combos), total_variants, "done", best_profit or 0, best_final_bank or 0, elapsed, session_id),
    )
    if progress:
        progress.update(len(combos), len(combos), "Done")
    return {"session_id": session_id, "total_combos": len(combos), "total_variants": total_variants, "time_sec": elapsed}


async def list_rsib_sessions(limit: int = 50) -> list[dict[str, Any]]:
    rows = await db.fetchall(
        """SELECT id, total_combos, completed, total_variants, status,
                  best_profit, best_final_bank, total_time_sec, created_at,
                  config_json, hmm_json, profit_json
           FROM rsib_sessions
           ORDER BY id DESC
           LIMIT %s""",
        (limit,),
    )
    results = []
    for row in rows:
        cfg = _safe_json_loads(row[9], {})
        results.append({
            "id": row[0],
            "total_combos": row[1],
            "completed": row[2],
            "total_variants": row[3],
            "status": row[4],
            "best_profit": row[5],
            "best_final_bank": row[6],
            "total_time_sec": row[7],
            "created_at": str(row[8]) if row[8] else None,
            "config": cfg,
            "hmm": _safe_json_loads(row[10], {}),
            "profit": _safe_json_loads(row[11], {}),
        })
    return results


async def get_rsib_session(session_id: int) -> Optional[dict[str, Any]]:
    row = await db.fetchone(
        """SELECT id, config_json, param_grid_json, hmm_json, profit_json,
                  total_combos, completed, total_variants, status,
                  best_profit, best_final_bank, total_time_sec, created_at
           FROM rsib_sessions WHERE id = %s""",
        (session_id,),
    )
    if not row:
        return None
    return {
        "id": row[0],
        "config": _safe_json_loads(row[1], {}),
        "param_grid": _safe_json_loads(row[2], {}),
        "hmm": _safe_json_loads(row[3], {}),
        "profit": _safe_json_loads(row[4], {}),
        "total_combos": row[5],
        "completed": row[6],
        "total_variants": row[7],
        "status": row[8],
        "best_profit": row[9],
        "best_final_bank": row[10],
        "total_time_sec": row[11],
        "created_at": str(row[12]) if row[12] else None,
    }


async def list_rsib_results(session_id: int, page: int = 1, page_size: int = 25) -> dict[str, Any]:
    page = max(1, int(page or 1))
    page_size = max(1, min(100, int(page_size or 25)))
    offset = (page - 1) * page_size
    count_row = await db.fetchone("SELECT COUNT(*) FROM rsib_results WHERE session_id = %s", (session_id,))
    total = int(count_row[0]) if count_row else 0
    rows = await db.fetchall(
        """SELECT id, session_id, super_run_id, combo_index, variant_type, variant_label,
                  strategy, strategy_params_json, sweep_id, sweep_combo_index,
                  accuracy_pct, signals, trades_taken, trades_skipped,
                  final_bank_half, profit_half, roi_half,
                  final_bank_full, profit_full, roi_full,
                  created_at
           FROM rsib_results
           WHERE session_id = %s
           ORDER BY profit_full DESC, final_bank_full DESC, id ASC
           LIMIT %s OFFSET %s""",
        (session_id, page_size, offset),
    )
    results = []
    for row in rows:
        results.append({
            "id": row[0],
            "session_id": row[1],
            "super_run_id": row[2],
            "combo_index": row[3],
            "variant_type": row[4],
            "variant_label": row[5],
            "strategy": row[6],
            "strategy_params": _safe_json_loads(row[7], {}),
            "sweep_id": row[8],
            "sweep_combo_index": row[9],
            "accuracy_pct": row[10],
            "signals": row[11],
            "trades_taken": row[12],
            "trades_skipped": row[13],
            "final_bank_half": row[14],
            "profit_half": row[15],
            "roi_half": row[16],
            "final_bank_full": row[17],
            "profit_full": row[18],
            "roi_full": row[19],
            "created_at": str(row[20]) if row[20] else None,
        })
    return {
        "results": results,
        "total": total,
        "page": page,
        "page_size": page_size,
        "pages": max(1, (total + page_size - 1) // page_size),
    }


async def get_rsib_result(result_id: int) -> Optional[dict[str, Any]]:
    row = await db.fetchone(
        """SELECT id, session_id, super_run_id, combo_index, variant_type, variant_label,
                  strategy, strategy_params_json, sweep_id, sweep_combo_index,
                  accuracy_pct, signals, correct, wrong, trades_taken, trades_skipped,
                  monthly_json, metrics_json,
                  final_bank_half, profit_half, roi_half,
                  final_bank_full, profit_full, roi_full,
                  created_at
           FROM rsib_results WHERE id = %s""",
        (result_id,),
    )
    if not row:
        return None
    return {
        "id": row[0],
        "session_id": row[1],
        "super_run_id": row[2],
        "combo_index": row[3],
        "variant_type": row[4],
        "variant_label": row[5],
        "strategy": row[6],
        "strategy_params": _safe_json_loads(row[7], {}),
        "sweep_id": row[8],
        "sweep_combo_index": row[9],
        "accuracy_pct": row[10],
        "signals": row[11],
        "correct": row[12],
        "wrong": row[13],
        "trades_taken": row[14],
        "trades_skipped": row[15],
        "monthly": _safe_json_loads(row[16], []),
        "metrics": _safe_json_loads(row[17], {}),
        "final_bank_half": row[18],
        "profit_half": row[19],
        "roi_half": row[20],
        "final_bank_full": row[21],
        "profit_full": row[22],
        "roi_full": row[23],
        "created_at": str(row[24]) if row[24] else None,
    }


async def _update_rsib_session_progress(session_id: int, completed: int, started: float) -> None:
    await db.execute(
        "UPDATE rsib_sessions SET completed = %s, total_time_sec = %s WHERE id = %s",
        (completed, round(time.time() - started, 2), session_id),
    )


async def _save_rsib_result(
    session_id: int,
    super_run_id: Optional[int],
    combo_index: int,
    variant_type: str,
    variant_label: str,
    strategy: str,
    strategy_params: dict[str, Any],
    monthly: list[dict[str, Any]],
    metrics: dict[str, Any],
    bankroll: dict[str, Any],
    sweep_id: Optional[int],
    sweep_combo_index: Optional[int],
    sim_half: Optional[dict[str, Any]] = None,
    sim_full: Optional[dict[str, Any]] = None,
) -> int:
    sim_half = sim_half or _simulate_monthly_bankroll(
        monthly,
        float(bankroll.get("start_bank") or 1000),
        float(bankroll.get("buy_price_cents") or 52),
        float(bankroll.get("max_bet") or 500),
        float(bankroll.get("half_kelly_pct") or 1.70) / 100.0,
        float(bankroll.get("fee_pct") or 1.56) / 100.0,
    )
    sim_full = sim_full or _simulate_monthly_bankroll(
        monthly,
        float(bankroll.get("start_bank") or 1000),
        float(bankroll.get("buy_price_cents") or 52),
        float(bankroll.get("max_bet") or 500),
        float(bankroll.get("full_kelly_pct") or 3.34) / 100.0,
        float(bankroll.get("fee_pct") or 1.56) / 100.0,
    )
    return await db.execute(
        """INSERT INTO rsib_results
           (session_id, super_run_id, combo_index, variant_type, variant_label,
            strategy, strategy_params_json, sweep_id, sweep_combo_index,
            accuracy_pct, signals, correct, wrong, trades_taken, trades_skipped,
            monthly_json, metrics_json,
            final_bank_half, profit_half, roi_half,
            final_bank_full, profit_full, roi_full)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
        (
            session_id,
            super_run_id,
            combo_index,
            variant_type,
            variant_label,
            strategy,
            json.dumps(strategy_params, ensure_ascii=False),
            sweep_id,
            sweep_combo_index,
            metrics.get("accuracy_pct"),
            metrics.get("signals"),
            metrics.get("correct"),
            metrics.get("wrong"),
            metrics.get("trades_taken"),
            metrics.get("trades_skipped"),
            json.dumps(monthly, ensure_ascii=False),
            json.dumps(metrics, ensure_ascii=False),
            sim_half.get("final_bank"),
            sim_half.get("profit"),
            sim_half.get("roi_pct"),
            sim_full.get("final_bank"),
            sim_full.get("profit"),
            sim_full.get("roi_pct"),
        ),
    )


async def _load_super_run_summary(super_run_id: int) -> dict[str, Any]:
    row = await db.fetchone(
        "SELECT signals, correct, wrong, accuracy_pct FROM super_backtest_runs WHERE id = %s",
        (super_run_id,),
    )
    if not row:
        return {"signals": 0, "correct": 0, "wrong": 0, "accuracy_pct": 0.0}
    return {
        "signals": row[0] or 0,
        "correct": row[1] or 0,
        "wrong": row[2] or 0,
        "accuracy_pct": row[3] or 0.0,
    }


async def _load_super_run_monthly(super_run_id: int) -> list[dict[str, Any]]:
    rows = await db.fetchall(
        """SELECT open_time, is_correct, vol_skip
           FROM super_backtest_predictions
           WHERE super_run_id = %s AND is_signal = 1
           ORDER BY candle_idx""",
        (super_run_id,),
    )
    monthly: dict[str, dict[str, Any]] = {}
    for row in rows:
        month = _month_key_from_open_time(row[0])
        bucket = monthly.setdefault(month, {"month": month, "total": 0, "wins": 0, "volatility_skips": 0})
        bucket["total"] += 1
        if row[1]:
            bucket["wins"] += 1
        if row[2]:
            bucket["volatility_skips"] += 1
    results = []
    for month in sorted(monthly.keys()):
        bucket = monthly[month]
        total = bucket["total"]
        bucket["accuracy"] = round(bucket["wins"] / total * 100, 2) if total else 0.0
        results.append(bucket)
    return results


def _normalize_sweep_monthly(raw: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for month in sorted(raw.keys()):
        bucket = raw[month] or {}
        total = int(bucket.get("taken") or 0)
        wins = int(bucket.get("filtered_correct") or 0)
        results.append({
            "month": month,
            "total": total,
            "wins": wins,
            "taken": total,
            "skipped": int(bucket.get("skipped") or 0),
            "baseline_total": int(bucket.get("total") or total),
            "baseline_correct": int(bucket.get("baseline_correct") or wins),
            "accuracy": round(wins / total * 100, 2) if total else 0.0,
            "volatility_skips": int(bucket.get("skipped") or 0),
        })
    return results


def _simulate_monthly_bankroll(
    monthly: list[dict[str, Any]],
    start_bank: float,
    buy_price_cents: float,
    max_bet: float,
    bet_pct: float,
    bet_fee_rate: float,
) -> dict[str, Any]:
    cost = buy_price_cents / 100.0
    profit_per_share = 1.0 - cost
    b = profit_per_share / cost if cost > 0 else 0.0

    def _sim_month(bank: float, n_signals: int, win_prob: float, kh: float) -> tuple[float, float, float, int]:
        ev_per_dollar = (win_prob * b) - (1 - win_prob)
        avg_stake = 0.0
        max_stake_used = 0.0
        edge_signals = 0
        if not (abs(kh) > 0) or n_signals <= 0:
            return bank, avg_stake, max_stake_used, edge_signals
        for _ in range(n_signals):
            raw_stake = bank * abs(kh)
            stake = max(0.0, min(raw_stake, max_bet or raw_stake, bank))
            if stake <= 0:
                break
            bank += stake * ev_per_dollar
            bank -= stake * bet_fee_rate
            if bank < 0.01:
                bank = 0.01
            avg_stake += stake
            if stake > max_stake_used:
                max_stake_used = stake
            edge_signals += 1
        avg_stake = (avg_stake / edge_signals) if edge_signals else 0.0
        return bank, avg_stake, max_stake_used, edge_signals

    bank = float(start_bank or 1000.0)
    month_entries = []
    for item in sorted(monthly, key=lambda x: str(x.get("month") or "")):
        signals = int(item.get("total") or 0)
        acc = float(item.get("accuracy") or 0.0)
        bank, avg_stake, max_stake_used, edge_signals = _sim_month(bank, signals, acc / 100.0, bet_pct)
        month_entries.append({
            "month": item.get("month"),
            "signals": signals,
            "accuracy": acc,
            "wins": item.get("wins"),
            "taken": item.get("taken", signals),
            "skipped": item.get("skipped", 0),
            "volatility_skips": item.get("volatility_skips", 0),
            "bank": round(bank, 2),
            "avg_stake": avg_stake,
            "max_stake": max_stake_used,
            "edge_signals": edge_signals,
        })
    return {
        "final_bank": round(bank, 2),
        "profit": round(bank - start_bank, 2),
        "roi_pct": round(((bank - start_bank) / start_bank * 100.0), 2) if start_bank else 0.0,
        "month_entries": month_entries,
    }


def _month_key_from_open_time(open_time: int) -> str:
    as_ms = int(open_time) // 1000
    return time.strftime("%Y-%m", time.gmtime(as_ms / 1000.0))


def _safe_json_loads(raw: Any, default: Any) -> Any:
    if not raw:
        return default
    try:
        return json.loads(raw)
    except Exception:
        return default


def _nullable_int(value: Any) -> Optional[int]:
    if value in (None, "", 0):
        return None
    try:
        return int(value)
    except Exception:
        return None


def _format_sweep_label(item: dict[str, Any]) -> str:
    return f"Sweep g{item.get('good_threshold')} b{item.get('bad_threshold')} p{item.get('filter_threshold')}"


def _update_best(
    best_profit: Optional[float],
    best_final_bank: Optional[float],
    sim: dict[str, Any],
) -> tuple[Optional[float], Optional[float]]:
    profit = sim.get("profit")
    final_bank = sim.get("final_bank")
    if best_profit is None or (profit is not None and profit > best_profit):
        return profit, final_bank
    return best_profit, best_final_bank
