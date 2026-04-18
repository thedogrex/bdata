import json
import time
import asyncio
import pathlib
import traceback
import logging
from typing import Optional
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from predictor.backtester import (
    run_backtest,
    preload_backtest_data,
    run_backtest_vectorized,
    RULE_BASED_STRATEGIES,
)
from predictor.data_loader import load_candles, add_direction, add_future_directions
from predictor.features import add_technical_features
from predictor.strategies import list_strategies, STRATEGY_REGISTRY
from predictor.strategies.lightgbm_strategy import LightGBMStrategy
from predictor.db_history import (
    save_backtest_run, get_history, get_history_detail,
    delete_run, clear_history,
    get_bruteforce_sessions, get_bruteforce_session_by_id, get_best_runs,
    get_runs_by_ids, delete_bruteforce_group, get_bf_runs_paginated,
)
from predictor.bruteforce import run_bruteforce, resume_bruteforce, get_default_grid, build_combos
from predictor.task_manager import task_mgr
from predictor import poly_service
from predictor import live_trading
from predictor import telegram_bot
from predictor import predict_4s
from predictor.binance_snapshot import start_snapshot_collector, stop_snapshot_collector
import app.config as config

BRUTEFORCE_ONLY_MODE = bool(getattr(config, "ONLY_BRUTEFORCE", False))
logger = logging.getLogger("predictor.api")
POLY_DISABLED_MESSAGE = {
    "error": "Polymarket endpoints are disabled when ONLY_BRUTEFORCE mode is enabled",
}

app = FastAPI(title="Candle Predictor & Backtester", version="3.0.0", root_path=config.FASTAPI_ROOT)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def block_poly_when_bruteforce_only(request: Request, call_next):
    if BRUTEFORCE_ONLY_MODE:
        path = request.url.path
        if path.startswith("/api/poly") or path.startswith("/api/analytics") or path.startswith("/api/compare_asume"):
            return JSONResponse(status_code=503, content=POLY_DISABLED_MESSAGE)
    return await call_next(request)


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Serve favicon.ico from project root."""
    favicon_path = pathlib.Path(__file__).resolve().parents[1] / "favicon.ico"
    if favicon_path.is_file():
        return FileResponse(favicon_path, media_type="image/x-icon")
    # Return 204 No Content if favicon does not exist
    return JSONResponse(status_code=204, content={})


poly_stop_event: asyncio.Event | None = None


class BacktestRequest(BaseModel):
    strategy: str = "xgboost"
    params: dict | None = None
    train_start: str = "2022-01-01"
    train_end: str = "2025-06-30"
    test_start: str = "2025-07-01"
    test_end: str = "2026-01-31"
    horizons: list[int] = [1]
    table: str = "c_5m"
    window_size: int = 5000
    retrain_every: int = 500


class CompareRequest(BaseModel):
    strategies: list[str] = ["xgboost", "rsi_mean_reversion", "momentum", "pattern_sequence", "ensemble"]
    train_start: str = "2022-01-01"
    train_end: str = "2025-06-30"
    test_start: str = "2025-07-01"
    test_end: str = "2026-01-31"
    horizons: list[int] = [1]
    table: str = "c_5m"
    window_size: int = 5000
    retrain_every: int = 500


class BruteforceRequest(BaseModel):
    strategy: str = "xgboost"
    param_grid: dict | None = None
    train_start: str = "2022-01-01"
    train_end: str = "2025-06-30"
    test_start: str = "2025-07-01"
    test_end: str = "2026-01-31"
    horizon: int = 1
    horizons: list[int] | None = None
    table: str = "c_5m"
    window_size: int = 5000
    retrain_every: int = 500
    max_combos: int = 100
    processes: int = 1  # See predictor.bruteforce.DEFAULT_BRUTEFORCE_PROCESSES


class SettingsRequest(BaseModel):
    autopredict: bool = False
    strategy: str = "rsi_mean_reversion"
    params: dict | None = None
    window_size: int = 1000


class LiveTradeSettingsRequest(BaseModel):
    auto_place: bool = False
    bet_size_usd: float = 5.0
    price_cap_cents: int = 52


class SimTradeRequest(BaseModel):
    slug: str
    asset_id: str
    qty: float
    outcome_side: str | None = None
    price: float | None = None


class PredictRequest(BaseModel):
    slug: str
    strategy: str = "rsi_mean_reversion"
    params: dict | None = None
    window_size: int = 1000
    horizon: int = 1
    table: str = "c_5m"


class TemplateRequest(BaseModel):
    name: str
    strategy: str = "rsi_mean_reversion"
    params: dict | None = None
    window_size: int = 1000
    horizon: int = 1


class TemplateUpdateRequest(BaseModel):
    name: str | None = None
    strategy: str | None = None
    params: dict | None = None
    window_size: int | None = None
    horizon: int | None = None
    active: bool | None = None
    sort_order: int | None = None


class BestCompareRequest(BaseModel):
    run_ids: list[int]
    horizon: int = 1


class BatchPredictRequest(BaseModel):
    slug: str
    quantum: bool = False
    table: str = "c_5m"


class LightGBMImportanceRequest(BaseModel):
    train_start: str = "2024-01-01"
    train_end: str = "2024-06-01"
    table: str = "c_5m"
    horizon: int = 2
    params: dict | None = None
    top_n: int = 40


class LiveBuyRequest(BaseModel):
    slug: str
    asset_id: str
    outcome_side: str = "UP"
    prediction_direction: str = "UP"
    amount_usd: float = 0.0
    snapshot_price: float = 0.50
    price_threshold: float = 0.52
    bank_usd: float | None = None
    bank_pct: float = 0.05
    min_buy_usd: float = 3.0
    max_buy_usd: float = 20.0
    batch_id: str | None = None
    template_id: int | None = None


class ManualResolveRequest(BaseModel):
    outcome: str


# ==================== API ROUTES ====================

@app.get("/api/strategies")
async def api_list_strategies():
    return list_strategies()


# ==================== POLYMARKET (ADMIN) ====================

@app.on_event("startup")
async def startup_event():
    global poly_stop_event
    if BRUTEFORCE_ONLY_MODE:
        logger.info("ONLY_BRUTEFORCE mode enabled: skipping Polymarket polling/telegram/snapshots startup")
        return
    poly_stop_event = asyncio.Event()
    asyncio.create_task(poly_service.poll_loop(poly_stop_event, orderbook_interval_sec=3))
    telegram_bot.start_polling()
    start_snapshot_collector()


@app.on_event("shutdown")
async def shutdown_event():
    global poly_stop_event
    if BRUTEFORCE_ONLY_MODE:
        return
    if poly_stop_event is not None:
        poly_stop_event.set()
    await telegram_bot.stop_polling()
    await stop_snapshot_collector()


@app.get("/api/poly/markets")
async def api_poly_markets(limit: int = Query(50), offset: int = Query(0)):
    return await poly_service.list_markets(limit=limit, offset=offset)


@app.get("/api/poly/status")
async def api_poly_status():
    return {
        "active_ts": poly_service.current_active_ts(),
        "emulate_down": bool(getattr(config, "EMULATE_DOWN", False)),
        "need_confirmation": bool(getattr(config, "NEED_CONFIRMATION", True)),
    }


@app.get("/api/poly/market/{slug}")
async def api_poly_market(slug: str):
    m = await poly_service.get_market(slug)
    if m is None:
        return JSONResponse(status_code=404, content={"error": "Not found"})
    return m


@app.get("/api/poly/market/{slug}/live")
async def api_poly_market_live(slug: str):
    m = await poly_service.get_market_live(slug)
    if m is None:
        return JSONResponse(status_code=404, content={"error": "Not found"})
    return m


@app.post("/api/poly/market/{slug}/resolve")
async def api_poly_market_resolve(slug: str, req: ManualResolveRequest):
    result = await poly_service.set_market_resolution_manual(slug, req.outcome)
    if result.get("error"):
        status = 404 if result.get("error") == "Market not found" else 400
        return JSONResponse(status_code=status, content=result)
    return result


@app.get("/api/poly/outcome/{asset_id}/series")
async def api_poly_series(asset_id: str, minutes: int = Query(60), limit: int = Query(2000)):
    return await poly_service.get_price_series(asset_id=asset_id, minutes=minutes, limit=limit)


@app.get("/api/poly/orderbook/{slug}/{asset_id}/analysis")
async def api_poly_orderbook_analysis(slug: str, asset_id: str, minutes: int = Query(60)):
    return await poly_service.get_orderbook_analysis(slug=slug, asset_id=asset_id, minutes=minutes)


@app.get("/api/poly/orderbook/{slug}/{asset_id}/latest")
async def api_poly_orderbook_latest(slug: str, asset_id: str):
    return await poly_service.get_latest_orderbook(slug=slug, asset_id=asset_id)


@app.get("/api/poly/live/quote")
async def api_poly_live_quote(slug: str = Query(""), asset_id: str = Query("")):
    """Return refreshed best ask for an outcome token.

    Used by frontend confirmation popup so it displays the same (refreshed) price
    the backend will attempt to use when placing a live order.
    """
    if not slug or not asset_id:
        return JSONResponse(status_code=400, content={"error": "slug and asset_id are required"})
    try:
        # Re-fetch market for debug parity (non-fatal)
        try:
            live_trading.trading_client.fetch_market(slug)
        except Exception:
            pass

        ask = live_trading.trading_client.get_best_ask(asset_id)
        if ask is None:
            return {"slug": slug, "asset_id": asset_id, "best_ask": None, "best_ask_cents": None}
        ask_f = float(ask)
        return {
            "slug": slug,
            "asset_id": asset_id,
            "best_ask": ask_f,
            "best_ask_cents": int(round(ask_f * 100.0)),
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/poly/predict")
async def api_poly_predict(req: PredictRequest):
    return await poly_service.predict_for_market(
        slug=req.slug,
        strategy_name=req.strategy,
        strategy_params=req.params,
        window_size=req.window_size,
        horizon=req.horizon,
        table=req.table,
    )


# ==================== PREDICTION TEMPLATES ====================

@app.get("/api/poly/pred_templates")
async def api_list_pred_templates():
    return await poly_service.list_pred_templates()


@app.post("/api/poly/pred_templates")
async def api_create_pred_template(req: TemplateRequest):
    return await poly_service.create_pred_template(
        name=req.name,
        strategy=req.strategy,
        params=req.params,
        window_size=req.window_size,
        horizon=req.horizon,
    )


@app.put("/api/poly/pred_templates/{template_id}")
async def api_update_pred_template(template_id: int, req: TemplateUpdateRequest):
    return await poly_service.update_pred_template(
        template_id=template_id,
        name=req.name,
        strategy=req.strategy,
        params=req.params,
        window_size=req.window_size,
        horizon=req.horizon,
        active=req.active,
        sort_order=req.sort_order,
    )


@app.delete("/api/poly/pred_templates/{template_id}")
async def api_delete_pred_template(template_id: int):
    return await poly_service.delete_pred_template(template_id=template_id)


@app.post("/api/poly/pred_templates/{template_id}/toggle")
async def api_toggle_pred_template(template_id: int):
    return await poly_service.toggle_pred_template(template_id=template_id)


@app.post("/api/poly/batch_predict")
async def api_poly_batch_predict(req: BatchPredictRequest):
    return await poly_service.batch_predict_for_market(
        slug=req.slug,
        quantum=req.quantum,
        table=req.table,
    )


@app.get("/api/poly/pred_runs/{slug:path}")
async def api_get_pred_runs(slug: str, limit: int = 200):
    return await poly_service.get_pred_runs_for_market(slug=slug, limit=limit)


@app.get("/api/poly/settings")
async def api_poly_get_settings():
    return await poly_service.get_settings()


@app.post("/api/poly/settings")
async def api_poly_save_settings(req: SettingsRequest):
    return await poly_service.save_settings(
        autopredict=req.autopredict,
        strategy=req.strategy,
        params=req.params,
        window_size=req.window_size,
    )


@app.post("/api/lightgbm/feature_importance")
async def api_lightgbm_feature_importance(req: LightGBMImportanceRequest):
    if req.horizon < 1:
        return JSONResponse(status_code=400, content={"error": "horizon must be >= 1"})

    df = await load_candles(req.table, req.train_start, req.train_end)
    if df is None or df.empty:
        return JSONResponse(status_code=404, content={"error": "No candle data for selected range"})

    df = add_direction(df)
    df = add_future_directions(df, [req.horizon])
    df = df.reset_index(drop=True)
    df = add_technical_features(df)

    strategy = LightGBMStrategy(req.params or {})
    await asyncio.to_thread(strategy.fit, df, req.horizon)
    feature_rows = strategy.get_feature_importance(top_n=req.top_n, normalize=True)
    payload = [
        {"feature": name, "weight": round(value, 6)}
        for name, value in feature_rows
    ]

    return {
        "strategy": "lightgbm",
        "train_start": req.train_start,
        "train_end": req.train_end,
        "table": req.table,
        "horizon": req.horizon,
        "top_n": req.top_n,
        "total_features": len(strategy.feature_cols or []),
        "feature_importance": payload,
    }


@app.get("/api/poly/live/trade_settings")
async def api_poly_get_live_trade_settings():
    return await poly_service.get_live_trade_settings()


@app.post("/api/poly/live/trade_settings")
async def api_poly_save_live_trade_settings(req: LiveTradeSettingsRequest):
    return await poly_service.request_bet_size_change(
        auto_place=req.auto_place,
        bet_size_usd=req.bet_size_usd,
        price_cap_cents=req.price_cap_cents,
    )


@app.get("/api/poly/live/trade_settings/request")
async def api_poly_bet_size_request_state():
    state = poly_service.get_bet_size_request_state()
    if not state:
        return {"status": "none"}
    return {"status": state.get("status", "pending"), "request": state}


@app.post("/api/poly/live/trade_settings/cancel")
async def api_poly_cancel_bet_size_request():
    return await poly_service.cancel_bet_size_request()


@app.get("/api/poly/prediction/{slug}")
async def api_poly_prediction(slug: str):
    return await poly_service.get_saved_prediction(slug=slug)


@app.get("/api/poly/pred_updates")
async def api_poly_pred_updates(since: int = Query(0), limit: int = Query(20)):
    return await poly_service.list_prediction_updates(since_ts=since, limit=limit)


@app.get("/api/poly/prediction_candles/{slug}")
async def api_poly_prediction_candles(slug: str, window: int = Query(1000), tail: int = Query(200)):
    return await poly_service.get_prediction_candles(slug=slug, window_size=window, tail=tail)


@app.post("/api/candles/sync")
async def api_candles_sync(target_ts: int = Query(...), window: int = Query(1100)):
    from predictor.candle_sync import sync_candles_up_to, check_and_fill_gaps
    sync = await sync_candles_up_to(target_ts, window_candles=window)
    fill = await check_and_fill_gaps(target_ts, window_candles=window)
    return {"sync": sync, "gap_fill": fill}


@app.get("/api/candles/sync_status")
async def api_candles_sync_status():
    from predictor.candle_sync import get_candle_sync_status
    return get_candle_sync_status()


@app.post("/api/backtest")
async def api_run_backtest(req: BacktestRequest):
    """Queue a backtest task. Returns task_id immediately."""
    async def _run(progress):
        if req.strategy in RULE_BASED_STRATEGIES:
            preloaded = await preload_backtest_data(
                train_start=req.train_start,
                train_end=req.train_end,
                test_start=req.test_start,
                test_end=req.test_end,
                horizons=req.horizons,
                table=req.table,
                progress=progress,
                strategy_name=req.strategy,
            )
            result = run_backtest_vectorized(
                strategy_name=req.strategy,
                strategy_params=req.params,
                preloaded=preloaded,
                horizons=req.horizons,
                window_size=req.window_size,
                retrain_every=req.retrain_every,
                train_start=req.train_start,
                train_end=req.train_end,
                test_start=req.test_start,
                test_end=req.test_end,
                table=req.table,
            )
        else:
            result = await run_backtest(
                strategy_name=req.strategy,
                strategy_params=req.params,
                train_start=req.train_start,
                train_end=req.train_end,
                test_start=req.test_start,
                test_end=req.test_end,
                horizons=req.horizons,
                table=req.table,
                window_size=req.window_size,
                retrain_every=req.retrain_every,
                progress=progress,
            )
        if "error" not in result:
            run_id = await save_backtest_run(result)
            result["id"] = run_id
        return result

    label = f"Backtest {req.strategy} [{req.test_start} -> {req.test_end}]"
    task_id = task_mgr.enqueue("backtest", label, _run)
    return {"task_id": task_id, "status": "queued", "label": label}


@app.post("/api/compare")
async def api_compare_strategies(req: CompareRequest):
    """Queue a compare task (runs all strategies sequentially)."""
    async def _run(progress):
        results = []
        strats = [s for s in req.strategies if s in STRATEGY_REGISTRY]
        progress.total = len(strats)

        preloaded = None
        if any(s in RULE_BASED_STRATEGIES for s in strats):
            preloaded = await preload_backtest_data(
                train_start=req.train_start,
                train_end=req.train_end,
                test_start=req.test_start,
                test_end=req.test_end,
                horizons=req.horizons,
                table=req.table,
                progress=progress,
            )

        for idx, strategy_name in enumerate(strats):
            await progress.check_pause_cancel()
            progress.update(idx, len(strats), f"Running {strategy_name} ({idx+1}/{len(strats)})")
            try:
                if strategy_name in RULE_BASED_STRATEGIES and preloaded is not None:
                    result = run_backtest_vectorized(
                        strategy_name=strategy_name,
                        strategy_params=None,
                        preloaded=preloaded,
                        horizons=req.horizons,
                        window_size=req.window_size,
                        retrain_every=req.retrain_every,
                        train_start=req.train_start,
                        train_end=req.train_end,
                        test_start=req.test_start,
                        test_end=req.test_end,
                        table=req.table,
                    )
                else:
                    result = await run_backtest(
                        strategy_name=strategy_name,
                        strategy_params=None,
                        train_start=req.train_start,
                        train_end=req.train_end,
                        test_start=req.test_start,
                        test_end=req.test_end,
                        horizons=req.horizons,
                        table=req.table,
                        window_size=req.window_size,
                        retrain_every=req.retrain_every,
                    )
                await save_backtest_run(result)
                results.append(result)
            except Exception as e:
                from predictor.task_manager import CancelledError
                if isinstance(e, CancelledError):
                    raise
                results.append({"strategy": strategy_name, "error": str(e)})
        progress.update(len(strats), len(strats), "Done")
        return results

    label = f"Compare {len(req.strategies)} strategies [{req.test_start} -> {req.test_end}]"
    task_id = task_mgr.enqueue("compare", label, _run, total=len(req.strategies))
    return {"task_id": task_id, "status": "queued", "label": label}


@app.get("/api/history")
async def api_history(
    limit: int = Query(100),
    strategy: str = Query(None),
    min_accuracy: float = Query(None),
    bruteforce_id: int = Query(None),
    exclude_bruteforce: bool = Query(False),
):
    return await get_history(limit, strategy, min_accuracy, bruteforce_id, exclude_bruteforce)


@app.get("/api/history/{run_id}")
async def api_history_detail(run_id: int):
    result = await get_history_detail(run_id)
    if result is None:
        return JSONResponse(status_code=404, content={"error": "Not found"})
    return result


@app.delete("/api/history/{run_id}")
async def api_delete_run(run_id: int):
    await delete_run(run_id)
    return {"status": "deleted", "id": run_id}


@app.delete("/api/history")
async def api_clear_history():
    await clear_history()
    return {"status": "cleared"}


@app.delete("/api/history/bruteforce/{bf_id}")
async def api_delete_bruteforce_group(bf_id: int):
    return await delete_bruteforce_group(bf_id)


@app.get("/api/history/bruteforce/{bf_id}/runs")
async def api_bf_runs_paginated(
    bf_id: int,
    offset: int = Query(0),
    limit: int = Query(20),
    min_accuracy: float | None = Query(None),
    window_size: int | None = Query(None),
):
    return await get_bf_runs_paginated(bf_id, offset, limit, min_accuracy, window_size)


@app.get("/api/best")
async def api_best_runs(
    limit: int = Query(20),
    horizon: int = Query(1),
    signals_min: int | None = Query(None),
    signals_max: int | None = Query(None),
    bruteforce_id: int | None = Query(None),
):
    return await get_best_runs(
        limit,
        horizon,
        signals_min=signals_min,
        signals_max=signals_max,
        bruteforce_id=bruteforce_id,
    )


@app.post("/api/best/compare")
async def api_best_compare(req: BestCompareRequest):
    return await get_runs_by_ids(run_ids=req.run_ids, horizon=req.horizon)


# ==================== BRUTE FORCE ====================

@app.get("/api/bruteforce/grid/{strategy}")
async def api_default_grid(strategy: str):
    grid = get_default_grid(strategy)
    combos = build_combos(grid) if grid else []

    # New UI format: return a full config object that can be pasted/edited as JSON.
    cfg = {
        "strategy": strategy,
        "train_start": BruteforceRequest.model_fields["train_start"].default,
        "train_end": BruteforceRequest.model_fields["train_end"].default,
        "test_start": BruteforceRequest.model_fields["test_start"].default,
        "test_end": BruteforceRequest.model_fields["test_end"].default,
        "horizon": BruteforceRequest.model_fields["horizon"].default,
        "table": BruteforceRequest.model_fields["table"].default,
        "window_size": BruteforceRequest.model_fields["window_size"].default,
        "retrain_every": BruteforceRequest.model_fields["retrain_every"].default,
        "max_combos": BruteforceRequest.model_fields["max_combos"].default,
        "param_grid": grid,
    }
    return {"strategy": strategy, "config": cfg, "total_combos": len(combos)}


@app.post("/api/bruteforce")
async def api_run_bruteforce(req: BruteforceRequest):
    """Queue a brute-force task. Returns task_id immediately."""
    grid = req.param_grid or get_default_grid(req.strategy)

    # UI may send a full config object by mistake (with nested param_grid).
    # In that case, extract the actual grid so combo counting works.
    if isinstance(grid, dict) and "param_grid" in grid and isinstance(grid.get("param_grid"), dict):
        grid = grid.get("param_grid")

    # If any non-grid metadata keys slipped into the grid, ignore them.
    if isinstance(grid, dict):
        grid = {
            k: v for k, v in grid.items()
            if k not in {
                "strategy", "train_start", "train_end", "test_start", "test_end",
                "horizon", "table", "retrain_every", "max_combos",
            }
        }
    if not grid:
        return JSONResponse(status_code=400, content={"error": f"No param grid for {req.strategy}"})

    combos_count = len(build_combos(grid))
    actual_count = min(combos_count, req.max_combos)

    selected_horizon = req.horizon
    if (selected_horizon is None or selected_horizon <= 0) and req.horizons:
        for h in req.horizons:
            if h and h > 0:
                selected_horizon = h
                break
    if selected_horizon is None or selected_horizon <= 0:
        selected_horizon = 1

    async def _run(progress):
        return await run_bruteforce(
            strategy=req.strategy,
            param_grid=grid,
            train_start=req.train_start,
            train_end=req.train_end,
            test_start=req.test_start,
            test_end=req.test_end,
            horizon=selected_horizon,
            table=req.table,
            window_size=req.window_size,
            retrain_every=req.retrain_every,
            max_combos=req.max_combos,
            processes=req.processes,
            progress=progress,
        )

    label = f"BruteForce {req.strategy} H{selected_horizon} ({actual_count} combos)"
    task_id = task_mgr.enqueue("bruteforce", label, _run, total=actual_count)
    return {"task_id": task_id, "status": "queued", "label": label, "combos": actual_count}


@app.get("/api/bruteforce/sessions")
async def api_bruteforce_sessions():
    return await get_bruteforce_sessions()


@app.post("/api/bruteforce/resume/{bf_id}")
async def api_resume_bruteforce(bf_id: int):
    """Resume a paused/interrupted brute-force session from its DB checkpoint."""
    session = await get_bruteforce_session_by_id(bf_id)
    if not session:
        return JSONResponse(status_code=404, content={"error": f"Session {bf_id} not found"})
    if session["status"] == "done":
        return JSONResponse(status_code=400, content={"error": f"Session {bf_id} is already completed"})

    remaining = session["total_combos"] - session["completed"]

    async def _run(progress):
        return await resume_bruteforce(bf_id=bf_id, progress=progress)

    label = f"Resume BF#{bf_id} {session['strategy']} H{session['horizon']} ({remaining} remaining)"
    task_id = task_mgr.enqueue("bruteforce", label, _run, total=session["total_combos"])
    return {"task_id": task_id, "status": "queued", "label": label, "bf_id": bf_id, "remaining": remaining}


@app.post("/api/bruteforce/stop/{bf_id}")
async def api_stop_bruteforce(bf_id: int):
    """Stop and cancel a brute-force session."""
    session = await get_bruteforce_session_by_id(bf_id)
    if not session:
        return JSONResponse(status_code=404, content={"error": f"Session {bf_id} not found"})
    
    # Try to cancel the task if it's running
    success = True
    if session.get("task_id"):
        success = task_mgr.cancel(session["task_id"])
    
    # Mark session as stopped in database
    from predictor.db_history import update_bruteforce_session
    await update_bruteforce_session(bf_id, {"status": "stopped"})
    
    return {"ok": success, "bf_id": bf_id, "status": "stopped"}


# ==================== TASK QUEUE ====================

@app.get("/api/tasks/status")
async def api_task_status():
    """Get current task, queue, and recent task history."""
    return task_mgr.get_status()


@app.get("/api/tasks/{task_id}")
async def api_task_progress(task_id: str):
    p = task_mgr.get_progress(task_id)
    if p is None:
        return JSONResponse(status_code=404, content={"error": "Task not found"})
    return p


@app.get("/api/tasks/{task_id}/result")
async def api_task_result(task_id: str):
    result = task_mgr.get_result(task_id)
    if result is None:
        return JSONResponse(status_code=404, content={"error": "No result yet"})
    return result


@app.post("/api/tasks/{task_id}/pause")
async def api_task_pause(task_id: str):
    ok = task_mgr.pause(task_id)
    return {"ok": ok, "task_id": task_id, "action": "pause"}


@app.post("/api/tasks/{task_id}/resume")
async def api_task_resume(task_id: str):
    ok = task_mgr.resume(task_id)
    return {"ok": ok, "task_id": task_id, "action": "resume"}


@app.post("/api/tasks/{task_id}/cancel")
async def api_task_cancel(task_id: str):
    ok = task_mgr.cancel(task_id)
    return {"ok": ok, "task_id": task_id, "action": "cancel"}


@app.delete("/api/tasks/queue/{task_id}")
async def api_task_remove_from_queue(task_id: str):
    ok = task_mgr.remove_from_queue(task_id)
    return {"ok": ok, "task_id": task_id, "action": "removed"}


@app.delete("/api/tasks/queue")
async def api_clear_queue():
    count = task_mgr.clear_queue()
    return {"cleared": count}


# ==================== ANALYTICS ====================

@app.get("/api/analytics/predictions")
async def api_predictions_analytics(
    date_from: str | None = Query(None),
    date_to: str | None = Query(None),
    hour_from: int | None = Query(None),
    hour_to: int | None = Query(None),
):
    return await poly_service.get_predictions_analytics(
        date_from=date_from,
        date_to=date_to,
        hour_from=hour_from,
        hour_to=hour_to,
    )


@app.get("/api/analytics/ask_prices")
async def api_ask_price_analysis(
    date_from: str | None = Query(None),
    date_to: str | None = Query(None),
    hour_from: int | None = Query(None),
    hour_to: int | None = Query(None),
    window_sec: int = Query(10),
):
    return await poly_service.get_ask_price_analysis(
        date_from=date_from,
        date_to=date_to,
        hour_from=hour_from,
        hour_to=hour_to,
        window_sec=window_sec,
    )


@app.get("/api/analytics/order_market_pricing")
async def api_order_market_pricing(
    date_from: str | None = Query(None),
    date_to: str | None = Query(None),
    price_threshold_cents: float = Query(52.0),
):
    return await poly_service.get_order_market_pricing(
        date_from=date_from,
        date_to=date_to,
        price_threshold_cents=price_threshold_cents,
    )


@app.get("/api/analytics/kelly_sim")
async def api_kelly_sim(
    date_from: str | None = Query(None),
    date_to: str | None = Query(None),
    hour_from: int | None = Query(None),
    hour_to: int | None = Query(None),
    start_bank: float = Query(100.0),
    max_bet: float | None = Query(None),
    fee_rate: float = Query(0.0156),
    max_price_cents: float = Query(51.0),
    hk_pct: float = Query(0.017),
    fk_pct: float = Query(0.0334),
):
    return await poly_service.get_kelly_simulation(
        date_from=date_from,
        date_to=date_to,
        hour_from=hour_from,
        hour_to=hour_to,
        start_bank=start_bank,
        max_bet=max_bet,
        fee_rate=fee_rate,
        max_price_cents=max_price_cents,
        hk_pct=hk_pct,
        fk_pct=fk_pct,
    )


# ==================== ADMIN PANEL ====================

# ==================== LIVE TRADING ====================

@app.on_event("startup")
async def startup_live_trading():
    await live_trading.ensure_tables()
    await predict_4s.ensure_tables()


@app.post("/api/poly/live/buy")
async def api_live_buy(req: LiveBuyRequest):
    """Buy an outcome token after prediction."""
    if float(req.price_threshold) > 0.53:
        return {"success": False, "error": "price_threshold must be <= 0.53"}
    return await live_trading.buy_after_prediction(
        slug=req.slug,
        asset_id=req.asset_id,
        outcome_side=req.outcome_side,
        prediction_direction=req.prediction_direction,
        snapshot_price=req.snapshot_price,
        price_threshold=req.price_threshold,
        batch_id=req.batch_id,
        template_id=req.template_id,
    )


@app.get("/api/poly/live/positions")
async def api_live_positions(status: str = Query("open"), slug: str | None = Query(None)):
    if status == "open":
        return await live_trading.list_open_positions(slug=slug)
    return await live_trading.list_all_positions(slug=slug)


@app.get("/api/poly/live/orders")
async def api_live_orders(limit: int = Query(100), slug: str | None = Query(None)):
    return await live_trading.list_orders(limit=limit, slug=slug)


@app.get("/api/poly/live/wallet")
async def api_live_wallet(limit: int = Query(25)):
    return await live_trading.wallet_summary(limit=limit)


@app.get("/api/poly/live/order_flow")
async def api_live_order_flow(
    date_from: str | None = Query(None, description="Inclusive start date (YYYY-MM-DD)"),
    date_to: str | None = Query(None, description="Inclusive end date (YYYY-MM-DD)"),
    fee_pct: float | None = Query(3.6, description="Fee percentage applied to winning payouts"),
):
    return await live_trading.order_flow_analytics(date_from=date_from, date_to=date_to, fee_pct=fee_pct)


# ==================== 4S-EARLY PREDICTIONS ====================

@app.post("/api/poly/predict_4s/{slug:path}")
async def api_predict_4s(slug: str):
    """Run a 4s-early prediction for a market slug."""
    return await predict_4s.predict_for_market_4s(slug=slug)


@app.get("/api/poly/predictions_4s/{slug:path}")
async def api_get_prediction_4s(slug: str):
    """Return saved 4s-early prediction payload for a market."""
    result = await predict_4s.get_saved_prediction_4s(slug=slug)
    if result is None:
        return {"error": "No 4s-early prediction found for this market"}
    return result


@app.get("/api/poly/predictions_4s")
async def api_list_predictions_4s(limit: int = Query(200, ge=1, le=1000)):
    """List recent 4s-early predictions."""
    return await predict_4s.list_predictions_4s(limit=limit)


@app.get("/api/poly/live/orders_4s")
async def api_live_orders_4s(
    limit: int = Query(100, ge=1, le=1000),
    slug: str | None = Query(None),
):
    """List recent orders placed via 4s-early predictions."""
    return await predict_4s.list_orders_4s(slug=slug, limit=limit)


# ==================== COMPARE ASUME ====================

@app.get("/api/compare_asume")
async def api_compare_asume(
    date_from: str | None = Query(None, description="Inclusive start date (YYYY-MM-DD)"),
    date_to: str | None = Query(None, description="Inclusive end date (YYYY-MM-DD)"),
    limit: int = Query(500, ge=1, le=5000, description="Max markets to compare in range"),
):
    from predictor.compare_asume import run_compare
    return await run_compare(date_from=date_from, date_to=date_to, limit=limit)


_TEMPLATE_DIR = pathlib.Path(__file__).parent / "templates"


def _load_template(name: str) -> str:
    return (_TEMPLATE_DIR / name).read_text(encoding="utf-8")


def _build_admin_html() -> str:
    base = _load_template("base.html")
    replacements = {
        "{{TAB_BACKTEST}}": _load_template("tabs_backtest.html"),
        "{{TAB_COMPARE}}": "",   # included in tabs_backtest.html
        "{{TAB_BRUTEFORCE}}": "",  # included in tabs_backtest.html
        "{{TAB_HISTORY}}": "",   # included in tabs_backtest.html
        "{{TAB_BEST}}": "",      # included in tabs_backtest.html
        "{{TAB_POLY}}": _load_template("tab_poly.html"),
        "{{TAB_WALLET}}": _load_template("tab_wallet.html"),
        "{{TAB_ORDERBOOKS}}": _load_template("tab_orderbooks.html"),
        "{{TAB_ANALYTICS}}": _load_template("tab_analytics.html"),
        "{{TAB_COMPARE_ASUME}}": _load_template("tab_compare_asume.html"),
        "{{TAB_ORDER_PRICING}}": _load_template("tab_order_pricing.html"),
        "{{TAB_LGBM}}": _load_template("tab_lgbm.html"),
        "{{JS_COMMON}}": _load_template("js_common.js"),
        "{{JS_POLY}}": _load_template("js_poly.js"),
        "{{JS_ORDERBOOKS}}": _load_template("js_orderbooks.js"),
        "{{JS_COMPARE_ASUME}}": _load_template("js_compare_asume.js"),
        "{{JS_BACKTEST}}": "",  # included in js_common.js
        "{{JS_LIGHTGBM}}": _load_template("js_lgbm.js"),
    }
    for key, val in replacements.items():
        base = base.replace(key, val)
    return base


_admin_html_cache: str | None = None


@app.get("/", response_class=HTMLResponse)
async def admin_panel():
    global _admin_html_cache
    if _admin_html_cache is None:
        _admin_html_cache = _build_admin_html()
    return _admin_html_cache


# Force rebuild on next request (useful during dev)
@app.get("/api/admin/reload")
async def admin_reload():
    global _admin_html_cache
    _admin_html_cache = None
    return {"status": "cache cleared — next GET / will rebuild"}


