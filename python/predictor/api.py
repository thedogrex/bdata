import json
import time
import asyncio
import pathlib
import traceback
from typing import Optional
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from predictor.backtester import (
    run_backtest,
    preload_backtest_data,
    run_backtest_vectorized,
    RULE_BASED_STRATEGIES,
)
from predictor.strategies import list_strategies, STRATEGY_REGISTRY
from predictor.db_history import (
    save_backtest_run, get_history, get_history_detail,
    delete_run, clear_history,
    get_bruteforce_sessions, get_bruteforce_session_by_id, get_best_runs,
    get_runs_by_ids, delete_bruteforce_group, get_bf_runs_paginated,
)
from predictor.bruteforce import run_bruteforce, resume_bruteforce, get_default_grid, build_combos
from predictor.task_manager import task_mgr
from predictor import poly_service

app = FastAPI(title="Candle Predictor & Backtester", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    table: str = "c_5m"
    window_size: int = 5000
    retrain_every: int = 500
    max_combos: int = 100


class SettingsRequest(BaseModel):
    autopredict: bool = False
    strategy: str = "rsi_mean_reversion"
    params: dict | None = None
    window_size: int = 1000


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


# ==================== API ROUTES ====================

@app.get("/api/strategies")
async def api_list_strategies():
    return list_strategies()


# ==================== POLYMARKET (ADMIN) ====================

@app.on_event("startup")
async def startup_event():
    global poly_stop_event
    poly_stop_event = asyncio.Event()
    asyncio.create_task(poly_service.poll_loop(poly_stop_event, orderbook_interval_sec=3))


@app.on_event("shutdown")
async def shutdown_event():
    global poly_stop_event
    if poly_stop_event is not None:
        poly_stop_event.set()


@app.get("/api/poly/markets")
async def api_poly_markets(limit: int = Query(50), offset: int = Query(0)):
    return await poly_service.list_markets(limit=limit, offset=offset)


@app.get("/api/poly/status")
async def api_poly_status():
    return {"active_ts": poly_service.current_active_ts()}


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


@app.get("/api/poly/outcome/{asset_id}/series")
async def api_poly_series(asset_id: str, minutes: int = Query(60), limit: int = Query(2000)):
    return await poly_service.get_price_series(asset_id=asset_id, minutes=minutes, limit=limit)


@app.get("/api/poly/orderbook/{slug}/{asset_id}/analysis")
async def api_poly_orderbook_analysis(slug: str, asset_id: str, minutes: int = Query(60)):
    return await poly_service.get_orderbook_analysis(slug=slug, asset_id=asset_id, minutes=minutes)


@app.get("/api/poly/orderbook/{slug}/{asset_id}/latest")
async def api_poly_orderbook_latest(slug: str, asset_id: str):
    return await poly_service.get_latest_orderbook(slug=slug, asset_id=asset_id)


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


@app.get("/api/poly/prediction/{slug}")
async def api_poly_prediction(slug: str):
    return await poly_service.get_saved_prediction(slug=slug)


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


@app.post("/api/poly/sim/trade")
async def api_poly_sim_trade(req: SimTradeRequest):
    try:
        return await poly_service.create_sim_trade(
            req.slug, req.asset_id, "BUY", req.qty,
            outcome_side=req.outcome_side, requested_price=req.price,
        )
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.get("/api/poly/sim/trades")
async def api_poly_sim_trades(limit: int = Query(200)):
    return await poly_service.list_sim_trades(limit=limit)


@app.get("/api/poly/sim/positions")
async def api_poly_sim_positions(slug: str | None = Query(None)):
    return await poly_service.get_sim_positions(slug=slug)


@app.get("/api/poly/sim/markets_with_positions")
async def api_poly_sim_markets_with_positions():
    return await poly_service.get_sim_markets_with_positions()


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
):
    return await get_best_runs(limit, horizon, signals_min=signals_min, signals_max=signals_max)


@app.post("/api/best/compare")
async def api_best_compare(req: BestCompareRequest):
    return await get_runs_by_ids(run_ids=req.run_ids, horizon=req.horizon)


# ==================== BRUTE FORCE ====================

@app.get("/api/bruteforce/grid/{strategy}")
async def api_default_grid(strategy: str):
    grid = get_default_grid(strategy)
    combos = build_combos(grid) if grid else []
    return {"strategy": strategy, "grid": grid, "total_combos": len(combos)}


@app.post("/api/bruteforce")
async def api_run_bruteforce(req: BruteforceRequest):
    """Queue a brute-force task. Returns task_id immediately."""
    grid = req.param_grid or get_default_grid(req.strategy)
    if not grid:
        return JSONResponse(status_code=400, content={"error": f"No param grid for {req.strategy}"})

    combos_count = len(build_combos(grid))
    actual_count = min(combos_count, req.max_combos)

    async def _run(progress):
        return await run_bruteforce(
            strategy=req.strategy,
            param_grid=grid,
            train_start=req.train_start,
            train_end=req.train_end,
            test_start=req.test_start,
            test_end=req.test_end,
            horizon=req.horizon,
            table=req.table,
            window_size=req.window_size,
            retrain_every=req.retrain_every,
            max_combos=req.max_combos,
            progress=progress,
        )

    label = f"BruteForce {req.strategy} H{req.horizon} ({actual_count} combos)"
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


# ==================== ADMIN PANEL ====================

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
        "{{TAB_ORDERBOOKS}}": _load_template("tab_orderbooks.html"),
        "{{TAB_ANALYTICS}}": _load_template("tab_analytics.html"),
        "{{JS_COMMON}}": _load_template("js_common.js"),
        "{{JS_POLY}}": _load_template("js_poly.js"),
        "{{JS_ORDERBOOKS}}": _load_template("js_orderbooks.js"),
        "{{JS_BACKTEST}}": "",  # included in js_common.js
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
