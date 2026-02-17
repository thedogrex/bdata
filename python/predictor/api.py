import json
import time
import asyncio
import traceback
from typing import Optional
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from predictor.backtester import run_backtest
from predictor.strategies import list_strategies, STRATEGY_REGISTRY
from predictor.db_history import (
    save_backtest_run, get_history, get_history_detail,
    delete_run, clear_history,
    get_bruteforce_sessions, get_bruteforce_session_by_id, get_best_runs,
)
from predictor.bruteforce import run_bruteforce, resume_bruteforce, get_default_grid, build_combos
from predictor.task_manager import task_mgr

app = FastAPI(title="Candle Predictor & Backtester", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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


# ==================== API ROUTES ====================

@app.get("/api/strategies")
async def api_list_strategies():
    return list_strategies()


@app.post("/api/backtest")
async def api_run_backtest(req: BacktestRequest):
    """Queue a backtest task. Returns task_id immediately."""
    async def _run(progress):
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
        for idx, strategy_name in enumerate(strats):
            await progress.check_pause_cancel()
            progress.update(idx, len(strats), f"Running {strategy_name} ({idx+1}/{len(strats)})")
            try:
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
):
    return await get_history(limit, strategy, min_accuracy, bruteforce_id)


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


@app.get("/api/best")
async def api_best_runs(limit: int = Query(20), horizon: int = Query(1)):
    return await get_best_runs(limit, horizon)


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


# ==================== ADMIN PANEL ====================

@app.get("/", response_class=HTMLResponse)
async def admin_panel():
    return ADMIN_HTML


ADMIN_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Candle Predictor v3 — Admin Panel</title>
<script src="https://cdn.tailwindcss.com"></script>
<style>
  body{background:#0f172a;color:#e2e8f0;font-family:'Inter',system-ui,sans-serif}
  .card{background:#1e293b;border:1px solid #334155;border-radius:12px}
  .btn{padding:6px 16px;border-radius:8px;font-size:13px;font-weight:600;cursor:pointer;border:none;color:#fff;transition:background .15s}
  .btn-primary{background:#3b82f6}.btn-primary:hover{background:#2563eb}
  .btn-green{background:#10b981}.btn-green:hover{background:#059669}
  .btn-red{background:#ef4444}.btn-red:hover{background:#dc2626}
  .btn-purple{background:#8b5cf6}.btn-purple:hover{background:#7c3aed}
  .btn-amber{background:#f59e0b}.btn-amber:hover{background:#d97706}
  .btn-slate{background:#475569}.btn-slate:hover{background:#64748b}
  .btn:disabled{opacity:.5;cursor:not-allowed}
  select,input,textarea{background:#0f172a;border:1px solid #475569;color:#e2e8f0;border-radius:8px;padding:8px 12px}
  select:focus,input:focus,textarea:focus{outline:none;border-color:#3b82f6}
  .accuracy-good{color:#10b981}.accuracy-ok{color:#f59e0b}.accuracy-bad{color:#ef4444}
  .tab-active{border-bottom:2px solid #3b82f6;color:#3b82f6}
  table{width:100%;border-collapse:collapse}
  th{text-align:left;padding:8px 10px;border-bottom:1px solid #334155;color:#94a3b8;font-weight:600;font-size:12px}
  td{padding:8px 10px;border-bottom:1px solid #1e293b;font-size:13px}
  tr:hover td{background:rgba(30,41,59,.5)}
  .badge{padding:2px 8px;border-radius:9999px;font-size:11px;font-weight:600}
  .badge-up{background:#064e3b;color:#6ee7b7}.badge-down{background:#7f1d1d;color:#fca5a5}
  .badge-bf{background:#4c1d95;color:#c4b5fd}.badge-run{background:#1e3a5f;color:#93c5fd}
  .badge-pause{background:#78350f;color:#fcd34d}.badge-err{background:#7f1d1d;color:#fca5a5}
  .badge-queue{background:#334155;color:#94a3b8}.badge-done{background:#064e3b;color:#6ee7b7}
  .badge-cancel{background:#374151;color:#9ca3af}
  .progress-bar{background:#334155;border-radius:6px;height:20px;overflow:hidden;position:relative}
  .progress-fill{height:100%;border-radius:6px;transition:width .3s;background:linear-gradient(90deg,#3b82f6,#10b981)}
  .progress-text{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:600;color:#fff}
  .tooltip{position:relative;display:inline-block}
  .tooltip .tt{visibility:hidden;width:300px;background:#0f172a;color:#e2e8f0;text-align:left;border-radius:6px;padding:8px 10px;position:absolute;z-index:1000;bottom:125%;left:50%;margin-left:-150px;opacity:0;transition:opacity .2s;font-size:11px;line-height:1.4;box-shadow:0 4px 12px rgba(0,0,0,.6);border:1px solid #334155}
  .tooltip:hover .tt{visibility:visible;opacity:1}
</style>
</head>
<body class="min-h-screen">
<div class="max-w-7xl mx-auto px-4 py-6">

  <!-- ===== LIVE TASK BAR ===== -->
  <div id="taskbar" class="hidden card p-4 mb-4">
    <div class="flex items-center justify-between mb-2">
      <div class="flex items-center gap-3">
        <span id="tb-status" class="badge badge-run">running</span>
        <span id="tb-label" class="text-sm font-medium"></span>
      </div>
      <div class="flex gap-2" id="tb-actions"></div>
    </div>
    <div class="progress-bar mb-1">
      <div id="tb-fill" class="progress-fill" style="width:0%"></div>
      <div id="tb-pct" class="progress-text">0%</div>
    </div>
    <div class="flex justify-between text-xs text-slate-400 mt-1">
      <span id="tb-phase"></span>
      <span id="tb-time"></span>
    </div>
  </div>

  <!-- ===== QUEUE BAR ===== -->
  <div id="queuebar" class="hidden card p-4 mb-4">
    <div class="flex items-center justify-between mb-2">
      <span class="text-sm font-semibold">Queue (<span id="q-count">0</span>)</span>
      <button onclick="clearQueue()" class="btn btn-red text-xs">Clear Queue</button>
    </div>
    <div id="q-list"></div>
  </div>

  <!-- Header -->
  <div class="flex items-center justify-between mb-6">
    <div>
      <h1 class="text-2xl font-bold text-white">Candle Predictor <span class="text-xs text-slate-400">v3</span></h1>
      <p class="text-slate-400 text-sm mt-1">Moving-window backtest &bull; brute-force &bull; task queue</p>
    </div>
  </div>

  <!-- Tabs -->
  <div class="flex gap-5 mb-6 border-b border-slate-700 pb-0">
    <button onclick="switchTab('backtest')" id="tab-backtest" class="pb-3 px-1 text-sm font-medium tab-active cursor-pointer">Backtest</button>
    <button onclick="switchTab('compare')" id="tab-compare" class="pb-3 px-1 text-sm font-medium text-slate-400 cursor-pointer">Compare</button>
    <button onclick="switchTab('bruteforce')" id="tab-bruteforce" class="pb-3 px-1 text-sm font-medium text-slate-400 cursor-pointer">Brute Force</button>
    <button onclick="switchTab('history')" id="tab-history" class="pb-3 px-1 text-sm font-medium text-slate-400 cursor-pointer">History</button>
    <button onclick="switchTab('best')" id="tab-best" class="pb-3 px-1 text-sm font-medium text-slate-400 cursor-pointer">Best Runs</button>
  </div>

  <!-- ============ TAB: BACKTEST ============ -->
  <div id="panel-backtest">
    <div class="card p-6 mb-6">
      <h2 class="text-lg font-semibold mb-4">Run Moving-Window Backtest</h2>
      <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
        <div><label class="block text-xs text-slate-400 mb-1">Strategy <span class="tooltip cursor-help text-blue-400" id="bt-strategy-info">&#9432;<span class="tt"></span></span></label><select id="bt-strategy" class="w-full"></select></div>
        <div><label class="block text-xs text-slate-400 mb-1">Train Start</label><input type="date" id="bt-train-start" value="2022-01-01" class="w-full"></div>
        <div><label class="block text-xs text-slate-400 mb-1">Train End</label><input type="date" id="bt-train-end" value="2025-06-30" class="w-full"></div>
        <div><label class="block text-xs text-slate-400 mb-1">Test Start</label><input type="date" id="bt-test-start" value="2025-07-01" class="w-full"></div>
        <div><label class="block text-xs text-slate-400 mb-1">Test End</label><input type="date" id="bt-test-end" value="2025-12-31" class="w-full"></div>
        <div><label class="block text-xs text-slate-400 mb-1">Horizons</label><input type="text" id="bt-horizons" value="1" class="w-full" placeholder="1,2,3"></div>
        <div><label class="block text-xs text-slate-400 mb-1">Window Size</label><input type="number" id="bt-window" value="5000" class="w-full"></div>
        <div><label class="block text-xs text-slate-400 mb-1">Retrain Every</label><input type="number" id="bt-retrain" value="500" class="w-full"></div>
      </div>
      <details class="mt-2"><summary class="text-xs text-slate-400 cursor-pointer">Custom Params (JSON) <span class="tooltip cursor-help text-blue-400" id="bt-params-info">&#9432;<span class="tt"></span></span></summary>
        <textarea id="bt-params" rows="3" class="w-full mt-2 text-xs font-mono" placeholder='{"n_estimators":300}'></textarea>
        <div class="flex gap-2 mt-1" id="bt-presets"></div>
      </details>
      <div id="strategy-desc" class="mt-2 text-xs text-slate-500 italic"></div>
      <div id="strategy-ref" class="hidden mt-3 p-3 rounded-lg text-xs" style="background:#0f172a;border:1px solid #334155"></div>
      <button onclick="runBacktest()" id="btn-run" class="btn btn-green mt-4">Run Backtest</button>
    </div>
    <div id="bt-results" class="hidden"></div>
  </div>

  <!-- ============ TAB: COMPARE ============ -->
  <div id="panel-compare" class="hidden">
    <div class="card p-6 mb-6">
      <h2 class="text-lg font-semibold mb-4">Compare All Strategies</h2>
      <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
        <div><label class="block text-xs text-slate-400 mb-1">Train Start</label><input type="date" id="cmp-train-start" value="2022-01-01" class="w-full"></div>
        <div><label class="block text-xs text-slate-400 mb-1">Train End</label><input type="date" id="cmp-train-end" value="2025-06-30" class="w-full"></div>
        <div><label class="block text-xs text-slate-400 mb-1">Test Start</label><input type="date" id="cmp-test-start" value="2025-07-01" class="w-full"></div>
        <div><label class="block text-xs text-slate-400 mb-1">Test End</label><input type="date" id="cmp-test-end" value="2025-12-31" class="w-full"></div>
        <div><label class="block text-xs text-slate-400 mb-1">Window Size</label><input type="number" id="cmp-window" value="5000" class="w-full"></div>
        <div><label class="block text-xs text-slate-400 mb-1">Retrain Every</label><input type="number" id="cmp-retrain" value="500" class="w-full"></div>
      </div>
      <button onclick="runCompare()" id="btn-compare" class="btn btn-green">Compare All</button>
    </div>
    <div id="cmp-results" class="hidden"></div>
  </div>

  <!-- ============ TAB: BRUTE FORCE ============ -->
  <div id="panel-bruteforce" class="hidden">
    <div class="card p-6 mb-6">
      <h2 class="text-lg font-semibold mb-4">Brute-Force Hyperparameter Search</h2>
      <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
        <div><label class="block text-xs text-slate-400 mb-1">Strategy <span class="tooltip cursor-help text-blue-400" id="bf-strategy-info">&#9432;<span class="tt"></span></span></label><select id="bf-strategy" class="w-full"></select></div>
        <div><label class="block text-xs text-slate-400 mb-1">Horizon</label><input type="number" id="bf-horizon" value="1" min="1" max="5" class="w-full"></div>
        <div><label class="block text-xs text-slate-400 mb-1">Max Combos</label><input type="number" id="bf-max" value="50" class="w-full"></div>
        <div><label class="block text-xs text-slate-400 mb-1">Window Size</label><input type="number" id="bf-window" value="5000" class="w-full"></div>
        <div><label class="block text-xs text-slate-400 mb-1">Train Start</label><input type="date" id="bf-train-start" value="2022-01-01" class="w-full"></div>
        <div><label class="block text-xs text-slate-400 mb-1">Train End</label><input type="date" id="bf-train-end" value="2025-06-30" class="w-full"></div>
        <div><label class="block text-xs text-slate-400 mb-1">Test Start</label><input type="date" id="bf-test-start" value="2025-07-01" class="w-full"></div>
        <div><label class="block text-xs text-slate-400 mb-1">Test End</label><input type="date" id="bf-test-end" value="2025-12-31" class="w-full"></div>
      </div>
      <div class="mt-2">
        <label class="block text-xs text-slate-400 mb-1">Param Grid (JSON) — <button onclick="loadDefaultGrid()" class="text-blue-400 underline text-xs">Load Default</button> <span class="tooltip cursor-help text-blue-400" id="bf-grid-info">&#9432;<span class="tt"></span></span></label>
        <textarea id="bf-grid" rows="6" class="w-full text-xs font-mono"></textarea>
        <div id="bf-combos" class="text-xs text-slate-400 mt-1"></div>
      </div>
      <button onclick="runBruteforce()" id="btn-bf" class="btn btn-purple mt-4">Queue Brute Force</button>
    </div>
    <div class="card p-6"><h3 class="font-semibold mb-3">Brute-Force Sessions (DB)</h3><div id="bf-sessions"></div></div>
  </div>

  <!-- ============ TAB: HISTORY ============ -->
  <div id="panel-history" class="hidden">
    <div class="card p-6 mb-4">
      <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div><label class="block text-xs text-slate-400 mb-1">Strategy</label><select id="hist-strategy" class="w-full"><option value="">All</option></select></div>
        <div><label class="block text-xs text-slate-400 mb-1">Min Accuracy %</label><input type="number" id="hist-min-acc" value="" placeholder="53" class="w-full"></div>
        <div><label class="block text-xs text-slate-400 mb-1">Limit</label><input type="number" id="hist-limit" value="50" class="w-full"></div>
        <div class="flex items-end gap-2">
          <button onclick="loadHistory()" class="btn btn-primary">Search</button>
          <button onclick="clearAllHistory()" class="btn btn-red">Clear All</button>
        </div>
      </div>
    </div>
    <div id="history-list"></div>
  </div>

  <!-- ============ TAB: BEST ============ -->
  <div id="panel-best" class="hidden">
    <div class="card p-6 mb-4">
      <div class="flex gap-4 items-end">
        <div><label class="block text-xs text-slate-400 mb-1">Horizon</label><input type="number" id="best-horizon" value="1" min="1" max="5" class="w-32"></div>
        <div><label class="block text-xs text-slate-400 mb-1">Limit</label><input type="number" id="best-limit" value="20" class="w-32"></div>
        <button onclick="loadBest()" class="btn btn-amber">Load Best</button>
      </div>
    </div>
    <div id="best-list"></div>
  </div>

</div>

<script>
const API='';
let strategiesData=[];
let pollTimer=null;
let activeTaskId=null;

function fmtTime(s){if(!s||s<=0)return'--';const m=Math.floor(s/60);const sec=Math.floor(s%60);return m>0?`${m}m ${sec}s`:`${sec}s`}
function accClass(a){return a>=54?'accuracy-good':a>=51?'accuracy-ok':'accuracy-bad'}
function statusBadge(s){const m={running:'badge-run',paused:'badge-pause',done:'badge-done',error:'badge-err',cancelled:'badge-cancel',queued:'badge-queue'};return `<span class="badge ${m[s]||'badge-queue'}">${s}</span>`}

// ===== POLLING =====
function startPolling(){if(pollTimer)return;pollTimer=setInterval(pollStatus,1500);pollStatus()}
function stopPolling(){if(pollTimer){clearInterval(pollTimer);pollTimer=null}}

async function pollStatus(){
  try{
    const res=await fetch(API+'/api/tasks/status');
    const d=await res.json();
    renderTaskbar(d.current);
    renderQueue(d.queue);
  }catch(e){}
}

function renderTaskbar(t){
  const bar=document.getElementById('taskbar');
  if(!t||t.status==='done'||t.status==='cancelled'||t.status==='error'){
    bar.classList.add('hidden');
    if(activeTaskId && t && (t.status==='done'||t.status==='error')){
      onTaskDone(activeTaskId, t.status);
      activeTaskId=null;
    }
    if(!t) activeTaskId=null;
    return;
  }
  bar.classList.remove('hidden');
  activeTaskId=t.task_id;
  document.getElementById('tb-label').textContent=t.label;
  document.getElementById('tb-status').innerHTML=statusBadge(t.status);
  const pct=t.total>0?Math.round(t.current/t.total*100):0;
  document.getElementById('tb-fill').style.width=pct+'%';
  document.getElementById('tb-pct').textContent=pct+'%';
  document.getElementById('tb-phase').textContent=t.phase||'';
  const elapsed=fmtTime(t.elapsed_sec);
  const eta=fmtTime(t.eta_sec);
  document.getElementById('tb-time').textContent=`Elapsed: ${elapsed} | ETA: ${eta}`;

  // Actions
  const acts=document.getElementById('tb-actions');
  if(t.status==='running'){
    acts.innerHTML=`<button onclick="taskAction('${t.task_id}','pause')" class="btn btn-amber text-xs">Pause</button><button onclick="taskAction('${t.task_id}','cancel')" class="btn btn-red text-xs">Cancel</button>`;
  }else if(t.status==='paused'){
    acts.innerHTML=`<button onclick="taskAction('${t.task_id}','resume')" class="btn btn-green text-xs">Resume</button><button onclick="taskAction('${t.task_id}','cancel')" class="btn btn-red text-xs">Cancel</button>`;
  }else{
    acts.innerHTML='';
  }
}

function renderQueue(q){
  const bar=document.getElementById('queuebar');
  if(!q||!q.length){bar.classList.add('hidden');return}
  bar.classList.remove('hidden');
  document.getElementById('q-count').textContent=q.length;
  let html='';
  q.forEach(t=>{
    html+=`<div class="flex items-center justify-between py-1 border-b border-slate-700 last:border-0">
      <span class="text-xs">${statusBadge(t.status)} <span class="ml-2">${t.label}</span></span>
      <button onclick="removeFromQueue('${t.task_id}')" class="text-red-400 text-xs hover:underline">remove</button>
    </div>`;
  });
  document.getElementById('q-list').innerHTML=html;
}

async function taskAction(id,action){
  await fetch(API+`/api/tasks/${id}/${action}`,{method:'POST'});
  pollStatus();
}
async function removeFromQueue(id){
  await fetch(API+`/api/tasks/queue/${id}`,{method:'DELETE'});
  pollStatus();
}
async function clearQueue(){
  if(!confirm('Clear entire queue?'))return;
  await fetch(API+'/api/tasks/queue',{method:'DELETE'});
  pollStatus();
}

async function onTaskDone(taskId, status){
  if(status==='error')return;
  try{
    const res=await fetch(API+'/api/tasks/'+taskId+'/result');
    if(!res.ok)return;
    const data=await res.json();
    // Auto-render result based on type
    const p=await(await fetch(API+'/api/tasks/'+taskId)).json();
    if(p.task_type==='backtest'){renderResult(data,'bt-results')}
    else if(p.task_type==='compare'){renderCompare(data)}
    else if(p.task_type==='bruteforce'){renderBfResult(data)}
  }catch(e){console.error(e)}
}

// ===== INIT =====
async function init(){
  const res=await fetch(API+'/api/strategies');
  strategiesData=await res.json();
  ['bt-strategy','bf-strategy'].forEach(id=>{
    const sel=document.getElementById(id);sel.innerHTML='';
    strategiesData.forEach(s=>{const o=document.createElement('option');o.value=s.name;o.textContent=s.name;sel.appendChild(o)});
  });
  const hsel=document.getElementById('hist-strategy');
  strategiesData.forEach(s=>{const o=document.createElement('option');o.value=s.name;o.textContent=s.name;hsel.appendChild(o)});
  document.getElementById('bt-strategy').addEventListener('change',updateDesc);
  document.getElementById('bf-strategy').addEventListener('change',()=>{loadDefaultGrid();updateDesc()});
  updateDesc();
  loadDefaultGrid();
  startPolling();
}

function updateDesc(){
  const n=document.getElementById('bt-strategy').value;
  const s=strategiesData.find(x=>x.name===n);
  document.getElementById('strategy-desc').textContent=s?s.description:'';

  // Tooltip for strategy info icons
  ['bt-strategy-info','bf-strategy-info'].forEach(id=>{
    const el=document.getElementById(id);if(!el)return;
    const tt=el.querySelector('.tt');if(!tt)return;
    if(s){
      const params=s.param_docs||{};
      const lines=Object.entries(params).map(([k,v])=>`<b>${k}:</b> ${v}`).join('<br>');
      const training=s.needs_training?'<br><b>Training:</b> Yes (retrains every N candles)':'<br><b>Training:</b> No (rule-based, instant)';
      const notes=s.recommended?.notes?`<br><b>Notes:</b> ${s.recommended.notes}`:'';
      tt.innerHTML=`${s.description}${training}${notes}<br><br><b>Params:</b><br>${lines}`;
    }else{tt.innerHTML=''}
  });
  ['bt-params-info','bf-grid-info'].forEach(id=>{
    const el=document.getElementById(id);if(!el)return;
    const tt=el.querySelector('.tt');if(!tt)return;
    if(s){
      const params=s.param_docs||{};
      tt.innerHTML=Object.entries(params).map(([k,v])=>`<b>${k}:</b> ${v}`).join('<br>');
    }else{tt.innerHTML=''}
  });

  // Strategy reference panel
  const ref=document.getElementById('strategy-ref');
  if(s){
    ref.classList.remove('hidden');
    const rec=s.recommended||{};
    const training=s.needs_training?'<span class="text-amber-400">Yes</span> (retrains every N candles — XGBoost training time scales with n_estimators)':'<span class="text-green-400">No</span> (rule-based, instant prediction)';
    let html=`<div class="flex items-center gap-2 mb-2"><b class="text-slate-200">Strategy Reference: ${s.name}</b>${s.needs_training?'<span class="badge badge-amber" style="background:#78350f;color:#fcd34d">Requires Training</span>':'<span class="badge badge-done">No Training</span>'}</div>`;
    html+=`<div class="mb-2"><b>Training:</b> ${training}</div>`;
    if(rec.notes) html+=`<div class="mb-2 text-slate-300"><b>Notes:</b> ${rec.notes}</div>`;

    // All params table
    html+=`<div class="mb-2"><b>All Parameters:</b></div>`;
    html+=`<table class="mb-3"><thead><tr><th>Param</th><th>Default</th><th>Description</th></tr></thead><tbody>`;
    const dp=s.default_params||{};
    const pd=s.param_docs||{};
    for(const[k,v] of Object.entries(dp)){
      const val=typeof v==='object'?JSON.stringify(v):String(v);
      html+=`<tr><td class="font-mono text-blue-300">${k}</td><td class="font-mono">${val}</td><td class="text-slate-400">${pd[k]||''}</td></tr>`;
    }
    html+=`</tbody></table>`;

    // Default params JSON
    html+=`<details class="mb-2"><summary class="cursor-pointer text-blue-400"><b>Default Params JSON (copy-paste)</b></summary>`;
    html+=`<pre class="mt-1 p-2 rounded text-xs overflow-x-auto" style="background:#1e293b">${JSON.stringify(dp,null,2)}</pre></details>`;

    // Presets
    const presetKeys=Object.keys(rec).filter(k=>k.endsWith('_preset'));
    if(presetKeys.length){
      html+=`<div class="mb-1"><b>Presets:</b></div><div class="flex flex-wrap gap-2 mb-2">`;
      presetKeys.forEach(pk=>{
        const label=pk.replace('_preset','').replace(/_/g,' ');
        html+=`<button onclick='applyPreset(${JSON.stringify(JSON.stringify(rec[pk]))})' class="btn btn-slate text-xs">${label}</button>`;
      });
      html+=`</div>`;
    }

    // BF recommended params
    if(rec.brute_force_include){
      html+=`<div class="text-slate-400"><b>Recommended brute-force params:</b> ${rec.brute_force_include.join(', ')}</div>`;
    }
    ref.innerHTML=html;
  }else{ref.classList.add('hidden')}

  // Preset buttons
  const presets=document.getElementById('bt-presets');
  if(s && s.recommended){
    const rec=s.recommended;
    const presetKeys=Object.keys(rec).filter(k=>k.endsWith('_preset'));
    if(presetKeys.length){
      let html='<span class="text-xs text-slate-400">Presets:</span> ';
      presetKeys.forEach(pk=>{
        const label=pk.replace('_preset','').replace(/_/g,' ');
        html+=`<button onclick='applyPreset(${JSON.stringify(JSON.stringify(rec[pk]))})' class="btn btn-slate text-xs">${label}</button> `;
      });
      presets.innerHTML=html;
    }else{presets.innerHTML=''}
  }else{presets.innerHTML=''}
}

function applyPreset(jsonStr){
  document.getElementById('bt-params').value=JSON.stringify(JSON.parse(jsonStr),null,2);
}

// ===== TABS =====
const TABS=['backtest','compare','bruteforce','history','best'];
function switchTab(tab){
  TABS.forEach(t=>{
    document.getElementById('panel-'+t).classList.toggle('hidden',t!==tab);
    const b=document.getElementById('tab-'+t);
    if(t===tab){b.classList.add('tab-active');b.classList.remove('text-slate-400')}
    else{b.classList.remove('tab-active');b.classList.add('text-slate-400')}
  });
  if(tab==='history')loadHistory();
  if(tab==='best')loadBest();
  if(tab==='bruteforce')loadBfSessions();
}

// ===== BACKTEST =====
async function runBacktest(){
  let params=null;
  const pt=document.getElementById('bt-params').value.trim();
  if(pt){try{params=JSON.parse(pt)}catch(e){alert('Invalid JSON');return}}
  const horizons=document.getElementById('bt-horizons').value.split(',').map(x=>parseInt(x.trim())).filter(x=>!isNaN(x));
  const body={strategy:document.getElementById('bt-strategy').value,params,
    train_start:document.getElementById('bt-train-start').value,train_end:document.getElementById('bt-train-end').value,
    test_start:document.getElementById('bt-test-start').value,test_end:document.getElementById('bt-test-end').value,
    horizons,table:'c_5m',window_size:parseInt(document.getElementById('bt-window').value)||5000,
    retrain_every:parseInt(document.getElementById('bt-retrain').value)||500};
  const res=await fetch(API+'/api/backtest',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const data=await res.json();
  if(data.error){alert(data.error);return}
  activeTaskId=data.task_id;
  document.getElementById('bt-results').classList.add('hidden');
}

// ===== RENDER RESULT =====
function renderResult(data,targetId){
  const el=document.getElementById(targetId);el.classList.remove('hidden');
  const ws=data.window_size||'?';const re=data.retrain_every||'?';
  const lt=data.load_time_sec?` | Load: ${data.load_time_sec}s`:'';
  const ft=data.feature_time_sec?` | Features: ${data.feature_time_sec}s`:'';
  let html=`<div class="card p-6 mb-6"><div class="mb-4">
    <h2 class="text-lg font-semibold">${data.strategy} ${data.id?'<span class="text-xs text-slate-400">#'+data.id+'</span>':''}</h2>
    <p class="text-xs text-slate-400">Train: ${data.train_period||''} | Test: ${data.test_period||''}</p>
    <p class="text-xs text-slate-400">Window: ${ws} | Retrain: ${re} | Total: ${data.total_time_sec}s${lt}${ft}</p></div>`;
  for(const[horizon,r]of Object.entries(data.horizons||{})){
    if(r.error){html+=`<div class="text-red-400 mb-4">H${horizon}: ${r.error}</div>`;continue}
    html+=`<div class="mb-6 p-4 rounded-lg" style="background:#0f172a">
      <h3 class="font-semibold mb-3">Horizon ${horizon}</h3>
      <div class="grid grid-cols-2 md:grid-cols-5 gap-4 mb-4">
        <div class="text-center"><div class="text-3xl font-bold ${accClass(r.accuracy_pct)}">${r.accuracy_pct}%</div><div class="text-xs text-slate-400">Accuracy</div></div>
        <div class="text-center"><div class="text-2xl font-bold">${r.signals?.toLocaleString()}</div><div class="text-xs text-slate-400">Signals</div></div>
        <div class="text-center"><div class="text-2xl font-bold text-green-400">${r.correct?.toLocaleString()}</div><div class="text-xs text-slate-400">Correct</div></div>
        <div class="text-center"><div class="text-2xl font-bold text-red-400">${r.wrong?.toLocaleString()}</div><div class="text-xs text-slate-400">Wrong</div></div>
        <div class="text-center"><div class="text-2xl font-bold text-slate-300">${r.skipped?.toLocaleString()}</div><div class="text-xs text-slate-400">Skipped</div></div>
      </div>
      <div class="grid grid-cols-2 gap-4 mb-3">
        <div class="p-2 rounded text-sm" style="background:#1e293b"><span class="badge badge-up">UP</span> ${r.up_predictions} preds, ${r.up_correct} correct (${r.up_accuracy}%)</div>
        <div class="p-2 rounded text-sm" style="background:#1e293b"><span class="badge badge-down">DOWN</span> ${r.down_predictions} preds, ${r.down_correct} correct (${r.down_accuracy}%)</div>
      </div>
      <div class="p-2 rounded text-xs mb-3" style="background:#1e293b">Win streak: <b class="text-green-400">${r.streaks?.max_win_streak||0}</b> | Lose streak: <b class="text-red-400">${r.streaks?.max_lose_streak||0}</b>${r.train_count?` | Trains: <b>${r.train_count}</b> (${r.total_train_time_sec}s) | Predict: ${r.predict_time_sec}s`:''}</div>`;
    if(r.monthly?.length){
      html+=`<details class="mb-2"><summary class="text-xs text-slate-400 cursor-pointer">Monthly (${r.monthly.length})</summary><table class="mt-1"><thead><tr><th>Month</th><th>Total</th><th>Correct</th><th>Acc</th></tr></thead><tbody>`;
      r.monthly.forEach(m=>{html+=`<tr><td>${m.month}</td><td>${m.total}</td><td>${m.correct}</td><td class="${accClass(m.accuracy)}">${m.accuracy}%</td></tr>`});
      html+=`</tbody></table></details>`}
    if(r.confidence_distribution){
      html+=`<details><summary class="text-xs text-slate-400 cursor-pointer">Confidence</summary><div class="grid grid-cols-7 gap-1 mt-1">`;
      Object.entries(r.confidence_distribution).forEach(([k,v])=>{html+=`<div class="text-center p-1 rounded text-xs" style="background:#1e293b"><div class="text-slate-400">${k}</div><div class="font-bold">${v}</div></div>`});
      html+=`</div></details>`}
    html+=`</div>`}
  html+=`</div>`;el.innerHTML=html;
}

// ===== COMPARE =====
async function runCompare(){
  const body={strategies:strategiesData.map(s=>s.name),
    train_start:document.getElementById('cmp-train-start').value,train_end:document.getElementById('cmp-train-end').value,
    test_start:document.getElementById('cmp-test-start').value,test_end:document.getElementById('cmp-test-end').value,
    horizons:[1,2,3],table:'c_5m',window_size:parseInt(document.getElementById('cmp-window').value)||5000,
    retrain_every:parseInt(document.getElementById('cmp-retrain').value)||500};
  const res=await fetch(API+'/api/compare',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const data=await res.json();
  if(data.error){alert(data.error);return}
  activeTaskId=data.task_id;
  document.getElementById('cmp-results').classList.add('hidden');
}
function renderCompare(results){
  if(!Array.isArray(results))return;
  const el=document.getElementById('cmp-results');el.classList.remove('hidden');
  const allH=new Set();results.forEach(r=>{if(r.horizons)Object.keys(r.horizons).forEach(h=>allH.add(h))});
  const horizons=[...allH].sort();
  let html='<div class="card p-6 mb-6"><h2 class="text-lg font-semibold mb-4">Comparison</h2>';
  for(const h of horizons){
    html+=`<h3 class="font-medium mt-4 mb-2">Horizon ${h}</h3>`;
    html+='<table><thead><tr><th>Strategy</th><th>Accuracy</th><th>Signals</th><th>Correct</th><th>Wrong</th><th>Skipped</th><th>W/L Streak</th></tr></thead><tbody>';
    const sorted=[...results].filter(r=>r.horizons&&r.horizons[h]&&!r.horizons[h].error).sort((a,b)=>(b.horizons[h].accuracy_pct||0)-(a.horizons[h].accuracy_pct||0));
    for(const r of sorted){const d=r.horizons[h];html+=`<tr><td class="font-medium">${r.strategy}</td><td class="${accClass(d.accuracy_pct)} font-bold">${d.accuracy_pct}%</td><td>${d.signals?.toLocaleString()}</td><td class="text-green-400">${d.correct?.toLocaleString()}</td><td class="text-red-400">${d.wrong?.toLocaleString()}</td><td>${d.skipped?.toLocaleString()}</td><td>${d.streaks?.max_win_streak||0}/${d.streaks?.max_lose_streak||0}</td></tr>`}
    html+='</tbody></table>'}
  html+='</div>';el.innerHTML=html;
}

// ===== BRUTE FORCE =====
async function loadDefaultGrid(){
  const s=document.getElementById('bf-strategy').value;if(!s)return;
  try{const res=await fetch(API+'/api/bruteforce/grid/'+s);const data=await res.json();
    document.getElementById('bf-grid').value=JSON.stringify(data.grid,null,2);
    document.getElementById('bf-combos').textContent=`Total combos: ${data.total_combos}`}catch(e){}
}
async function runBruteforce(){
  let grid;try{grid=JSON.parse(document.getElementById('bf-grid').value)}catch(e){alert('Invalid grid JSON');return}
  const body={strategy:document.getElementById('bf-strategy').value,param_grid:grid,
    train_start:document.getElementById('bf-train-start').value,train_end:document.getElementById('bf-train-end').value,
    test_start:document.getElementById('bf-test-start').value,test_end:document.getElementById('bf-test-end').value,
    horizon:parseInt(document.getElementById('bf-horizon').value)||1,table:'c_5m',
    window_size:parseInt(document.getElementById('bf-window').value)||5000,
    max_combos:parseInt(document.getElementById('bf-max').value)||50};
  const res=await fetch(API+'/api/bruteforce',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const data=await res.json();
  if(data.error){alert(data.error);return}
  activeTaskId=data.task_id;
}
function renderBfResult(data){
  if(!data||!data.best_accuracy)return;
  // Just reload sessions and switch to history
  loadBfSessions();
}
async function loadBfSessions(){
  try{const res=await fetch(API+'/api/bruteforce/sessions');const data=await res.json();
    const el=document.getElementById('bf-sessions');
    if(!data.length){el.innerHTML='<p class="text-slate-400 text-sm">No sessions yet.</p>';return}
    let html='<table><thead><tr><th>ID</th><th>Strategy</th><th>H</th><th>Combos</th><th>Best</th><th>Status</th><th>Time</th><th>Date</th><th></th></tr></thead><tbody>';
    data.forEach(s=>{
      const canResume=s.status==='paused'||s.status==='running';
      const resumeBtn=canResume?`<button onclick="resumeBf(${s.id})" class="btn btn-green text-xs">Resume</button>`:'';
      const viewBtn=`<button onclick="loadHistory();document.getElementById('hist-strategy').value='';switchTab('history')" class="text-blue-400 text-xs hover:underline ml-1">runs</button>`;
      html+=`<tr><td>${s.id}</td><td class="font-medium">${s.strategy}</td><td>${s.horizon}</td><td>${s.completed}/${s.total_combos}</td><td class="${accClass(s.best_accuracy)} font-bold">${s.best_accuracy}%</td><td>${statusBadge(s.status)}</td><td>${s.total_time_sec}s</td><td class="text-slate-400 text-xs">${s.created_at}</td><td class="flex gap-1">${resumeBtn}${viewBtn}</td></tr>`});
    html+='</tbody></table>';el.innerHTML=html}catch(e){}
}
async function resumeBf(bfId){
  const res=await fetch(API+'/api/bruteforce/resume/'+bfId,{method:'POST'});
  const data=await res.json();
  if(data.error){alert(data.error);return}
  activeTaskId=data.task_id;
  loadBfSessions();
}

// ===== HISTORY =====
async function loadHistory(){
  const strategy=document.getElementById('hist-strategy')?.value||'';
  const minAcc=document.getElementById('hist-min-acc')?.value||'';
  const limit=document.getElementById('hist-limit')?.value||'50';
  let url=API+'/api/history?limit='+limit;
  if(strategy)url+='&strategy='+strategy;if(minAcc)url+='&min_accuracy='+minAcc;
  try{const res=await fetch(url);const data=await res.json();const el=document.getElementById('history-list');
    if(!data.length){el.innerHTML='<div class="card p-6 text-center text-slate-400">No results.</div>';return}

    // Group BF runs by bruteforce_id
    const bfGroups={};const standalone=[];
    data.forEach(r=>{
      if(r.is_bruteforce && r.bruteforce_id){
        if(!bfGroups[r.bruteforce_id])bfGroups[r.bruteforce_id]={runs:[],strategy:r.strategy,bf_id:r.bruteforce_id};
        bfGroups[r.bruteforce_id].runs.push(r);
      }else{standalone.push(r)}
    });

    let html='<div class="card p-6">';

    // Render BF groups first
    const bfIds=Object.keys(bfGroups).sort((a,b)=>b-a);
    bfIds.forEach(bfId=>{
      const g=bfGroups[bfId];
      const best=g.runs.reduce((b,r)=>{
        const acc=Object.values(r.horizons||{}).reduce((m,h)=>Math.max(m,h.accuracy_pct||0),0);
        return acc>b.acc?{acc,r}:b;
      },{acc:0,r:null});
      html+=`<details class="mb-3 p-3 rounded-lg" style="background:#0f172a;border:1px solid #334155">
        <summary class="cursor-pointer flex items-center justify-between">
          <span><span class="badge badge-bf">BF#${bfId}</span> <b class="ml-2">${g.strategy}</b> <span class="text-slate-400 text-xs ml-2">${g.runs.length} runs</span></span>
          <span class="${accClass(best.acc)} font-bold">Best: ${best.acc}%</span>
        </summary>
        <table class="mt-2"><thead><tr><th>ID</th><th>Params</th><th>Win</th><th>Horizons</th><th>Time</th><th></th></tr></thead><tbody>`;
      g.runs.forEach(r=>{
        const hs=Object.entries(r.horizons||{}).map(([h,d])=>d.error?`H${h}:err`:`H${h}:<span class="${accClass(d.accuracy_pct)}">${d.accuracy_pct}%</span>`).join(' | ');
        const ps=JSON.stringify(r.params||{}).substring(0,80);
        html+=`<tr class="cursor-pointer" onclick="showDetail(${r.id})"><td>${r.id}</td><td class="text-xs text-slate-400 max-w-xs truncate">${ps}</td><td>${r.window_size||'?'}</td><td>${hs}</td><td>${r.total_time_sec}s</td><td><button onclick="event.stopPropagation();deleteRun(${r.id})" class="text-red-400 text-xs hover:underline">del</button></td></tr>`});
      html+=`</tbody></table></details>`;
    });

    // Render standalone runs
    if(standalone.length){
      html+=`<table><thead><tr><th>ID</th><th>Strategy</th><th>Test Period</th><th>Win</th><th>Horizons</th><th>Time</th><th>Date</th><th></th></tr></thead><tbody>`;
      standalone.forEach(r=>{
        const hs=Object.entries(r.horizons||{}).map(([h,d])=>d.error?`H${h}:err`:`H${h}:<span class="${accClass(d.accuracy_pct)}">${d.accuracy_pct}%</span>`).join(' | ');
        html+=`<tr class="cursor-pointer" onclick="showDetail(${r.id})"><td>${r.id}</td><td class="font-medium">${r.strategy}</td><td class="text-xs">${r.test_period||''}</td><td>${r.window_size||'?'}</td><td>${hs}</td><td>${r.total_time_sec}s</td><td class="text-slate-400 text-xs">${r.created_at||''}</td><td><button onclick="event.stopPropagation();deleteRun(${r.id})" class="text-red-400 text-xs hover:underline">del</button></td></tr>`});
      html+=`</tbody></table>`;
    }
    html+='</div>';el.innerHTML=html}catch(e){console.error(e)}
}
async function showDetail(id){try{const res=await fetch(API+'/api/history/'+id);const data=await res.json();if(data.error){alert(data.error);return}switchTab('backtest');renderResult(data,'bt-results')}catch(e){alert(e.message)}}
async function deleteRun(id){if(!confirm('Delete #'+id+'?'))return;await fetch(API+'/api/history/'+id,{method:'DELETE'});loadHistory()}
async function clearAllHistory(){if(!confirm('Delete ALL?'))return;await fetch(API+'/api/history',{method:'DELETE'});loadHistory()}

// ===== BEST =====
async function loadBest(){
  const horizon=document.getElementById('best-horizon').value||1;const limit=document.getElementById('best-limit').value||20;
  try{const res=await fetch(API+`/api/best?horizon=${horizon}&limit=${limit}`);const data=await res.json();const el=document.getElementById('best-list');
    if(!data.length){el.innerHTML='<div class="card p-6 text-center text-slate-400">No results.</div>';return}
    let html='<div class="card p-6"><h2 class="text-lg font-semibold mb-4">Top Runs (H'+horizon+')</h2><table><thead><tr><th>#</th><th>Strategy</th><th>Accuracy</th><th>Signals</th><th>Correct</th><th>Wrong</th><th>W/L</th><th>Win</th><th>Params</th></tr></thead><tbody>';
    data.forEach((r,i)=>{const ps=JSON.stringify(r.params||{}).substring(0,60);
      html+=`<tr class="cursor-pointer" onclick="showDetail(${r.id})"><td>${i+1}</td><td class="font-medium">${r.strategy}</td><td class="${accClass(r.accuracy_pct)} font-bold text-lg">${r.accuracy_pct}%</td><td>${r.signals}</td><td class="text-green-400">${r.correct}</td><td class="text-red-400">${r.wrong}</td><td>${r.max_win_streak}/${r.max_lose_streak}</td><td>${r.window_size}</td><td class="text-xs text-slate-400 max-w-xs truncate">${ps}</td></tr>`});
    html+='</tbody></table></div>';el.innerHTML=html}catch(e){console.error(e)}
}

// ===== BOOT =====
init();
</script>
</body>
</html>"""
