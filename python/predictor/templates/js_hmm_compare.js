// ==================== HMM COMPARE ====================

let hmmCompareInitialized = false;
let hmmCompareRuns = [];
let hmmCompareResults = [];

async function initHmmCompare() {
  if (hmmCompareInitialized) return;
  hmmCompareInitialized = true;
  await hmmCompareLoadRuns();
  hmmCompareRun();
}

async function hmmCompareLoadRuns() {
  const select = document.getElementById('hc-run');
  if (!select) return;
  select.innerHTML = '<option value="">All runs</option>';
  try {
    const res = await fetch(API + '/api/super_backtest/list?limit=200');
    const runs = await res.json();
    if (!Array.isArray(runs) || runs.length === 0) {
      select.innerHTML = '<option value="">No runs available</option>';
      return;
    }
    hmmCompareRuns = runs;
    const opts = ['<option value="">All runs</option>'];
    for (const run of runs) {
      const label = `#${run.id} · ${run.strategy} · h${run.horizon}`;
      opts.push(`<option value="${run.id}">${label}</option>`);
    }
    select.innerHTML = opts.join('');
  } catch (e) {
    console.error('Failed to load runs', e);
    select.innerHTML = '<option value="">Failed to load runs</option>';
  }
}

async function hmmCompareRun() {
  const btn = document.getElementById('hc-run-btn');
  const status = document.getElementById('hc-status');
  if (btn) btn.disabled = true;
  if (status) status.textContent = 'Loading...';

  const params = new URLSearchParams();
  const runId = document.getElementById('hc-run')?.value || '';
  const minTotal = parseInt(document.getElementById('hc-min-total')?.value || '', 10);
  const minTaken = parseInt(document.getElementById('hc-min-taken')?.value || '', 10);
  const minWin = parseFloat(document.getElementById('hc-min-winrate')?.value || '');
  let limit = parseInt(document.getElementById('hc-limit')?.value || '150', 10);
  if (runId) params.append('super_run_id', runId);
  if (Number.isFinite(minTotal) && minTotal > 0) params.append('min_total_signals', String(minTotal));
  if (Number.isFinite(minTaken) && minTaken > 0) params.append('min_taken_signals', String(minTaken));
  if (Number.isFinite(minWin) && minWin > 0) params.append('min_winrate', String(minWin));
  if (!Number.isFinite(limit) || limit < 1) limit = 150;
  limit = Math.min(Math.max(limit, 1), 500);
  params.append('limit', String(limit));

  try {
    const res = await fetch(API + '/api/super_backtest/hmm_sweep_results?' + params.toString());
    const data = await res.json();
    if (!Array.isArray(data)) {
      throw new Error('Unexpected response');
    }
    hmmCompareResults = data;
    hmmCompareRenderSummary(data);
    hmmCompareRenderTable(data);
    if (status) status.textContent = `Loaded ${data.length} combos`;
  } catch (e) {
    console.error('Failed to load sweep results', e);
    if (status) status.textContent = 'Failed to load results';
    const body = document.getElementById('hc-results-body');
    if (body) body.innerHTML = '<tr><td colspan="8" class="p-4 text-center text-red-300">' + e + '</td></tr>';
  } finally {
    if (btn) btn.disabled = false;
  }
}

function hmmCompareRenderSummary(results) {
  const wrap = document.getElementById('hc-summary');
  if (!wrap) return;
  if (!Array.isArray(results) || results.length === 0) {
    wrap.innerHTML = '<div class="text-xs text-slate-500">No results match the current filters. Try lowering thresholds.</div>';
    const noteEmpty = document.getElementById('hc-table-note');
    if (noteEmpty) noteEmpty.textContent = '0 results';
    return;
  }

  const bestImp = results.reduce((best, cur) => {
    if (cur.improvement == null) return best;
    if (!best || cur.improvement > best.improvement) return cur;
    return best;
  }, null);
  const bestWr = results.reduce((best, cur) => {
    if (cur.filtered_winrate == null) return best;
    if (!best || cur.filtered_winrate > best.filtered_winrate) return cur;
    return best;
  }, null);
  const avgTaken = results.reduce((sum, cur) => sum + (cur.filtered_trades || 0), 0) / results.length;
  const avgSkip = results.reduce((sum, cur) => sum + (cur.trades_skipped || 0), 0) / results.length;

  const cards = [];
  if (bestImp) {
    cards.push(`
      <div class="p-3 rounded border border-emerald-500/30 bg-emerald-500/5">
        <div class="text-[11px] uppercase tracking-wide text-emerald-300 mb-1">Best Δ vs Baseline</div>
        <div class="text-2xl font-semibold text-emerald-200">${bestImp.improvement.toFixed(2)}%</div>
        <div class="text-[11px] text-slate-400">Sweep #${bestImp.sweep_id} · Combo ${bestImp.combo_index}</div>
      </div>`);
  }
  if (bestWr) {
    cards.push(`
      <div class="p-3 rounded border border-indigo-500/30 bg-indigo-500/5">
        <div class="text-[11px] uppercase tracking-wide text-indigo-300 mb-1">Top Filtered Winrate</div>
        <div class="text-2xl font-semibold text-indigo-200">${bestWr.filtered_winrate.toFixed(2)}%</div>
        <div class="text-[11px] text-slate-400">${bestWr.filtered_trades || 0} taken · h${bestWr.horizon}</div>
      </div>`);
  }
  cards.push(`
    <div class="p-3 rounded border border-slate-600 bg-slate-800/60">
      <div class="text-[11px] uppercase tracking-wide text-slate-300 mb-1">Avg Taken Signals</div>
      <div class="text-2xl font-semibold text-white">${avgTaken.toFixed(1)}</div>
      <div class="text-[11px] text-slate-400">Avg skipped ${avgSkip.toFixed(1)}</div>
    </div>`);
  cards.push(`
    <div class="p-3 rounded border border-slate-600 bg-slate-800/60">
      <div class="text-[11px] uppercase tracking-wide text-slate-300 mb-1">Result Count</div>
      <div class="text-2xl font-semibold text-white">${results.length}</div>
      <div class="text-[11px] text-slate-400">Sorted by Δ then winrate</div>
    </div>`);

  wrap.innerHTML = cards.join('');
  const note = document.getElementById('hc-table-note');
  if (note) note.textContent = `Showing ${results.length} combo(s)`;
}

function hmmCompareRenderTable(results) {
  const body = document.getElementById('hc-results-body');
  if (!body) return;
  if (!Array.isArray(results) || results.length === 0) {
    body.innerHTML = '<tr><td colspan="8" class="p-4 text-center text-slate-500">No results match filters.</td></tr>';
    return;
  }

  body.innerHTML = results.map(r => {
    const imp = r.improvement;
    const impStr = imp == null ? '—' : (imp >= 0 ? '+' : '') + imp.toFixed(2) + '%';
    const impClass = imp == null ? 'text-slate-400' : (imp >= 0 ? 'text-emerald-300' : 'text-rose-300');
    const baseWr = r.baseline_winrate != null ? r.baseline_winrate.toFixed(2) + '%' : '—';
    const filtWr = r.filtered_winrate != null ? r.filtered_winrate.toFixed(2) + '%' : '—';
    const features = Array.isArray(r.features) ? r.features : [];
    const thresholds = `Good ≥ ${r.good_threshold}% · Bad ≤ ${r.bad_threshold}% · P(bad) > ${r.filter_threshold}`;
    const sweepLabel = `#${r.sweep_id} ${r.sweep_name || ''}`.trim();
    const runLabel = `Run #${r.super_run_id} · ${r.strategy} · h${r.horizon}`;
    return `<tr class="border-b border-slate-800">
      <td class="p-2">
        <div class="text-[12px] font-semibold text-amber-200">${sweepLabel}</div>
        <div class="text-[11px] text-slate-400">${r.n_states} states · ${r.fit_mode}</div>
      </td>
      <td class="p-2 text-[11px] text-slate-300">${runLabel}<div class="text-slate-500">${r.test_start || ''} → ${r.test_end || ''}</div></td>
      <td class="p-2 text-[11px] text-slate-300">${thresholds}</td>
      <td class="p-2 text-center">
        <div>${baseWr}</div>
        <div class="text-[10px] text-slate-500">${r.baseline_trades || 0} signals</div>
      </td>
      <td class="p-2 text-center">
        <div class="text-emerald-300 font-semibold">${filtWr}</div>
        <div class="text-[10px] text-slate-500">${r.filtered_trades || 0} taken · ${r.trades_skipped || 0} skipped</div>
      </td>
      <td class="p-2 text-center ${impClass} font-semibold">${impStr}</td>
      <td class="p-2 font-mono text-[10px] text-slate-400">${features.slice(0,6).join(', ')}${features.length>6?'…':''}</td>
      <td class="p-2 text-center">
        <button class="btn btn-amber text-[10px] py-0.5 px-2" onclick="hmmCompareOpenSweep(${r.super_run_id}, ${r.sweep_id})">Inspect</button>
      </td>
    </tr>`;
  }).join('');
}

async function hmmCompareOpenSweep(superRunId, sweepId) {
  switchTab('super_backtest');
  try {
    await showSuperDetails(superRunId);
    showHmmSweepDetail(sweepId);
  } catch (e) {
    console.error('Failed to open sweep detail', e);
  }
}
