let rsibInitialized = false;
let rsibSessions = [];
let rsibCurrentSession = null;
let rsibCurrentPage = 1;
let rsibCurrentPages = 1;
let rsibCurrentResults = [];

async function initRsib() {
  if (rsibInitialized) return;
  rsibInitialized = true;
  await rsibLoadDefaultConfig();
  await loadRsibSessions();
}

function _showRsibPeriodFromConfig(cfg) {
  const el = document.getElementById('rsib-period');
  if (!el) return;
  const test = (cfg && (cfg.test_start || cfg.test_end)) ? `${cfg.test_start || '—'} → ${cfg.test_end || '—'}` : '';
  el.textContent = test ? `test: ${test}` : '';
}

async function rsibLoadDefaultConfig() {
  const el = document.getElementById('rsib-config');
  if (!el) return;
  try {
    const res = await fetch(API + '/api/rsib/default_config');
    const data = await res.json();
    el.value = JSON.stringify(data.config || {}, null, 2);
    _showRsibPeriodFromConfig(data.config || {});
  } catch (e) {
    console.error('Failed to load RSIB config', e);
  }
}

async function runRsib() {
  const btn = document.getElementById('btn-rsib-run');
  const status = document.getElementById('rsib-status');
  let cfg;
  try {
    cfg = JSON.parse(document.getElementById('rsib-config').value || '{}');
  } catch (e) {
    alert('Invalid RSIB JSON: ' + e.message);
    return;
  }
  btn.disabled = true;
  if (status) status.textContent = 'Queueing RSIB...';
  try {
    const res = await fetch(API + '/api/rsib/run', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(cfg),
    });
    const data = await res.json();
    if (data.error) {
      alert(data.error);
      if (status) status.textContent = data.error;
      return;
    }
    activeTaskId = data.task_id;
    if (status) status.textContent = data.label || 'Queued';
    await loadRsibSessions();
  } catch (e) {
    if (status) status.textContent = 'Failed: ' + e.message;
  } finally {
    btn.disabled = false;
  }
}

async function loadRsibSessions() {
  const select = document.getElementById('rsib-session');
  if (!select) return;
  try {
    const res = await fetch(API + '/api/rsib/sessions?limit=100');
    const data = await res.json();
    rsibSessions = Array.isArray(data) ? data : [];
    const current = select.value;
    const opts = ['<option value="">Select session</option>'];
    rsibSessions.forEach(s => {
      opts.push(`<option value="${s.id}">#${s.id} · ${s.status} · best ${fmtMoney(s.best_profit)}</option>`);
    });
    select.innerHTML = opts.join('');
    if (current && rsibSessions.some(s => String(s.id) === String(current))) {
      select.value = current;
    } else if (!current && rsibSessions[0]) {
      select.value = String(rsibSessions[0].id);
    }
    if (select.value) {
      await rsibChangeSession();
    }
  } catch (e) {
    console.error('Failed to load RSIB sessions', e);
  }
}

async function rsibChangeSession() {
  const select = document.getElementById('rsib-session');
  const sessionId = parseInt(select?.value || '', 10);
  rsibCurrentSession = rsibSessions.find(s => s.id === sessionId) || null;
  rsibCurrentPage = 1;
  if (rsibCurrentSession && rsibCurrentSession.config) {
    _showRsibPeriodFromConfig(rsibCurrentSession.config);
  }
  renderRsibSessionSummary();
  if (sessionId) {
    await rsibReloadResults(1);
  }
}

function rsibRecalcCurrentSession() {
  renderRsibSessionSummary();
  renderRsibResultsTable(rsibCurrentResults);
}

async function rsibReloadResults(page) {
  const sessionId = parseInt(document.getElementById('rsib-session')?.value || '', 10);
  if (!sessionId) return;
  const pageSize = parseInt(document.getElementById('rsib-page-size')?.value || '25', 10) || 25;
  rsibCurrentPage = Math.max(1, page || 1);
  try {
    const res = await fetch(API + `/api/rsib/sessions/${sessionId}/results?page=${rsibCurrentPage}&page_size=${pageSize}`);
    const data = await res.json();
    rsibCurrentResults = Array.isArray(data.results) ? data.results : [];
    rsibCurrentPages = data.pages || 1;
    renderRsibResultsTable(rsibCurrentResults);
    const pag = document.getElementById('rsib-pagination');
    if (pag) pag.textContent = `Page ${data.page || 1} / ${data.pages || 1}`;
    const note = document.getElementById('rsib-page-note');
    if (note) note.textContent = `${data.total || 0} result(s) sorted by final profit`;
  } catch (e) {
    console.error('Failed to load RSIB results', e);
  }
}

function rsibPrevPage() {
  if (rsibCurrentPage > 1) rsibReloadResults(rsibCurrentPage - 1);
}

function rsibNextPage() {
  if (rsibCurrentPage < rsibCurrentPages) rsibReloadResults(rsibCurrentPage + 1);
}

function getRsibProfitSettings() {
  return {
    startBank: parseFloat(document.getElementById('rsib-bank')?.value || '1000') || 1000,
    buyPriceCents: parseFloat(document.getElementById('rsib-buy-price')?.value || '52') || 52,
    maxBet: parseFloat(document.getElementById('rsib-max-bet')?.value || '500') || 500,
    fullKellyPct: parseFloat(document.getElementById('rsib-full-kelly')?.value || '3.34') || 3.34,
    feePct: parseFloat(document.getElementById('rsib-fee')?.value || '1.56') || 1.56,
  };
}

function renderRsibSessionSummary() {
  const wrap = document.getElementById('rsib-summary');
  if (!wrap) return;
  if (!rsibCurrentSession) {
    wrap.innerHTML = '';
    return;
  }
  const cfg = rsibCurrentSession.config || {};
  wrap.innerHTML = `
    <div class="p-3 rounded bg-slate-800 border border-slate-700"><div class="text-[11px] text-slate-400 uppercase">Status</div><div class="text-lg font-semibold">${rsibCurrentSession.status || '—'}</div></div>
    <div class="p-3 rounded bg-slate-800 border border-slate-700"><div class="text-[11px] text-slate-400 uppercase">Combos</div><div class="text-lg font-semibold">${rsibCurrentSession.completed || 0}/${rsibCurrentSession.total_combos || 0}</div></div>
    <div class="p-3 rounded bg-slate-800 border border-slate-700"><div class="text-[11px] text-slate-400 uppercase">Variants</div><div class="text-lg font-semibold">${rsibCurrentSession.total_variants || 0}</div></div>
    <div class="p-3 rounded bg-slate-800 border border-slate-700"><div class="text-[11px] text-slate-400 uppercase">Best Profit</div><div class="text-lg font-semibold ${(rsibCurrentSession.best_profit || 0) >= 0 ? 'text-emerald-300' : 'text-rose-300'}">${fmtMoney(rsibCurrentSession.best_profit)}</div></div>
    <div class="p-3 rounded bg-slate-800 border border-slate-700"><div class="text-[11px] text-slate-400 uppercase">Best Final Bank</div><div class="text-lg font-semibold text-indigo-200">${fmtMoney(rsibCurrentSession.best_final_bank)}</div></div>
    <div class="p-3 rounded bg-slate-800 border border-slate-700"><div class="text-[11px] text-slate-400 uppercase">Period</div><div class="text-sm font-semibold">${cfg.test_start || '—'} → ${cfg.test_end || '—'}</div></div>`;
}

function renderRsibResultsTable(results) {
  const body = document.getElementById('rsib-results-body');
  if (!body) return;
  if (!Array.isArray(results) || results.length === 0) {
    body.innerHTML = '<tr><td colspan="12" class="p-4 text-slate-500 text-center">No results.</td></tr>';
    return;
  }
  body.innerHTML = results.map(r => {
    const profit = r.profit_full || 0;
    const profitClass = profit >= 0 ? 'text-emerald-300' : 'text-rose-300';
    const threshold = (r.strategy_params && r.strategy_params.threshold != null)
      ? Number(r.strategy_params.threshold).toFixed(3)
      : '—';
    return `<tr class="border-b border-slate-800">
      <td class="p-2">#${r.id}</td>
      <td class="p-2">${r.combo_index}</td>
      <td class="p-2">${r.variant_label || r.variant_type}</td>
      <td class="p-2 text-center font-mono">${threshold}</td>
      <td class="p-2 text-center">${fmtPct(r.accuracy_pct)}</td>
      <td class="p-2 text-center">${r.signals ?? '—'}</td>
      <td class="p-2 text-center">${r.trades_taken ?? '—'}</td>
      <td class="p-2 text-center">${r.trades_skipped ?? '—'}</td>
      <td class="p-2 text-center font-mono ${profitClass}">${fmtMoney(r.final_bank_full)}</td>
      <td class="p-2 text-center ${profitClass}">${fmtPct(r.roi_full)}</td>
      <td class="p-2 text-center font-mono ${profitClass}">${fmtMoney(r.profit_full, true)}</td>
      <td class="p-2 text-center"><button onclick="openRsibDetail(${r.id})" class="btn btn-slate text-[10px] py-0.5 px-2">View</button></td>
    </tr>`;
  }).join('');
}

async function openRsibDetail(resultId) {
  const modal = document.getElementById('rsib-detail-modal');
  const title = document.getElementById('rsib-detail-title');
  const summary = document.getElementById('rsib-detail-summary');
  const monthlyBody = document.getElementById('rsib-detail-monthly');
  const config = document.getElementById('rsib-detail-config');
  if (!modal || !title || !summary || !monthlyBody || !config) return;
  modal.classList.remove('hidden');
  title.textContent = 'RSIB Result #' + resultId;
  summary.innerHTML = '<div class="text-slate-400 text-sm">Loading...</div>';
  monthlyBody.innerHTML = '';
  config.textContent = '';
  try {
    const res = await fetch(API + '/api/rsib/results/' + resultId);
    const data = await res.json();
    if (data.error) {
      summary.innerHTML = '<div class="text-red-400">' + data.error + '</div>';
      return;
    }
    title.textContent = `RSIB Result #${data.id} — ${data.variant_label || data.variant_type}`;
    const threshold = (data.strategy_params && data.strategy_params.threshold != null)
      ? Number(data.strategy_params.threshold).toFixed(3)
      : '—';
    summary.innerHTML = `
      <div class="p-2 rounded bg-slate-800"><div class="text-slate-400 text-[11px]">Accuracy</div><div class="font-semibold">${fmtPct(data.accuracy_pct)}</div></div>
      <div class="p-2 rounded bg-slate-800"><div class="text-slate-400 text-[11px]">Threshold</div><div class="font-semibold font-mono">${threshold}</div></div>
      <div class="p-2 rounded bg-slate-800"><div class="text-slate-400 text-[11px]">Signals</div><div class="font-semibold">${data.signals ?? '—'}</div></div>
      <div class="p-2 rounded bg-slate-800"><div class="text-slate-400 text-[11px]">Taken</div><div class="font-semibold">${data.trades_taken ?? '—'}</div></div>
      <div class="p-2 rounded bg-slate-800"><div class="text-slate-400 text-[11px]">Skipped</div><div class="font-semibold">${data.trades_skipped ?? '—'}</div></div>
      <div class="p-2 rounded bg-slate-800"><div class="text-slate-400 text-[11px]">Final Bank</div><div class="font-semibold">${fmtMoney(data.final_bank_full)}</div></div>
      <div class="p-2 rounded bg-slate-800"><div class="text-slate-400 text-[11px]">Profit</div><div class="font-semibold ${(data.profit_full || 0) >= 0 ? 'text-emerald-300' : 'text-rose-300'}">${fmtMoney(data.profit_full, true)}</div></div>`;
    const monthly = Array.isArray(data.monthly) ? data.monthly : [];
    if (!monthly.length) {
      monthlyBody.innerHTML = '<tr><td colspan="6" class="p-4 text-center text-slate-500">No monthly data.</td></tr>';
    } else {
      monthlyBody.innerHTML = monthly.map(m => `<tr class="border-b border-slate-800">
        <td class="p-2 font-mono">${m.month || '—'}</td>
        <td class="p-2 text-center">${m.total ?? m.taken ?? 0}</td>
        <td class="p-2 text-center">${m.wins ?? m.filtered_correct ?? 0}</td>
        <td class="p-2 text-center">${fmtPct(m.accuracy)}</td>
        <td class="p-2 text-center">${m.skipped ?? 0}</td>
        <td class="p-2 text-center font-mono">${m.bank != null ? fmtMoney(m.bank) : '—'}</td>
      </tr>`).join('');
    }
    const cfgPayload = {
      strategy: data.strategy,
      strategy_params: data.strategy_params,
      metrics: data.metrics,
      super_run_id: data.super_run_id,
      sweep_id: data.sweep_id,
      sweep_combo_index: data.sweep_combo_index,
    };
    if (cfgPayload.strategy_params && cfgPayload.strategy_params.threshold != null) {
      cfgPayload.strategy_params.threshold = Number(cfgPayload.strategy_params.threshold).toFixed(3);
    }
    config.textContent = JSON.stringify(cfgPayload, null, 2);
  } catch (e) {
    summary.innerHTML = '<div class="text-red-400">Failed: ' + e.message + '</div>';
  }
}

function closeRsibDetail() {
  document.getElementById('rsib-detail-modal')?.classList.add('hidden');
}

function fmtMoney(v, signed) {
  const n = Number(v || 0);
  if (signed) return (n >= 0 ? '+$' : '-$') + Math.abs(n).toFixed(2);
  return '$' + n.toFixed(2);
}

function fmtPct(v) {
  if (v == null || Number.isNaN(Number(v))) return '—';
  const n = Number(v);
  return (n >= 0 ? '' : '') + n.toFixed(2) + '%';
}
