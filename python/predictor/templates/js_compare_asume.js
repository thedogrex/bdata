// ===== COMPARE ASUME =====
let _caData = null;

function _caToday() {
  const d = new Date();
  const y = d.getUTCFullYear();
  const m = String(d.getUTCMonth() + 1).padStart(2, '0');
  const day = String(d.getUTCDate()).padStart(2, '0');
  return `${y}-${m}-${day}`;
}

function caInit() {
  const fromEl = document.getElementById('ca-date-from');
  const toEl = document.getElementById('ca-date-to');
  if (!fromEl || !toEl) return;
  const today = _caToday();
  if (!fromEl.value) fromEl.value = today;
  if (!toEl.value) toEl.value = today;
}

async function caRun() {
  const dateFrom = document.getElementById('ca-date-from').value || _caToday();
  const dateTo = document.getElementById('ca-date-to').value || dateFrom;
  const limit = parseInt(document.getElementById('ca-limit').value) || 500;
  const btn = document.getElementById('ca-run-btn');
  const status = document.getElementById('ca-status');

  btn.disabled = true;
  status.textContent = 'Running comparison...';

  try {
    const url = `${API}/api/compare_asume?date_from=${encodeURIComponent(dateFrom)}&date_to=${encodeURIComponent(dateTo)}&limit=${limit}`;
    const resp = await fetch(url);
    const data = await resp.json();
    if (data.error) {
      status.textContent = 'Error: ' + data.error;
      btn.disabled = false;
      return;
    }
    _caData = data;
    status.textContent = `Done in ${data.elapsed_sec}s`;
    caRender();
  } catch (e) {
    status.textContent = 'Request failed: ' + e.message;
  } finally {
    btn.disabled = false;
  }
}

const _CA_OFFSETS = ['c_5m_3s', 'c_5m_4s', 'c_5m_5s', 'c_5m_7s', 'c_5m_8s'];

function _caSignalBadge(signal, diff) {
  if (!signal) return '<span class="text-slate-600">—</span>';
  let cls = 'badge ';
  if (signal === 'UP') cls += 'badge-up';
  else if (signal === 'DOWN') cls += 'badge-down';
  else cls += 'badge-queue';
  if (diff) cls += ' ring-2 ring-yellow-400';
  return `<span class="${cls}">${signal}</span>`;
}

function _caOffsetCell(info) {
  if (!info || !info.has_data) {
    const msg = info && info.message ? info.message : 'candle not found';
    return `<td class="text-center align-top px-1 py-1 leading-tight"><div class="text-red-300 text-[9px] break-words leading-tight">${msg}</div></td>`;
  }
  const diff = info.diff;
  const bg = diff ? 'background:rgba(251,191,36,.12)' : '';
  const closeDelta = info.close != null ? `<div class="text-[9px] text-slate-500 leading-tight">${info.close}</div>` : '';
  const rsiTxt = info.rsi != null ? `<div class="text-[9px] text-slate-500 leading-tight">${info.rsi}</div>` : '';
  return `<td style="${bg}" class="text-center align-top px-1 py-1 leading-tight">${_caSignalBadge(info.signal, diff)}<div class="text-[9px] text-slate-500 leading-tight">${info.prob != null ? (info.prob * 100).toFixed(0) + '%' : ''}</div>${closeDelta}${rsiTxt}</td>`;
}

function caRender() {
  if (!_caData) return;
  const data = _caData;

  // Template info
  const tplInfo = document.getElementById('ca-template-info');
  tplInfo.classList.remove('hidden');
  document.getElementById('ca-tpl-name').textContent = data.template.name;
  document.getElementById('ca-tpl-strategy').textContent = data.template.strategy;
  document.getElementById('ca-tpl-horizon').textContent = 'H' + data.template.horizon;
  document.getElementById('ca-tpl-elapsed').textContent = `${data.elapsed_sec}s | ${data.processed_markets}/${data.requested_markets} markets (${data.date_from}..${data.date_to})`;

  // Summary cards
  const summaryDiv = document.getElementById('ca-summary');
  let cardsHtml = '';
  const refSignals = data.processed_markets || data.comparison.length;
  const refUp = data.comparison.filter(c => c.ref_signal === 'UP').length;
  const refDown = data.comparison.filter(c => c.ref_signal === 'DOWN').length;
  const refUndef = refSignals - refUp - refDown;
  cardsHtml += `<div class="card p-4"><div class="text-xs text-slate-400 mb-1">c_5m (base)</div>
    <div class="text-xl font-bold">${refSignals} markets</div>
    <div class="text-xs mt-1"><span class="text-green-400">${refUp} UP</span> &bull; <span class="text-red-400">${refDown} DOWN</span> &bull; <span class="text-slate-400">${refUndef} UNDEF</span></div>
    <div class="text-[11px] mt-1 text-slate-500">Skipped: ${data.skipped_markets || 0}</div></div>`;

  for (const tbl of _CA_OFFSETS) {
    const s = (data.summary || {})[tbl];
    if (!s) continue;
    const matchColor = s.match_pct >= 95 ? 'text-green-400' : s.match_pct >= 85 ? 'text-yellow-400' : s.match_pct >= 70 ? 'text-orange-400' : 'text-red-400';
    cardsHtml += `<div class="card p-4"><div class="text-xs text-slate-400 mb-1">${s.label}</div>
      <div class="text-xl font-bold ${matchColor}">${s.match_pct}% match</div>
      <div class="text-xs mt-1">${s.total_compared} compared &bull; <span class="text-green-400">${s.same_signal} same</span> &bull; <span class="text-yellow-400">${s.diff_signal} diff</span></div>
      <div class="text-[11px] mt-1 text-slate-500">Missing candle: ${s.missing_offset_candle || 0}</div>
      <div class="text-[11px] text-amber-300 mt-1">False-positive vs UNDEFINED: ${s.false_positive || 0}</div></div>`;
  }
  summaryDiv.innerHTML = cardsHtml;

  // Table
  const wrap = document.getElementById('ca-table-wrap');
  wrap.classList.remove('hidden');
  const tbody = document.getElementById('ca-tbody');
  const diffOnly = document.getElementById('ca-diff-only').checked;

  // Backend already returns markets DESC (newest first)
  const rows = [...data.comparison];

  let html = '';
  for (const r of rows) {
    if (diffOnly && !r.any_diff) continue;
    const rowBg = r.any_diff ? 'background:rgba(251,191,36,.06)' : '';

    html += `<tr style="${rowBg}">`;
    html += `<td class="px-1 py-1 align-top truncate"><a href="#" onclick="caGotoMarket('${r.slug}');return false;" class="text-blue-300 hover:text-blue-200 text-[10px]">${r.slug}</a></td>`;
    html += `<td class="px-1 py-1 align-top whitespace-nowrap font-mono text-[10px]">${r.dt}</td>`;
    html += `<td class="px-1 py-1 align-top whitespace-nowrap font-mono text-[9px] text-slate-400">${r.signal_open_time || ''}</td>`;
    html += `<td class="px-1 py-1 align-top text-right font-mono text-[10px]">${r.ref_close}</td>`;
    html += `<td class="px-1 py-1 align-top text-center">${_caSignalBadge(r.ref_signal, false)}<div class="text-[9px] text-slate-500 leading-tight">${(r.ref_prob * 100).toFixed(0)}%</div></td>`;
    html += `<td class="px-1 py-1 align-top text-right text-slate-500 text-[10px]">${r.ref_rsi}</td>`;

    for (const tbl of _CA_OFFSETS) {
      html += _caOffsetCell(r.offsets[tbl]);
    }
    html += '</tr>';
  }
  tbody.innerHTML = html || '<tr><td colspan="11" class="text-center text-slate-500 py-4">No data</td></tr>';
}

function caGotoMarket(slug) {
  try {
    localStorage.setItem('poly_last_selected_market_slug_v1', String(slug));
  } catch (e) {}
  switchTab('poly');
  loadPolyMarkets();
  setTimeout(() => {
    try {
      if (typeof showPolyMarket === 'function') {
        showPolyMarket(slug);
      }
    } catch (e) {}
  }, 50);
}

document.addEventListener('DOMContentLoaded', caInit);
