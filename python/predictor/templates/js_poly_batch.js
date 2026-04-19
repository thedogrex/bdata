// ===== POLY BATCH =====
let polyBatchData = null;
let polyBatchFilter = 'ALL';
let polyBatchLoading = false;

async function polyBatchLoadLatest(){
  try{
    const res = await fetch(API + '/api/poly/batch_recent');
    if(!res.ok) throw new Error('Failed to fetch recent batch data');
    polyBatchData = await res.json();
    polyBatchRender();
  }catch(err){
    const target = document.getElementById('poly-batch-results');
    if(target) target.innerHTML = `<div class="text-red-400 text-sm">${err.message||err}</div>`;
  }
}

async function polyBatchRun(){
  if(polyBatchLoading) return;
  const limitInput = document.getElementById('poly-batch-limit');
  const tableInput = document.getElementById('poly-batch-table');
  const statusEl = document.getElementById('poly-batch-status');
  const limit = Math.max(1, Number(limitInput?.value||20));
  const table = tableInput?.value || 'c_5m';
  statusEl && (statusEl.textContent = 'Running predictions...');
  polyBatchLoading = true;
  try{
    const res = await fetch(API + `/api/poly/batch_recent?limit=${limit}&table=${encodeURIComponent(table)}`,{
      method:'POST'
    });
    if(!res.ok) throw new Error(`Server error ${res.status}`);
    polyBatchData = await res.json();
    statusEl && (statusEl.textContent = 'Completed');
    polyBatchRender();
  }catch(err){
    statusEl && (statusEl.textContent = err.message||String(err));
  }finally{
    polyBatchLoading = false;
  }
}

function polyBatchSetFilter(filter){
  polyBatchFilter = filter;
  polyBatchRender();
}

function polyBatchClearFilter(){
  polyBatchFilter = 'ALL';
  polyBatchRender();
}

function polyBatchRender(){
  const resultsEl = document.getElementById('poly-batch-results');
  const updatedEl = document.getElementById('poly-batch-updated');
  const summaryEl = document.getElementById('poly-batch-summary');
  if(!resultsEl) return;
  if(!polyBatchData){
    resultsEl.innerHTML = '<div class="text-slate-500 text-sm">No data yet.</div>';
    if(updatedEl) updatedEl.textContent='';
    if(summaryEl) summaryEl.textContent='';
    return;
  }
  const summary = polyBatchData.summary || {};
  if(summaryEl){
    summaryEl.innerHTML = `UP <span class="text-green-400">${summary.UP||0}</span> · DOWN <span class="text-red-400">${summary.DOWN||0}</span> · UNDEF ${summary.UNDEFINED||0} · ERR ${summary.ERROR||0}`;
  }
  if(updatedEl){
    updatedEl.textContent = `Updated at ${polyBatchData.updated_at || '—'} (table ${polyBatchData.table||'c_5m'})`;
  }

  const filter = polyBatchFilter;
  const cards = [];
  (polyBatchData.markets||[]).forEach((m,idx)=>{
    const label = m.main_label || 'UNDEFINED';
    if(filter !== 'ALL' && label !== filter) return;
    const tagClass = label==='UP'?'text-green-400':label==='DOWN'?'text-red-400':label==='ERROR'?'text-red-300':'text-amber-300';
    const tplRows = (m.results||[]).map(r=>{
      const rClass = r.label==='UP'?'text-green-400':r.label==='DOWN'?'text-red-400':r.label==='ERROR'?'text-red-300':'text-amber-300';
      const prob = r.probability!=null ? `${Math.round(Number(r.probability||0)*100)}%` : '—';
      const errMsg = r.error ? `<div class="text-xs text-red-400">${r.error}</div>` : '';
      return `<tr><td class="text-xs text-slate-400">${r.template_name||'-'} <span class="text-slate-600">H${r.horizon||1}</span></td><td class="${rClass} font-semibold text-sm">${r.label}</td><td class="text-xs text-slate-400">${prob}</td></tr>${errMsg?`<tr><td colspan="3">${errMsg}</td></tr>`:''}`;
    }).join('') || '<tr><td class="text-xs text-slate-500" colspan="3">No template results</td></tr>';
    cards.push(`
      <div class="rounded-lg border border-slate-800 bg-slate-900/40 p-4">
        <div class="flex items-center justify-between mb-2">
          <div>
            <div class="text-sm font-semibold text-slate-200">${m.slug||'unknown'}</div>
            <div class="text-xs text-slate-500">${m.market_dt||''}</div>
          </div>
          <div class="text-right">
            <div class="${tagClass} font-bold">${label}</div>
            <div class="text-xs text-slate-500">${m.table||''}</div>
          </div>
        </div>
        ${m.error?`<div class="text-xs text-red-400 mb-2">${m.error}</div>`:''}
        <table class="text-xs w-full"><thead><tr><th class="text-left">Template</th><th>Label</th><th>Prob</th></tr></thead><tbody>${tplRows}</tbody></table>
      </div>
    `);
  });
  resultsEl.innerHTML = cards.length ? cards.join('') : '<div class="text-slate-500 text-sm">No markets match the current filter.</div>';
  const filterButtons = document.querySelectorAll('#poly-batch-filter-buttons button');
  filterButtons.forEach(btn=>{
    const isActive = btn.getAttribute('data-filter') === filter;
    btn.classList.toggle('btn-primary', isActive);
    btn.classList.toggle('btn-slate', !isActive);
  });
}

window.polyBatchRun = polyBatchRun;
window.polyBatchSetFilter = polyBatchSetFilter;
window.polyBatchClearFilter = polyBatchClearFilter;
window.polyBatchLoadLatest = polyBatchLoadLatest;
