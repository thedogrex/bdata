let lgbmFiLastResponse = null;
let lgbmFiInitialized = false;

function lgbmFiInit(){
  if(lgbmFiInitialized) return;
  lgbmFiInitialized = true;
  const end = new Date();
  const start = new Date();
  start.setDate(end.getDate() - 90);
  const fmt = d => `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,'0')}-${String(d.getDate()).padStart(2,'0')}`;
  document.getElementById('lgbm-train-start').value = fmt(start);
  document.getElementById('lgbm-train-end').value = fmt(end);
}

function lgbmFiSetRangePreset(days){
  const end = new Date();
  const start = new Date();
  start.setDate(end.getDate() - Math.max(1, parseInt(days,10)||30));
  const fmt = d => `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,'0')}-${String(d.getDate()).padStart(2,'0')}`;
  document.getElementById('lgbm-train-start').value = fmt(start);
  document.getElementById('lgbm-train-end').value = fmt(end);
}

function lgbmFiReadParams(){
  const raw = (document.getElementById('lgbm-params').value||'').trim();
  if(!raw) return {};
  try{
    return JSON.parse(raw);
  }catch(e){
    alert('Params JSON is invalid.');
    throw e;
  }
}

async function lgbmFiRun(){
  lgbmFiInit();
  const btn = document.getElementById('lgbm-run-btn');
  const statusEl = document.getElementById('lgbm-run-status');
  const errEl = document.getElementById('lgbm-fi-error');
  const tableEl = document.getElementById('lgbm-fi-table');
  errEl.classList.add('hidden');
  statusEl.textContent = 'Training snapshot...';
  btn.disabled = true;
  try{
    const payload = {
      train_start: document.getElementById('lgbm-train-start').value,
      train_end: document.getElementById('lgbm-train-end').value,
      table: document.getElementById('lgbm-table').value || 'c_5m',
      horizon: Math.max(1, parseInt(document.getElementById('lgbm-horizon').value, 10)||1),
      top_n: Math.max(5, parseInt(document.getElementById('lgbm-topn').value, 10)||40),
      params: lgbmFiReadParams(),
    };
    const res = await fetch(API + '/api/lightgbm/feature_importance', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify(payload),
    });
    if(!res.ok){
      const data = await res.json().catch(()=>({error:'Request failed'}));
      throw new Error(data.error || `HTTP ${res.status}`);
    }
    const data = await res.json();
    lgbmFiLastResponse = data;
    lgbmFiRender(data);
    statusEl.textContent = 'Done';
  }catch(e){
    console.error(e);
    errEl.textContent = e.message || 'Failed to compute feature importance';
    errEl.classList.remove('hidden');
    statusEl.textContent = 'Error';
    tableEl.innerHTML = '';
  }finally{
    btn.disabled = false;
    setTimeout(()=>{ statusEl.textContent=''; }, 4000);
  }
}

function lgbmFiRender(data){
  const summary = document.getElementById('lgbm-fi-summary');
  const table = document.getElementById('lgbm-fi-table');
  if(!data || !Array.isArray(data.feature_importance) || !data.feature_importance.length){
    summary.textContent = 'No feature importance data available.';
    table.innerHTML = '';
    return;
  }
  summary.innerHTML = `Train ${data.train_start} → ${data.train_end} | `+
    `H${data.horizon} | Top ${data.feature_importance.length} / ${data.total_features} features.`;
  let rows = `
    <table class="w-full text-left">
      <thead>
        <tr>
          <th class="text-[11px] uppercase text-slate-400">#</th>
          <th class="text-[11px] uppercase text-slate-400">Feature</th>
          <th class="text-[11px] uppercase text-slate-400">Weight %</th>
          <th class="text-[11px] uppercase text-slate-400">Bar</th>
        </tr>
      </thead>
      <tbody>`;
  data.feature_importance.forEach((item, idx)=>{
    const pct = (item.weight * 100).toFixed(2);
    const barWidth = Math.min(100, item.weight * 100 * 3);
    rows += `
      <tr class="border-b border-slate-800">
        <td class="py-1 pr-2 text-slate-400">${idx+1}</td>
        <td class="py-1 pr-2 font-mono text-xs text-blue-200">${item.feature}</td>
        <td class="py-1 pr-2 text-sm text-slate-100">${pct}%</td>
        <td class="py-1">
          <div class="h-2 rounded bg-slate-800" style="min-width:120px">
            <div class="h-2 rounded" style="width:${barWidth}%;background:linear-gradient(90deg,#34d399,#3b82f6)"></div>
          </div>
        </td>
      </tr>`;
  });
  rows += '</tbody></table>';
  table.innerHTML = rows;
}

function lgbmFiCopyJson(){
  if(!lgbmFiLastResponse){
    alert('No data yet. Run the analyzer first.');
    return;
  }
  const text = JSON.stringify(lgbmFiLastResponse, null, 2);
  navigator.clipboard.writeText(text).then(()=>{
    const statusEl = document.getElementById('lgbm-run-status');
    statusEl.textContent = 'JSON copied';
    setTimeout(()=>statusEl.textContent='', 3000);
  });
}
