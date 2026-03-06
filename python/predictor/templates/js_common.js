function fmtTime(s){if(!s||s<=0)return'--';const m=Math.floor(s/60);const sec=Math.floor(s%60);return m>0?`${m}m ${sec}s`:`${sec}s`}
function accClass(a){return a>=54?'accuracy-good':a>=51?'accuracy-ok':'accuracy-bad'}
function statusBadge(s){const m={running:'badge-run',paused:'badge-pause',done:'badge-done',error:'badge-err',cancelled:'badge-cancel',queued:'badge-queue'};return `<span class="badge ${m[s]||'badge-queue'}">${s}</span>`}

function taskExtraLine(t){
  const e=(t&&t.extra)||{};
  const parts=[];
  const state=e.state;

  // State pill
  if(state==='training'){
    const trainSec=e.train_elapsed_sec!=null?` ${fmtTime(e.train_elapsed_sec)}`:""
    parts.push(`⚙️ TRAINING${trainSec}`);
  } else if(state==='walking'){
    parts.push('▶ Walking candles');
  } else if(state==='preloading'){
    parts.push('⏳ Preloading data...');
  } else if(state==='done'){
    parts.push('✓ Combo done');
  }

  // Train X/Y — always show when available
  const tc=e.train_count!=null?e.train_count:null;
  const tt=e.train_total!=null?e.train_total:null;
  if(tc!=null && tt!=null){
    parts.push(`Train: ${state==='training'?tc+1:tc}/${tt}`);
  } else if(tc!=null){
    parts.push(`Trains done: ${tc}`);
  }

  // Candles X/Y
  if(e.candles_total!=null){
    const done=e.candles_done!=null?e.candles_done:0;
    const tot=e.candles_total;
    const pct=tot>0?Math.round(done/tot*100):0;
    parts.push(`Candles: ${done}/${tot} (${pct}%)`);
  }

  // Horizon
  if(e.horizon!=null) parts.push(`H${e.horizon}`);

  return parts.length?parts.join(' | '):'';
}

// ===== POLLING =====
function startPolling(){if(pollTimer)return;pollTimer=setInterval(pollStatus,1500);pollStatus()}
function stopPolling(){if(pollTimer){clearInterval(pollTimer);pollTimer=null}}

async function pollStatus(){
  try{
    // Silence network logs for /api/tasks/status polling
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
  const extraEl=document.getElementById('tb-extra');
  if(extraEl){
    const line=taskExtraLine(t);
    extraEl.innerHTML=line?`<span style="font-size:12px;font-weight:600;color:#94a3b8">${line}</span>`:""
    if(!line) extraEl.classList.add('hidden'); else extraEl.classList.remove('hidden');
  }
  const elapsed=fmtTime(t.elapsed_sec);
  const eta=fmtTime(t.eta_sec);
  document.getElementById('tb-time').textContent=`Elapsed: ${elapsed} | ETA: ${eta}`;
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
    const p=await(await fetch(API+'/api/tasks/'+taskId)).json();
    if(p.task_type==='backtest'){renderResult(data,'bt-results');switchTab('history');loadHistory();}
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
  ensureBacktestConfig();
  ensureCompareConfig();
  histUpdateHorizonButtons();
  loadDefaultGrid();
  startPolling();
  loadAutopredictState();
}

function updateDesc(){
  const n=document.getElementById('bt-strategy').value;
  const s=strategiesData.find(x=>x.name===n);
  document.getElementById('strategy-desc').textContent=s?s.description:'';
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
  const ref=document.getElementById('strategy-ref');
  if(s){
    ref.classList.remove('hidden');
    const rec=s.recommended||{};
    const training=s.needs_training?'<span class="text-amber-400">Yes</span> (retrains every N candles — XGBoost training time scales with n_estimators)':'<span class="text-green-400">No</span> (rule-based, instant prediction)';
    let html=`<div class="flex items-center gap-2 mb-2"><b class="text-slate-200">Strategy Reference: ${s.name}</b>${s.needs_training?'<span class="badge badge-amber" style="background:#78350f;color:#fcd34d">Requires Training</span>':'<span class="badge badge-done">No Training</span>'}</div>`;
    html+=`<div class="mb-2"><b>Training:</b> ${training}</div>`;
    if(rec.notes) html+=`<div class="mb-2 text-slate-300"><b>Notes:</b> ${rec.notes}</div>`;
    html+=`<div class="mb-2"><b>All Parameters:</b></div>`;
    html+=`<table class="mb-3"><thead><tr><th>Param</th><th>Default</th><th>Description</th></tr></thead><tbody>`;
    const dp=s.default_params||{};
    const pd=s.param_docs||{};
    for(const[k,v] of Object.entries(dp)){
      const val=typeof v==='object'?JSON.stringify(v):String(v);
      html+=`<tr><td class="font-mono text-blue-300">${k}</td><td class="font-mono">${val}</td><td class="text-slate-400">${pd[k]||''}</td></tr>`;
    }
    html+=`</tbody></table>`;
    html+=`<details class="mb-2"><summary class="cursor-pointer text-blue-400"><b>Default Params JSON (copy-paste)</b></summary>`;
    html+=`<pre class="mt-1 p-2 rounded text-xs overflow-x-auto" style="background:#1e293b">${JSON.stringify(dp,null,2)}</pre></details>`;
    const presetKeys=Object.keys(rec).filter(k=>k.endsWith('_preset'));
    if(presetKeys.length){
      html+=`<div class="mb-1"><b>Presets:</b></div><div class="flex flex-wrap gap-2 mb-2">`;
      presetKeys.forEach(pk=>{
        const label=pk.replace('_preset','').replace(/_/g,' ');
        html+=`<button onclick='applyPreset(${JSON.stringify(JSON.stringify(rec[pk]))})' class="btn btn-slate text-xs">${label}</button>`;
      });
      html+=`</div>`;
    }
    if(rec.brute_force_include){
      html+=`<div class="text-slate-400"><b>Recommended brute-force params:</b> ${rec.brute_force_include.join(', ')}</div>`;
    }
    ref.innerHTML=html;
  }else{ref.classList.add('hidden')}
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
  ensureBacktestConfig();
}

function applyPreset(jsonStr){
  const cfgEl=document.getElementById('bt-config');
  if(!cfgEl) return;
  let cfg={};
  const txt=(cfgEl.value||'').trim();
  if(txt){
    try{cfg=JSON.parse(txt)||{}}catch(e){cfg={}}
  }
  try{
    cfg.params=JSON.parse(jsonStr);
  }catch(e){
    alert('Invalid preset JSON');
    return;
  }
  if(!cfg.strategy) cfg.strategy=document.getElementById('bt-strategy')?.value||'xgboost';
  cfgEl.value=JSON.stringify(cfg,null,2);
}

function ensureBacktestConfig(){
  const el=document.getElementById('bt-config');
  if(!el) return;
  const existing=(el.value||'').trim();
  if(existing){
    try{
      const cfg=JSON.parse(existing);
      if(cfg && typeof cfg==='object') return;
    }catch(e){}
  }
  const strategy=document.getElementById('bt-strategy')?.value||'xgboost';
  const cfg={
    strategy,
    train_start:'2022-01-01',
    train_end:'2025-06-30',
    test_start:'2025-07-01',
    test_end:'2025-12-31',
    horizons:[1],
    table:'c_5m',
    window_size:5000,
    retrain_every:500,
    params:{},
  };
  el.value=JSON.stringify(cfg,null,2);
}

// ===== TABS =====
const TABS=['backtest','bruteforce','history','best','best_compare','poly','wallet','analytics'];
function switchTab(tab){
  TABS.forEach(t=>{
    const panel=document.getElementById('panel-'+t);
    if(panel) panel.classList.toggle('hidden',t!==tab);
    const b=document.getElementById('tab-'+t);
    if(b){
      if(t===tab){b.classList.add('tab-active');b.classList.remove('text-slate-400')}
      else{b.classList.remove('tab-active');b.classList.add('text-slate-400')}
    }
  });
  // Highlight parent tab for sub-panels
  if(tab==='best_compare'){
    const b=document.getElementById('tab-best');
    if(b){b.classList.add('tab-active');b.classList.remove('text-slate-400')}
  }
  if(tab==='history')loadHistory();
  if(tab==='best')loadBest();
  if(tab==='bruteforce')loadBfSessions();
  if(tab==='poly'){loadPolyMarkets();loadSimTrades();loadSimPositions();}
  else{
    clearPolySelectionComplete();
    stopPolyOrderBookUpdates();
    if(typeof stopLiveMarketPoll === 'function') stopLiveMarketPoll();
  }
  if(tab==='wallet') loadWallet();
}

// ===== WALLET =====
async function loadWallet(){
  const balEl = document.getElementById('wallet-balance');
  const posEl = document.getElementById('wallet-positions');
  const ordEl = document.getElementById('wallet-orders');
  if(balEl) balEl.textContent = 'Loading...';
  if(posEl) posEl.textContent = 'Loading...';
  if(ordEl) ordEl.textContent = 'Loading...';
  try{
    const res = await fetch(API + '/api/poly/live/wallet?limit=25');
    const data = await res.json();
    renderWallet(data);
  }catch(e){
    if(balEl) balEl.textContent = 'Wallet load error';
    if(posEl) posEl.textContent = 'Wallet load error';
    if(ordEl) ordEl.textContent = 'Wallet load error';
  }
}

function renderWallet(data){
  const balEl = document.getElementById('wallet-balance');
  const posEl = document.getElementById('wallet-positions');
  const ordEl = document.getElementById('wallet-orders');
  if(!balEl || !posEl || !ordEl) return;

  const balance = data?.balance || {};
  const positions = Array.isArray(data?.positions) ? data.positions : [];
  const orders = Array.isArray(data?.orders) ? data.orders : [];

  // Helper to normalize USDC balance (same as js_poly.js)
  const normUsdc = (x) => {
    if(x === undefined || x === null) return null;
    if(typeof x === 'number'){
      if(!Number.isFinite(x)) return null;
      return Number.isInteger(x) ? x / 1e6 : x;
    }
    if(typeof x === 'string'){
      const s = x.trim();
      if(!s) return null;
      if(/^[0-9]+$/.test(s)){
        const n = parseInt(s, 10);
        return Number.isFinite(n) ? n / 1e6 : null;
      }
      const v = parseFloat(s);
      if(!Number.isFinite(v)) return null;
      return v;
    }
    return null;
  };

  // Extract and display a friendly USDC balance line
  let friendlyBalance = '';
  const coll = balance?.collateral;
  if(coll){
    let usd = null;
    if(typeof coll === 'number' || typeof coll === 'string'){
      usd = normUsdc(coll);
    } else if(typeof coll === 'object'){
      for(const k of ['availableBalance','available_balance','balance','totalBalance','total_balance','amount','value']){
        if(coll[k] !== undefined && coll[k] !== null){
          const v = normUsdc(coll[k]);
          if(v !== null && Number.isFinite(v)){ usd = v; break; }
        }
      }
    }
    if(usd !== null && Number.isFinite(usd)){
      friendlyBalance = `<div class="mb-2"><div class="text-slate-400 text-[11px]">Balance (USDC)</div><div class="text-green-400 text-[12px] font-semibold">$${usd.toFixed(2)}</div></div>`;
    }
  }

  const balanceBlock = (title, obj) => {
    const content = obj ? JSON.stringify(obj, null, 2) : '{}';
    return `<div class="mb-2"><div class="text-slate-400 text-[11px]">${title}</div><pre class="text-[11px] font-mono whitespace-pre-wrap">${content}</pre></div>`;
  };
  balEl.innerHTML = friendlyBalance || '<div class="text-slate-500 text-xs">No balance data.</div>';

  if(!positions.length){
    posEl.innerHTML = '<div class="text-slate-500 text-xs">No open positions</div>';
  } else {
    let html = '<div class="space-y-2">';
    positions.forEach(p=>{
      const resolved = (p.resolved_outcome||'').toUpperCase();
      const side = (p.outcome_side||'').toUpperCase();
      let badge = '';
      if(resolved && side){
        if(resolved === side){
          badge = `<span class="ml-2 px-2 py-0.5 rounded-full text-[10px] font-bold bg-green-900 text-green-300">WIN</span>`;
        } else {
          badge = `<span class="ml-2 px-2 py-0.5 rounded-full text-[10px] font-bold bg-red-900 text-red-300">LOST</span>`;
        }
      }
      html += `<div class="p-2 rounded" style="background:#0f172a;border:1px solid #334155">
        <div class="flex items-center justify-between">
          <div class="text-xs font-semibold">${p.slug}${badge}</div>
          <div class="text-[11px] text-slate-500">${side}</div>
        </div>
        <div class="text-[11px] text-slate-400">${side}</div>
        <div class="grid grid-cols-3 gap-2 text-[11px] mt-1">
          <div>shares: <b>${(p.shares||0).toFixed(4)}</b></div>
          <div>avg: <b>${(p.avg_price||0).toFixed(4)}</b></div>
          <div>cost: <b>$${(p.total_cost||0).toFixed(2)}</b></div>
        </div>
      </div>`;
    });
    html += '</div>';
    posEl.innerHTML = html;
  }

  if(!orders.length){
    ordEl.innerHTML = '<div class="text-slate-500 text-xs">No recent orders</div>';
  } else {
    let html = '<div class="space-y-2">';
    orders.forEach(o=>{
      const fullId = o.clob_order_id || '';
      const shortId = fullId ? fullId.slice(0, 8) : '-';
      const idHtml = fullId
        ? `<span class="flex items-center gap-1"><b>${shortId}</b><button onclick="navigator.clipboard.writeText('${fullId}')" title="Copy full ID" style="background:none;border:none;color:#60a5fa;cursor:pointer;padding:0;font-size:12px;line-height:1">📋</button></span>`
        : '-';
      html += `<div class="p-2 rounded" style="background:#0f172a;border:1px solid #334155">
        <div class="flex items-center justify-between">
          <div class="text-xs font-semibold">${o.side} ${o.outcome_side || ''}</div>
          <div class="text-[11px] text-slate-500">${o.clob_status || ''}</div>
        </div>
        <div class="text-[11px] text-slate-400">${o.slug}</div>
        <div class="grid grid-cols-3 gap-2 text-[11px] mt-1">
          <div>price: <b>${(o.price||0).toFixed(4)}</b></div>
          <div>amt: <b>${(o.amount||0).toFixed(2)}</b></div>
          <div>id: ${idHtml}</div>
        </div>
      </div>`;
    });
    html += '</div>';
    ordEl.innerHTML = html;
  }
}

// ===== BACKTEST =====
async function runBacktest(){
  const txt=(document.getElementById('bt-config')?.value||'').trim();
  if(!txt){alert('Backtest config JSON is empty');return}
  let cfg;
  try{cfg=JSON.parse(txt)}catch(e){alert('Invalid backtest config JSON');return}

  // Allow horizons to be "1" or [1,2]
  let horizons=cfg.horizons;
  if(typeof horizons==='string'){
    horizons=horizons.split(',').map(x=>parseInt(x.trim())).filter(x=>!isNaN(x));
  }else if(typeof horizons==='number'){
    horizons=[parseInt(horizons)];
  }
  if(!Array.isArray(horizons) || !horizons.length) horizons=[1];

  const body={
    strategy: cfg.strategy || document.getElementById('bt-strategy').value,
    params: cfg.params || null,
    train_start: String(cfg.train_start||''),
    train_end: String(cfg.train_end||''),
    test_start: String(cfg.test_start||''),
    test_end: String(cfg.test_end||''),
    horizons,
    table: String(cfg.table||'c_5m'),
    window_size: parseInt(cfg.window_size)||5000,
    retrain_every: parseInt(cfg.retrain_every)||500,
  };
  const res=await fetch(API+'/api/backtest',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const data=await res.json();
  if(data.error){alert(data.error);return}
  activeTaskId=data.task_id;
  document.getElementById('bt-results').classList.add('hidden');
}

function renderResult(data,targetId){
  const el=document.getElementById(targetId);el.classList.remove('hidden');
  const ws=data.window_size||'?';const re=data.retrain_every||'?';
  const lt=data.load_time_sec?` | Load: ${data.load_time_sec}s`:'';
  const ft=data.feature_time_sec?` | Features: ${data.feature_time_sec}s`:'';
  let html=`<div class="card p-6 mb-6"><div class="mb-4 flex items-center justify-between">
    <div>
      <h2 class="text-lg font-semibold">${data.strategy} ${data.id?'<span class="text-xs text-slate-400">#'+data.id+'</span>':''}</h2>
      <p class="text-xs text-slate-400">Train: ${data.train_period||''} | Test: ${data.test_period||''}</p>
      <p class="text-xs text-slate-400">Window: ${ws} | Retrain: ${re} | Total: ${data.total_time_sec}s${lt}${ft}</p>
    </div>
    ${data.params ? `<button onclick="event.stopPropagation();toggleConfigModal(${data.id||0}, '${data.strategy}', ${JSON.stringify(data.params).replace(/'/g, "\\'")})" class="text-blue-400 hover:text-blue-200 text-xs px-2 py-1 rounded border border-slate-600" title="View Config">⚙ ${data.strategy}</button>` : ''}
  </div></div>`;
  if(data.params){
    const pJson=JSON.stringify(data.params,null,2);
    html+=`<details class="mb-4"><summary class="text-xs text-blue-400 cursor-pointer font-semibold">Strategy Settings (JSON) — click to copy</summary><div class="relative mt-1"><pre id="result-params-json" class="p-3 rounded text-xs font-mono overflow-x-auto" style="background:#0f172a;border:1px solid #334155;cursor:pointer" onclick="navigator.clipboard.writeText(this.textContent).then(()=>{this.style.borderColor='#22c55e';setTimeout(()=>this.style.borderColor='#334155',1000)})">${pJson}</pre></div></details>`;
  }
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
  const txt=(document.getElementById('cmp-config')?.value||'').trim();
  if(!txt){alert('Compare config JSON is empty');return}
  let cfg;
  try{cfg=JSON.parse(txt)}catch(e){alert('Invalid compare config JSON');return}

  let horizons=cfg.horizons;
  if(typeof horizons==='string'){
    horizons=horizons.split(',').map(x=>parseInt(x.trim())).filter(x=>!isNaN(x));
  }else if(typeof horizons==='number'){
    horizons=[parseInt(horizons)];
  }
  if(!Array.isArray(horizons) || !horizons.length) horizons=[1,2,3];

  const body={
    strategies: cfg.strategies || strategiesData.map(s=>s.name),
    train_start: String(cfg.train_start||''),
    train_end: String(cfg.train_end||''),
    test_start: String(cfg.test_start||''),
    test_end: String(cfg.test_end||''),
    horizons,
    table: String(cfg.table||'c_5m'),
    window_size: parseInt(cfg.window_size)||5000,
    retrain_every: parseInt(cfg.retrain_every)||500,
  };
  const res=await fetch(API+'/api/compare',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const data=await res.json();
  if(data.error){alert(data.error);return}
  activeTaskId=data.task_id;
  document.getElementById('cmp-results').classList.add('hidden');
}

function ensureCompareConfig(){
  const el=document.getElementById('cmp-config');
  if(!el) return;
  const existing=(el.value||'').trim();
  if(existing){
    try{
      const cfg=JSON.parse(existing);
      if(cfg && typeof cfg==='object') return;
    }catch(e){}
  }
  const cfg={
    train_start:'2022-01-01',
    train_end:'2025-06-30',
    test_start:'2025-07-01',
    test_end:'2025-12-31',
    horizons:[1,2,3],
    table:'c_5m',
    window_size:5000,
    retrain_every:500,
  };
  el.value=JSON.stringify(cfg,null,2);
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
    const cfg=data.config||data.grid;
    document.getElementById('bf-grid').value=JSON.stringify(cfg,null,2);
    document.getElementById('bf-combos').textContent=`Total combos: ${data.total_combos}`}catch(e){}
}
async function runBruteforce(){
  let cfg;
  try{cfg=JSON.parse(document.getElementById('bf-grid').value)}catch(e){alert('Invalid grid JSON');return}

  // Backward compatible:
  // - old format: bf-grid contains ONLY the param grid
  // - new format: bf-grid can contain {param_grid, retrain_every, window_size, max_combos, horizon}
  const grid = (cfg && typeof cfg==='object' && cfg.param_grid) ? cfg.param_grid : cfg;

  const body={
    strategy: (cfg && cfg.strategy) ? String(cfg.strategy) : document.getElementById('bf-strategy').value,
    param_grid: grid,
    train_start: String((cfg&&cfg.train_start)!=null?cfg.train_start:''),
    train_end: String((cfg&&cfg.train_end)!=null?cfg.train_end:''),
    test_start: String((cfg&&cfg.test_start)!=null?cfg.test_start:''),
    test_end: String((cfg&&cfg.test_end)!=null?cfg.test_end:''),
    horizon: parseInt((cfg&&cfg.horizon)!=null?cfg.horizon:1)||1,
    table: String((cfg&&cfg.table)!=null?cfg.table:'c_5m'),
    window_size: parseInt((cfg&&cfg.window_size)!=null?cfg.window_size:5000)||5000,
    retrain_every: parseInt((cfg&&cfg.retrain_every)!=null?cfg.retrain_every:500)||500,
    max_combos: parseInt((cfg&&cfg.max_combos)!=null?cfg.max_combos:100)||100,
  };
  const res=await fetch(API+'/api/bruteforce',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const data=await res.json();
  if(data.error){alert(data.error);return}
  activeTaskId=data.task_id;
}
function renderBfResult(data){
  if(!data||!data.best_accuracy)return;
  loadBfSessions();
}
async function loadBfSessions(){
  try{const res=await fetch(API+'/api/bruteforce/sessions');const data=await res.json();
    const el=document.getElementById('bf-sessions');
    if(!data.length){el.innerHTML='<p class="text-slate-400 text-sm">No sessions yet.</p>';return}
    let html='<table><thead><tr><th>ID</th><th>Strategy</th><th>H</th><th>Combos</th><th>Best</th><th>Avg signals/mo</th><th>Status</th><th>Time</th><th>Date</th><th></th></tr></thead><tbody>';
    data.forEach(s=>{
      const canResume=s.status==='paused'||s.status==='running';
      const resumeBtn=canResume?`<button onclick="resumeBf(${s.id})" class="btn btn-green text-xs">Resume</button>`:'';
      const stopBtn=canResume?`<button onclick="stopBf(${s.id})" class="btn btn-red text-xs ml-1">Stop</button>`:'';
      const viewBtn=`<button onclick="loadHistory();document.getElementById('hist-strategy').value='';switchTab('history')" class="text-blue-400 text-xs hover:underline ml-1">runs</button>`;
      // Compute avg signals per month if we have dates and signals
      let avgSignalsMo = '--';
      if(s.avg_signals_per_month != null){
        avgSignalsMo = s.avg_signals_per_month;
      } else if(s.test_start && s.test_end && s.signals && s.signals>0){
        const start = new Date(s.test_start);
        const end = new Date(s.test_end);
        if(!isNaN(start) && !isNaN(end) && end > start){
          const months = (end.getFullYear() - start.getFullYear())*12 + (end.getMonth() - start.getMonth()) + 1;
          if(months > 0){
            avgSignalsMo = Math.round(s.signals / months);
          }
        }
      }
      html+=`<tr><td>${s.id}</td><td class="font-medium">${s.strategy}</td><td>${s.horizon}</td><td>${s.completed}/${s.total_combos}</td><td class="${accClass(s.best_accuracy)} font-bold">${s.best_accuracy}%</td><td class="text-xs text-slate-300">${avgSignalsMo}</td><td>${statusBadge(s.status)}</td><td>${s.total_time_sec}s</td><td class="text-slate-400 text-xs">${s.created_at}</td><td class="flex gap-1">${resumeBtn}${stopBtn}${viewBtn}</td></tr>`});
    html+='</tbody></table>';el.innerHTML=html}catch(e){}
 }
async function resumeBf(bfId){
  const res=await fetch(API+'/api/bruteforce/resume/'+bfId,{method:'POST'});
  const data=await res.json();
  if(data.error){alert(data.error);return}
  activeTaskId=data.task_id;
  loadBfSessions();
}
async function stopBf(bfId){
  if(!confirm(`Stop and cancel brute-force session BF#${bfId}? This will terminate all remaining tasks and cannot be undone.`))return;
  try{
    const res=await fetch(API+'/api/bruteforce/stop/'+bfId,{method:'POST'});
    const data=await res.json();
    if(data.error){alert(data.error);return}
    loadBfSessions();
  }catch(e){
    alert('Error stopping brute-force session: '+e.message);
  }
}

// ===== HISTORY =====
const histBfState={};

// Standalone (non-BF) compare mode
let histStandaloneCompareMode = false;
let histStandaloneCompareSelected = new Set();
let histStandaloneRunCache = {}; // runId -> run (as returned by /api/history)

// Helper: shorten params for display (sorted keys, no vowels, tiny font)
function histShortParams(params){
  if(!params || typeof params !== 'object') return '';
  const keys = Object.keys(params).sort();
  const parts = [];
  keys.forEach(k => {
    const v = params[k];
    const ks = String(k).replace(/[aeiouAEIOU]/g, '').substring(0,6);
    const vs = String(v).replace(/[aeiouAEIOU]/g, '').substring(0,6);
    parts.push(`${ks}=${vs}`);
  });
  return parts.join(', ');
}

function histGetSelectedHorizons(){
  const v=document.getElementById('hist-horizons')?.value||'1,2,3';
  return String(v).split(',').map(s=>parseInt(String(s).trim(),10)).filter(n=>!isNaN(n));
}

function histUpdateHorizonButtons(){
  const hs=new Set(histGetSelectedHorizons());
  [1,2,3].forEach(h=>{
    const b=document.getElementById('hist-h'+h);
    if(!b)return;
    const on=hs.has(h);
    b.style.background=on?'#052e16':'transparent';
    b.style.color=on?'#22c55e':'#94a3b8';
  });
}

function histToggleHorizon(h){
  const el=document.getElementById('hist-horizons');
  if(!el)return;
  const hs=new Set(histGetSelectedHorizons());
  const hh=parseInt(h,10);
  if(hs.has(hh))hs.delete(hh);else hs.add(hh);
  if(hs.size===0)hs.add(hh);
  el.value=[...hs].sort((a,b)=>a-b).join(',');
  histUpdateHorizonButtons();
}

function histMaxAccBySelectedHorizons(run, selectedHorizons){
  const h=run?.horizons||{};
  let m=0;
  selectedHorizons.forEach(k=>{
    const d=h[String(k)];
    const a=(d&&d.accuracy_pct!=null)?Number(d.accuracy_pct):0;
    if(a>m)m=a;
  });
  return m;
}

function histHorizonsHtml(run, selectedHorizons){
  const h=run?.horizons||{};
  const parts=[];
  selectedHorizons.forEach(k=>{
    const d=h[String(k)];
    if(!d){parts.push(`H${k}:--`);return;}
    if(d.error){parts.push(`H${k}:err`);return;}
    const a=(d.accuracy_pct!=null)?Number(d.accuracy_pct):0;
    parts.push(`H${k}:<span class="${accClass(a)}">${a}%</span>`);
  });
  return parts.join(' | ');
}

// ===== BF COMPARE MODE (per-session) =====
let bfCompareMode = {};       // bfId -> bool
let bfCompareSelected = {};   // bfId -> Set of run ids

function bfToggleCompareMode(bfId){
  bfCompareMode[bfId] = !bfCompareMode[bfId];
  if(!bfCompareMode[bfId]){
    if(bfCompareSelected[bfId]) bfCompareSelected[bfId].clear();
  }
  histRenderBfBody(bfId);
}

function bfToggleSelect(bfId, runId){
  if(!bfCompareSelected[bfId]) bfCompareSelected[bfId] = new Set();
  const sel = bfCompareSelected[bfId];
  if(sel.has(runId)) sel.delete(runId); else sel.add(runId);
  // Update checkbox + row highlight without full re-render
  const cb = document.querySelector(`input[data-bf-id="${bfId}"][data-bf-run="${runId}"]`);
  if(cb) cb.checked = sel.has(runId);
  const row = cb ? cb.closest('tr') : null;
  if(row) row.style.background = sel.has(runId) ? 'rgba(139,92,246,0.12)' : '';
  // Update counter
  const countEl = document.getElementById('bf-cmp-count-'+bfId);
  if(countEl) countEl.textContent = sel.size;
  const btn = document.getElementById('bf-cmp-btn-'+bfId);
  if(btn) btn.disabled = sel.size < 2;
}

function bfToggleAllSelect(bfId, checked){
  const st = histBfState[bfId];
  if(!st || !st.runs) return;
  if(!bfCompareSelected[bfId]) bfCompareSelected[bfId] = new Set();
  const sel = bfCompareSelected[bfId];
  if(checked){ st.runs.forEach(r => sel.add(r.id)); } else { st.runs.forEach(r => sel.delete(r.id)); }
  document.querySelectorAll(`input[data-bf-id="${bfId}"]`).forEach(cb => {
    const rid = parseInt(cb.getAttribute('data-bf-run'));
    cb.checked = sel.has(rid);
    const row = cb.closest('tr');
    if(row) row.style.background = cb.checked ? 'rgba(139,92,246,0.12)' : '';
  });
  const countEl = document.getElementById('bf-cmp-count-'+bfId);
  if(countEl) countEl.textContent = sel.size;
  const btn = document.getElementById('bf-cmp-btn-'+bfId);
  if(btn) btn.disabled = sel.size < 2;
}

async function bfRunCompare(bfId){
  const sel = bfCompareSelected[bfId];
  if(!sel || sel.size < 2){ alert('Select at least 2 runs'); return; }
  // Derive horizon from the actual loaded runs (not the filter selection)
  const st = histBfState[bfId];
  const firstRun = st && st.runs && st.runs.length ? st.runs[0] : null;
  const horizonKeys = firstRun ? Object.keys(firstRun.horizons || {}).map(Number).filter(n=>n>0).sort((a,b)=>a-b) : [];
  const horizon = horizonKeys.length ? horizonKeys[0] : 1;
  const ids = [...sel];
  try{
    const res = await fetch(API+'/api/best/compare', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({run_ids: ids, horizon})
    });
    bestCompareData = await res.json();
    _bcSetSource('history');
    switchTab('best_compare');
    bestCompareRecalc();
  }catch(e){ console.error(e); alert('Error loading comparison data'); }
}

async function histLoadBfRuns(bfId){
  const st=histBfState[bfId]||{page:1,pageSize:20,win:'',total:0};
  histBfState[bfId]=st;
  const minAcc=document.getElementById('hist-min-acc')?.value||'';
  const offset=(st.page-1)*st.pageSize;
  let url=API+`/api/history/bruteforce/${bfId}/runs?offset=${offset}&limit=${st.pageSize}`;
  if(minAcc)url+='&min_accuracy='+minAcc;
  const win=String(st.win||'').trim();
  if(win)url+='&window_size='+win;
  try{
    const res=await fetch(url);
    const data=await res.json();
    st.runs=Array.isArray(data.runs)?data.runs:[];
    st.total=data.total||0;
    st.page=Math.max(1,Math.min(st.page, Math.max(1, Math.ceil(st.total/st.pageSize))));
  }catch(e){
    st.runs=[];
    st.total=0;
    st.page=1;
  }
}

function histRenderBfBody(bfId){
  const st=histBfState[bfId]||{page:1,pageSize:20,win:'',runs:[],total:0};
  const selectedHorizons=histGetSelectedHorizons();
  const total=st.total||0;
  const pages=Math.max(1,Math.ceil(total/st.pageSize));
  const page=Math.max(1,Math.min(st.page,pages));
  const runs=st.runs||[];
  const body=document.getElementById('hist-bf-body-'+bfId);
  if(!body)return;

  const prevDisabled=page<=1?'disabled':'';
  const nextDisabled=page>=pages?'disabled':'';
  const winVal=st.win!=null?String(st.win):'';
  const cmp = !!bfCompareMode[bfId];
  const sel = bfCompareSelected[bfId] || new Set();
  const cmpBtnLabel = cmp ? 'Exit Compare' : 'Compare Mode';
  const cmpBtnClass = cmp ? 'btn-purple' : 'btn-slate';
  let html=`
    <div class="flex flex-wrap items-end justify-between gap-3 mt-3">
      <div class="flex items-end gap-3">
        <div>
          <label class="block text-xs text-slate-400 mb-1">Window Size</label>
          <input type="number" value="${winVal}" placeholder="5000" class="w-32" oninput="histSetBfWin(${bfId},this.value)" />
        </div>
        <button onclick="bfToggleCompareMode(${bfId})" class="btn ${cmpBtnClass} text-xs">${cmpBtnLabel}</button>
        ${cmp ? `<button id="bf-cmp-btn-${bfId}" onclick="bfRunCompare(${bfId})" class="btn btn-purple text-xs" ${sel.size<2?'disabled':''}>Compare Selected (<span id="bf-cmp-count-${bfId}">${sel.size}</span>)</button>` : ''}
      </div>
      <div class="flex items-center gap-2">
        <button class="btn btn-slate text-xs" ${prevDisabled} onclick="histSetBfPage(${bfId},${page-1})">Prev</button>
        <span class="text-xs text-slate-400">Page ${page} / ${pages} (${total})</span>
        <button class="btn btn-slate text-xs" ${nextDisabled} onclick="histSetBfPage(${bfId},${page+1})">Next</button>
      </div>
    </div>
    <table class="mt-2"><thead><tr>${cmp ? `<th style="width:32px"><input type="checkbox" onchange="bfToggleAllSelect(${bfId},this.checked)" title="Select all"></th>` : ''}<th>ID</th><th>Params</th><th>Win</th><th>Horizons</th><th>Time</th><th></th></tr></thead><tbody>
  `;

  runs.forEach(r=>{
    const hs=histHorizonsHtml(r, selectedHorizons);
    const ps=JSON.stringify(r.params||{}).substring(0,80);
    const checked = (cmp && sel.has(r.id)) ? 'checked' : '';
    const rowClick = cmp ? `event.stopPropagation();bfToggleSelect(${bfId},${r.id})` : `event.stopPropagation();showDetail(${r.id})`;
    const selectedBg = (cmp && sel.has(r.id)) ? 'background:rgba(139,92,246,0.12)' : '';
    html+=`<tr class="cursor-pointer hover:bg-slate-700" onclick="${rowClick}" style="${selectedBg}">`;
    if(cmp) html+=`<td><input type="checkbox" ${checked} onclick="event.stopPropagation();bfToggleSelect(${bfId},${r.id})" data-bf-id="${bfId}" data-bf-run="${r.id}"></td>`;
    html+=`<td>${r.id}</td><td class="text-xs text-slate-400 max-w-xs truncate">${ps}</td><td>${r.window_size||'?'} </td><td>${hs}</td><td>${r.total_time_sec}s</td><td><button onclick="event.stopPropagation();deleteRun(${r.id})" class="text-red-400 text-xs hover:underline">del</button></td></tr>`;
  });
  html+='</tbody></table>';
  body.innerHTML=html;
}

async function histOnToggleBf(bfId, detailsEl){
  if(!detailsEl||!detailsEl.open)return;
  await histLoadBfRuns(bfId);
  histRenderBfBody(bfId);
}

async function histSetBfPage(bfId, page){
  const st=histBfState[bfId]||{page:1,pageSize:20,win:'',total:0};
  histBfState[bfId]=st;
  st.page=Math.max(1,parseInt(page,10)||1);
  await histLoadBfRuns(bfId);
  histRenderBfBody(bfId);
}

async function histSetBfWin(bfId, win){
  const st=histBfState[bfId]||{page:1,pageSize:20,win:''};
  histBfState[bfId]=st;
  st.win=String(win||'').trim();
  st.page=1;
  await histLoadBfRuns(bfId);
  histRenderBfBody(bfId);
}

async function loadHistory(){
  histUpdateHorizonButtons();
  const strategy=document.getElementById('hist-strategy')?.value||'';
  const minAcc=document.getElementById('hist-min-acc')?.value||'';
  const limit=document.getElementById('hist-limit')?.value||'50';
  const selectedHorizons=histGetSelectedHorizons();
  const itemLimit=Math.max(1,parseInt(limit,10)||50);
  try{
    // Fetch standalone (non-BF) runs and BF session headers in parallel
    let runUrl=API+'/api/history?limit='+itemLimit+'&exclude_bruteforce=true';
    if(strategy)runUrl+='&strategy='+strategy;
    if(minAcc)runUrl+='&min_accuracy='+minAcc;
    const [runsRes, bfRes] = await Promise.all([
      fetch(runUrl),
      fetch(API+'/api/bruteforce/sessions')
    ]);
    const allRuns=await runsRes.json();
    const bfSessions=await bfRes.json();
    const el=document.getElementById('history-list');

    const standalone=Array.isArray(allRuns)?allRuns:[];

    // Build items list: BF sessions + standalone runs, sorted by date
    const items=[];
    (Array.isArray(bfSessions)?bfSessions:[]).forEach(s=>{
      if(strategy && s.strategy!==strategy) return;
      items.push({type:'bf',session:s,createdAt:new Date(s.created_at||0).getTime()||0});
    });
    standalone.forEach(r=>{
      items.push({type:'run',run:r,createdAt:new Date(r.created_at||0).getTime()||0});
    });
    items.sort((a,b)=>b.createdAt-a.createdAt);
    const limited=items.slice(0,itemLimit);

    if(!limited.length){el.innerHTML='<div class="card p-6 text-center text-slate-400">No results.</div>';return}

    let html='<div class="card p-6">';
    limited.filter(x=>x.type==='bf').forEach(it=>{
      const s=it.session;
      const bfId=s.id;
      const bestAcc=s.best_accuracy||0;
      html+=`<details class="mb-3 p-3 rounded-lg" style="background:#0f172a;border:1px solid #334155" ontoggle="histOnToggleBf(${bfId},this)">
        <summary class="cursor-pointer flex items-center justify-between">
          <span><span class="badge badge-bf">BF#${bfId}</span> <b class="ml-2">${s.strategy}</b> <span class="text-slate-400 text-xs ml-2">${s.completed||0} runs</span></span>
          <span class="${accClass(bestAcc)} font-bold">Best: ${bestAcc}%</span>
        </summary>
        <div class="flex items-center justify-end mt-2">
          <button onclick="event.stopPropagation();deleteBruteforceGroup(${bfId})" class="text-red-400 text-xs hover:underline">remove pack</button>
        </div>
        <div id="hist-bf-body-${bfId}"></div>
      </details>`;
    });

    const standaloneLimited=limited.filter(x=>x.type==='run').map(x=>x.run);
    if(standaloneLimited.length){
      // Cache standalone runs so Compare can infer horizon from actual run data.
      histStandaloneRunCache = {};
      standaloneLimited.forEach(r=>{ if(r && r.id!=null) histStandaloneRunCache[String(r.id)] = r; });

      const cmp = !!histStandaloneCompareMode;
      const sel = histStandaloneCompareSelected || new Set();
      const cmpBtnLabel = cmp ? 'Exit Compare' : 'Compare Mode';
      const cmpBtnClass = cmp ? 'btn-purple' : 'btn-slate';
      html+=`<div class="mt-4 flex items-center justify-between gap-2 flex-wrap">
        <div class="text-xs text-slate-400">Standalone backtests</div>
        <div class="flex items-center gap-2">
          <button onclick="histStandaloneToggleCompareMode()" class="btn ${cmpBtnClass} text-xs">${cmpBtnLabel}</button>
          ${cmp ? `<button id="hist-standalone-cmp-btn" onclick="histStandaloneRunCompare()" class="btn btn-purple text-xs" ${sel.size<1?'disabled':''}>Compare Selected (<span id="hist-standalone-cmp-count">${sel.size}</span>)</button>` : ''}
        </div>
      </div>`;

      html+=`<table class="mt-2"><thead><tr>`;
      if(cmp) html+=`<th style="width:32px"><input type="checkbox" onchange="histStandaloneToggleAllSelect(this.checked)" title="Select all"></th>`;
      html+=`<th>ID</th><th>Strategy</th><th>Test Period</th><th>Win</th><th>Horizons</th><th>Time</th><th>Params</th><th>Date</th><th></th></tr></thead><tbody>`;
      standaloneLimited.forEach(r=>{
        const hs=histHorizonsHtml(r, selectedHorizons);
        const shortParams = histShortParams(r.params);
        const checked = (cmp && sel.has(r.id)) ? 'checked' : '';
        const rowClick = cmp ? `event.stopPropagation();histStandaloneToggleSelect(${r.id})` : `showDetail(${r.id})`;
        const selectedBg = (cmp && sel.has(r.id)) ? 'background:rgba(139,92,246,0.12)' : 'background:#0b1220';
        html+=`<tr class="cursor-pointer hover:bg-slate-700" onclick="${rowClick}" style="${selectedBg}">`;
        if(cmp) html+=`<td><input type="checkbox" ${checked} onclick="event.stopPropagation();histStandaloneToggleSelect(${r.id})" data-standalone-run="${r.id}"></td>`;
        html+=`<td>${r.id}</td><td class="font-medium">${r.strategy}</td><td class="text-xs">${r.test_period||''}</td><td>${r.window_size||'?'} </td><td>${hs}</td><td>${r.total_time_sec}s</td><td class="text-xs font-mono text-slate-500" style="font-size:0.65em;line-height:1">${shortParams}</td><td class="text-slate-400 text-xs">${r.created_at||''}</td><td><button onclick="event.stopPropagation();deleteRun(${r.id})" class="text-red-400 text-xs hover:underline">del</button></td></tr>`
      });
      html+=`</tbody></table>`;
    }
    html+='</div>';el.innerHTML=html}catch(e){console.error(e)}
}

function histStandaloneToggleCompareMode(){
  histStandaloneCompareMode = !histStandaloneCompareMode;
  if(!histStandaloneCompareMode){
    if(histStandaloneCompareSelected) histStandaloneCompareSelected.clear();
  }
  loadHistory();
}

function histStandaloneToggleSelect(runId){
  if(!histStandaloneCompareSelected) histStandaloneCompareSelected = new Set();
  if(histStandaloneCompareSelected.has(runId)) histStandaloneCompareSelected.delete(runId);
  else histStandaloneCompareSelected.add(runId);
  // Update checkbox + row highlight + count without full re-render
  const cb = document.querySelector(`input[data-standalone-run="${runId}"]`);
  if(cb) cb.checked = histStandaloneCompareSelected.has(runId);
  const cnt = document.getElementById('hist-standalone-cmp-count');
  if(cnt) cnt.textContent = String(histStandaloneCompareSelected.size);
  const btn = document.getElementById('hist-standalone-cmp-btn');
  if(btn) btn.disabled = histStandaloneCompareSelected.size < 1;
}

function histStandaloneToggleAllSelect(checked){
  if(!histStandaloneCompareSelected) histStandaloneCompareSelected = new Set();
  const all = Array.from(document.querySelectorAll('input[data-standalone-run]')).map(cb=>parseInt(cb.getAttribute('data-standalone-run'))).filter(n=>!isNaN(n));
  if(checked){ all.forEach(id=>histStandaloneCompareSelected.add(id)); } else { all.forEach(id=>histStandaloneCompareSelected.delete(id)); }
  document.querySelectorAll('input[data-standalone-run]').forEach(cb=>{
    const rid=parseInt(cb.getAttribute('data-standalone-run'));
    cb.checked = histStandaloneCompareSelected.has(rid);
  });
  const cnt = document.getElementById('hist-standalone-cmp-count');
  if(cnt) cnt.textContent = String(histStandaloneCompareSelected.size);
  const btn = document.getElementById('hist-standalone-cmp-btn');
  if(btn) btn.disabled = histStandaloneCompareSelected.size < 1;
}

async function histStandaloneRunCompare(){
  const sel = histStandaloneCompareSelected;
  if(!sel || sel.size < 1){ alert('Select at least 1 run'); return; }
  // Derive horizon from the first selected run (prefer actual run data).
  // Fallback: use the first horizon from History filter.
  let horizon = 1;
  try{
    const firstId = [...sel][0];
    const run = histStandaloneRunCache ? histStandaloneRunCache[String(firstId)] : null;
    const keys = run && run.horizons ? Object.keys(run.horizons).map(Number).filter(n=>n>0).sort((a,b)=>a-b) : [];
    if(keys.length){
      horizon = keys[0];
    } else {
      const hs = histGetSelectedHorizons();
      if(hs && hs.length) horizon = hs[0];
    }
  }catch(e){}

  const ids = [...sel];
  try{
    const res = await fetch(API + '/api/best/compare', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({run_ids: ids, horizon})
    });
    bestCompareData = await res.json();
    if(!Array.isArray(bestCompareData) || bestCompareData.length < 1){
      alert('Compare returned no data (check that selected runs have the chosen horizon)');
      return;
    }
    _bcSetSource('history');
    switchTab('best_compare');
    bestCompareRecalc();
  }catch(e){ console.error(e); alert('Error loading comparison data'); }
}
async function showDetail(id){try{const res=await fetch(API+'/api/history/'+id);const data=await res.json();if(data.error){alert(data.error);return}switchTab('backtest');renderResult(data,'bt-results');document.getElementById('bt-results').scrollIntoView({behavior:'smooth',block:'start'})}catch(e){alert(e.message)}}
async function deleteRun(id){if(!confirm('Delete #'+id+'?'))return;await fetch(API+'/api/history/'+id,{method:'DELETE'});loadHistory()}
async function deleteBruteforceGroup(bfId){
  if(!confirm('Remove brute-force pack BF#'+bfId+' and ALL its runs?'))return;
  const res=await fetch(API+'/api/history/bruteforce/'+bfId,{method:'DELETE'});
  const data=await res.json();
  if(data&&data.error){alert(data.error);return}
  loadHistory();
}
async function clearAllHistory(){if(!confirm('Delete ALL?'))return;await fetch(API+'/api/history',{method:'DELETE'});loadHistory()}

// ===== BEST =====
let bestCompareMode = false;
let bestCompareSelected = new Set();
let bestLastData = [];

async function loadBest(){
  const horizon=document.getElementById('best-horizon').value||1;const limit=document.getElementById('best-limit').value||20;
  const sMinRaw = document.getElementById('best-signals-min')?.value;
  const sMaxRaw = document.getElementById('best-signals-max')?.value;
  const sMin = (sMinRaw !== undefined && sMinRaw !== null && String(sMinRaw).trim() !== '') ? Number(sMinRaw) : null;
  const sMax = (sMaxRaw !== undefined && sMaxRaw !== null && String(sMaxRaw).trim() !== '') ? Number(sMaxRaw) : null;
  const qs = new URLSearchParams({horizon: String(horizon), limit: String(limit)});
  if(sMin !== null && Number.isFinite(sMin)) qs.set('signals_min', String(Math.max(0, Math.floor(sMin))));
  if(sMax !== null && Number.isFinite(sMax)) qs.set('signals_max', String(Math.max(0, Math.floor(sMax))));
  try{const res=await fetch(API+`/api/best?${qs.toString()}`);const data=await res.json();bestLastData=data;const el=document.getElementById('best-list');
    if(!data.length){el.innerHTML='<div class="card p-6 text-center text-slate-400">No results.</div>';return}
    renderBestList(data, horizon);
  }catch(e){console.error(e)}
}

function renderBestList(data, horizon){
  const el=document.getElementById('best-list');
  const cmp = bestCompareMode;
  let html='<div class="card p-6"><div class="flex items-center justify-between mb-4"><h2 class="text-lg font-semibold">Top Runs (H'+horizon+')</h2><button onclick="event.stopPropagation();toggleConfigModal(0, \'Best List\', {horizon:'+horizon+', limit:'+data.length+'})" class="text-blue-400 hover:text-blue-200 text-xs px-2 py-1 rounded border border-slate-600" title="View Config">⚙ Config</button></div><table><thead><tr>';
  if(cmp) html+='<th style="width:32px"><input type="checkbox" onchange="bestToggleAll(this.checked)" title="Select all"></th>';
  html+='<th>#</th><th>Strategy</th><th>Accuracy</th><th>Signals</th><th>Correct</th><th>Wrong</th><th>W/L</th><th>Win</th><th>Params</th></tr></thead><tbody>';
  data.forEach((r,i)=>{const ps=JSON.stringify(r.params||{}).substring(0,60);
    const checked = bestCompareSelected.has(r.id) ? 'checked' : '';
    const rowClick = cmp ? `bestToggleSelect(${r.id})` : `showDetail(${r.id})`;
    const selectedBg = (cmp && bestCompareSelected.has(r.id)) ? 'background:rgba(139,92,246,0.12)' : '';
    html+=`<tr class="cursor-pointer" onclick="${rowClick}" style="${selectedBg}">`;
    if(cmp) html+=`<td><input type="checkbox" ${checked} onclick="event.stopPropagation();bestToggleSelect(${r.id})" data-best-cb="${r.id}"></td>`;
    html+=`<td>${i+1}</td><td class="font-medium">${r.strategy}</td><td class="${accClass(r.accuracy_pct)} font-bold text-lg">${r.accuracy_pct}%</td><td>${r.signals}</td><td class="text-green-400">${r.correct}</td><td class="text-red-400">${r.wrong}</td><td>${r.max_win_streak}/${r.max_lose_streak}</td><td>${r.window_size}</td><td class="text-xs text-slate-400 max-w-xs truncate">${ps}</td></tr>`});
  html+='</tbody></table></div>';el.innerHTML=html;
}

function bestToggleCompareMode(){
  bestCompareMode = !bestCompareMode;
  const btn = document.getElementById('best-compare-toggle');
  const cmpBtn = document.getElementById('best-compare-btn');
  if(bestCompareMode){
    btn.classList.remove('btn-slate'); btn.classList.add('btn-purple');
    btn.textContent = 'Exit Compare';
    cmpBtn.classList.remove('hidden');
  } else {
    btn.classList.remove('btn-purple'); btn.classList.add('btn-slate');
    btn.textContent = 'Compare Mode';
    cmpBtn.classList.add('hidden');
    bestCompareSelected.clear();
  }
  bestUpdateCompareCount();
  if(bestLastData.length){
    const h = document.getElementById('best-horizon').value||1;
    renderBestList(bestLastData, h);
  }
}

function bestToggleSelect(id){
  if(bestCompareSelected.has(id)) bestCompareSelected.delete(id);
  else bestCompareSelected.add(id);
  bestUpdateCompareCount();
  // Update checkbox and row style without full re-render
  const cb = document.querySelector(`input[data-best-cb="${id}"]`);
  if(cb) cb.checked = bestCompareSelected.has(id);
  const row = cb ? cb.closest('tr') : null;
  if(row) row.style.background = bestCompareSelected.has(id) ? 'rgba(139,92,246,0.12)' : '';
}

function bestToggleAll(checked){
  if(checked){
    bestLastData.forEach(r => bestCompareSelected.add(r.id));
  } else {
    bestCompareSelected.clear();
  }
  bestUpdateCompareCount();
  document.querySelectorAll('input[data-best-cb]').forEach(cb => {
    const rid = parseInt(cb.getAttribute('data-best-cb'));
    cb.checked = bestCompareSelected.has(rid);
    const row = cb.closest('tr');
    if(row) row.style.background = cb.checked ? 'rgba(139,92,246,0.12)' : '';
  });
}

function bestUpdateCompareCount(){
  const el = document.getElementById('best-compare-count');
  if(el) el.textContent = bestCompareSelected.size;
  const btn = document.getElementById('best-compare-btn');
  if(btn) btn.disabled = bestCompareSelected.size < 2;
}

// ===== BEST COMPARE: Half-Kelly Simulation =====
let bestCompareData = [];
let bestCompareSource = 'best'; // 'best' or 'history'

function bestCompareGoBack(){
  switchTab(bestCompareSource === 'history' ? 'history' : 'best');
}

function _bcSetSource(src){
  bestCompareSource = src;
  const btn = document.getElementById('bc-back-btn');
  if(btn) btn.innerHTML = src === 'history' ? '&larr; Back to History' : '&larr; Back to Best Runs';
}

const DEFAULT_BET_FEE_RATE = 0.0156;

const BC_COLORS = [
  '#3b82f6','#10b981','#f59e0b','#ef4444','#8b5cf6','#ec4899','#06b6d4','#84cc16',
  '#f97316','#6366f1','#14b8a6','#e11d48','#a855f7','#0ea5e9','#eab308','#d946ef',
  '#22d3ee','#4ade80','#fb923c','#c084fc'
];

async function bestRunCompare(){
  if(bestCompareSelected.size < 2){ alert('Select at least 2 runs'); return; }
  const horizon = parseInt(document.getElementById('best-horizon').value) || 1;
  const ids = [...bestCompareSelected];
  try{
    const res = await fetch(API+'/api/best/compare', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({run_ids: ids, horizon})
    });
    bestCompareData = await res.json();
    _bcSetSource('best');
    switchTab('best_compare');
    bestCompareRecalc();
  }catch(e){ console.error(e); alert('Error loading comparison data'); }
}

function bestCompareRecalc(){
  if(!bestCompareData.length) return;
  const startBank = parseFloat(document.getElementById('bc-bank').value) || 1000;
  const buyPriceCents = parseFloat(document.getElementById('bc-buy-price').value) || 52;
  const maxBet = parseFloat(document.getElementById('bc-max-bet').value) || 500;
  const hkPctEl = document.getElementById('bc-hkpct');
  const fkPctEl = document.getElementById('bc-fkpct');
  const hkPct = hkPctEl ? (parseFloat(hkPctEl.value) / 100) : 0.017;
  const fkPct = fkPctEl ? (parseFloat(fkPctEl.value) / 100) : 0.0334;

  const feePctEl = document.getElementById('bc-fee');
  const feePct = feePctEl ? parseFloat(feePctEl.value) : NaN;
  const betFeeRate = (isFinite(feePct) && feePct >= 0) ? (feePct / 100) : DEFAULT_BET_FEE_RATE;

  // Simulate half Kelly + full Kelly using fixed % of bank per signal
  const simHalf = bestCompareData.map(run => bestSimKelly(run, startBank, buyPriceCents, maxBet, hkPct, betFeeRate));
  const simFull = bestCompareData.map(run => bestSimKelly(run, startBank, buyPriceCents, maxBet, fkPct, betFeeRate));

  bestRenderCompareChart(simHalf, startBank, 'bc-chart', 'Bank Growth (Half-Kelly)');
  bestRenderCompareTable(simHalf, startBank, 'bc-table', 'Month-by-Month Bank Progression (Half-Kelly)');

  bestRenderCompareChart(simFull, startBank, 'bc-chart-full', 'Bank Growth (Full Kelly)');
  bestRenderCompareTable(simFull, startBank, 'bc-table-full', 'Month-by-Month Bank Progression (Full Kelly)');

  // Summary remains half-kelly based (primary view)
  bestRenderCompareSummary(simHalf, startBank, buyPriceCents, maxBet);
}

/**
 * Polymarket economics:
 *   Buy 1 share at `cost` cents.
 *   Win  → get 100¢ back → profit = (100 - cost) cents.
 *   Lose → get   0¢ back → loss   = cost cents.
 *
 * Net odds ratio b = profit / loss = (100 - cost) / cost.
 * Kelly fraction  f* = (b·p − q) / b   where p = accuracy, q = 1−p.
 * Half-Kelly      f½ = f* / 2.
 * Breakeven accuracy = cost / 100  (e.g. 52¢ → need >52% to have edge).
 *
 * Per signal we wager `stake = bank × f½` dollars.
 *   Win  → bank += stake × b   (we risked stake, got back stake + stake·b)
 *   Lose → bank -= stake        (we lost the cost)
 */
function bestSimKelly(run, startBank, buyPriceCents, maxBet, betPct, betFeeRate){
  const cost = buyPriceCents / 100;                // e.g. 0.52
  const profitPerShare = 1.0 - cost;               // e.g. 0.48
  const b = profitPerShare / cost;                  // net odds ≈ 0.923

  const acc = (run.accuracy_pct || 50) / 100;
  const kellyFull = (b * acc - (1 - acc)) / b;
  const kellyApplied = (typeof betPct === 'number' ? betPct : 0.017);

  const monthly = (run.monthly || []).slice().sort((a,b)=>{
    const sa=String((a&&a.month)!=null?a.month:'');
    const sb=String((b&&b.month)!=null?b.month:'');
    const ta=Date.parse(sa+'-01');
    const tb=Date.parse(sb+'-01');
    if(!isNaN(ta) && !isNaN(tb)) return ta-tb;
    return sa.localeCompare(sb);
  });
  const label = `#${run.id} ${run.strategy} (${run.accuracy_pct}%, ${run.signals}sig)`;

  function _calcKellyApplied(){
    return (typeof betPct === 'number' ? betPct : 0.017);
  }

  function _simMonth(bank, nSignals, winProb, kh){
    // Expected value per $1 staked: EV = p*b - (1-p)
    // We use expected-value update (not a randomized path) to avoid artifacts
    // like growth in months with low accuracy due to rounding/order.
    const evPerDollar = (winProb * b) - (1 - winProb);

    let avgStake = 0;
    let maxStakeUsed = 0;
    let edgeSignals = 0;

    // If no bet fraction (or no signals), don't bet.
    // NOTE: kh may be negative; we still simulate betting (and losses) using |kh|.
    if(!(Math.abs(kh) > 0) || nSignals <= 0){
      return {bank, avgStake: 0, maxStakeUsed: 0, edgeSignals: 0};
    }

    for(let j = 0; j < nSignals; j++){
      // Cap by max bet and current bank.
      const rawStake = bank * Math.abs(kh);
      const stake = Math.max(0, Math.min(rawStake, maxBet || rawStake, bank));
      if(stake <= 0) break;

      bank += stake * evPerDollar;
      bank -= stake * (typeof betFeeRate === 'number' ? betFeeRate : DEFAULT_BET_FEE_RATE);
      if(bank < 0.01) bank = 0.01;

      avgStake += stake;
      if(stake > maxStakeUsed) maxStakeUsed = stake;
      edgeSignals += 1;
    }

    avgStake = edgeSignals ? (avgStake / edgeSignals) : 0;
    return {bank, avgStake, maxStakeUsed, edgeSignals};
  }

  if(!monthly.length){
    // If monthly breakdown is missing, fall back to no month-by-month sim.
    // (We keep bank flat because we can't reliably derive monthly accuracy/signals.)
    return {run, label, kellyHalf: kellyFull/2, kellyFull, kellyApplied, b, monthEntries: [], finalBank: startBank};
  }

  const monthEntries = [];
  let bank = startBank;
  for(const m of monthly){
    const mSignals = (m.total || 0);
    const mAcc = ((m.accuracy || 0) / 100);
    // Always use per-month accuracy for Kelly sizing, even if signals/month is overridden.
    // This avoids "49% month but still grows" artifacts.
    const mK = _calcKellyApplied(mAcc);
    const res = _simMonth(bank, mSignals, mAcc, mK);
    bank = res.bank;
    monthEntries.push({
      month: m.month || '?',
      bank: Math.round(bank*100)/100,
      signals: mSignals,
      edge_signals: res.edgeSignals,
      avg_stake: res.avgStake,
      max_stake: res.maxStakeUsed,
      accuracy: (m.accuracy||0)
    });
  }
  return {run, label, kellyHalf: kellyFull/2, kellyFull, kellyApplied, b, monthEntries, finalBank: bank};
}

function bestRenderCompareChart(simResults, startBank, targetElId, title){
  const el = document.getElementById(targetElId || 'bc-chart');
  if(!el) return;
  // Collect all unique months
  const allMonths = [];
  const monthSet = new Set();
  simResults.forEach(s => s.monthEntries.forEach(m => {
    if(!monthSet.has(m.month)){monthSet.add(m.month); allMonths.push(m.month);}
  }));

  // Build SVG chart
  const W = 900, H = 350, pad = {t:30,r:20,b:60,l:80};
  const plotW = W - pad.l - pad.r;
  const plotH = H - pad.t - pad.b;

  // Data series: [startBank, month1bank, month2bank, ...]
  const series = simResults.map(s => {
    const vals = [startBank];
    allMonths.forEach(month => {
      const entry = s.monthEntries.find(e => e.month === month);
      vals.push(entry ? entry.bank : vals[vals.length-1]);
    });
    return vals;
  });

  const allVals = series.flat();
  const minV = Math.min(...allVals) * 0.95;
  const maxV = Math.max(...allVals) * 1.05;
  const xLabels = ['Start', ...allMonths];
  const xStep = plotW / Math.max(1, xLabels.length - 1);

  function toX(i){ return pad.l + i * xStep; }
  function toY(v){ return pad.t + plotH - ((v - minV) / (maxV - minV)) * plotH; }

  let svg = `<svg viewBox="0 0 ${W} ${H}" style="width:100%;max-width:${W}px;height:auto" xmlns="http://www.w3.org/2000/svg">`;
  svg += `<rect width="${W}" height="${H}" fill="#0f172a" rx="8"/>`;

  // Grid lines
  const yTicks = 5;
  for(let i = 0; i <= yTicks; i++){
    const v = minV + (maxV - minV) * i / yTicks;
    const y = toY(v);
    svg += `<line x1="${pad.l}" y1="${y}" x2="${W-pad.r}" y2="${y}" stroke="#334155" stroke-width="0.5"/>`;
    svg += `<text x="${pad.l-8}" y="${y+4}" text-anchor="end" fill="#94a3b8" font-size="10">$${Math.round(v)}</text>`;
  }

  // X labels
  xLabels.forEach((lbl, i) => {
    const x = toX(i);
    svg += `<text x="${x}" y="${H-pad.b+18}" text-anchor="middle" fill="#94a3b8" font-size="10">${lbl}</text>`;
  });

  // Lines + dots
  series.forEach((vals, si) => {
    const color = BC_COLORS[si % BC_COLORS.length];
    let path = '';
    vals.forEach((v, i) => {
      const x = toX(i), y = toY(v);
      path += (i === 0 ? `M${x},${y}` : ` L${x},${y}`);
    });
    svg += `<path d="${path}" fill="none" stroke="${color}" stroke-width="2.5" stroke-linejoin="round"/>`;
    vals.forEach((v, i) => {
      const x = toX(i), y = toY(v);
      svg += `<circle cx="${x}" cy="${y}" r="3.5" fill="${color}" stroke="#0f172a" stroke-width="1"/>`;
    });
  });

  // Start bank reference line
  const sY = toY(startBank);
  svg += `<line x1="${pad.l}" y1="${sY}" x2="${W-pad.r}" y2="${sY}" stroke="#f59e0b" stroke-width="1" stroke-dasharray="6,3"/>`;
  svg += `<text x="${W-pad.r+2}" y="${sY+3}" fill="#f59e0b" font-size="9">start</text>`;

  svg += '</svg>';

  // Legend
  let legend = '<div class="flex flex-wrap gap-3 mt-3">';
  simResults.forEach((s, i) => {
    const color = BC_COLORS[i % BC_COLORS.length];
    legend += `<span class="flex items-center gap-1 text-xs"><span style="display:inline-block;width:12px;height:12px;border-radius:3px;background:${color}"></span>${s.label}</span>`;
  });
  legend += '</div>';

  el.innerHTML = `<div class="flex items-center justify-between mb-3"><h3 class="font-semibold">${title || 'Bank Growth'}</h3>${simResults.length === 1 ? `<button onclick="event.stopPropagation();showConfigPopover(this)" data-run-id="${simResults[0].run.id}" data-strategy="${simResults[0].run.strategy}" data-params='${JSON.stringify(simResults[0].run.params)}' class="text-blue-400 hover:text-blue-200 text-xs px-2 py-1 rounded border border-slate-600" title="View Config">⚙ ${simResults[0].run.strategy}</button>` : ''}</div>${svg}${legend}`;
}

function bestRenderCompareTable(simResults, startBank, targetElId, title){
  const el = document.getElementById(targetElId || 'bc-table');
  if(!el) return;
  if(!simResults.length){el.innerHTML='';return;}

  const isFull = (targetElId === 'bc-table-full');
  const kLabel = isFull ? 'K%' : '½K%';

  // Collect all months
  const allMonths = [];
  const monthSet = new Set();
  simResults.forEach(s => s.monthEntries.forEach(m => {
    if(!monthSet.has(m.month)){monthSet.add(m.month); allMonths.push(m.month);}
  }));

  // Ensure month columns are chronological so comparisons are against the previous month.
  // Month format is typically YYYY-MM which is lexicographically sortable.
  allMonths.sort((a,b)=>{
    const sa=String(a||'');
    const sb=String(b||'');
    const ta=Date.parse(sa+'-01');
    const tb=Date.parse(sb+'-01');
    if(!isNaN(ta) && !isNaN(tb)) return ta-tb;
    return sa.localeCompare(sb);
  });

  let html = `<div class="flex items-center justify-between mb-3"><h3 class="font-semibold">${title || 'Month-by-Month Bank Progression'}</h3>${simResults.length === 1 ? `<button onclick="event.stopPropagation();showConfigPopover(this)" data-run-id="${simResults[0].run.id}" data-strategy="${simResults[0].run.strategy}" data-params='${JSON.stringify(simResults[0].run.params)}' class="text-blue-400 hover:text-blue-200 text-xs px-2 py-1 rounded border border-slate-600" title="View Config">⚙ ${simResults[0].run.strategy}</button>` : ''}</div>`;
  html += `<div style="overflow-x:auto"><table><thead><tr><th>Run</th><th>${kLabel}</th><th>Start</th>`;
  allMonths.forEach(m => html += `<th>${m}</th>`);
  html += '<th>Final</th><th>ROI</th></tr></thead><tbody>';

  simResults.forEach((s, i) => {
    const color = BC_COLORS[i % BC_COLORS.length];
    const roi = ((s.finalBank - startBank) / startBank * 100).toFixed(1);
    const roiClass = s.finalBank >= startBank ? 'text-green-400' : 'text-red-400';
    html += `<tr><td><span style="display:inline-block;width:8px;height:8px;border-radius:2px;background:${color};margin-right:6px"></span><span class="text-xs">${s.label}</span></td>`;
    const kPct = ((isFull ? (s.kellyApplied != null ? s.kellyApplied : s.kellyFull) : s.kellyHalf) * 100);
    const kClass = kPct >= 0 ? 'text-slate-200' : 'text-red-400';
    html += `<td class="text-xs font-mono ${kClass}">${kPct.toFixed(2)}%</td>`;
    html += `<td class="font-mono text-xs">$${startBank.toFixed(0)}</td>`;
    let prevBank = startBank;
    allMonths.forEach(month => {
      const entry = s.monthEntries.find(e => e.month === month);
      if(entry){
        const bankClass = entry.bank >= startBank ? 'text-green-400' : 'text-red-400';
        const avgStake = entry.avg_stake ? entry.avg_stake.toFixed(0) : '0';
        const maxStake = entry.max_stake ? entry.max_stake.toFixed(0) : '0';
        const edgeSigs = entry.edge_signals != null ? entry.edge_signals : 0;
        // Highlight negative profit months (bank decreased from previous month)
        const bgStyle = (entry.bank < prevBank) ? 'background:rgba(239,68,68,0.12)' : '';
        html += `<td class="font-mono text-xs ${bankClass}" style="${bgStyle}" title="Signals: ${entry.signals}, Edge signals: ${edgeSigs}, Avg stake: $${avgStake}, Max stake: $${maxStake}">$${entry.bank.toFixed(0)}</td>`;
        prevBank = entry.bank;
      } else {
        html += '<td class="text-slate-500">—</td>';
      }
    });
    html += `<td class="font-mono font-bold ${roiClass}">$${s.finalBank.toFixed(0)}</td>`;
    html += `<td class="font-bold ${roiClass}">${roi}%</td>`;
    html += '</tr>';
  });

  html += '</tbody></table></div>';

  // Detail table with signals breakdown
  html += '<details class="mt-4"><summary class="text-xs text-slate-400 cursor-pointer font-semibold">Monthly Signal Details</summary>';
  html += '<div style="overflow-x:auto" class="mt-2"><table><thead><tr><th>Run</th>';
  allMonths.forEach(m => html += `<th colspan="2">${m}</th>`);
  html += '</tr><tr><th></th>';
  allMonths.forEach(() => html += '<th class="text-xs text-slate-500">Sig</th><th class="text-xs text-slate-500">Acc%</th>');
  html += '</tr></thead><tbody>';

  simResults.forEach((s, i) => {
    const color = BC_COLORS[i % BC_COLORS.length];
    html += `<tr><td><span style="display:inline-block;width:8px;height:8px;border-radius:2px;background:${color};margin-right:6px"></span><span class="text-xs">#${s.run.id}</span></td>`;
    allMonths.forEach(month => {
      const entry = s.monthEntries.find(e => e.month === month);
      if(entry){
        const a = entry.accuracy != null ? entry.accuracy : (entry.wins/Math.max(1,entry.signals)*100);
        html += `<td class="text-xs font-mono">${entry.signals}</td><td class="text-xs ${accClass(a)}">${a.toFixed(1)}%</td>`;
      } else {
        html += '<td>—</td><td>—</td>';
      }
    });
    html += '</tr>';
  });
  html += '</tbody></table></div></details>';

  el.innerHTML = html;
}

function bestRenderCompareSummary(simResults, startBank, buyPriceCents, maxBet){
  const el = document.getElementById('bc-summary');
  if(!simResults.length){el.innerHTML='';return;}

  const cost = (buyPriceCents || 52) / 100;
  const profitCents = 100 - (buyPriceCents || 52);
  const lossCents = buyPriceCents || 52;
  const bOdds = profitCents / lossCents;
  const breakeven = (cost * 100).toFixed(1);

  // Sort by final bank descending
  const sorted = [...simResults].sort((a,b) => b.finalBank - a.finalBank);

  let html = '<h3 class="font-semibold mb-3">Summary — Ranked by Final Bank</h3>';
  html += '<table><thead><tr><th>Rank</th><th>Run</th><th>Strategy</th><th>Accuracy</th><th>Signals</th><th>Kelly%</th><th>½Kelly%</th><th>Max Lose Streak</th><th>Final Bank</th><th>ROI</th><th>Profit</th><th></th></tr></thead><tbody>';

  sorted.forEach((s, i) => {
    const r = s.run;
    const roi = ((s.finalBank - startBank) / startBank * 100).toFixed(1);
    const profit = s.finalBank - startBank;
    const color = BC_COLORS[simResults.indexOf(s) % BC_COLORS.length];
    const roiClass = profit >= 0 ? 'text-green-400' : 'text-red-400';
    const hasEdge = r.accuracy_pct > parseFloat(breakeven);
    const kFullPct = (s.kellyFull*100);
    const kHalfPct = (s.kellyHalf*100);
    const kFullClass = kFullPct >= 0 ? 'text-slate-400' : 'text-red-400';
    const kHalfClass = kHalfPct >= 0 ? 'text-slate-200' : 'text-red-400';
    html += `<tr>`;
    html += `<td class="font-bold">${i+1}</td>`;
    html += `<td><span style="display:inline-block;width:8px;height:8px;border-radius:2px;background:${color};margin-right:6px"></span>#${r.id}</td>`;
    html += `<td class="font-medium">${r.strategy}</td>`;
    html += `<td class="${accClass(r.accuracy_pct)} font-bold">${r.accuracy_pct}%${!hasEdge ? ' <span class="text-red-500 text-xs">no edge</span>' : ''}</td>`;
    html += `<td>${r.signals}</td>`;
    html += `<td class="font-mono ${kFullClass}">${kFullPct.toFixed(2)}%</td>`;
    html += `<td class="font-mono ${kHalfClass}">${kHalfPct.toFixed(2)}%</td>`;
    html += `<td class="text-red-400">${r.max_lose_streak}</td>`;
    html += `<td class="font-bold font-mono ${roiClass}">$${s.finalBank.toFixed(0)}</td>`;
    html += `<td class="font-bold ${roiClass}">${roi}%</td>`;
    html += `<td class="font-mono ${roiClass}">${profit >= 0 ? '+' : ''}$${profit.toFixed(0)}</td>`;
    html += `<td><button onclick="event.stopPropagation();toggleConfigModal(${r.id}, '${r.strategy}', ${JSON.stringify(r.params).replace(/'/g, "\\'")})" class="text-blue-400 text-xs hover:underline">Config</button></td>`;
    html += '</tr>';
  });

  html += '</tbody></table>';

  // Economics box
  html += `<div class="mt-4 p-4 rounded-lg text-xs" style="background:#0f172a;border:1px solid #334155">
    <div class="text-slate-200 font-semibold mb-2">Polymarket Economics (Buy @ ${buyPriceCents||52}¢)</div>
    <div class="grid grid-cols-2 md:grid-cols-4 gap-3 mb-3">
      <div class="p-2 rounded" style="background:#1e293b"><span class="text-slate-400">Buy price:</span> <b>${lossCents}¢</b></div>
      <div class="p-2 rounded" style="background:#1e293b"><span class="text-slate-400">Win profit:</span> <b class="text-green-400">+${profitCents}¢</b></div>
      <div class="p-2 rounded" style="background:#1e293b"><span class="text-slate-400">Lose loss:</span> <b class="text-red-400">−${lossCents}¢</b></div>
      <div class="p-2 rounded" style="background:#1e293b"><span class="text-slate-400">Odds (b):</span> <b>${bOdds.toFixed(4)}</b></div>
      <div class="p-2 rounded" style="background:#1e293b"><span class="text-slate-400">Breakeven:</span> <b class="text-amber-400">${breakeven}%</b></div>
      <div class="p-2 rounded" style="background:#1e293b"><span class="text-slate-400">Max bet:</span> <b>$${(maxBet||0).toFixed(0)}</b></div>
    </div>
    <div class="text-slate-400 leading-relaxed">
      <b>Half-Kelly Criterion (asymmetric payoff):</b><br>
      <code>b = (100 − cost) / cost = ${profitCents}/${lossCents} ≈ ${bOdds.toFixed(4)}</code><br>
      <code>f* = (b·p − q) / b</code> &nbsp; where <code>p</code> = accuracy, <code>q = 1−p</code><br>
      <code>f½ = f* / 2</code> — we bet this fraction of our current bank per signal.<br><br>
      <b>On win:</b> bank += stake × b (profit is only ${profitCents}¢ per share, not 1:1).<br>
      <b>On lose:</b> bank -= stake (we lose our ${lossCents}¢ cost).<br><br>
      <b>Risk cap:</b> stake per signal is capped by <b>Max bet</b> input.<br>
      Because bets are always a <b>% of current bank</b>, lose streaks automatically shrink bet size, preventing ruin.<br>
      Strategies below <b>${breakeven}%</b> accuracy have <b>zero edge</b> at this buy price — Kelly = 0%, no bets placed.
    </div>
  </div>`;

  // Modal container for config
  html += `<div id="bc-config-modal" class="hidden fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50" onclick="if(event.target===this)toggleConfigModal()">
    <div class="bg-slate-800 rounded-lg p-6 max-w-2xl w-full mx-4 max-h-[80vh] overflow-auto">
      <div class="flex justify-between items-center mb-4">
        <h3 class="text-lg font-semibold">Strategy Config</h3>
        <button onclick="toggleConfigModal()" class="text-slate-400 hover:text-white text-2xl leading-none">&times;</button>
      </div>
      <div id="bc-config-content" class="text-xs font-mono bg-slate-900 p-3 rounded whitespace-pre-wrap"></div>
    </div>
  </div>`;

  el.innerHTML = html;
}

// Modal to show strategy config JSON (fallback for other pages)
function toggleConfigModal(runId, strategy, params){
  const modal = document.getElementById('bc-config-modal');
  const content = document.getElementById('bc-config-content');
  if(!modal || !content) return;
  if(modal.classList.contains('hidden')){
    content.innerHTML = `<div class="mb-2"><b>Run #${runId} — ${strategy}</b></div><pre>${JSON.stringify(params, null, 2)}</pre>`;
    modal.classList.remove('hidden');
  } else {
    modal.classList.add('hidden');
  }
}

// Popover for Best Compare: small floating tooltip next to button
function showConfigPopover(button){
  // Remove any existing popover
  const existing = document.getElementById('bc-config-popover');
  if(existing) existing.remove();

  const runId = button.getAttribute('data-run-id');
  const strategy = button.getAttribute('data-strategy');
  let params = {};
  try {
    params = JSON.parse(button.getAttribute('data-params') || '{}');
  } catch(e){
    console.error('Failed to parse data-params', e);
  }

  const popover = document.createElement('div');
  popover.id = 'bc-config-popover';
  popover.className = 'absolute z-50 p-3 rounded-lg shadow-lg text-xs font-mono';
  popover.style.cssText = 'background:#0f172a;border:1px solid #334155;min-width:280px;max-width:420px;max-height:320px;overflow:auto';
  popover.innerHTML = `
    <div class="flex items-center justify-between mb-2">
      <div><b>#${runId} — ${strategy}</b></div>
      <button onclick="this.parentElement.parentElement.remove()" class="text-slate-400 hover:text-white text-xs px-1">✕</button>
    </div>
    <pre style="margin:0;white-space:pre-wrap;word-break:break-all">${JSON.stringify(params, null, 2)}</pre>
    <div class="mt-2">
      <button onclick="navigator.clipboard.writeText(this.previousElementSibling.textContent).then(()=>{this.textContent='Copied!';setTimeout(()=>this.textContent='Copy JSON',1500)})" class="text-blue-400 hover:text-blue-200 text-xs font-semibold">Copy JSON</button>
    </div>
  `;

  // Position near button (bottom-left)
  const rect = button.getBoundingClientRect();
  popover.style.position = 'fixed';
  popover.style.top = `${rect.bottom + window.scrollY + 4}px`;
  popover.style.left = `${rect.left + window.scrollX}px`;

  document.body.appendChild(popover);

  // Auto-close on click outside
  const closeOnOutside = (e) => {
    if (!popover.contains(e.target) && e.target !== button) {
      popover.remove();
      document.removeEventListener('click', closeOnOutside);
    }
  };
  setTimeout(() => document.addEventListener('click', closeOnOutside), 0);
}

// ===== PREDICT CONFIG PANEL =====

let predictConfigData = null;

function togglePredictConfigPanel(){
  const panel = document.getElementById('predict-config-panel');
  if(!panel) return;
  if(panel.classList.contains('hidden')){
    panel.classList.remove('hidden');
    loadPredictConfig();
    loadPredictTemplateConfigs();
  } else {
    panel.classList.add('hidden');
  }
}

async function loadPredictConfig(){
  const textarea = document.getElementById('predict-config-json');
  if(!textarea) return;
  try{
    const res = await fetch(API + '/api/poly/settings');
    if(res.ok){
      const data = await res.json();
      predictConfigData = data;
      textarea.value = JSON.stringify(data, null, 2);
    } else {
      textarea.value = 'Error loading config';
    }
  }catch(e){
    textarea.value = 'Network error';
  }
}

async function savePredictConfig(){
  const textarea = document.getElementById('predict-config-json');
  if(!textarea) return;
  try{
    const json = JSON.parse(textarea.value);
    const res = await fetch(API + '/api/poly/settings', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify(json)
    });
    if(res.ok){
      alert('Config saved');
      predictConfigData = json;
    } else {
      alert('Failed to save config');
    }
  }catch(e){
    alert('Invalid JSON');
  }
}

async function loadPredictTemplateConfigs(){
  const container = document.getElementById('predict-template-configs');
  if(!container) return;
  try{
    const res = await fetch(API + '/api/poly/pred_templates');
    if(!res.ok){ container.innerHTML = '<div class="text-red-400">Failed to load templates</div>'; return; }
    const templates = await res.json();
    if(!Array.isArray(templates)){ container.innerHTML = '<div class="text-red-400">Invalid response</div>'; return; }
    let html = '';
    templates.forEach(t => {
      const paramsJson = JSON.stringify(t.params || {}, null, 2);
      const statusDot = t.active ? '<span class="inline-block w-2 h-2 rounded-full" style="background:#22c55e"></span>' : '<span class="inline-block w-2 h-2 rounded-full" style="background:#64748b"></span>';
      html += `<div class="mb-2 p-2 rounded-lg" style="background:#0b1220;border:1px solid #334155">
        <div class="flex items-center justify-between gap-2 mb-1">
          <div class="flex items-center gap-2">
            ${statusDot}
            <span class="font-semibold text-xs text-slate-200">${t.name || '(unnamed)'}</span>
            <span class="text-slate-500 text-[11px] font-mono">${t.strategy}</span>
            <span class="text-slate-600 text-[11px]">win=${t.window_size} H${t.horizon}</span>
          </div>
          <div class="flex items-center gap-3">
            <button onclick="reloadPredictTemplateConfig(${t.id})" class="text-blue-400 hover:text-blue-200 text-xs font-semibold" title="Reload this template">↺</button>
            <button onclick="savePredictTemplateConfig(${t.id})" class="text-green-400 hover:text-green-200 text-xs font-semibold" title="Save params JSON">Save</button>
          </div>
        </div>
        <textarea id="predict-template-config-${t.id}" rows="4" class="w-full text-xs font-mono" style="background:#0f172a;border:1px solid #334155;border-radius:8px;padding:8px 10px">${paramsJson}</textarea>
      </div>`;
    });
    container.innerHTML = html || '<div class="text-slate-500">No templates</div>';
  }catch(e){
    container.innerHTML = '<div class="text-red-400">Network error</div>';
  }
}

function reloadPredictTemplateConfig(templateId){
  // Simple approach: reload all templates and re-render.
  // Keeps UI consistent and avoids partial DOM diff complexity.
  loadPredictTemplateConfigs();
}

async function savePredictTemplateConfig(templateId){
  const textarea = document.getElementById(`predict-template-config-${templateId}`);
  if(!textarea) return;
  try{
    const params = JSON.parse(textarea.value);
    const res = await fetch(API + `/api/poly/pred_templates/${templateId}`, {
      method:'PUT',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ params })
    });
    if(res.ok){
      // Silent success to match the compact UX in Polymarket popup.
      await loadPredictTemplateConfigs();
    } else {
      alert('Failed to save template config');
    }
  }catch(e){
    alert('Invalid JSON');
  }
}

// ===== ANALYTICS =====
(function _initAnalyticsHours(){
  const hours = Array.from({length:24},(_,i)=>i);
  ['an-hour-from','an-hour-to'].forEach(id=>{
    const sel = document.getElementById(id);
    if(!sel) return;
    hours.forEach(h=>{
      const opt = document.createElement('option');
      opt.value = h;
      opt.textContent = String(h).padStart(2,'0')+':00';
      sel.appendChild(opt);
    });
  });
})();

function analyticsClearFilters(){
  ['an-date-from','an-date-to'].forEach(id=>{
    const el=document.getElementById(id); if(el) el.value='';
  });
  ['an-hour-from','an-hour-to'].forEach(id=>{
    const el=document.getElementById(id); if(el) el.value='';
  });
  analyticsLoad();
}

function analyticsSetLastDays(days){
  const d = new Date();
  const to = new Date(d.getFullYear(), d.getMonth(), d.getDate());
  const from = new Date(to);
  from.setDate(from.getDate() - Math.max(1, parseInt(days||1,10)) + 1);
  const fmt = (x)=>{
    const y=x.getFullYear();
    const m=String(x.getMonth()+1).padStart(2,'0');
    const dd=String(x.getDate()).padStart(2,'0');
    return `${y}-${m}-${dd}`;
  };
  const fromEl=document.getElementById('an-date-from');
  const toEl=document.getElementById('an-date-to');
  if(fromEl) fromEl.value = fmt(from);
  if(toEl) toEl.value = fmt(to);
  analyticsLoad();
}

async function analyticsLoad(){
  const dateFrom = document.getElementById('an-date-from')?.value || '';
  const dateTo   = document.getElementById('an-date-to')?.value   || '';
  const hourFrom = document.getElementById('an-hour-from')?.value ?? '';
  const hourTo   = document.getElementById('an-hour-to')?.value   ?? '';

  let url = API+'/api/analytics/predictions';
  const ps=[];
  if(dateFrom) ps.push('date_from='+encodeURIComponent(dateFrom));
  if(dateTo)   ps.push('date_to='+encodeURIComponent(dateTo));
  if(hourFrom!=='') ps.push('hour_from='+encodeURIComponent(hourFrom));
  if(hourTo!=='')   ps.push('hour_to='+encodeURIComponent(hourTo));
  if(ps.length) url+='?'+ps.join('&');

  try{
    const res = await fetch(url);
    const data = await res.json();
    analyticsRenderSummary(data.summary);
    analyticsRenderTemplates(data.per_template);
    analyticsRenderPerDay(data.per_day);
    analyticsRenderPerHour(data.per_hour);
  }catch(e){ console.error('Analytics error',e); }
  analyticsAskLoad();
}

async function analyticsAskLoad(){
  const dateFrom = document.getElementById('an-date-from')?.value || '';
  const dateTo   = document.getElementById('an-date-to')?.value   || '';
  const hourFrom = document.getElementById('an-hour-from')?.value ?? '';
  const hourTo   = document.getElementById('an-hour-to')?.value   ?? '';
  const windowSec = parseInt(document.getElementById('an-ask-window')?.value || '10', 10) || 10;

  let url = API+'/api/analytics/ask_prices?window_sec='+windowSec;
  if(dateFrom) url += '&date_from='+encodeURIComponent(dateFrom);
  if(dateTo)   url += '&date_to='+encodeURIComponent(dateTo);
  if(hourFrom!=='') url += '&hour_from='+encodeURIComponent(hourFrom);
  if(hourTo!=='')   url += '&hour_to='+encodeURIComponent(hourTo);

  // Show loading state
  ['an-ask-summary','an-ask-buckets','an-ask-hist','an-ask-perday'].forEach(id=>{
    const el=document.getElementById(id);
    if(el && !el.innerHTML) el.innerHTML='<div class="text-slate-500 text-xs">Loading…</div>';
  });

  try{
    const res = await fetch(url);
    const data = await res.json();
    analyticsRenderAskSummary(data.summary, data.window_sec);
    analyticsRenderAskBuckets(data.buckets, data.summary);
    analyticsRenderAskDepth(data.depth);
    analyticsRenderAskPerDay(data.per_day);
  }catch(e){ console.error('Ask analysis error',e); }
}

function analyticsRenderAskSummary(s, windowSec){
  const el=document.getElementById('an-ask-summary');
  if(!el||!s) return;
  const na = v => v!=null ? v+'¢' : '—';
  const covColor = s.coverage_pct>=80?'#22c55e':s.coverage_pct>=50?'#eab308':'#ef4444';
  const cards=[
    {label:'Predictions Analyzed', value:s.total_preds,       sub:'UP/DOWN before market start',           color:'#8b5cf6'},
    {label:'With Snapshot',        value:s.preds_with_snap,   sub:`orderbook found within ${windowSec}s`,  color:'#3b82f6'},
    {label:'Coverage',             value:s.coverage_pct+'%',  sub:'% of predictions with data',            color:covColor},
    {label:'Avg Best Ask',         value:na(s.avg_min_ask),   sub:'best (lowest) ask per prediction',      color:'#06b6d4'},
    {label:'Lowest Ask Seen',      value:na(s.overall_min_ask),sub:'cheapest ever available',              color:'#10b981'},
  ];
  el.innerHTML = cards.map(c=>`<div class="card p-5" style="border-left:3px solid ${c.color}">
    <div class="text-xs text-slate-400 mb-1">${c.label}</div>
    <div class="text-2xl font-bold" style="color:${c.color}">${c.value}</div>
    <div class="text-xs text-slate-500 mt-1">${c.sub}</div>
  </div>`).join('');
}

function analyticsRenderAskBuckets(buckets, summary){
  const el=document.getElementById('an-ask-buckets');
  if(!el||!buckets) return;
  const defs=[
    {key:'51', label:'Ask ≤ 51¢', color:'#22c55e'},
    {key:'52', label:'Ask ≤ 52¢', color:'#eab308'},
    {key:'53', label:'Ask ≤ 53¢', color:'#f97316'},
  ];
  el.innerHTML = defs.map(def=>{
    const b = buckets[def.key] || {};
    const pct = b.pct || 0;
    const cnt = b.cnt || 0;
    const avgAsk = b.avg_ask != null ? b.avg_ask+'¢' : '—';
    const avgAmt = b.avg_amount != null ? b.avg_amount : '—';
    return `<div class="card p-5" style="border-left:3px solid ${def.color}">
      <div class="text-xs text-slate-400 mb-2 font-semibold">${def.label}</div>
      <div class="text-2xl font-bold" style="color:${def.color}">${pct}%</div>
      <div class="text-xs text-slate-300 mt-1">${cnt} predictions</div>
      <div class="mt-3">
        <div class="w-full bg-slate-700 rounded-full" style="height:4px">
          <div style="width:${Math.min(pct,100)}%;height:4px;background:${def.color};border-radius:9999px"></div>
        </div>
      </div>
      <div class="text-xs text-slate-400 mt-2">Avg ask: <span class="font-mono" style="color:${def.color}">${avgAsk}</span></div>
      <div class="text-xs text-slate-400 mt-1">Avg available qty: <span class="font-mono font-bold" style="color:${def.color}">${avgAmt}</span></div>
    </div>`;
  }).join('');
}

function analyticsRenderAskDepth(rows){
  const el=document.getElementById('an-ask-hist');
  if(!el) return;
  if(!rows||!rows.length){el.innerHTML='<div class="text-slate-500 text-xs">No depth data</div>';return;}

  // Keep chart/table readable: hide very high price levels
  rows = rows.filter(r => {
    if(!(r && typeof r.price === 'number')) return false;
    // Only integer-cent levels, and only the range you care about
    if(r.price < 48 || r.price > 55) return false;
    return Math.abs(r.price - Math.round(r.price)) < 1e-9;
  });
  if(!rows.length){el.innerHTML='<div class="text-slate-500 text-xs">No depth data</div>';return;}

  // Bar chart by avg_cumul amount
  const maxAmt = Math.max(...rows.map(r=>r.avg_cumul||0), 1);
  let html=`<h3 class="font-semibold mb-1">Orderbook Depth — Avg Cumulative Amount Available</h3>`;
  html+=`<p class="text-xs text-slate-500 mb-3">At each price (¢), the average total quantity available at that price or cheaper in the first snapshot after prediction.</p>`;
  html+=`<div class="flex items-end gap-px mb-2" style="height:90px">`;
  rows.forEach(r=>{
    const barPct = Math.round((r.avg_cumul||0)/maxAmt*100);
    const color = r.price<=51?'#22c55e':r.price<=52?'#eab308':r.price<=53?'#f97316':'#64748b';
    html+=`<div style="flex:1;display:flex;flex-direction:column;align-items:center;justify-content:flex-end;height:100%"
         title="≤${r.price}¢: ${r.avg_cumul} avg qty (${r.pct}% of preds have any)">
      <div style="width:100%;background:${color};height:${barPct}%;min-height:${r.avg_cumul?2:0}px;border-radius:2px 2px 0 0"></div>
      <div class="text-slate-500" style="font-size:9px;margin-top:2px;writing-mode:vertical-rl;transform:rotate(180deg)">${r.price}</div>
    </div>`;
  });
  html+=`</div>`;

  // Table
  html+=`<div style="overflow-x:auto" class="mt-3"><table>
    <thead><tr>
      <th>Price ≤ (¢)</th>
      <th>Preds with liq.</th>
      <th>% of preds</th>
      <th>Avg cumul. qty</th>
    </tr></thead><tbody>`;
  rows.forEach(r=>{
    const color = r.price<=51?'text-green-400':r.price<=52?'text-yellow-400':r.price<=53?'text-orange-400':'text-slate-300';
    html+=`<tr>
      <td class="font-mono text-xs font-bold ${color}">${r.price}¢</td>
      <td class="font-mono text-xs">${r.count}</td>
      <td class="font-mono text-xs">${r.pct}%</td>
      <td class="font-mono text-xs font-bold ${color}">${r.avg_cumul}</td>
    </tr>`;
  });
  html+=`</tbody></table></div>`;
  el.innerHTML=html;
}

function analyticsRenderAskPerDay(rows){
  const el=document.getElementById('an-ask-perday');
  if(!el) return;
  if(!rows||!rows.length){el.innerHTML='';return;}
  let html=`<h3 class="font-semibold mb-3">By Day — Ask Price &amp; Available Quantity</h3><div style="overflow-x:auto"><table>
    <thead><tr>
      <th>Date</th><th>Preds</th><th>Avg Ask</th>
      <th>≤51¢ cnt</th><th>≤51¢ avg qty</th>
      <th>≤52¢ cnt</th><th>≤52¢ avg qty</th>
      <th>≤53¢ cnt</th><th>≤53¢ avg qty</th>
    </tr></thead><tbody>`;
  rows.forEach(r=>{
    const pct51=r.preds>0?Math.round((r.cnt_le51||0)/r.preds*100):0;
    const pct52=r.preds>0?Math.round((r.cnt_le52||0)/r.preds*100):0;
    const pct53=r.preds>0?Math.round((r.cnt_le53||0)/r.preds*100):0;
    const amt = (v,cls) => v!=null?`<span class="font-bold ${cls}">${v}</span>`:'—';
    html+=`<tr>
      <td class="font-mono text-xs">${r.day}</td>
      <td class="font-mono text-xs">${r.preds}</td>
      <td class="font-mono text-xs">${r.avg_ask!=null?r.avg_ask+'¢':'—'}</td>
      <td class="font-mono text-xs text-green-400">${r.cnt_le51||0} <span class="text-slate-500">(${pct51}%)</span></td>
      <td class="font-mono text-xs">${amt(r.avg_amt_le51,'text-green-400')}</td>
      <td class="font-mono text-xs text-yellow-400">${r.cnt_le52||0} <span class="text-slate-500">(${pct52}%)</span></td>
      <td class="font-mono text-xs">${amt(r.avg_amt_le52,'text-yellow-400')}</td>
      <td class="font-mono text-xs text-orange-400">${r.cnt_le53||0} <span class="text-slate-500">(${pct53}%)</span></td>
      <td class="font-mono text-xs">${amt(r.avg_amt_le53,'text-orange-400')}</td>
    </tr>`;
  });
  html+='</tbody></table></div>';
  el.innerHTML=html;
}

async function analyticsOpenPolyMarket(slug){
  if(!slug) return;
  try{
    switchTab('poly');
    if(typeof selectPolyMarket === 'function'){
      await selectPolyMarket(String(slug));
    } else if(typeof showPolyMarket === 'function'){
      await showPolyMarket(String(slug));
    }
  }catch(e){
    console.error('analyticsOpenPolyMarket error:', e);
  }
}

async function analyticsKellyLoad(){
  const dateFrom  = document.getElementById('an-date-from')?.value  || '';
  const dateTo    = document.getElementById('an-date-to')?.value    || '';
  const hourFrom  = document.getElementById('an-hour-from')?.value  ?? '';
  const hourTo    = document.getElementById('an-hour-to')?.value    ?? '';
  const startBank = parseFloat(document.getElementById('an-kelly-bank')?.value  || '100') || 100;
  const maxBetRaw = document.getElementById('an-kelly-maxbet')?.value?.trim();
  const maxBet    = maxBetRaw ? parseFloat(maxBetRaw) : null;
  const feePctRaw = document.getElementById('an-kelly-fee')?.value?.trim();
  const feeRate   = feePctRaw !== '' && feePctRaw != null ? (parseFloat(feePctRaw) / 100) : 0.0156;
  const maxPrice  = parseFloat(document.getElementById('an-kelly-maxprice')?.value || '51') || 51;
  const hkPctRaw  = document.getElementById('an-kelly-hkpct')?.value?.trim();
  const fkPctRaw  = document.getElementById('an-kelly-fkpct')?.value?.trim();
  const hkPct     = hkPctRaw !== '' && hkPctRaw != null ? (parseFloat(hkPctRaw) / 100) : 0.017;
  const fkPct     = fkPctRaw !== '' && fkPctRaw != null ? (parseFloat(fkPctRaw) / 100) : 0.0334;

  const ps = [`start_bank=${startBank}`, `fee_rate=${feeRate}`, `max_price_cents=${maxPrice}`, `hk_pct=${hkPct}`, `fk_pct=${fkPct}`];
  if(dateFrom) ps.push('date_from='+encodeURIComponent(dateFrom));
  if(dateTo)   ps.push('date_to='+encodeURIComponent(dateTo));
  if(hourFrom !== '') ps.push('hour_from='+hourFrom);
  if(hourTo   !== '') ps.push('hour_to='+hourTo);
  if(maxBet != null)  ps.push('max_bet='+maxBet);
  const url = API+'/api/analytics/kelly_sim?'+ps.join('&');

  ['an-kelly-summary','an-kelly-table'].forEach(id=>{
    const el=document.getElementById(id);
    if(el) el.innerHTML='<div class="text-slate-500 text-xs">Loading…</div>';
  });

  try{
    const res = await fetch(url);
    const data = await res.json();
    analyticsKellyRenderSummary(data);
    analyticsKellyRenderTable(data);
  }catch(e){ console.error('Kelly sim error',e); }
}

function analyticsKellyRenderSummary(d){
  const el = document.getElementById('an-kelly-summary');
  if(!el) return;
  const hk = d.half_kelly || {};
  const fk = d.full_kelly || {};
  const roi = (v, sb) => {
    if(v==null) return '—';
    const pct = ((v - sb) / sb * 100).toFixed(2);
    const cls = v >= sb ? 'text-green-400' : 'text-red-400';
    return `<span class="${cls}">${pct}%</span>`;
  };
  const $b = (v) => v!=null ? `$${v.toFixed(2)}` : '—';
  const maxB = d.max_bet != null ? `$${d.max_bet}` : 'none';
  const mpc  = d.max_price_cents != null ? d.max_price_cents+'¢' : '—';
  const feeStr = d.fee_rate!=null ? (d.fee_rate*100).toFixed(2)+'%' : '—';
  const hkPctStr = d.hk_pct!=null ? (d.hk_pct*100).toFixed(2)+'%' : '—';
  const fkPctStr = d.fk_pct!=null ? (d.fk_pct*100).toFixed(2)+'%' : '—';
  const skipTotal = d.skipped_price||0;
  const skipSub = `ask > ${mpc}: ${skipTotal}`;
  const cards = [
    {label:'Settings',          value:`${mpc} max`,                 sub:`bank: $${d.start_bank} · ½K: ${hkPctStr} · FK: ${fkPctStr} · cap: ${maxB} · fee: ${feeStr}`, color:'#64748b'},
    {label:'Resolved w/ Snap',  value:d.total_resolved||0,          sub:`predictions with orderbook data`,          color:'#8b5cf6'},
    {label:'Trades Executed',   value:d.total_trades,               sub:`${d.total_wins} wins · ${d.win_pct}% acc`, color:'#06b6d4'},
    {label:'Skipped',           value:skipTotal,                    sub:skipSub,                                    color:'#475569'},
    {label:'½ Kelly End',       value:$b(hk.end_bank),              sub:roi(hk.end_bank, d.start_bank)+' ROI',      color: (hk.end_bank||0)>=d.start_bank?'#22c55e':'#ef4444'},
    {label:'Full Kelly End',    value:$b(fk.end_bank),              sub:roi(fk.end_bank, d.start_bank)+' ROI',      color: (fk.end_bank||0)>=d.start_bank?'#3b82f6':'#ef4444'},
  ];
  el.innerHTML = cards.map(c=>`<div class="card p-5" style="border-left:3px solid ${c.color}">
    <div class="text-xs text-slate-400 mb-1">${c.label}</div>
    <div class="text-2xl font-bold" style="color:${c.color}">${c.value}</div>
    <div class="text-xs text-slate-500 mt-1">${c.sub}</div>
  </div>`).join('');
}

function analyticsKellyRenderTable(d){
  const el = document.getElementById('an-kelly-table');
  if(!el) return;
  const trades = d.trades || [];

  if(!trades.length){
    el.innerHTML='<div class="text-slate-500 text-xs">No trades — no resolved predictions with ask ≤ '+
      (d.max_price_cents||51)+'¢ found for the selected filters.</div>';
    return;
  }

  const $  = (v,dp) => v!=null ? v.toFixed(dp??2) : '—';
  const profitTd = (v, skipped) => {
    if(v==null) return '<td class="font-mono text-xs text-slate-500">—</td>';
    if(skipped) return `<td class="font-mono text-xs text-slate-400">${v>=0?'+':''}$${v.toFixed(2)}</td>`;
    const cls = v>=0?'text-green-400':'text-red-400';
    const sign = v>=0?'+':'';
    return `<td class="font-mono text-xs ${cls}">${sign}$${v.toFixed(2)}</td>`;
  };
  const bankTd = (v, sb) => {
    if(v==null) return '<td class="font-mono text-xs text-slate-500">—</td>';
    const cls = v>=sb?'text-green-400':'text-red-400';
    return `<td class="font-mono text-xs font-bold ${cls}">$${v.toFixed(2)}</td>`;
  };
  const sb = d.start_bank || 100;

  const calcAvgFill = (fillKey, sharesKey) => {
    let totalShares = 0;
    let weightedCents = 0;
    trades.forEach(t => {
      if(t.skipped) return;
      const shares = Number(t[sharesKey] || 0);
      const fill = Number(t[fillKey] || 0);
      if(!isFinite(shares) || !isFinite(fill) || shares <= 0 || fill <= 0) return;
      totalShares += shares;
      weightedCents += shares * fill;
    });
    if(totalShares <= 0) return null;
    return weightedCents / totalShares;
  };

  const hkAvgFill = calcAvgFill('hk_fill', 'hk_shares');
  const fkAvgFill = calcAvgFill('fk_fill', 'fk_shares');
  const avgFillText = (() => {
    if(hkAvgFill == null && fkAvgFill == null) return '';
    const fmt = (cents) => `${cents.toFixed(2)}¢ ($${(cents/100).toFixed(4)})`;
    const parts = [];
    if(hkAvgFill != null) parts.push(`<span class="text-slate-200">½K: <strong>${fmt(hkAvgFill)}</strong></span>`);
    if(fkAvgFill != null) parts.push(`<span class="text-slate-200">Full K: <strong>${fmt(fkAvgFill)}</strong></span>`);
    return `<div class="text-xs text-slate-400 mb-2">Средняя реальная стоимость выкупа · ${parts.join(' · ')}</div>`;
  })();

  let html = `${avgFillText}<h3 class="font-semibold mb-3">All Simulated Trades (${trades.length})</h3>
  <div style="overflow-x:auto"><table style="white-space:nowrap">
    <thead><tr>
      <th>#</th><th>Date</th><th>Pr Time</th><th>Market</th><th>Pred</th><th>✓</th>
      <th>Ask ¢</th>
      <th>½K Bet $</th><th>½K Fill ¢</th><th>½K P&L</th><th>½K Bank</th>
      <th>FK Bet $</th><th>FK Fill ¢</th><th>FK P&L</th><th>FK Bank</th>
    </tr></thead><tbody>`;

  trades.forEach((t,i) => {
    const okCls = t.correct ? 'text-green-400' : 'text-red-400';
    const okTxt = t.correct ? '✓' : '✗';
    const skipped = !!t.skipped;
    const rowCls = skipped ? 'opacity:0.7' : '';
    const reason = skipped ? (t.skip_reason||'skipped') : '';
    const askTxt = (t.best_ask==null) ? '—' : `${$(t.best_ask,2)}¢`;
    const slugEsc = String(t.slug||'').replace(/'/g,"\\'");
    const slugHtml = (t.slug||'—')==='—'
      ? '—'
      : `<a href="#" class="${skipped ? 'text-slate-400 hover:underline' : 'text-blue-400 hover:underline'}" onclick="analyticsOpenPolyMarket('${slugEsc}');return false;">${t.slug}</a>`;
    html+=`<tr>
      <td class="font-mono text-xs ${skipped ? 'text-slate-400' : 'text-slate-500'}" style="${rowCls}">${i+1}</td>
      <td class="font-mono text-xs ${skipped ? 'text-slate-400 italic' : ''}">${skipped ? reason : t.date}</td>
      <td class="font-mono text-xs ${skipped ? 'text-slate-400' : ''}">${t.time ? t.time.slice(0,5) : '—'}</td>
      <td class="font-mono text-xs ${skipped ? 'text-slate-400' : 'text-slate-300'}">${slugHtml}</td>
      <td class="font-mono text-xs ${skipped ? 'text-slate-400' : ''}">${t.pred}</td>
      <td class="font-mono text-xs font-bold ${skipped ? 'text-slate-400' : okCls}">${okTxt}</td>
      <td class="font-mono text-xs ${skipped ? 'text-slate-400' : ''}">${askTxt}</td>
      <td class="font-mono text-xs ${skipped ? 'text-slate-400' : ''}">${skipped ? '—' : `$${$(t.hk_bet)}`}</td>
      <td class="font-mono text-xs ${skipped ? 'text-slate-400' : ''}">${skipped ? '—' : `${$(t.hk_fill,2)}¢`}</td>
      ${skipped ? '<td class="font-mono text-xs text-slate-400">—</td>' : profitTd(t.hk_profit, false)}
      <td class="font-mono text-xs font-bold ${skipped ? 'text-slate-400' : (t.hk_profit >= 0 ? 'text-green-400' : 'text-red-400')}" style="background:#1e293b">
        ${skipped ? '—' : `$${(t.hk_bank||0).toFixed(2)} ${t.hk_profit >= 0 ? '▲' : '▼'}`}
      </td>
      <td class="font-mono text-xs ${skipped ? 'text-slate-400' : ''}">${skipped ? '—' : `$${$(t.fk_bet)}`}</td>
      <td class="font-mono text-xs ${skipped ? 'text-slate-400' : ''}">${skipped ? '—' : `${$(t.fk_fill,2)}¢`}</td>
      ${skipped ? '<td class="font-mono text-xs text-slate-400">—</td>' : profitTd(t.fk_profit, false)}
      <td class="font-mono text-xs font-bold ${skipped ? 'text-slate-400' : (t.correct ? 'text-green-400' : 'text-red-400')}" style="background:#1e293b">
        ${skipped ? '—' : `$${$(t.fk_bank)} ${t.correct ? '▲' : '▼'}`}
      </td>
    </tr>`;
  });

  html+='</tbody></table></div>';
  el.innerHTML = html;
}

function _anAccBadge(pct){
  if(pct===null||pct===undefined) return '<span class="text-slate-500 text-xs">n/a</span>';
  const cls = pct>=55?'text-green-400':pct>=50?'text-yellow-400':'text-red-400';
  return `<span class="font-bold ${cls}">${pct}%</span>`;
}

function analyticsRenderSummary(s){
  const el=document.getElementById('an-summary');
  if(!el||!s) return;
  const cards=[
    {label:'Predicted Markets',    value: s.total_markets,     sub:'UP or DOWN before market start'},
    {label:'Total Predictions',    value: s.total_predictions,  sub:'individual template runs'},
    {label:'Correct Predictions',  value: s.correct_count !== undefined ? `${s.correct_count} / ${s.resolved_count}` : '—', sub:'of resolved markets'},
    {label:'Accuracy',             value: s.correct_pct!==null&&s.correct_pct!==undefined ? s.correct_pct+'%' : '—', sub:'% correct on resolved', accent: s.correct_pct!=null?(s.correct_pct>=55?'green':s.correct_pct>=50?'yellow':'red'):null},
    {label:'Avg / Day',            value: s.avg_per_day,       sub:'predictions per active day'},
    {label:'Avg / Hour',           value: s.avg_per_hour,      sub:'predictions per active hour'},
  ];
  const accentColor={'green':'#22c55e','yellow':'#eab308','red':'#ef4444'};
  el.innerHTML = cards.map(c=>{
    const color = c.accent ? accentColor[c.accent] : '#8b5cf6';
    return `<div class="card p-5" style="border-left:3px solid ${color}">
      <div class="text-xs text-slate-400 mb-1">${c.label}</div>
      <div class="text-2xl font-bold" style="color:${color}">${c.value}</div>
      <div class="text-xs text-slate-500 mt-1">${c.sub}</div>
    </div>`;
  }).join('');
}

function analyticsRenderTemplates(rows){
  const el=document.getElementById('an-templates');
  if(!el) return;
  if(!rows||!rows.length){el.innerHTML='<div class="text-slate-400 text-sm">No data</div>';return;}
  let html=`<h3 class="font-semibold mb-3">By Template</h3><div style="overflow-x:auto"><table>
    <thead><tr><th>Template</th><th>Predictions</th><th>Markets</th><th>Resolved</th><th>Correct</th><th>Accuracy</th></tr></thead><tbody>`;
  rows.forEach(r=>{
    html+=`<tr>
      <td class="font-medium text-xs">${r.template}</td>
      <td class="font-mono text-xs">${r.predictions}</td>
      <td class="font-mono text-xs">${r.markets}</td>
      <td class="font-mono text-xs">${r.resolved}</td>
      <td class="font-mono text-xs">${r.correct}</td>
      <td>${_anAccBadge(r.correct_pct)}</td>
    </tr>`;
  });
  html+='</tbody></table></div>';
  el.innerHTML=html;
}

function analyticsRenderPerDay(rows){
  const el=document.getElementById('an-per-day');
  if(!el) return;
  if(!rows||!rows.length){el.innerHTML='';return;}
  let html=`<h3 class="font-semibold mb-3">By Day</h3><div style="overflow-x:auto"><table>
    <thead><tr><th>Date</th><th>Predictions</th><th>Markets</th><th>Resolved</th><th>Correct</th><th>Accuracy</th></tr></thead><tbody>`;
  rows.forEach(r=>{
    html+=`<tr>
      <td class="font-mono text-xs">${r.day}</td>
      <td class="font-mono text-xs">${r.predictions}</td>
      <td class="font-mono text-xs">${r.markets}</td>
      <td class="font-mono text-xs">${r.resolved}</td>
      <td class="font-mono text-xs">${r.correct}</td>
      <td>${_anAccBadge(r.correct_pct)}</td>
    </tr>`;
  });
  html+='</tbody></table></div>';
  el.innerHTML=html;
}

function analyticsRenderPerHour(rows){
  const el=document.getElementById('an-per-hour');
  if(!el) return;
  if(!rows||!rows.length){el.innerHTML='';return;}

  // Bar chart for predictions per hour
  const maxPred = Math.max(...rows.map(r=>r.predictions),1);
  let html=`<h3 class="font-semibold mb-3">By Hour (UTC)</h3>`;

  // Mini bar chart
  html+=`<div class="flex items-end gap-1 mb-4" style="height:80px">`;
  for(let h=0;h<24;h++){
    const r=rows.find(x=>x.hour===h);
    const pct=r?Math.round(r.predictions/maxPred*100):0;
    const acc=r?.correct_pct;
    const barColor=acc!=null?(acc>=55?'#22c55e':acc>=50?'#eab308':'#ef4444'):'#334155';
    const title=r?`${String(h).padStart(2,'0')}:00 — ${r.predictions} preds, ${acc!=null?acc+'%':'n/a'} acc`:`${String(h).padStart(2,'0')}:00 — no data`;
    html+=`<div style="flex:1;display:flex;flex-direction:column;align-items:center;justify-content:flex-end;height:100%" title="${title}">
      <div style="width:100%;background:${barColor};height:${pct}%;min-height:${r?2:0}px;border-radius:2px 2px 0 0"></div>
      <div class="text-slate-500" style="font-size:9px;margin-top:2px">${h}</div>
    </div>`;
  }
  html+=`</div>`;

  // Table
  html+=`<div style="overflow-x:auto"><table>
    <thead><tr><th>Hour (UTC)</th><th>Predictions</th><th>Markets</th><th>Resolved</th><th>Correct</th><th>Accuracy</th></tr></thead><tbody>`;
  rows.forEach(r=>{
    html+=`<tr>
      <td class="font-mono text-xs">${String(r.hour).padStart(2,'0')}:00</td>
      <td class="font-mono text-xs">${r.predictions}</td>
      <td class="font-mono text-xs">${r.markets}</td>
      <td class="font-mono text-xs">${r.resolved}</td>
      <td class="font-mono text-xs">${r.correct}</td>
      <td>${_anAccBadge(r.correct_pct)}</td>
    </tr>`;
  });
  html+='</tbody></table></div>';
  el.innerHTML=html;
}

// ===== COMPARE DEVELOPMENT POPUP =====
function showCompareDevPopup(){
  const button = document.getElementById('tab-compare');
  const popup = document.getElementById('compare-dev-popup');
  if(!button || !popup) return;
  
  // Position popup above the button
  const rect = button.getBoundingClientRect();
  popup.style.position = 'fixed';
  popup.style.left = (rect.left + rect.width/2 - 60) + 'px'; // Center horizontally
  popup.style.top = (rect.top - 40) + 'px'; // Position above button
  popup.classList.remove('hidden');
  
  // Auto-hide after 2 seconds
  setTimeout(() => hideCompareDevPopup(), 2000);
}
function hideCompareDevPopup(){
  const popup = document.getElementById('compare-dev-popup');
  if(popup) popup.classList.add('hidden');
}
