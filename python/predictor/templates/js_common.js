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
}

function applyPreset(jsonStr){
  document.getElementById('bt-params').value=JSON.stringify(JSON.parse(jsonStr),null,2);
}

// ===== TABS =====
const TABS=['backtest','compare','bruteforce','history','best','poly','orderbooks'];
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
  if(tab==='poly'){loadPolyMarkets();loadSimTrades();loadSimPositions();}
  else{
    clearPolySelectionComplete();
    stopPolyOrderBookUpdates();
  }
  if(tab==='orderbooks'){obLoadMarkets();}
  else{obStopAll();}
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

function renderResult(data,targetId){
  const el=document.getElementById(targetId);el.classList.remove('hidden');
  const ws=data.window_size||'?';const re=data.retrain_every||'?';
  const lt=data.load_time_sec?` | Load: ${data.load_time_sec}s`:'';
  const ft=data.feature_time_sec?` | Features: ${data.feature_time_sec}s`:'';
  let html=`<div class="card p-6 mb-6"><div class="mb-4">
    <h2 class="text-lg font-semibold">${data.strategy} ${data.id?'<span class="text-xs text-slate-400">#'+data.id+'</span>':''}</h2>
    <p class="text-xs text-slate-400">Train: ${data.train_period||''} | Test: ${data.test_period||''}</p>
    <p class="text-xs text-slate-400">Window: ${ws} | Retrain: ${re} | Total: ${data.total_time_sec}s${lt}${ft}</p></div>`;
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
    const bfGroups={};const standalone=[];
    data.forEach(r=>{
      if(r.is_bruteforce && r.bruteforce_id){
        if(!bfGroups[r.bruteforce_id])bfGroups[r.bruteforce_id]={runs:[],strategy:r.strategy,bf_id:r.bruteforce_id};
        bfGroups[r.bruteforce_id].runs.push(r);
      }else{standalone.push(r)}
    });
    let html='<div class="card p-6">';
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
        html+=`<tr class="cursor-pointer hover:bg-slate-700" onclick="event.stopPropagation();showDetail(${r.id})"><td>${r.id}</td><td class="text-xs text-slate-400 max-w-xs truncate">${ps}</td><td>${r.window_size||'?'}</td><td>${hs}</td><td>${r.total_time_sec}s</td><td><button onclick="event.stopPropagation();deleteRun(${r.id})" class="text-red-400 text-xs hover:underline">del</button></td></tr>`});
      html+=`</tbody></table></details>`;
    });
    if(standalone.length){
      html+=`<table><thead><tr><th>ID</th><th>Strategy</th><th>Test Period</th><th>Win</th><th>Horizons</th><th>Time</th><th>Date</th><th></th></tr></thead><tbody>`;
      standalone.forEach(r=>{
        const hs=Object.entries(r.horizons||{}).map(([h,d])=>d.error?`H${h}:err`:`H${h}:<span class="${accClass(d.accuracy_pct)}">${d.accuracy_pct}%</span>`).join(' | ');
        html+=`<tr class="cursor-pointer" onclick="showDetail(${r.id})"><td>${r.id}</td><td class="font-medium">${r.strategy}</td><td class="text-xs">${r.test_period||''}</td><td>${r.window_size||'?'}</td><td>${hs}</td><td>${r.total_time_sec}s</td><td class="text-slate-400 text-xs">${r.created_at||''}</td><td><button onclick="event.stopPropagation();deleteRun(${r.id})" class="text-red-400 text-xs hover:underline">del</button></td></tr>`});
      html+=`</tbody></table>`;
    }
    html+='</div>';el.innerHTML=html}catch(e){console.error(e)}
}
async function showDetail(id){try{const res=await fetch(API+'/api/history/'+id);const data=await res.json();if(data.error){alert(data.error);return}switchTab('backtest');renderResult(data,'bt-results');document.getElementById('bt-results').scrollIntoView({behavior:'smooth',block:'start'})}catch(e){alert(e.message)}}
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
