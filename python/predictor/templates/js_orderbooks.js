// ===== ORDER BOOKS TAB =====
let obSelectedMarket=null;
let obSelectedSlug=null;
let obMode='live'; // 'live' or 'history'
let obLiveInterval=null;
let obHistoryData=[];

function obSetMode(mode){
  obMode=mode;
  document.getElementById('ob-mode-live').className = mode==='live' ? 'btn btn-green text-xs' : 'btn btn-slate text-xs';
  document.getElementById('ob-mode-history').className = mode==='history' ? 'btn btn-green text-xs' : 'btn btn-slate text-xs';
  document.getElementById('ob-live-panel').classList.toggle('hidden', mode!=='live');
  document.getElementById('ob-history-panel').classList.toggle('hidden', mode!=='history');
  if(mode==='live' && obSelectedMarket){
    obStartLive();
  } else if(mode==='history'){
    obStopLive();
  }
}

function obStopLive(){
  if(obLiveInterval){ clearInterval(obLiveInterval); obLiveInterval=null; }
}

function obStartLive(){
  obStopLive();
  if(!obSelectedMarket) return;
  obUpdateLive();
  obLiveInterval = setInterval(()=>obUpdateLive(), 3000);
}

function obStopAll(){
  obStopLive();
  obSelectedMarket=null;
  obSelectedSlug=null;
}

async function obLoadMarkets(){
  const el=document.getElementById('ob-market-list');
  el.textContent='Loading...';
  try{
    const [res, posRes] = await Promise.all([
      fetch(API+'/api/poly/markets?limit=80'),
      fetch(API+'/api/poly/sim/markets_with_positions')
    ]);
    const data=await res.json();
    const marketsWithPosRaw = await posRes.json();
    const marketsWithPos = new Set(Array.isArray(marketsWithPosRaw) ? marketsWithPosRaw : []);
    if(!Array.isArray(data)||!data.length){el.innerHTML='<div class="text-slate-400">No markets.</div>';return}

    let html='<table><thead><tr><th>Time</th><th>Status</th></tr></thead><tbody>';
    data.forEach(m=>{
      const d=new Date((m.ts||0)*1000);
      const dateStr = d.toLocaleDateString('ru-RU',{day:'2-digit',month:'2-digit'}) + ' ' + d.toLocaleTimeString('ru-RU',{hour:'2-digit',minute:'2-digit'});
      const status = m.status || (m.closed ? '[DONE]' : 'open');
      const statusClass = status==='[DONE]' ? 'badge badge-queue' : 'badge badge-done';
      const isActive = (polyActiveTs!==null && (m.ts||0)===polyActiveTs && !m.closed);
      const dot = isActive ? '<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#ef4444;margin-right:4px"></span>' : '';
      const posDot = marketsWithPos.has(m.slug) ? '<span title="has position" style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#22c55e;margin-right:4px"></span>' : '';
      const sel = obSelectedSlug===m.slug ? 'bg-blue-900' : '';
      html+=`<tr class="cursor-pointer ${sel}" onclick="obSelectMarket('${m.slug}')"><td class="text-xs text-slate-400">${dot}${posDot}${dateStr}</td><td><span class="${statusClass}">${status}</span></td></tr>`;
    });
    html+='</tbody></table>';
    el.innerHTML=html;
  }catch(e){el.textContent='Error loading markets.';}
}

async function obSelectMarket(slug){
  obSelectedSlug=slug;
  // Highlight
  document.querySelectorAll('#ob-market-list tr').forEach(tr=>tr.classList.remove('bg-blue-900'));
  document.querySelectorAll('#ob-market-list tr').forEach(tr=>{
    if(tr.getAttribute('onclick') && tr.getAttribute('onclick').includes(`'${slug}'`)) tr.classList.add('bg-blue-900');
  });

  // Load market detail
  try{
    const res=await fetch(API+'/api/poly/market/'+encodeURIComponent(slug));
    const m=await res.json();
    if(m.error){ document.getElementById('ob-market-info').textContent=m.error; return; }
    obSelectedMarket=m;

    const isDone = m.closed || (m.ts && (m.ts+300) < Math.floor(Date.now()/1000));
    const statusStr = isDone ? '<span class="badge badge-queue">[DONE]</span>' : '<span class="badge badge-done">open</span>';
    const d = new Date((m.ts||0)*1000);
    const dateStr = d.toLocaleDateString('ru-RU') + ' ' + d.toLocaleTimeString('ru-RU');
    document.getElementById('ob-market-info').innerHTML = `<span class="font-mono">${m.slug}</span> · ${dateStr} · ${statusStr}`;

    // Populate outcome selector for history
    const sel = document.getElementById('ob-hist-outcome');
    sel.innerHTML='<option value="">All outcomes</option>';
    (m.outcomes||[]).forEach(o=>{
      sel.innerHTML+=`<option value="${o.asset_id}">${o.name}</option>`;
    });

    // Set default date range for history (last 30 min)
    const now = new Date();
    const from = new Date(now.getTime() - 30*60*1000);
    document.getElementById('ob-hist-to').value = toLocalISOString(now);
    document.getElementById('ob-hist-from').value = toLocalISOString(from);

    if(obMode==='live') obStartLive();
    else obLoadHistory();
  }catch(e){ document.getElementById('ob-market-info').textContent='Error loading market.'; }
}

function toLocalISOString(d){
  const pad=(n)=>String(n).padStart(2,'0');
  return `${d.getFullYear()}-${pad(d.getMonth()+1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

async function obUpdateLive(){
  if(!obSelectedMarket) return;
  const pair = findUpDownOutcomes(obSelectedMarket);
  if(!pair){
    document.getElementById('ob-live-up').innerHTML='<span class="text-slate-400">Need 2 outcomes.</span>';
    document.getElementById('ob-live-down').innerHTML='<span class="text-slate-400">Need 2 outcomes.</span>';
    return;
  }
  try{
    const [upRes, downRes] = await Promise.all([
      fetch(API+`/api/poly/orderbook/${encodeURIComponent(obSelectedMarket.slug)}/${encodeURIComponent(pair.up.asset_id)}/latest`),
      fetch(API+`/api/poly/orderbook/${encodeURIComponent(obSelectedMarket.slug)}/${encodeURIComponent(pair.down.asset_id)}/latest`)
    ]);
    const [upData, downData] = await Promise.all([upRes.json(), downRes.json()]);
    document.getElementById('ob-live-up').innerHTML = upData ? renderAsks(upData, pair.up.name) : '<span class="text-slate-400">No data.</span>';
    document.getElementById('ob-live-down').innerHTML = downData ? renderAsks(downData, pair.down.name) : '<span class="text-slate-400">No data.</span>';
    document.getElementById('ob-live-status').innerHTML = `<span class="text-green-400">Updated ${new Date().toLocaleTimeString()}</span>`;
  }catch(e){
    document.getElementById('ob-live-up').innerHTML='<span class="text-red-400">Error.</span>';
    document.getElementById('ob-live-down').innerHTML='<span class="text-red-400">Error.</span>';
  }
}

async function obLoadHistory(){
  if(!obSelectedMarket) return;
  const el=document.getElementById('ob-hist-list');
  el.textContent='Loading...';
  document.getElementById('ob-hist-detail').innerHTML='<span class="text-slate-400">Click a snapshot to view.</span>';

  const assetId = document.getElementById('ob-hist-outcome').value;
  const fromStr = document.getElementById('ob-hist-from').value;
  const toStr = document.getElementById('ob-hist-to').value;

  // Determine which outcomes to load
  let outcomes = [];
  if(assetId){
    const o = (obSelectedMarket.outcomes||[]).find(x=>x.asset_id===assetId);
    if(o) outcomes=[o];
  } else {
    outcomes = obSelectedMarket.outcomes||[];
  }

  if(!outcomes.length){ el.innerHTML='<span class="text-slate-400">No outcomes.</span>'; return; }

  // Calculate minutes from date range
  const fromTs = fromStr ? Math.floor(new Date(fromStr).getTime()/1000) : Math.floor(Date.now()/1000) - 1800;
  const toTs = toStr ? Math.floor(new Date(toStr).getTime()/1000) : Math.floor(Date.now()/1000);
  const minutes = Math.max(1, Math.ceil((toTs - fromTs) / 60));

  try{
    // Fetch snapshots for all selected outcomes
    let allSnapshots = [];
    for(const o of outcomes){
      const res = await fetch(API+`/api/poly/orderbook/${encodeURIComponent(obSelectedMarket.slug)}/${encodeURIComponent(o.asset_id)}/analysis?minutes=${minutes}`);
      const data = await res.json();
      if(Array.isArray(data)){
        data.forEach(snap=>{
          // Filter by date range
          if(snap.ts >= fromTs && snap.ts <= toTs){
            allSnapshots.push({...snap, outcome_name: o.name, asset_id: o.asset_id});
          }
        });
      }
    }

    // Sort by timestamp descending (newest first)
    allSnapshots.sort((a,b)=>b.ts - a.ts);
    obHistoryData = allSnapshots;

    if(!allSnapshots.length){
      el.innerHTML='<span class="text-slate-400">No snapshots in this range.</span>';
      return;
    }

    // Render snapshot list
    let html=`<div class="text-xs text-slate-500 mb-2">${allSnapshots.length} snapshots</div>`;
    html+='<table><thead><tr><th>Date/Time</th><th>Outcome</th><th>Best Ask</th></tr></thead><tbody>';
    allSnapshots.forEach((snap, idx)=>{
      const d = new Date(snap.ts*1000);
      const dateStr = d.toLocaleDateString('ru-RU',{day:'2-digit',month:'2-digit'}) + ' ' + d.toLocaleTimeString('ru-RU');
      const isUp = (snap.outcome_name||'').toUpperCase().includes('UP');
      const nameClass = isUp ? 'text-green-400' : 'text-red-400';
      html+=`<tr class="ob-snapshot-row cursor-pointer" onclick="obShowSnapshot(${idx})"><td class="text-xs">${dateStr}</td><td class="${nameClass} text-xs">${snap.outcome_name}</td><td class="font-mono text-xs">${snap.best_ask_cents!==null?snap.best_ask_cents+'¢':'-'}</td></tr>`;
    });
    html+='</tbody></table>';
    el.innerHTML=html;
  }catch(e){
    el.innerHTML='<span class="text-red-400">Error loading history.</span>';
  }
}

function obShowSnapshot(idx){
  const snap = obHistoryData[idx];
  if(!snap) return;

  // Highlight selected row
  document.querySelectorAll('.ob-snapshot-row').forEach((r,i)=>{
    r.classList.toggle('active', i===idx);
  });

  const el = document.getElementById('ob-hist-detail');
  const d = new Date(snap.ts*1000);
  const fullDate = d.toLocaleDateString('ru-RU') + ' ' + d.toLocaleTimeString('ru-RU');
  const isUp = (snap.outcome_name||'').toUpperCase().includes('UP');
  const nameClass = isUp ? 'text-green-400' : 'text-red-400';

  let html=`<div class="mb-2"><span class="${nameClass} font-semibold">${snap.outcome_name}</span> · <span class="text-slate-400">${fullDate}</span></div>`;

  // Summary
  html+=`<div class="grid grid-cols-1 gap-2 mb-3">`;
  html+=`<div class="p-2 rounded text-center" style="background:#0f172a"><div class="text-xs text-slate-400">Best Ask</div><div class="font-mono font-bold text-red-400">${snap.best_ask_cents!==null?snap.best_ask_cents+'¢':'-'}</div></div>`;
  html+=`</div>`;

  if(snap.ask_depth!==null && snap.ask_depth!==undefined){
    const ad = (typeof snap.ask_depth === 'number') ? snap.ask_depth.toFixed(0) : snap.ask_depth;
    html+=`<div class="text-xs text-slate-400 mb-2">Ask depth: ${ad}</div>`;
  }

  // Asks table (all asks in this snapshot)
  const asks = snap.asks||[];
  if(asks.length){
    const sorted = [...asks].sort((a,b)=>parseFloat(a.price)-parseFloat(b.price));
    html+=`<div class="mb-2"><span class="text-xs text-red-400 font-semibold">Asks (${sorted.length})</span>`;
    html+='<table class="text-xs"><thead><tr><th>Price (¢)</th><th>Size</th></tr></thead><tbody>';
    sorted.forEach(a=>{
      html+=`<tr class="text-red-300"><td class="font-mono">${a.price}</td><td>${a.size}</td></tr>`;
    });
    html+='</tbody></table></div>';
  }

  el.innerHTML=html;
}
