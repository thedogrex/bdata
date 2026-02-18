// ===== POLYMARKET =====
let polySelectedMarket=null;
let polySelectedOutcome=null;
let polySelectedMarketSlug=null;
let polySelectedOutcomeAssetId=null;
let polyOrderBookInterval=null;
let polyCountdownInterval=null;

function clearPolySelection(){
  polySelectedMarket=null;
  polySelectedOutcome=null;
  polySelectedOutcomeAssetId=null;
  stopPolyOrderBookUpdates();
  document.getElementById('poly-market-title').textContent='';
  document.getElementById('poly-market-detail').innerHTML='<span class="text-slate-400">Select a market.</span>';
  document.getElementById('poly-orderbook-up').innerHTML='<span class="text-slate-400">Select a market.</span>';
  document.getElementById('poly-orderbook-down').innerHTML='<span class="text-slate-400">Select a market.</span>';
  document.getElementById('poly-ob-status').textContent='';
  document.getElementById('poly-sim-msg').textContent='';
  document.querySelectorAll('#poly-markets tr').forEach(tr=>tr.classList.remove('bg-blue-900'));
  document.querySelectorAll('[data-outcome-asset-id]').forEach(el=>el.classList.remove('ring-2','ring-blue-500','bg-blue-800'));
}

function clearPolySelectionComplete(){
  polySelectedMarket=null;
  polySelectedOutcome=null;
  polySelectedMarketSlug=null;
  polySelectedOutcomeAssetId=null;
  stopPolyOrderBookUpdates();
  stopPolyCountdown();
  document.getElementById('poly-market-title').textContent='';
  document.getElementById('poly-market-detail').innerHTML='<span class="text-slate-400">Select a market.</span>';
  document.getElementById('poly-orderbook-up').innerHTML='<span class="text-slate-400">Select a market.</span>';
  document.getElementById('poly-orderbook-down').innerHTML='<span class="text-slate-400">Select a market.</span>';
  document.getElementById('poly-ob-status').textContent='';
  document.getElementById('poly-sim-msg').textContent='';
  document.querySelectorAll('#poly-markets tr').forEach(tr=>tr.classList.remove('bg-blue-900'));
  document.querySelectorAll('[data-outcome-asset-id]').forEach(el=>el.classList.remove('ring-2','ring-blue-500','bg-blue-800'));
}

function startPolyOrderBookUpdates(){
  stopPolyOrderBookUpdates();
  if(polySelectedMarket && polySelectedMarket.outcomes && polySelectedMarket.outcomes.length >= 2){
    polyOrderBookInterval = setInterval(()=>{ updatePolyOrderBooks(); }, 3000);
  }
}

function stopPolyOrderBookUpdates(){
  if(polyOrderBookInterval){ clearInterval(polyOrderBookInterval); polyOrderBookInterval=null; }
}

function startPolyCountdown(marketTs, intervalSeconds){
  stopPolyCountdown();
  polyCountdownInterval = setInterval(()=>{ updatePolyCountdown(marketTs, intervalSeconds); }, 1000);
  updatePolyCountdown(marketTs, intervalSeconds);
}

function stopPolyCountdown(){
  if(polyCountdownInterval){ clearInterval(polyCountdownInterval); polyCountdownInterval=null; }
}

function updatePolyCountdown(marketTs, intervalSeconds){
  const now = Math.floor(Date.now() / 1000);
  const marketEnd = marketTs + intervalSeconds;
  const remaining = marketEnd - now;
  const countdownEl = document.getElementById('poly-countdown');
  if(!countdownEl) return;
  if(remaining <= 0){
    countdownEl.innerHTML = '<span class="text-red-400 font-semibold">EXPIRED</span>';
    stopPolyCountdown();
    loadPolyMarkets();
  } else {
    const minutes = Math.floor(remaining / 60);
    const seconds = remaining % 60;
    countdownEl.innerHTML = `<span class="text-green-400 font-semibold">${minutes.toString().padStart(2,'0')}:${seconds.toString().padStart(2,'0')}</span>`;
  }
}

async function loadPolyMarkets(){
  const el=document.getElementById('poly-markets');
  el.textContent='Loading...';
  try{
    try{
      const st=await fetch(API+'/api/poly/status');
      const s=await st.json();
      polyActiveTs=s.active_ts||null;
    }catch(e){polyActiveTs=null;}

    const [res, posRes] = await Promise.all([
      fetch(API+'/api/poly/markets?limit=80'),
      fetch(API+'/api/poly/sim/markets_with_positions')
    ]);
    const data=await res.json();
    const marketsWithPosRaw = await posRes.json();
    const marketsWithPos = new Set(Array.isArray(marketsWithPosRaw) ? marketsWithPosRaw : []);
    if(!Array.isArray(data)||!data.length){el.innerHTML='<div class="text-slate-400">No markets yet.</div>';return}
    let html='<table><thead><tr><th>TS</th><th>Slug</th><th>Status</th></tr></thead><tbody>';
    data.forEach(m=>{
      const d=new Date((m.ts||0)*1000).toISOString().replace('T',' ').substring(0,19);
      const status = m.status || (m.closed ? '[DONE]' : 'open');
      const statusClass = status === '[DONE]' ? 'badge badge-queue' : 'badge badge-done';
      const st = `<span class="${statusClass}">${status}</span>`;
      const isActive = (polyActiveTs!==null && (m.ts||0)===polyActiveTs && !m.closed);
      const dot = isActive ? '<span title="active" style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#ef4444;margin-right:6px"></span>' : '';
      const posDot = marketsWithPos.has(m.slug) ? '<span title="has position" style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#22c55e;margin-right:6px"></span>' : '';
      const isSelected = polySelectedMarketSlug === m.slug;
      const selectedClass = isSelected ? 'bg-blue-900' : '';
      html+=`<tr class="cursor-pointer ${selectedClass}" onclick="selectPolyMarket('${m.slug}')"><td class="text-xs text-slate-400">${dot}${posDot}${d}</td><td class="font-mono text-blue-300">${m.slug}</td><td>${st}</td></tr>`;
    });
    html+='</tbody></table>';
    el.innerHTML=html;
  }catch(e){el.textContent='Error loading markets';}
}

async function selectPolyMarket(slug){
  polySelectedMarketSlug = slug;
  document.querySelectorAll('#poly-markets tr').forEach(tr=>tr.classList.remove('bg-blue-900'));
  const rows = document.querySelectorAll('#poly-markets tr');
  rows.forEach(tr=>{
    if(tr.getAttribute('onclick') && tr.getAttribute('onclick').includes(`'${slug}'`)){
      tr.classList.add('bg-blue-900');
    }
  });
  await showPolyMarket(slug);
}

async function showPolyMarket(slug){
  polySelectedOutcome=null;
  polySelectedOutcomeAssetId=null;
  stopPolyOrderBookUpdates();
  stopPolyCountdown();
  document.getElementById('poly-orderbook-up').innerHTML='<span class="text-slate-400">Loading...</span>';
  document.getElementById('poly-orderbook-down').innerHTML='<span class="text-slate-400">Loading...</span>';
  document.getElementById('poly-sim-msg').textContent='';

  const title=document.getElementById('poly-market-title');
  const el=document.getElementById('poly-market-detail');
  el.textContent='Loading...';
  try{
    const res=await fetch(API+'/api/poly/market/'+encodeURIComponent(slug));
    const m=await res.json();
    if(m.error){el.textContent=m.error;return}
    polySelectedMarket=m;
    title.textContent=m.question||m.slug;
    const extUrl=`https://polymarket.com/event/${m.slug}`;
    const isActive = (polyActiveTs!==null && m.ts===polyActiveTs && !m.closed);
    const activeBadge = isActive ? '<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#ef4444;margin-right:6px"></span><span class="text-red-400 font-semibold">ACTIVE</span>' : '';
    const isDone = m.closed || (m.ts && (m.ts + 300) < Math.floor(Date.now() / 1000));
    const countdownDisplay = isActive ? '<div class="mb-2 text-xs">Time remaining: <span id="poly-countdown" class="font-mono">--:--</span></div>' : '';

    let html=`<div class="mb-2 text-xs text-slate-400"><span class="font-mono">${m.slug}</span> | ts: ${m.ts} | closed: ${m.closed} ${activeBadge}</div>`;
    html+=countdownDisplay;
    html+=`<div class="mb-3 text-xs"><a href="${extUrl}" target="_blank" class="text-blue-400 hover:underline">Open on Polymarket</a></div>`;
    if(m.description) html+=`<div class="text-xs text-slate-400 mb-3">${m.description}</div>`;
    html+='<div class="font-semibold mb-2">Outcomes</div>';
    html+='<div class="grid grid-cols-1 md:grid-cols-2 gap-2">';
    (m.outcomes||[]).forEach(o=>{
      const shortId = o.asset_id.length > 12 ? o.asset_id.substring(0,12)+'...' : o.asset_id;
      const isSelected = polySelectedOutcomeAssetId === o.asset_id;
      const selectedClass = isSelected ? 'ring-2 ring-blue-500 bg-blue-800' : '';
      let tri = '';
      if(isDone){
        if((o.name||'').toUpperCase().includes('UP')) tri='<span class="text-red-500 ml-2">▼</span>';
        else if((o.name||'').toUpperCase().includes('DOWN')) tri='<span class="text-green-500 ml-2">▲</span>';
      }
      html+=`<div class="p-2 rounded cursor-pointer transition-all duration-200 ${selectedClass}" style="background:#1e293b;border:1px solid #334155" data-outcome-asset-id="${o.asset_id}" onclick="selectPolyOutcome('${o.asset_id}','${(o.name||'').replace(/'/g,"\\'")}')"><div class="text-sm font-medium">${o.name}${tri}</div><div class="text-xs text-slate-400 font-mono flex items-center gap-1"><span title="${o.asset_id}">${shortId}</span><button onclick="event.stopPropagation();copyToClipboard('${o.asset_id}')" class="text-blue-400 hover:text-blue-300 text-xs px-1 py-0.5 rounded hover:bg-blue-900/20" title="Copy">📋</button></div></div>`;
    });
    html+='</div>';
    el.innerHTML=html;

    if(isActive) startPolyCountdown(m.ts, 300);

    // Auto-start order book updates
    if(m.outcomes && m.outcomes.length >= 2){
      updatePolyOrderBooks();
      startPolyOrderBookUpdates();
    }
  }catch(e){el.textContent='Error loading market';}
}

function selectPolyOutcome(assetId, name){
  polySelectedOutcome={asset_id:assetId,name:name};
  polySelectedOutcomeAssetId=assetId;
  document.querySelectorAll('[data-outcome-asset-id]').forEach(el=>el.classList.remove('ring-2','ring-blue-500','bg-blue-800'));
  const selectedEl = document.querySelector(`[data-outcome-asset-id="${assetId}"]`);
  if(selectedEl) selectedEl.classList.add('ring-2','ring-blue-500','bg-blue-800');
  document.getElementById('poly-sim-msg').textContent='';
}

function copyToClipboard(text){
  navigator.clipboard.writeText(text).then(()=>{
    const btn = event.target;
    const orig = btn.textContent;
    btn.textContent = '✓';
    btn.classList.add('text-green-400');
    setTimeout(()=>{ btn.textContent=orig; btn.classList.remove('text-green-400'); }, 1000);
  }).catch(err=>console.error('Copy failed:', err));
}

// ===== Order Book rendering (asks only, 5 lowest) =====

function findUpDownOutcomes(market){
  if(!market || !market.outcomes || market.outcomes.length < 2) return null;
  let up=null, down=null;
  market.outcomes.forEach(o=>{
    const n=(o.name||'').toUpperCase();
    if(n.includes('UP')) up=o;
    else if(n.includes('DOWN')) down=o;
  });
  if(!up||!down){ up=market.outcomes[0]; down=market.outcomes[1]; }
  return {up, down};
}

async function updatePolyOrderBooks(){
  const pair = findUpDownOutcomes(polySelectedMarket);
  if(!pair){
    document.getElementById('poly-orderbook-up').innerHTML='<span class="text-slate-400">Need 2 outcomes.</span>';
    document.getElementById('poly-orderbook-down').innerHTML='<span class="text-slate-400">Need 2 outcomes.</span>';
    return;
  }
  try{
    const [upRes, downRes] = await Promise.all([
      fetch(API+`/api/poly/orderbook/${encodeURIComponent(polySelectedMarket.slug)}/${encodeURIComponent(pair.up.asset_id)}/latest`),
      fetch(API+`/api/poly/orderbook/${encodeURIComponent(polySelectedMarket.slug)}/${encodeURIComponent(pair.down.asset_id)}/latest`)
    ]);
    const [upData, downData] = await Promise.all([upRes.json(), downRes.json()]);
    document.getElementById('poly-orderbook-up').innerHTML = upData ? renderAsks(upData, pair.up.name) : '<span class="text-slate-400">No data.</span>';
    document.getElementById('poly-orderbook-down').innerHTML = downData ? renderAsks(downData, pair.down.name) : '<span class="text-slate-400">No data.</span>';
    const now = new Date().toLocaleTimeString();
    document.getElementById('poly-ob-status').innerHTML = `<span class="text-green-400">Updated ${now}</span>`;
  }catch(e){
    document.getElementById('poly-orderbook-up').innerHTML='<span class="text-red-400">Error.</span>';
    document.getElementById('poly-orderbook-down').innerHTML='<span class="text-red-400">Error.</span>';
  }
}

function renderAsks(data, outcomeName){
  if(!data || !data.asks || !data.asks.length) return `<div class="text-slate-400">${outcomeName}: no asks</div>`;
  const ts = new Date((data.ts||0)*1000);
  const timeStr = ts.toLocaleTimeString();
  // Sort ascending by price, take 5 lowest
  const sorted = [...data.asks].sort((a,b)=>parseFloat(a.price)-parseFloat(b.price));
  const top5 = sorted.slice(0,5);
  let html=`<div class="text-slate-500 text-xs mb-1">${outcomeName} · ${timeStr}</div>`;
  html+='<table class="text-xs"><thead><tr><th>Price (¢)</th><th>Size</th></tr></thead><tbody>';
  top5.forEach(a=>{
    html+=`<tr class="text-red-300"><td class="font-mono">${a.price}</td><td>${a.size}</td></tr>`;
  });
  html+='</tbody></table>';
  if(data.best_ask_cents!==null){
    html+=`<div class="mt-1 text-xs text-slate-500">Best ask: ${data.best_ask_cents}¢</div>`;
  }
  return html;
}

// ===== Sim trades =====

async function submitSimTrade(){
  const msg=document.getElementById('poly-sim-msg');
  msg.textContent='';
  if(!polySelectedOutcome){msg.textContent='Select an outcome first';return}
  if(!polySelectedMarket || !polySelectedMarket.slug){msg.textContent='Select a market first';return}
  const qty=parseFloat(document.getElementById('poly-sim-qty').value)||0;
  try{
    const res=await fetch(API+'/api/poly/sim/trade',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({slug: polySelectedMarket.slug, asset_id:polySelectedOutcome.asset_id, qty})});
    const data=await res.json();
    if(data.error){msg.textContent=data.error;return}
    msg.textContent=`Trade #${data.id} filled @ ${data.fill_price_cents} cents`;
    loadSimTrades();
    loadSimPositions();
    loadPolyMarkets();
  }catch(e){msg.textContent='Error submitting trade';}
}

async function loadSimTrades(){
  const el=document.getElementById('poly-trades');
  el.textContent='Loading...';
  try{
    const res=await fetch(API+'/api/poly/sim/trades?limit=200');
    const data=await res.json();
    if(!Array.isArray(data)||!data.length){el.innerHTML='<div class="text-slate-400">No trades yet.</div>';return}
    let html='<div class="max-h-56 overflow-y-auto"><table><thead><tr><th>Time</th><th>Asset</th><th>Side</th><th>Qty</th><th>Fill (c)</th></tr></thead><tbody>';
    data.forEach(t=>{
      const d=new Date((t.ts||0)*1000).toISOString().substring(11,19);
      const side=t.side==='BUY'?'<span class="badge badge-up">BUY</span>':'<span class="badge badge-down">SELL</span>';
      html+=`<tr><td class="text-xs text-slate-400">${d}</td><td class="font-mono">${t.asset_id}</td><td>${side}</td><td>${t.qty}</td><td class="font-bold">${t.fill_price_cents}</td></tr>`;
    });
    html+='</tbody></table></div>';
    el.innerHTML=html;
  }catch(e){el.textContent='Error loading trades';}
}

async function loadSimPositions(){
  const el=document.getElementById('poly-positions');
  el.textContent='Loading...';
  try{
    const res=await fetch(API+'/api/poly/sim/positions');
    const data=await res.json();
    if(!Array.isArray(data)||!data.length){el.innerHTML='<div class="text-slate-400">No positions.</div>';return}
    let html='<table><thead><tr><th>Asset</th><th>Pos</th><th>Mark</th><th>PnL (c)</th></tr></thead><tbody>';
    data.forEach(p=>{
      const pnl=p.pnl_cents;
      const cls=pnl>0?'text-green-400':pnl<0?'text-red-400':'text-slate-300';
      const shortId = p.asset_id.length > 8 ? p.asset_id.substring(0,8)+'...' : p.asset_id;
      html+=`<tr><td class="font-mono text-xs"><span title="${p.asset_id}">${shortId}</span><button onclick="event.stopPropagation();copyToClipboard('${p.asset_id}')" class="text-blue-400 hover:text-blue-300 text-xs px-1 py-0.5 rounded hover:bg-blue-900/20 ml-1" title="Copy">📋</button></td><td>${p.pos_qty}</td><td>${p.mark_cents??''}</td><td class="font-bold ${cls}">${p.pnl_cents??''}</td></tr>`;
    });
    html+='</tbody></table>';
    el.innerHTML=html;
  }catch(e){el.textContent='Error loading positions';}
}
