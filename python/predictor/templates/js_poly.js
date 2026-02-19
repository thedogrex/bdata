// ===== POLYMARKET =====
let polySelectedMarket=null;
let polySelectedOutcome=null;
let polySelectedMarketSlug=null;
let polySelectedOutcomeAssetId=null;
let polyOrderBookInterval=null;
let polyCountdownInterval=null;

const POLY_PRED_SETTINGS_KEY = 'poly_pred_settings_v1';

function loadPolyPredictionSettings(){
  try{
    const raw = localStorage.getItem(POLY_PRED_SETTINGS_KEY);
    if(!raw) return;
    const d = JSON.parse(raw);
    const wEl = document.getElementById('poly-pred-window');
    const pEl = document.getElementById('poly-pred-params');
    if(wEl && d && typeof d.window_size === 'number') wEl.value = String(d.window_size);
    if(pEl && d && typeof d.params_json === 'string') pEl.value = d.params_json;
  }catch(e){}
}

function savePolyPredictionSettings(){
  const msgEl = document.getElementById('poly-pred-save-msg');
  if(msgEl) msgEl.textContent='';

  const wEl = document.getElementById('poly-pred-window');
  const pEl = document.getElementById('poly-pred-params');

  const windowSize = parseInt(wEl?.value||'') || 1000;
  const paramsText = (pEl?.value||'').trim();

  if(paramsText){
    try{ JSON.parse(paramsText); }
    catch(e){ if(msgEl) msgEl.textContent='Invalid JSON'; return; }
  }

  try{
    localStorage.setItem(POLY_PRED_SETTINGS_KEY, JSON.stringify({
      window_size: windowSize,
      params_json: paramsText,
      saved_ts: Math.floor(Date.now()/1000)
    }));
    if(msgEl){
      msgEl.textContent='Saved';
      setTimeout(()=>{ if(msgEl.textContent==='Saved') msgEl.textContent=''; }, 1200);
    }
  }catch(e){
    if(msgEl) msgEl.textContent='Save failed';
  }
}

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
    countdownEl.innerHTML = '<span class="text-red-400 font-semibold">00:00</span>';
    stopPolyCountdown();
    // Refresh markets and show resolved outcome if available
    loadPolyMarkets();
    if(polySelectedMarket && polySelectedMarket.slug){
      showPolyMarketResolvedOutcome(polySelectedMarket.slug);
    }
  } else {
    const minutes = Math.floor(remaining / 60);
    const seconds = remaining % 60;
    countdownEl.innerHTML = `<span class="text-green-400 font-semibold">${minutes.toString().padStart(2,'0')}:${seconds.toString().padStart(2,'0')}</span>`;
  }
}

async function showPolyMarketResolvedOutcome(slug){
  try{
    const res = await fetch(API+`/api/poly/market/${encodeURIComponent(slug)}/live`);
    if(!res.ok) return;
    const m = await res.json();
    if(!m || !Array.isArray(m.outcomes) || !m.outcomes.length) return;

    // Determine winner as the outcome with max price
    const sorted = [...m.outcomes].filter(o=>typeof o.price==='number').sort((a,b)=>b.price-a.price);
    const win = sorted[0];
    if(!win) return;

    const pct = Math.round(win.price * 1000) / 10; // 1 decimal
    const upDown = ((win.name||'').toUpperCase().includes('UP') ? 'UP' : ((win.name||'').toUpperCase().includes('DOWN') ? 'DOWN' : win.name));
    const badge = `<span class="badge badge-done" style="margin-left:8px">RESULT: ${upDown} (${pct}%)</span>`;

    // Insert badge near the market detail header line
    const detail = document.getElementById('poly-market-detail');
    if(detail && typeof detail.innerHTML === 'string'){
      // If badge already present, avoid duplicating
      if(detail.innerHTML.includes('RESULT:')) return;
      detail.innerHTML = detail.innerHTML.replace('</div>', `${badge}</div>`);
    }
  }catch(e){/* ignore */}
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
    let html='<table><thead><tr><th>Time</th><th>Status</th></tr></thead><tbody>';
    data.forEach(m=>{
      const d=new Date((m.ts||0)*1000);
      const dateStr = d.toLocaleDateString('ru-RU',{day:'2-digit',month:'2-digit'}) + ' ' + d.toLocaleTimeString('ru-RU',{hour:'2-digit',minute:'2-digit'});
      const slugSuffix = (m.slug||'').split('-').pop();
      const status = m.status || (m.closed ? '[DONE]' : 'open');
      const statusClass = status === '[DONE]' ? 'badge badge-queue' : 'badge badge-done';
      const resolved = (m.resolved_outcome||'');
      const resolvedTri = resolved === 'UP'
        ? '<span style="margin-left:8px;color:#22c55e;font-weight:700">▲</span>'
        : (resolved === 'DOWN'
          ? '<span style="margin-left:8px;color:#ef4444;font-weight:700">▼</span>'
          : '');
      const pred = (m.prediction_outcome||'');
      const predTri = pred === 'UP'
        ? '<span style="margin-left:8px;color:#22c55e;font-weight:700">▲</span>'
        : (pred === 'DOWN'
          ? '<span style="margin-left:8px;color:#ef4444;font-weight:700">▼</span>'
          : (pred === 'UNDEFINED'
            ? '<span style="margin-left:8px;color:#94a3b8;font-weight:700">?</span>'
            : ''));
      const predTs = (m.prediction_ts||null);
      const predTitle = predTs ? ` title="pred @ ${new Date(predTs*1000).toLocaleString('ru-RU')}"` : '';
      const predWrap = predTri ? `<span${predTitle}>${predTri}</span>` : '';
      const stHtml = `<span class="${statusClass}">${status}</span>${resolvedTri}${predWrap}`;
      const isActive = (polyActiveTs!==null && (m.ts||0)===polyActiveTs && !m.closed);
      const dot = isActive ? '<span title="active" style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#ef4444;margin-right:6px"></span>' : '';
      const posDot = marketsWithPos.has(m.slug) ? '<span title="has position" style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#22c55e;margin-right:6px"></span>' : '';
      const isSelected = polySelectedMarketSlug === m.slug;
      const selectedClass = isSelected ? 'bg-blue-900' : '';
      html+=`<tr class="cursor-pointer ${selectedClass}" onclick="selectPolyMarket('${m.slug}')"><td class="text-xs text-slate-400" style="white-space:nowrap">${dot}${posDot}${dateStr} <span class="font-mono text-blue-300">${slugSuffix}</span></td><td>${stHtml}</td></tr>`;
    });
    html+='</tbody></table>';
    el.innerHTML=html;

    // Apply saved prediction settings whenever the Poly tab is loaded.
    loadPolyPredictionSettings();
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
  const buyBtn = document.getElementById('poly-sim-submit');
  if(buyBtn){buyBtn.disabled = true;}

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
    const obPanel = document.getElementById('poly-ob-panel');
    const buyPanel = document.getElementById('poly-buy-panel');
    const predPanel = document.getElementById('poly-predict-panel');
    if(obPanel) obPanel.classList.toggle('hidden', !!isDone);
    if(buyPanel) buyPanel.classList.toggle('hidden', !!isDone);
    if(predPanel) predPanel.classList.toggle('hidden', !!isDone);
    const predResult = document.getElementById('poly-pred-result');
    if(predResult) predResult.innerHTML='';
    loadPolyPredictionSettings();
    const countdownDisplay = isActive ? '<div class="mb-2 text-xs">Time remaining: <span id="poly-countdown" class="font-mono">--:--</span></div>' : '';

    const pred = (m.prediction_outcome||'');
    const predTs = (m.prediction_ts||null);
    let predHtml = '';
    if(pred){
      const c = pred==='UP' ? '#22c55e' : (pred==='DOWN' ? '#ef4444' : '#94a3b8');
      const a = pred==='UP' ? '▲' : (pred==='DOWN' ? '▼' : '?');
      const t = predTs ? new Date(predTs*1000).toLocaleString('ru-RU') : '';
      predHtml = ` | pred: <span style="color:${c};font-weight:700">${a} ${pred}</span>${t?` <span class=\"text-slate-500\">@ ${t}</span>`:''}`;
    }
    let html=`<div class="mb-2 text-xs text-slate-400"><span class="font-mono">${m.slug}</span> | ts: ${m.ts} | closed: ${m.closed} ${activeBadge}${predHtml}</div>`;
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

    // Auto-start order book updates (skip for DONE markets)
    if(!isDone && m.outcomes && m.outcomes.length >= 2){
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
    const p = Math.round(parseFloat(a.price));
    const s = Math.round(parseFloat(a.size));
    html+=`<tr class="text-red-300"><td class="font-mono">${Number.isFinite(p)?p:a.price}</td><td>${Number.isFinite(s)?s:a.size}</td></tr>`;
  });
  html+='</tbody></table>';
  if(data.best_ask_cents!==null){
    html+=`<div class="mt-1 text-xs text-slate-500">Best ask: ${data.best_ask_cents}¢</div>`;
  }
  return html;
}

// ===== Prediction =====

async function runPolyPrediction(){
  if(!polySelectedMarket || !polySelectedMarket.slug){
    document.getElementById('poly-pred-result').innerHTML='<span class="text-red-400">Select a market first.</span>';
    return;
  }
  const btn = document.getElementById('poly-pred-btn');
  const resultEl = document.getElementById('poly-pred-result');
  btn.disabled = true;
  btn.textContent = 'Predicting...';
  resultEl.innerHTML='<span class="text-slate-400">Running prediction...</span>';

  let params = null;
  const paramsText = document.getElementById('poly-pred-params').value.trim();
  if(paramsText){
    try{ params = JSON.parse(paramsText); }catch(e){
      resultEl.innerHTML='<span class="text-red-400">Invalid JSON in strategy settings.</span>';
      btn.disabled = false; btn.textContent = 'PREDICT'; return;
    }
  }
  const windowSize = parseInt(document.getElementById('poly-pred-window').value) || 1000;

  try{
    const res = await fetch(API+'/api/poly/predict', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({
        slug: polySelectedMarket.slug,
        strategy: 'rsi_mean_reversion',
        params: params,
        window_size: windowSize,
        table: 'c_5m'
      })
    });
    const data = await res.json();
    if(data.error){
      resultEl.innerHTML=`<div class="p-3 rounded-lg" style="background:#7f1d1d;border:1px solid #ef4444"><span class="text-red-300 font-semibold">Error:</span> <span class="text-red-200">${data.error}</span></div>`;
    } else {
      const color = data.prediction === 'UP' ? '#22c55e' : (data.prediction === 'DOWN' ? '#ef4444' : '#94a3b8');
      const arrow = data.prediction === 'UP' ? '\u25b2' : (data.prediction === 'DOWN' ? '\u25bc' : '\u2014');
      const prob = Math.round(data.probability * 100);
      resultEl.innerHTML=`<div class="p-4 rounded-lg text-center" style="background:#1e293b;border:2px solid ${color}">`
        +`<div style="font-size:48px;font-weight:800;color:${color}">${arrow} ${data.prediction}</div>`
        +`<div class="text-lg text-slate-300 mt-1">Probability: <b>${prob}%</b></div>`
        +`<div class="text-xs text-slate-400 mt-2">Strategy: ${data.strategy} | Window: ${data.window_size} candles | Last candle: ${data.last_candle_dt}</div>`
        +`<div class="text-xs text-slate-500 mt-1">Params: <span class="font-mono">${JSON.stringify(data.params)}</span></div>`
        +`</div>`;
    }
  }catch(e){
    resultEl.innerHTML=`<span class="text-red-400">Request failed: ${e.message}</span>`;
  }
  btn.disabled = false;
  btn.textContent = 'PREDICT';
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
