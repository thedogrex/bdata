// ===== ORDER BOOKS TAB =====
let obSelectedMarket=null;
let obSelectedSlug=null;
let obMode='live'; // 'live' or 'history'
let obHistView='chart'; // 'chart' or 'table'
let obLiveInterval=null;
let obHistoryData=[];
let obChartData={}; // {outcomeName: [{ts, best_ask, asks}]}

let obMarketsPage = 1;
const OB_MARKETS_PER_PAGE = 20;
let obHistPage = 1;
const OB_HIST_PER_PAGE = 20;

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
    if(obSelectedMarket){
      const sideEl = document.getElementById('ob-hist-side');
      if(sideEl && !sideEl.value) sideEl.value = 'UP';
      obUpdateHistSideButtons();
      obLoadHistory();
    }
  }
}

function obUpdateHistSideButtons(){
  const side = (document.getElementById('ob-hist-side')?.value || 'UP').toUpperCase();
  const upBtn = document.getElementById('ob-hist-side-up');
  const downBtn = document.getElementById('ob-hist-side-down');
  if(upBtn){
    upBtn.style.background = side === 'UP' ? '#052e16' : 'transparent';
    upBtn.style.color = '#22c55e';
  }
  if(downBtn){
    downBtn.style.background = side === 'DOWN' ? '#450a0a' : 'transparent';
    downBtn.style.color = '#ef4444';
  }
}

function obSetHistSide(side){
  const s = String(side || 'UP').toUpperCase();
  const sideEl = document.getElementById('ob-hist-side');
  if(sideEl) sideEl.value = (s === 'DOWN' ? 'DOWN' : 'UP');
  obUpdateHistSideButtons();
  if(obMode === 'history' && obSelectedMarket){
    // Hide chart/table to avoid flicker while loading
    const chartPanel = document.getElementById('ob-hist-chart-panel');
    const tablePanel = document.getElementById('ob-hist-table-panel');
    chartPanel.classList.add('hidden');
    tablePanel.classList.add('hidden');
    obLoadHistory();
  }
}

function obRenderHistoryPage(){
  const listEl=document.getElementById('ob-hist-list');
  if(!listEl) return;
  const total = obHistoryData.length;
  if(!total){
    listEl.innerHTML='<span class="text-slate-400">No snapshots in this range.</span>';
    const pgEl=document.getElementById('ob-hist-page');
    if(pgEl) pgEl.textContent='';
    return;
  }

  const totalPages = Math.max(1, Math.ceil(total / OB_HIST_PER_PAGE));
  if(obHistPage > totalPages) obHistPage = totalPages;
  if(obHistPage < 1) obHistPage = 1;
  const startIdx = (obHistPage - 1) * OB_HIST_PER_PAGE;
  const endIdx = Math.min(startIdx + OB_HIST_PER_PAGE, total);

  const prevBtn=document.getElementById('ob-hist-prev');
  const nextBtn=document.getElementById('ob-hist-next');
  if(prevBtn) prevBtn.disabled = obHistPage <= 1;
  if(nextBtn) nextBtn.disabled = obHistPage >= totalPages;
  const pgEl=document.getElementById('ob-hist-page');
  if(pgEl) pgEl.textContent = `Page ${obHistPage}/${totalPages} (${total})`;

  let html=`<div class="text-xs text-slate-500 mb-2">${total} snapshots</div>`;
  html+='<table><thead><tr><th>Date/Time</th><th>Outcome</th><th>Best Ask</th></tr></thead><tbody>';
  obHistoryData.slice(startIdx, endIdx).forEach((snap, localIdx)=>{
    const idx = startIdx + localIdx;
    const d = new Date(snap.ts*1000);
    const dateStr = d.toLocaleDateString('ru-RU',{day:'2-digit',month:'2-digit',timeZone:'UTC'}) + ' ' + d.toLocaleTimeString('ru-RU',{timeZone:'UTC'});
    const isUp = (snap.outcome_name||'').toUpperCase().includes('UP');
    const nameClass = isUp ? 'text-green-400' : 'text-red-400';
    html+=`<tr class="ob-snapshot-row cursor-pointer" onclick="obShowSnapshot(${idx})"><td class="text-xs">${dateStr}</td><td class="${nameClass} text-xs">${snap.outcome_name}</td><td class="font-mono text-xs">${snap.best_ask_cents!==null?snap.best_ask_cents+'¢':'-'}</td></tr>`;
  });
  html+='</tbody></table>';
  listEl.innerHTML=html;
}

function obHistPrevPage(){
  obHistPage = Math.max(1, obHistPage - 1);
  obRenderHistoryPage();
}

function obHistNextPage(){
  obHistPage = obHistPage + 1;
  obRenderHistoryPage();
}

function obSetHistView(view){
  obHistView=view;
  const chartBtn = document.getElementById('ob-histview-chart');
  const tableBtn = document.getElementById('ob-histview-table');
  if(chartBtn) chartBtn.classList.toggle('ob-histview-active', view==='chart');
  if(tableBtn) tableBtn.classList.toggle('ob-histview-active', view==='table');
  const chartPanel = document.getElementById('ob-hist-chart-panel');
  const tablePanel = document.getElementById('ob-hist-table-panel');
  // Hide both first to avoid flicker
  chartPanel.classList.add('hidden');
  tablePanel.classList.add('hidden');
  // Then show the target
  if(view==='chart'){
    chartPanel.classList.remove('hidden');
  } else {
    tablePanel.classList.remove('hidden');
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
      fetch(API+'/api/poly/markets?limit=500'),
      fetch(API+'/api/poly/sim/markets_with_positions')
    ]);
    const data=await res.json();
    const marketsWithPosRaw = await posRes.json();
    const marketsWithPos = new Set(Array.isArray(marketsWithPosRaw) ? marketsWithPosRaw : []);
    if(!Array.isArray(data)||!data.length){
      el.innerHTML='<div class="text-slate-400">No markets.</div>';
      const pgEl=document.getElementById('ob-markets-page');
      if(pgEl) pgEl.textContent='';
      return;
    }

    const total = data.length;
    const totalPages = Math.max(1, Math.ceil(total / OB_MARKETS_PER_PAGE));
    if(obMarketsPage > totalPages) obMarketsPage = totalPages;
    if(obMarketsPage < 1) obMarketsPage = 1;
    const startIdx = (obMarketsPage - 1) * OB_MARKETS_PER_PAGE;
    const endIdx = Math.min(startIdx + OB_MARKETS_PER_PAGE, total);

    const prevBtn=document.getElementById('ob-markets-prev');
    const nextBtn=document.getElementById('ob-markets-next');
    if(prevBtn) prevBtn.disabled = obMarketsPage <= 1;
    if(nextBtn) nextBtn.disabled = obMarketsPage >= totalPages;
    const pgEl=document.getElementById('ob-markets-page');
    if(pgEl) pgEl.textContent = `Page ${obMarketsPage}/${totalPages} (${total})`;

    let html='<table><thead><tr><th>Time</th><th>Status</th></tr></thead><tbody>';
    data.slice(startIdx, endIdx).forEach(m=>{
      const d=new Date((m.ts||0)*1000);
      const dateStr = d.toLocaleDateString('ru-RU',{day:'2-digit',month:'2-digit',timeZone:'UTC'}) + ' ' + d.toLocaleTimeString('ru-RU',{hour:'2-digit',minute:'2-digit',timeZone:'UTC'});
      const status = m.status || (m.closed ? 'ended' : 'open');
      const statusClass = status==='ended' ? 'badge badge-queue' : 'badge badge-done';
      const resolved = (m.resolved_outcome||'');
      const resolvedBadge = resolved === 'UP'
        ? '<span style="margin-left:8px;color:#22c55e;font-weight:700">▲</span>'
        : (resolved === 'DOWN'
          ? '<span style="margin-left:8px;color:#ef4444;font-weight:700">▼</span>'
          : '');

      const pred = (m.prediction_outcome||'');
      const predTri = pred === 'UP'
        ? '<span style="margin-left:8px;color:#22c55e;font-weight:700;background:rgba(245,158,11,0.22);padding:1px 6px;border-radius:6px">▲</span>'
        : (pred === 'DOWN'
          ? '<span style="margin-left:8px;color:#ef4444;font-weight:700;background:rgba(245,158,11,0.22);padding:1px 6px;border-radius:6px">▼</span>'
          : (pred === 'UNDEFINED'
            ? '<span style="margin-left:8px;color:#94a3b8;font-weight:700;background:rgba(245,158,11,0.22);padding:1px 6px;border-radius:6px">?</span>'
            : ''));
      const predTs = (m.prediction_ts||null);
      const predTitle = predTs ? ` title="pred @ ${new Date(predTs*1000).toLocaleString('ru-RU',{timeZone:'UTC'})}"` : '';
      const predWrap = predTri ? `<span${predTitle}>${predTri}</span>` : '';

      const isActive = (polyActiveTs!==null && (m.ts||0)===polyActiveTs && !m.closed);
      const dot = isActive ? '<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#ef4444;margin-right:4px"></span>' : '';
      const posDot = marketsWithPos.has(m.slug) ? '<span title="has position" style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#22c55e;margin-right:4px"></span>' : '';
      const sel = obSelectedSlug===m.slug ? 'bg-blue-900' : '';
      html+=`<tr class="cursor-pointer ${sel}" onclick="obSelectMarket('${m.slug}')"><td class="text-xs text-slate-400">${dot}${posDot}${dateStr}</td><td><span class="${statusClass}">${status}</span>${resolvedBadge}${predWrap}</td></tr>`;
    });
    html+='</tbody></table>';
    el.innerHTML=html;
  }catch(e){el.textContent='Error loading markets.';}
}

function obMarketsPrevPage(){
  obMarketsPage = Math.max(1, obMarketsPage - 1);
  obLoadMarkets();
}

function obMarketsNextPage(){
  obMarketsPage = obMarketsPage + 1;
  obLoadMarkets();
}

async function obSelectMarket(slug){
  obSelectedSlug=slug;
  document.querySelectorAll('#ob-market-list tr').forEach(tr=>tr.classList.remove('bg-blue-900'));
  document.querySelectorAll('#ob-market-list tr').forEach(tr=>{
    if(tr.getAttribute('onclick') && tr.getAttribute('onclick').includes(`'${slug}'`)) tr.classList.add('bg-blue-900');
  });

  try{
    const res=await fetch(API+'/api/poly/market/'+encodeURIComponent(slug));
    const m=await res.json();
    if(m.error){ document.getElementById('ob-market-info').textContent=m.error; return; }
    obSelectedMarket=m;

    const isDone = m.closed || (m.ts && (m.ts+300) < Math.floor(Date.now()/1000));
    const statusStr = isDone ? '<span class="badge badge-queue">ended</span>' : '<span class="badge badge-done">open</span>';
    const d = new Date((m.ts||0)*1000);
    const dateStr = d.toLocaleDateString('ru-RU',{timeZone:'UTC'}) + ' ' + d.toLocaleTimeString('ru-RU',{timeZone:'UTC'});
    document.getElementById('ob-market-info').innerHTML = `<span class="font-mono">${m.slug}</span> · ${dateStr} · ${statusStr}`;

    const liveBtn = document.getElementById('ob-mode-live');
    if(liveBtn) liveBtn.disabled = !!isDone;
    if(isDone){
      obSetMode('history');
    }

    const sideSel = document.getElementById('ob-hist-side');
    if(sideSel){
      const pred = String(m.prediction_outcome || '').toUpperCase();
      const nowUtc = Math.floor(Date.now() / 1000);
      const notStartedYet = !!(m.ts && nowUtc < Number(m.ts));
      if(notStartedYet && (pred === 'UP' || pred === 'DOWN')){
        sideSel.value = pred;
      } else if(!sideSel.value){
        sideSel.value = 'UP';
      }
    }
    obUpdateHistSideButtons();

    if(obMode==='live' && !isDone) obStartLive();
    else obLoadHistory();
  }catch(e){ document.getElementById('ob-market-info').textContent='Error loading market.'; }
}

function toLocalISOString(d){
  const pad=(n)=>String(n).padStart(2,'0');
  return `${d.getUTCFullYear()}-${pad(d.getUTCMonth()+1)}-${pad(d.getUTCDate())}T${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}`;
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
    document.getElementById('ob-live-status').innerHTML = `<span class="text-green-400">Updated ${new Date().toLocaleTimeString('ru-RU',{timeZone:'UTC'})}</span>`;
  }catch(e){
    document.getElementById('ob-live-up').innerHTML='<span class="text-red-400">Error.</span>';
    document.getElementById('ob-live-down').innerHTML='<span class="text-red-400">Error.</span>';
  }
}

// ===== History: load data for both chart and table =====

async function obLoadHistory(){
  if(!obSelectedMarket) return;
  const listEl=document.getElementById('ob-hist-list');
  listEl.textContent='Loading...';
  document.getElementById('ob-hist-detail').innerHTML='<span class="text-slate-400">Click a snapshot to view.</span>';

  // If we have a defined prediction for this market, auto-select that side for AskPriceHistory.
  try{
    let predSide = String(obSelectedMarket.prediction_outcome || '').toUpperCase();
    if(predSide !== 'UP' && predSide !== 'DOWN'){
      try{
        const prRes = await fetch(API + '/api/poly/pred_runs/' + encodeURIComponent(obSelectedMarket.slug) + '?limit=50');
        const runs = await prRes.json();
        if(Array.isArray(runs)){
          const r0 = runs.find(r => r && !r.error && (String(r.prediction || '').toUpperCase() === 'UP' || String(r.prediction || '').toUpperCase() === 'DOWN'));
          if(r0) predSide = String(r0.prediction || '').toUpperCase();
        }
      }catch(e){/* ignore */}
    }
    if(predSide === 'UP' || predSide === 'DOWN'){
      const sideEl = document.getElementById('ob-hist-side');
      if(sideEl && sideEl.value !== predSide){
        sideEl.value = predSide;
        obUpdateHistSideButtons();
      }
    }
  }catch(e){/* ignore */}

  const side = (document.getElementById('ob-hist-side')?.value || 'UP').toUpperCase();
  const pair = findUpDownOutcomes(obSelectedMarket);
  let outcomes = [];
  if(pair && (side === 'UP' || side === 'DOWN')){
    outcomes = [side === 'DOWN' ? pair.down : pair.up];
  } else {
    const allOutcomes = obSelectedMarket.outcomes || [];
    const matched = allOutcomes.find(o => (o.name||'').toUpperCase().includes(side));
    outcomes = matched ? [matched] : (allOutcomes.length ? [allOutcomes[0]] : []);
  }

  if(!outcomes.length){ listEl.innerHTML='<span class="text-slate-400">No outcomes.</span>'; return; }

  try{
    let allSnapshots = [];
    obChartData = {};
    for(const o of outcomes){
      const res = await fetch(API+`/api/poly/orderbook/${encodeURIComponent(obSelectedMarket.slug)}/${encodeURIComponent(o.asset_id)}/analysis`);
      const data = await res.json();
      if(Array.isArray(data)){
        const seriesPoints = [];
        data.forEach(snap=>{
          allSnapshots.push({...snap, outcome_name: o.name, asset_id: o.asset_id});
          seriesPoints.push({ts: snap.ts, best_ask: snap.best_ask_cents, asks: snap.asks||[]});
        });
        if(seriesPoints.length) obChartData[o.name] = seriesPoints;
      }
    }

    allSnapshots.sort((a,b)=>b.ts - a.ts);
    obHistoryData = allSnapshots;

    obHistPage = 1;

    // Show correct panel before drawing to avoid squished canvas
    const chartPanel = document.getElementById('ob-hist-chart-panel');
    const tablePanel = document.getElementById('ob-hist-table-panel');
    if(obHistView === 'chart'){
      chartPanel.classList.remove('hidden');
      tablePanel.classList.add('hidden');
    } else {
      chartPanel.classList.add('hidden');
      tablePanel.classList.remove('hidden');
    }

    if(!allSnapshots.length){
      listEl.innerHTML='<span class="text-slate-400">No snapshots in this range.</span>';
      obDrawChart();
      return;
    }

    obRenderHistoryPage();
    obDrawChart();

    // Auto-scroll chart to the latest data point
    const chartContainer = document.getElementById('ob-chart-container');
    if(chartContainer){
      chartContainer.scrollLeft = chartContainer.scrollWidth;
    }
  }catch(e){
    listEl.innerHTML='<span class="text-red-400">Error loading history.</span>';
  }
}

// ===== Chart rendering (canvas) =====

function obDrawChart(){
  const canvas = document.getElementById('ob-chart-canvas');
  const container = document.getElementById('ob-chart-container');
  if(!canvas || !container) return;
  if(!container.clientWidth){
    requestAnimationFrame(() => obDrawChart());
    return;
  }

  const seriesNames = Object.keys(obChartData);
  if(!seriesNames.length){
    canvas.width=container.clientWidth;
    canvas.height=320;
    const ctx=canvas.getContext('2d');
    ctx.fillStyle='#1e293b';
    ctx.fillRect(0,0,canvas.width,canvas.height);
    ctx.fillStyle='#64748b';
    ctx.font='13px monospace';
    ctx.textAlign='center';
    ctx.fillText('No data to display',canvas.width/2,160);
    return;
  }

  // Collect all timestamps and price range
  let allTs=[], minPrice=Infinity, maxPrice=-Infinity;
  seriesNames.forEach(name=>{
    obChartData[name].forEach(p=>{
      allTs.push(p.ts);
      if(p.best_ask!==null && p.best_ask!==undefined){
        minPrice=Math.min(minPrice,p.best_ask);
        maxPrice=Math.max(maxPrice,p.best_ask);
      }
    });
  });
  allTs = [...new Set(allTs)].sort((a,b)=>a-b);
  if(!allTs.length) return;

  // Add padding to price range
  const priceRange = maxPrice - minPrice;
  const pricePad = Math.max(priceRange * 0.15, 1);
  minPrice -= pricePad;
  maxPrice += pricePad;

  // Chart dimensions
  const dpr = window.devicePixelRatio || 1;
  const paddingLeft=60, paddingRight=30, paddingTop=30, paddingBottom=40;
  const pointSpacing = 6; // px per data point for horizontal scroll
  const chartW = Math.max(container.clientWidth - paddingLeft - paddingRight, allTs.length * pointSpacing);
  const totalW = chartW + paddingLeft + paddingRight;
  const totalH = 320;
  const chartH = totalH - paddingTop - paddingBottom;

  canvas.width = totalW * dpr;
  canvas.height = totalH * dpr;
  canvas.style.width = totalW + 'px';
  canvas.style.height = totalH + 'px';

  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);

  // Background
  ctx.fillStyle='#0f172a';
  ctx.fillRect(0,0,totalW,totalH);

  // Helper: ts -> x
  const tsMin=allTs[0], tsMax=allTs[allTs.length-1];
  const tsRange = Math.max(tsMax-tsMin, 1);
  function tsToX(ts){ return paddingLeft + ((ts-tsMin)/tsRange)*chartW; }
  function priceToY(p){ return paddingTop + chartH - ((p-minPrice)/(maxPrice-minPrice))*chartH; }

  // Grid lines (horizontal)
  ctx.strokeStyle='#1e293b';
  ctx.lineWidth=1;
  const priceSteps = 6;
  const priceStep = (maxPrice-minPrice)/priceSteps;
  ctx.font='10px monospace';
  ctx.fillStyle='#64748b';
  ctx.textAlign='right';
  for(let i=0;i<=priceSteps;i++){
    const p=minPrice+i*priceStep;
    const y=priceToY(p);
    ctx.beginPath();ctx.moveTo(paddingLeft,y);ctx.lineTo(totalW-paddingRight,y);ctx.stroke();
    ctx.fillText(p.toFixed(1)+'¢',paddingLeft-6,y+3);
  }

  // Time labels on X axis
  ctx.textAlign='center';
  ctx.fillStyle='#64748b';
  const maxLabels=Math.min(allTs.length,20);
  const labelStep=Math.max(1,Math.floor(allTs.length/maxLabels));
  for(let i=0;i<allTs.length;i+=labelStep){
    const x=tsToX(allTs[i]);
    const d=new Date(allTs[i]*1000);
    const lbl=d.toLocaleTimeString('ru-RU',{hour:'2-digit',minute:'2-digit',second:'2-digit',timeZone:'UTC'});
    ctx.fillText(lbl,x,totalH-paddingBottom+16);
    ctx.beginPath();ctx.moveTo(x,paddingTop);ctx.lineTo(x,totalH-paddingBottom);ctx.strokeStyle='#1e293b';ctx.stroke();
  }

  // Market start line
  if(obSelectedMarket && obSelectedMarket.ts){
    const marketStartTs = obSelectedMarket.ts;
    if(marketStartTs >= tsMin && marketStartTs <= tsMax){
      const sx = tsToX(marketStartTs);
      ctx.save();
      ctx.strokeStyle='#f59e0b';
      ctx.lineWidth=2;
      ctx.setLineDash([6,4]);
      ctx.beginPath();ctx.moveTo(sx,paddingTop);ctx.lineTo(sx,totalH-paddingBottom);ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle='#f59e0b';
      ctx.font='bold 10px monospace';
      ctx.textAlign='center';
      ctx.fillText('MARKET',sx,paddingTop-6);
      ctx.restore();
    }
  }

  // Series colors
  const colors = {};
  seriesNames.forEach(name=>{
    colors[name] = (name||'').toUpperCase().includes('UP') ? '#22c55e' : '#ef4444';
  });

  // Draw lines
  seriesNames.forEach(name=>{
    const pts = obChartData[name].filter(p=>p.best_ask!==null && p.best_ask!==undefined);
    if(pts.length<2) return;
    ctx.strokeStyle=colors[name];
    ctx.lineWidth=2;
    ctx.beginPath();
    pts.forEach((p,i)=>{
      const x=tsToX(p.ts), y=priceToY(p.best_ask);
      if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
    });
    ctx.stroke();

    // Draw dots
    ctx.fillStyle=colors[name];
    pts.forEach(p=>{
      const x=tsToX(p.ts), y=priceToY(p.best_ask);
      ctx.beginPath();ctx.arc(x,y,2.5,0,Math.PI*2);ctx.fill();
    });
  });

  // Legend
  ctx.font='11px sans-serif';
  let legendX=paddingLeft+10;
  seriesNames.forEach(name=>{
    ctx.fillStyle=colors[name];
    ctx.fillRect(legendX,8,12,12);
    ctx.fillStyle='#e2e8f0';
    ctx.textAlign='left';
    ctx.fillText(name,legendX+16,18);
    legendX+=ctx.measureText(name).width+36;
  });

  // --- Prediction markers on Ask Price chart ---
  const PRED_MARKER_R = 13;
  const predMarkerHits = []; // [{cx, cy, batch_id, batchData, runs}] for hover
  const marketStartTsForMarkers = (typeof obSelectedMarket !== 'undefined' && obSelectedMarket && obSelectedMarket.ts)
    ? obSelectedMarket.ts
    : null;
  const markerCutoffTs = (marketStartTsForMarkers !== null) ? (marketStartTsForMarkers + 300) : null;
  try{
    const predRuns = (typeof polyPredRunsCache !== 'undefined') ? polyPredRunsCache : [];
    if(predRuns.length){
      // Group runs by batch_id, preserving full run list per batch
      const batches = {};
      predRuns.forEach(r => {
        // Ignore failed runs (error present)
        if(r && r.error) return;
        if(!r.batch_id) return;
        if(!batches[r.batch_id]) batches[r.batch_id] = {started_at: r.started_at, quantum: r.quantum, up:0, down:0, unk:0, qCount:0, runs:[]};
        const b = batches[r.batch_id];
        b.runs.push(r);
        if(r.quantum){ b.qCount++; }
        else if(r.prediction === 'UP') b.up++;
        else if(r.prediction === 'DOWN') b.down++;
        else b.unk++;
      });

      const markerY = totalH - paddingBottom - PRED_MARKER_R - 6;
      Object.entries(batches).forEach(([bid, b]) => {
        if(!b.runs || b.runs.length === 0) return;
        if(!b.started_at) return;
        const batchTs = Math.floor(new Date(b.started_at + (b.started_at.endsWith('Z') ? '' : 'Z')).getTime() / 1000);
        if(isNaN(batchTs) || batchTs < tsMin || batchTs > tsMax) return;
        // Only show predictions made before market start or within 5 minutes after market start
        if(markerCutoffTs !== null && batchTs > markerCutoffTs) return;

        const mx = tsToX(batchTs);
        const total = b.up + b.down + b.unk;
        let icon, iconColor;
        if(b.up > b.down && b.up > b.unk){
          icon = '\u25b2'; iconColor = '#22c55e';
        } else if(b.down > b.up && b.down > b.unk){
          icon = '\u25bc'; iconColor = '#ef4444';
        } else if(total === 0 && b.qCount > 0){
          icon = '\u269b'; iconColor = '#8b5cf6';
        } else {
          icon = '?'; iconColor = '#f59e0b';
        }

        // Vertical dashed line
        ctx.save();
        ctx.strokeStyle = iconColor;
        ctx.globalAlpha = 0.25;
        ctx.lineWidth = 1;
        ctx.setLineDash([2, 3]);
        ctx.beginPath(); ctx.moveTo(mx, paddingTop); ctx.lineTo(mx, markerY - PRED_MARKER_R); ctx.stroke();
        ctx.setLineDash([]);
        ctx.globalAlpha = 1.0;

        // Circle border
        ctx.strokeStyle = iconColor;
        ctx.lineWidth = 1.5;
        ctx.globalAlpha = 0.7;
        ctx.beginPath(); ctx.arc(mx, markerY, PRED_MARKER_R, 0, Math.PI * 2); ctx.stroke();

        // Circle fill
        ctx.fillStyle = iconColor;
        ctx.globalAlpha = 0.18;
        ctx.beginPath(); ctx.arc(mx, markerY, PRED_MARKER_R, 0, Math.PI * 2); ctx.fill();
        ctx.globalAlpha = 1.0;

        // Icon
        ctx.fillStyle = iconColor;
        ctx.font = 'bold 13px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(icon, mx, markerY);

        // Count
        if(total > 1){
          ctx.font = 'bold 8px sans-serif';
          ctx.fillStyle = '#cbd5e1';
          ctx.textBaseline = 'alphabetic';
          ctx.fillText(String(total), mx + PRED_MARKER_R + 3, markerY + 4);
        }
        ctx.textBaseline = 'alphabetic';
        ctx.restore();

        predMarkerHits.push({cx: mx, cy: markerY, r: PRED_MARKER_R, batch_id: bid, b, iconColor});
      });
    }
  }catch(e){ /* polyPredRunsCache may not exist in non-poly context */ }

  // Store chart metadata for hover
  canvas._obChartMeta = {tsMin,tsMax,tsRange,chartW,chartH,paddingLeft,paddingTop,paddingBottom,paddingRight,totalW,totalH,minPrice,maxPrice,seriesNames,colors,tsToX,priceToY,predMarkerHits};

  // Attach hover handler (once)
  if(!canvas._obHoverBound){
    canvas._obHoverBound=true;
    canvas.addEventListener('mousemove', obChartHover);
    canvas.addEventListener('mouseleave', ()=>{
      document.getElementById('ob-chart-tooltip').style.display='none';
    });
  }
}

function obChartHover(e){
  const canvas = e.target;
  const meta = canvas._obChartMeta;
  if(!meta) return;

  const rect = canvas.getBoundingClientRect();
  const mx = (e.clientX - rect.left);
  const my = (e.clientY - rect.top);

  const tooltip = document.getElementById('ob-chart-tooltip');

  // --- Check prediction marker hover FIRST ---
  const hits = meta.predMarkerHits || [];
  for(const m of hits){
    const dx = mx - m.cx, dy = my - m.cy;
    if(Math.sqrt(dx*dx + dy*dy) <= m.r + 4){
      // Build prediction detail tooltip
      const dt = m.b.started_at ? m.b.started_at.replace('T',' ').substring(0,19) : '?';
      const isQ = m.b.qCount > 0 && (m.b.up + m.b.down + m.b.unk) === 0;
      let html = `<div style="color:${m.iconColor};font-weight:700;font-size:13px">Prediction Batch</div>`;
      html += `<div style="color:#94a3b8;font-size:10px;margin-bottom:4px">${dt}${isQ ? ' <span style="color:#8b5cf6">⚛ QUANTUM</span>' : ''}</div>`;
      // Aggregate line
      if(!isQ){
        html += `<div style="font-size:11px;margin-bottom:4px">`
          + (m.b.up ? `<span style="color:#22c55e;font-weight:700">▲ ${m.b.up} UP</span>  ` : '')
          + (m.b.down ? `<span style="color:#ef4444;font-weight:700">▼ ${m.b.down} DOWN</span>  ` : '')
          + (m.b.unk ? `<span style="color:#f59e0b">? ${m.b.unk} unknown</span>` : '')
          + `</div>`;
      }
      // Per-template rows
      html += '<div style="border-top:1px solid #334155;padding-top:4px;display:flex;flex-direction:column;gap:2px">';
      m.b.runs.forEach(r => {
        const predColor = r.prediction==='UP' ? '#22c55e' : (r.prediction==='DOWN' ? '#ef4444' : '#f59e0b');
        const predIcon = r.prediction==='UP' ? '▲' : (r.prediction==='DOWN' ? '▼' : '?');
        const prob = r.probability !== null && r.probability !== undefined ? Math.round(r.probability*100)+'%' : '';
        const dur = r.duration_ms ? r.duration_ms+'ms' : '';
        const scBadge = r.quantum_scenario ? ` <span style="color:#8b5cf6">[${r.quantum_scenario}]</span>` : '';
        html += `<div style="display:flex;align-items:center;gap:5px;font-size:10px">`;
        html += `<span style="color:#94a3b8;min-width:70px">${r.template_name||'?'}${scBadge}</span>`;
        html += `<span style="color:#64748b;min-width:55px">${r.strategy||'—'} H${r.horizon||1}</span>`;
        html += `<span style="color:${predColor};font-weight:700">${predIcon} ${r.prediction||'ERR'}</span>`;
        if(prob) html += `<span style="color:#94a3b8">${prob}</span>`;
        if(dur) html += `<span style="color:#475569">${dur}</span>`;
        html += `</div>`;
        if(r.error) html += `<div style="color:#ef4444;font-size:9px;padding-left:4px">${r.error.substring(0,80)}</div>`;
      });
      html += '</div>';
      tooltip.innerHTML = html;
      tooltip.style.display = 'block';
      tooltip.style.left = (e.clientX + 14) + 'px';
      tooltip.style.top = (e.clientY - 10) + 'px';
      const tr = tooltip.getBoundingClientRect();
      if(tr.right > window.innerWidth) tooltip.style.left = (e.clientX - tr.width - 10) + 'px';
      if(tr.bottom > window.innerHeight) tooltip.style.top = (e.clientY - tr.height - 10) + 'px';
      return; // don't show ask tooltip when hovering a prediction marker
    }
  }

  // Check if inside chart area for normal hover
  if(mx < meta.paddingLeft || mx > meta.totalW - meta.paddingRight || my < meta.paddingTop || my > meta.totalH - meta.paddingBottom){
    tooltip.style.display='none';
    return;
  }

  // Find closest timestamp
  const tsAtMouse = meta.tsMin + ((mx - meta.paddingLeft) / meta.chartW) * meta.tsRange;

  // Find closest data point across all series
  let bestDist=Infinity, bestSnap=null, bestName=null;
  meta.seriesNames.forEach(name=>{
    (obChartData[name]||[]).forEach(p=>{
      if(p.best_ask===null || p.best_ask===undefined) return;
      const dist = Math.abs(p.ts - tsAtMouse);
      if(dist < bestDist){ bestDist=dist; bestSnap=p; bestName=name; }
    });
  });

  if(!bestSnap || bestDist > meta.tsRange * 0.03){
    tooltip.style.display='none';
    return;
  }

  const d = new Date(bestSnap.ts*1000);
  const timeStr = d.toLocaleTimeString('ru-RU',{timeZone:'UTC'});
  const dateStr = d.toLocaleDateString('ru-RU',{day:'2-digit',month:'2-digit',timeZone:'UTC'});

  // Build volume breakdown from asks
  let volHtml='';
  if(bestSnap.asks && bestSnap.asks.length){
    const sorted=[...bestSnap.asks].sort((a,b)=>parseFloat(a.price)-parseFloat(b.price)).slice(0,8);
    volHtml='<div style="margin-top:4px;border-top:1px solid #334155;padding-top:4px">';
    volHtml+='<div style="color:#94a3b8;margin-bottom:2px">Asks:</div>';
    sorted.forEach(a=>{
      const p = Math.round(parseFloat(a.price));
      const s = Math.round(parseFloat(a.size));
      const pStr = Number.isFinite(p) ? String(p) : String(a.price);
      const sStr = Number.isFinite(s) ? String(s) : String(a.size);
      volHtml+=`<div><span style="color:#f87171">${pStr}¢</span> × <span style="color:#e2e8f0">${sStr}</span></div>`;
    });
    if(bestSnap.asks.length>8) volHtml+=`<div style="color:#64748b">...and ${bestSnap.asks.length-8} more</div>`;
    volHtml+='</div>';
  }

  const nameColor = (bestName||'').toUpperCase().includes('UP') ? '#22c55e' : '#ef4444';
  tooltip.innerHTML=`<div><span style="color:${nameColor};font-weight:600">${bestName}</span></div>`
    +`<div>${dateStr} ${timeStr}</div>`
    +`<div style="font-size:14px;font-weight:700;color:#f8fafc;margin:2px 0">Best Ask: ${bestSnap.best_ask}¢</div>`
    +volHtml;

  tooltip.style.display='block';
  tooltip.style.left=(e.clientX+14)+'px';
  tooltip.style.top=(e.clientY-10)+'px';

  const tr=tooltip.getBoundingClientRect();
  if(tr.right>window.innerWidth) tooltip.style.left=(e.clientX-tr.width-10)+'px';
  if(tr.bottom>window.innerHeight) tooltip.style.top=(e.clientY-tr.height-10)+'px';
}

// ===== Table view: snapshot detail =====

function obShowSnapshot(idx){
  const snap = obHistoryData[idx];
  if(!snap) return;

  document.querySelectorAll('.ob-snapshot-row').forEach((r,i)=>{
    r.classList.toggle('active', i===idx);
  });

  const el = document.getElementById('ob-hist-detail');
  const d = new Date(snap.ts*1000);
  const fullDate = d.toLocaleDateString('ru-RU',{timeZone:'UTC'}) + ' ' + d.toLocaleTimeString('ru-RU',{timeZone:'UTC'});
  const isUp = (snap.outcome_name||'').toUpperCase().includes('UP');
  const nameClass = isUp ? 'text-green-400' : 'text-red-400';

  let html=`<div class="mb-2"><span class="${nameClass} font-semibold">${snap.outcome_name}</span> · <span class="text-slate-400">${fullDate}</span></div>`;

  html+=`<div class="grid grid-cols-1 gap-2 mb-3">`;
  html+=`<div class="p-2 rounded text-center" style="background:#0f172a"><div class="text-xs text-slate-400">Best Ask</div><div class="font-mono font-bold text-red-400">${snap.best_ask_cents!==null?snap.best_ask_cents+'¢':'-'}</div></div>`;
  html+=`</div>`;

  if(snap.ask_depth!==null && snap.ask_depth!==undefined){
    const ad = (typeof snap.ask_depth === 'number') ? snap.ask_depth.toFixed(0) : snap.ask_depth;
    html+=`<div class="text-xs text-slate-400 mb-2">Ask depth: ${ad}</div>`;
  }

  const asks = snap.asks||[];
  if(asks.length){
    const sorted = [...asks].sort((a,b)=>parseFloat(a.price)-parseFloat(b.price));
    html+=`<div class="mb-2"><span class="text-xs text-red-400 font-semibold">Asks (${sorted.length})</span>`;
    html+='<table class="text-xs"><thead><tr><th>Price (¢)</th><th>Size</th></tr></thead><tbody>';
    sorted.forEach(a=>{
      const p = Math.round(parseFloat(a.price));
      const s = Math.round(parseFloat(a.size));
      html+=`<tr class="text-red-300"><td class="font-mono">${Number.isFinite(p)?p:a.price}</td><td>${Number.isFinite(s)?s:a.size}</td></tr>`;
    });
    html+='</tbody></table></div>';
  }

  el.innerHTML=html;
}
