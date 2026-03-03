// ===== POLYMARKET =====
let polySelectedMarket=null;
let polySelectedOutcome=null;
let polySelectedMarketSlug=null;
let polySelectedOutcomeAssetId=null;
let polyOrderBookInterval=null;
let polyCountdownInterval=null;
let polySelectedSide=null;
let polySelectedPriceCents=null;

// Autopredict guards
let polyLastActiveTs = null;
let polyAutopredictLastTriggeredForEndedTs = null;

let polyMarketsPage = 1;
const POLY_MARKETS_PER_PAGE = 20;

let polyMarketsCache = null;
let polyMarketsWithPosCache = null;

let polyDetailTab = 'live';
let polyPredRunsCache = [];

const POLY_PRED_SETTINGS_KEY = 'poly_pred_settings_v1';

const POLY_LAST_MARKET_KEY = 'poly_last_selected_market_slug_v1';

function polyMarketsPrevPage(){
  polyMarketsPage = Math.max(1, polyMarketsPage - 1);
  loadPolyMarkets();
}

function polyMarketsNextPage(){
  polyMarketsPage = polyMarketsPage + 1;
  loadPolyMarkets();
}

function renderPolyMarkets(){
  const el = document.getElementById('poly-markets');
  if(!el) return;
  const data = Array.isArray(polyMarketsCache) ? polyMarketsCache : [];

  const prevBtn = document.getElementById('poly-markets-prev');
  const nextBtn = document.getElementById('poly-markets-next');
  if(prevBtn) prevBtn.disabled = polyMarketsPage <= 1;
  if(nextBtn) nextBtn.disabled = data.length < POLY_MARKETS_PER_PAGE;

  const pgEl = document.getElementById('poly-markets-page');
  if(pgEl) pgEl.textContent = `Page ${polyMarketsPage}`;

  if(!data.length){
    el.innerHTML = '<div class="text-slate-500 text-xs">No markets.</div>';
    return;
  }

  const marketsWithPos = (polyMarketsWithPosCache instanceof Set) ? polyMarketsWithPosCache : new Set();
  let html = '<table class="w-full"><tbody>';
  data.forEach(m => {
    if(!m) return;
    const dateStr = m.ts ? new Date(m.ts*1000).toLocaleString('ru-RU', {timeZone:'UTC'}) : '';
    const slugSuffix = (m.slug||'').split('-').slice(-1)[0] || '';

    const status = m.status || (m.closed ? 'done' : 'open');
    const statusClass = status === 'active' ? 'badge badge-active' : (status === 'done' ? 'badge badge-done' : 'badge');

    const resolved = (m.resolved_outcome||'');
    const resolvedTri = resolved === 'UP'
      ? '<span title="resolved: UP" style="margin-left:8px;color:#22c55e;font-weight:800">▲</span>'
      : (resolved === 'DOWN'
        ? '<span title="resolved: DOWN" style="margin-left:8px;color:#ef4444;font-weight:800">▼</span>'
        : '');

    let predBadge = '';
    if(m.has_pred){
      let bg = 'rgba(148,163,184,0.14)';
      let fg = '#e2e8f0';
      let title = 'Has predictions';
      if(m.pred_badge === 'green'){
        bg = 'rgba(34,197,94,0.16)';
        fg = '#22c55e';
        title = 'At least one prediction matched resolved outcome';
      } else if(m.pred_badge === 'red'){
        bg = 'rgba(239,68,68,0.16)';
        fg = '#ef4444';
        title = 'Predictions exist, but none matched resolved outcome';
      }
      // For future markets (not resolved), make P gold if there is at least one defined prediction
      const isResolved = !!(m.resolved_outcome && m.resolved_outcome !== '');
      if(!isResolved && m.has_pred_defined){
        fg = '#fbbf24'; // gold
        title = 'Has defined predictions (future market)';
      }
      predBadge = `<span title="${title}" style="margin-left:8px;display:inline-flex;align-items:center;justify-content:center;width:18px;height:18px;border-radius:6px;background:${bg};color:${fg};font-weight:900;font-size:11px;border:1px solid rgba(51,65,85,0.8)">P</span>`;
    }

    const stHtml = `<span class="${statusClass}">${status}</span>${resolvedTri}${predBadge}`;
    const isActive = (polyActiveTs!==null && (m.ts||0)===polyActiveTs && !m.closed);
    const dot = isActive ? '<span title="active" style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#ef4444;margin-right:6px"></span>' : '';
    const posDot = marketsWithPos.has(m.slug) ? '<span title="has position" style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#22c55e;margin-right:6px"></span>' : '';
    const isSelected = polySelectedMarketSlug === m.slug;
    let predTint = '';
    if(m.pred_badge === 'green') predTint = 'style="background:rgba(34,197,94,0.08)"';
    else if(m.pred_badge === 'red') predTint = 'style="background:rgba(239,68,68,0.08)"';
    const selectedClass = isSelected ? 'bg-blue-900' : '';

    html += `<tr ${predTint} class="cursor-pointer ${selectedClass}" data-slug="${m.slug}" onclick="selectPolyMarket('${m.slug}')">`
      + `<td class="text-xs text-slate-400" style="white-space:nowrap">${dot}${posDot}${dateStr} <span class="font-mono text-blue-300">${slugSuffix}</span></td>`
      + `<td>${stHtml}</td>`
      + '</tr>';
  });
  html += '</tbody></table>';
  el.innerHTML = html;
}

let liveMarketPollInterval = null;
function startLiveMarketPoll(){
  stopLiveMarketPoll();
  liveMarketPollInterval = setInterval(async () => {
    try{
      const st = await fetch(API+'/api/poly/status');
      const s = await st.json();
      polyActiveTs = s.active_ts || null;
      if(polyActiveTs === null) return;

      // Fetch first page to likely include the active market (sorted DESC)
      const res = await fetch(API+`/api/poly/markets?limit=${POLY_MARKETS_PER_PAGE}&offset=0`);
      const data = await res.json();
      if(!Array.isArray(data) || !data.length) return;
      polyMarketsCache = data;
      renderPolyMarkets();

      const liveMarket = data.find(m => polyActiveTs !== null && (m.ts||0) === polyActiveTs && !m.closed);
      if(liveMarket){
        stopLiveMarketPoll();
        await selectPolyMarket(liveMarket.slug);
      }
    }catch(e){
      // ignore errors, keep polling
    }
  }, 5000);
}

function stopLiveMarketPoll(){
  if(liveMarketPollInterval){
    clearInterval(liveMarketPollInterval);
    liveMarketPollInterval = null;
  }
}

// ===== AUTOPREDICT =====
let autopredictEnabled = false;

async function loadAutopredictState(){
  try{
    const res = await fetch(API+'/api/poly/settings');
    const data = await res.json();
    autopredictEnabled = !!data.autopredict;
    updateAutopredictUI();
  }catch(e){ console.error('loadAutopredictState error:', e); }

}

function polySetDetailTab(tab, isEnded = null){
  const nowUtc = Math.floor(Date.now() / 1000);
  const isMarketPast = (isEnded !== null)
    ? isEnded
    : (polySelectedMarket && (polySelectedMarket.closed || (polySelectedMarket.ts && (polySelectedMarket.ts + 300) < nowUtc)));

  polyDetailTab = (isMarketPast || tab === 'history') ? 'history' : 'live';

  const liveBtn = document.getElementById('poly-detail-tab-live');
  const histBtn = document.getElementById('poly-detail-tab-history');
  if(liveBtn){
    liveBtn.classList.toggle('hidden', !!isMarketPast);
    liveBtn.classList.toggle('poly-subtab-active', polyDetailTab === 'live');
  }
  if(histBtn){
    histBtn.classList.toggle('poly-subtab-active', polyDetailTab === 'history');
  }

  const lbl = document.getElementById('poly-tab-content-label');
  if(lbl) lbl.textContent = polyDetailTab === 'history' ? 'HISTORY' : 'LIVE';

  const tabsDiv = document.getElementById('poly-detail-tabs');
  if(tabsDiv) tabsDiv.classList.remove('hidden');

  const livePanel = document.getElementById('poly-live-panel');
  const histPanel = document.getElementById('poly-history-panel');
  if(livePanel) livePanel.classList.toggle('hidden', polyDetailTab !== 'live');
  if(histPanel) histPanel.classList.toggle('hidden', polyDetailTab !== 'history');

  if(polyDetailTab === 'live'){
    if(polySelectedMarket && !polySelectedMarket.closed){
      updatePolyOrderBooks();
      startPolyOrderBookUpdates();
    }
  } else {
    stopPolyOrderBookUpdates();
    try{
      if(typeof obStopLive === 'function') obStopLive();
      if(polySelectedMarket){
        // Bind embedded history viewer to current selected market
        obSelectedMarket = polySelectedMarket;
        obSelectedSlug = polySelectedMarket.slug;
        obMode = 'history';
        const sideEl = document.getElementById('ob-hist-side');
        if(sideEl){
          const pred = String(polySelectedMarket.prediction_outcome || '').toUpperCase();
          const nowUtc = Math.floor(Date.now() / 1000);
          const notStartedYet = !!(polySelectedMarket.ts && nowUtc < Number(polySelectedMarket.ts));
          if(notStartedYet && (pred === 'UP' || pred === 'DOWN')){
            sideEl.value = pred;
          } else if(!sideEl.value){
            sideEl.value = 'UP';
          }
        }
        if(typeof obUpdateHistSideButtons === 'function') obUpdateHistSideButtons();
        if(typeof obLoadHistory === 'function') obLoadHistory();
        // Pre-fetch pred_runs so markers appear on the ask price chart
        polyFetchPredRunsForChart(polySelectedMarket.slug);
      }
    }catch(e){/* ignore */}
  }
}

function updateAutopredictUI(){
  const btn = document.getElementById('autopredict-toggle');
  const dot = document.getElementById('autopredict-dot');
  const lbl = document.getElementById('autopredict-label');
  if(!btn||!dot||!lbl) return;
  if(autopredictEnabled){
    btn.style.background='#10b981';
    dot.style.transform='translateX(22px)';
    lbl.textContent='ON';
    lbl.style.color='#10b981';
  } else {
    btn.style.background='#475569';
    dot.style.transform='translateX(2px)';
    lbl.textContent='OFF';
    lbl.style.color='#94a3b8';
  }
}

async function toggleAutopredict(){
  autopredictEnabled = !autopredictEnabled;
  updateAutopredictUI();
  try{
    await fetch(API+'/api/poly/settings', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({
        autopredict: autopredictEnabled,
        strategy: 'rsi_mean_reversion',
        params: null,
        window_size: 1000
      })
    });
  }catch(e){ console.error('toggleAutopredict error:', e); }
}

async function polyAutopredictTrigger(endedMarketTs){
  if(!autopredictEnabled) return;
  // De-dupe: avoid re-triggering for same ended market
  if(polyAutopredictLastTriggeredForEndedTs === endedMarketTs) return;
  polyAutopredictLastTriggeredForEndedTs = endedMarketTs;
  const activeCount = Array.isArray(polyTemplatesCache) ? polyTemplatesCache.filter(t => t.active).length : 0;
  if(!activeCount){
    polyAutopredictSetStatus('⚠ no templates', '#f59e0b');
    return;
  }
  polyAutopredictSetStatus('⏳ searching…', '#94a3b8');

  // Horizon=2 => target markets start >= endedMarketTs + 10 minutes
  const minTargetTs = (endedMarketTs || 0) + 600;

  // Find next 2 markets with ts >= minTargetTs, sorted ascending
  let upcoming = (Array.isArray(polyMarketsCache) ? polyMarketsCache : [])
    .filter(m => m.ts >= minTargetTs)
    .sort((a, b) => a.ts - b.ts)
    .slice(0, 2);

  // If fewer than 2 in cache, fetch fresh list
  if(upcoming.length < 2){
    try{
      const res = await fetch(API + '/api/poly/markets?limit=500');
      const fresh = await res.json();
      if(Array.isArray(fresh)){
        polyMarketsCache = fresh;
        renderPolyMarkets();
        upcoming = fresh
          .filter(m => m.ts >= minTargetTs)
          .sort((a, b) => a.ts - b.ts)
          .slice(0, 2);
      }
    }catch(e){ console.error('[autopredict] market fetch failed:', e); }
  }

  if(!upcoming.length){
    polyAutopredictSetStatus('⚠ no next mkt', '#f59e0b');
    return;
  }

  polyAutopredictSetStatus(`⏳ 0/${upcoming.length}…`, '#60a5fa');
  let done = 0;
  for(const mkt of upcoming){
    await polyAutopredictRunForSlug(mkt.slug, mkt.ts);
    done++;
    polyAutopredictSetStatus(`⏳ ${done}/${upcoming.length}…`, '#60a5fa');
  }
  polyAutopredictSetStatus(`✓ done (${done})`, '#22c55e');
  // Fade status after 6 seconds
  setTimeout(() => polyAutopredictSetStatus('', '#94a3b8'), 6000);
}

async function polyAutopredictRunForSlug(slug, marketTs){
  try{
    const res = await fetch(API + '/api/poly/batch_predict', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({slug, quantum: false, table: 'c_5m'})
    });
    const data = await res.json();
    if(data && data.results){
      // Update markets cache pred_badge so list re-renders with new P badge state
      if(Array.isArray(polyMarketsCache)){
        const mm = polyMarketsCache.find(x => x && x.slug === slug);
        if(mm){
          const up = data.results.filter(r => r.result?.prediction === 'UP').length;
          const dn = data.results.filter(r => r.result?.prediction === 'DOWN').length;
          const unk = data.results.length - up - dn;
          mm.has_pred = true;
          mm.pred_votes = {up, down: dn, unk, ts: Math.floor(Date.now()/1000)};
        }
      }
      renderPolyMarkets();
    }
    console.log(`[autopredict] ${slug} done`, data?.results?.length, 'runs');
  }catch(e){
    console.error(`[autopredict] ${slug} failed:`, e);
  }
}

function polyAutopredictSetStatus(text, color){
  const el = document.getElementById('autopredict-status');
  if(!el) return;
  el.textContent = text;
  el.style.color = color || '#94a3b8';
}

function loadPolyPredictionSettings(){
  // Load templates from server (async, fire-and-forget)
  polyLoadTemplates();
  polyPopulateStrategySelect();
}

function savePolyPredictionSettings(){
  // Templates now manage prediction configs; this is kept for autopredict DB sync
  try{
    fetch(API+'/api/poly/settings', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({
        autopredict: autopredictEnabled,
        strategy: 'rsi_mean_reversion',
        params: null,
        window_size: 1000
      })
    }).catch(()=>{});
  }catch(e){}
}

function clearPolySelection(){
  polySelectedMarket=null;
  polySelectedOutcome=null;
  polySelectedOutcomeAssetId=null;
  stopPolyOrderBookUpdates();
  document.getElementById('poly-market-title').textContent='';
  const emptyEl = document.getElementById('poly-market-empty');
  const secEl = document.getElementById('poly-market-sections');
  if(emptyEl) emptyEl.classList.remove('hidden');
  if(secEl) secEl.classList.add('hidden');
  const detailEl = document.getElementById('poly-market-detail');
  if(detailEl) detailEl.innerHTML='';
  document.getElementById('poly-orderbook-up').innerHTML='<span class="text-slate-400">Select a market.</span>';
  document.getElementById('poly-orderbook-down').innerHTML='<span class="text-slate-400">Select a market.</span>';
  document.getElementById('poly-ob-status').textContent='';
  document.getElementById('poly-sim-msg').textContent='';
  const priceEl = document.getElementById('poly-sim-price');
  if(priceEl) priceEl.value = '';
  polySelectedPriceCents = null;
  const buyBtn = document.getElementById('poly-sim-submit');
  if(buyBtn) buyBtn.disabled = true;
  const predPopup = document.getElementById('poly-pred-popup');
  if(predPopup) predPopup.classList.add('hidden');
  polySelectedSide = null;
  polyUpdateSideButtons();
  document.querySelectorAll('#poly-markets tr').forEach(tr=>tr.classList.remove('bg-blue-900'));
}

function clearPolySelectionComplete(){
  polySelectedMarket=null;
  polySelectedOutcome=null;
  polySelectedMarketSlug=null;
  polySelectedOutcomeAssetId=null;
  stopPolyOrderBookUpdates();
  stopPolyCountdown();
  try{ localStorage.removeItem(POLY_LAST_MARKET_KEY); }catch(e){}
  document.getElementById('poly-market-title').textContent='';
  const emptyEl = document.getElementById('poly-market-empty');
  const secEl = document.getElementById('poly-market-sections');
  if(emptyEl) emptyEl.classList.remove('hidden');
  if(secEl) secEl.classList.add('hidden');
  const detailEl = document.getElementById('poly-market-detail');
  if(detailEl) detailEl.innerHTML='';
  document.getElementById('poly-orderbook-up').innerHTML='<span class="text-slate-400">Select a market.</span>';
  document.getElementById('poly-orderbook-down').innerHTML='<span class="text-slate-400">Select a market.</span>';
  document.getElementById('poly-ob-status').textContent='';
  document.getElementById('poly-sim-msg').textContent='';
  const priceEl = document.getElementById('poly-sim-price');
  if(priceEl) priceEl.value = '';
  polySelectedPriceCents = null;
  const buyBtn = document.getElementById('poly-sim-submit');
  if(buyBtn) buyBtn.disabled = true;
  const predPopup = document.getElementById('poly-pred-popup');
  if(predPopup) predPopup.classList.add('hidden');
  polySelectedSide = null;
  polyUpdateSideButtons();
  document.querySelectorAll('#poly-markets tr').forEach(tr=>tr.classList.remove('bg-blue-900'));
}

function startPolyOrderBookUpdates(){
  stopPolyOrderBookUpdates();
  if(polySelectedMarket && polySelectedMarket.outcomes && polySelectedMarket.outcomes.length >= 2){
    polyOrderBookInterval = setInterval(updatePolyOrderBooks, 800);
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
    const isFuture = marketTs > now;
    // Subtract 5 minutes for future markets
    const adjustedRemaining = isFuture ? remaining - 300 : remaining;
    const minutes = Math.floor(adjustedRemaining / 60);
    const seconds = adjustedRemaining % 60;
    const colorClass = isFuture ? 'text-orange-400' : 'text-green-400';
    countdownEl.innerHTML = `<span class="${colorClass} font-semibold">${minutes.toString().padStart(2,'0')}:${seconds.toString().padStart(2,'0')}</span>`;
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
      polyActiveTs = s.active_ts||null;
    }catch(e){polyActiveTs=null;}

    const [res, posRes] = await Promise.all([
      fetch(API+`/api/poly/markets?limit=${POLY_MARKETS_PER_PAGE}&offset=${(Math.max(1, polyMarketsPage)-1)*POLY_MARKETS_PER_PAGE}`),
      fetch(API+'/api/poly/sim/markets_with_positions')
    ]);
    const data=await res.json();
    const marketsWithPosRaw = await posRes.json();
    const marketsWithPos = new Set(Array.isArray(marketsWithPosRaw) ? marketsWithPosRaw : []);
    if(!Array.isArray(data)||!data.length){
      el.textContent='No markets found';
      return;
    }
    polyMarketsCache=data;
    polyMarketsWithPosCache = marketsWithPos;
    renderPolyMarkets();

    // Only restore last selected market if none is currently selected
    if(!polySelectedMarketSlug){
      let lastSlug = null;
      try{ lastSlug = localStorage.getItem(POLY_LAST_MARKET_KEY); }catch(e){ lastSlug = null; }
      const hasLast = !!(lastSlug && data.find(m => m && m.slug === lastSlug));
      if(hasLast){
        await selectPolyMarket(lastSlug);
      } else {
        // No auto-selection; wait for user to click a market
        startLiveMarketPoll();
      }
    }

    // Apply saved prediction settings whenever the Poly tab is loaded.
    loadPolyPredictionSettings();
  }catch(e){
    console.error('loadPolyMarkets error:', e);
    el.textContent='Error loading markets';
  }
}

async function selectPolyMarket(slug){
  stopLiveMarketPoll(); // Stop auto-polling if user manually selects
  polySelectedMarketSlug = slug;
  try{ if(slug) localStorage.setItem(POLY_LAST_MARKET_KEY, String(slug)); }catch(e){}
  document.querySelectorAll('#poly-markets tr').forEach(tr=>tr.classList.remove('bg-blue-900'));
  const rows = document.querySelectorAll('#poly-markets tr');
  rows.forEach(tr=>{
    if(tr.getAttribute('data-slug')===slug) tr.classList.add('bg-blue-900');
  });
  await showPolyMarket(slug);
}

async function showPolyMarket(slug){
  polySelectedOutcome=null;
  polySelectedOutcomeAssetId=null;
  stopPolyOrderBookUpdates();
  stopPolyCountdown();
  polyClosePredHistoryPanel();
  polyPredRunsCache = [];
  document.getElementById('poly-orderbook-up').innerHTML='<span class="text-slate-400">Loading...</span>';
  document.getElementById('poly-orderbook-down').innerHTML='<span class="text-slate-400">Loading...</span>';
  document.getElementById('poly-sim-msg').textContent='';
  const buyBtn = document.getElementById('poly-sim-submit');
  if(buyBtn){buyBtn.disabled = true;}
  const priceEl = document.getElementById('poly-sim-price');
  if(priceEl) priceEl.value = '';
  polySelectedPriceCents = null;
  polySelectedSide = null;
  polyUpdateSideButtons();

  const title=document.getElementById('poly-market-title');
  const el=document.getElementById('poly-market-detail');
  el.textContent='Loading...';
  try{
    const res=await fetch(API+'/api/poly/market/'+encodeURIComponent(slug));
    const m=await res.json();
    if(m.error){el.textContent=m.error;return}
    polySelectedMarket=m;
    const extUrl=`https://polymarket.com/event/${m.slug}`;
    const isActive = (polyActiveTs!==null && m.ts===polyActiveTs && !m.closed);
    const isEnded = m.closed || (m.ts && (m.ts + 300) < Math.floor(Date.now() / 1000));
    const isFuture = m.ts && (m.ts > Math.floor(Date.now() / 1000));
    let badge = '';
    if(isActive){
      badge = '<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#ef4444;margin-right:6px"></span><span class="text-red-400 font-semibold">ACTIVE</span>';
    } else if(isEnded){
      badge = '<span class="text-slate-400 font-semibold">[ENDED]</span>';
    }
    title.innerHTML = (badge ? badge + ' ' : '') + (m.question||m.slug);

    const emptyEl = document.getElementById('poly-market-empty');
    const secEl = document.getElementById('poly-market-sections');
    if(emptyEl) emptyEl.classList.add('hidden');
    if(secEl) secEl.classList.remove('hidden');
    const isDone = isEnded;
    const obPanel = document.getElementById('poly-ob-panel');
    if(obPanel) obPanel.classList.toggle('hidden', !!isDone);
    const predPanel = document.getElementById('poly-predict-panel');
    if(predPanel){
      predPanel.classList.remove('hidden');
      predPanel.style.display = '';
    }

    // Quantum predict is only valid for markets AFTER the current active one
    const isQuantumEligible = (polyActiveTs !== null && m.ts > polyActiveTs);
    const btnQ = document.getElementById('poly-pred-quantum-btn');
    if(btnQ){
      btnQ.style.display = isQuantumEligible ? '' : 'none';
    }

    const predPopup = document.getElementById('poly-pred-popup');
    if(predPopup) predPopup.classList.add('hidden');
    const predResult = document.getElementById('poly-pred-result');
    if(predResult) predResult.innerHTML='';
    const predResultInline = document.getElementById('poly-pred-result-inline');
    if(predResultInline) predResultInline.innerHTML='';
    loadPolyPredictionSettings();
    const countdownDisplay = (isActive || isFuture) ? `<div class="mb-2 text-xs">${isFuture ? 'Starts in: ' : 'Time remaining: '}<span id="poly-countdown" class="font-mono">--:--</span></div>` : '';

    const pred = (m.prediction_outcome||'');
    const predTs = (m.prediction_ts||null);
    let predHtml = '';
    if(pred){
      const c = pred==='UP' ? '#22c55e' : (pred==='DOWN' ? '#ef4444' : '#94a3b8');
      const a = pred==='UP' ? '▲' : (pred==='DOWN' ? '▼' : '?');
      const t = predTs ? new Date(predTs*1000).toLocaleString('ru-RU',{timeZone:'UTC'}) : '';
      predHtml = ` | pred: <span style="color:${c};font-weight:700;background:rgba(245,158,11,0.22);padding:1px 6px;border-radius:6px">${a} ${pred}</span>${t?` <span class=\"text-slate-500\">@ ${t}</span>`:''}`;
      // Auto-load prediction details if available
      polyLoadPredictionDetails(m.slug);
    }
    let html=`<div class="mb-2 text-xs text-slate-400"><span class="font-mono">${m.slug}</span> | ts: ${m.ts} | closed: ${m.closed}${predHtml}</div>`;
    html+=countdownDisplay;
    html+=`<div class="mb-3 text-xs"><a href="${extUrl}" target="_blank" class="text-blue-400 hover:underline">Open on Polymarket</a></div>`;
    if(m.description) html+=`<div class="text-xs text-slate-400 mb-3">${m.description}</div>`;
    el.innerHTML=html;

    // Wait for orderbook click to choose side + price
    polySelectedOutcome = null;
    polySelectedOutcomeAssetId = null;

    if(isActive || isFuture) startPolyCountdown(m.ts, 300);

    // For ended markets: show only history, no tab buttons
    if(isDone){
      polySetDetailTab('history', true);
    } else if(m.outcomes && m.outcomes.length >= 2){
      polySetDetailTab('live', false);
    } else {
      polySetDetailTab('history', false);
    }

    // Refresh positions/trades for the selected market (panels auto-hide if empty)
    try{
      loadSimPositions();
      loadSimTrades();
    }catch(e){/* ignore */}
  }catch(e){el.textContent='Error loading market';}
}

function updateTabVisibilityForMarket(){
  const tabsDiv = document.getElementById('poly-detail-tabs');
  const lbl = document.getElementById('poly-tab-content-label');
  const livePanel = document.getElementById('poly-live-panel');
  const histPanel = document.getElementById('poly-history-panel');
  const liveBtn = document.getElementById('poly-detail-tab-live');
  if(!tabsDiv || !lbl || !livePanel || !histPanel) return;
  const nowUtc = Math.floor(Date.now() / 1000);
  const isMarketPast = polySelectedMarket && (polySelectedMarket.closed || (polySelectedMarket.ts && (polySelectedMarket.ts + 300) < nowUtc));
  if(liveBtn) liveBtn.classList.toggle('hidden', !!isMarketPast);
  tabsDiv.classList.remove('hidden');
  if(isMarketPast) polyDetailTab = 'history';
  lbl.textContent = polyDetailTab === 'history' ? 'HISTORY' : 'LIVE';
  livePanel.classList.toggle('hidden', polyDetailTab !== 'live');
  histPanel.classList.toggle('hidden', polyDetailTab !== 'history');
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

// ===== Direction selection + Prediction popup =====

function polySelectSide(side){
  polySelectedSide = (side||'UP').toUpperCase();
  polyUpdateSideButtons();
  // Auto-select the matching outcome
  const pair = findUpDownOutcomes(polySelectedMarket);
  if(pair){
    const o = polySelectedSide === 'DOWN' ? pair.down : pair.up;
    polySelectedOutcome = {asset_id: o.asset_id, name: o.name};
    polySelectedOutcomeAssetId = o.asset_id;
  }
  document.getElementById('poly-sim-msg').textContent='';
}

function polyUpdateSideButtons(){
  const upWrap = document.getElementById('poly-ob-side-up');
  const downWrap = document.getElementById('poly-ob-side-down');
  if(upWrap){
    upWrap.style.border = polySelectedSide === 'UP' ? '2px solid #22c55e' : '1px solid #334155';
    upWrap.style.background = polySelectedSide === 'UP' ? '#052e16' : '#0b1220';
  }
  if(downWrap){
    downWrap.style.border = polySelectedSide === 'DOWN' ? '2px solid #ef4444' : '1px solid #334155';
    downWrap.style.background = polySelectedSide === 'DOWN' ? '#450a0a' : '#0b1220';
  }

  const sideEl = document.getElementById('poly-sim-side');
  if(sideEl){
    if(polySelectedSide === 'UP'){
      sideEl.innerHTML = '<span style="display:inline-flex;align-items:center;gap:6px;padding:2px 8px;border-radius:999px;background:rgba(34,197,94,0.15);border:1px solid #22c55e;color:#22c55e;font-weight:800">▲ UP</span>';
    } else if(polySelectedSide === 'DOWN'){
      sideEl.innerHTML = '<span style="display:inline-flex;align-items:center;gap:6px;padding:2px 8px;border-radius:999px;background:rgba(239,68,68,0.15);border:1px solid #ef4444;color:#ef4444;font-weight:800">▼ DOWN</span>';
    } else {
      sideEl.innerHTML = '';
    }
  }
}

function polySetBuyPrice(priceCents, side){
  polySelectedPriceCents = priceCents;
  polySelectSide(side);
  const priceEl = document.getElementById('poly-sim-price');
  if(priceEl) priceEl.value = priceCents;
  const buyBtn = document.getElementById('poly-sim-submit');
  if(buyBtn) buyBtn.disabled = false;
  document.getElementById('poly-sim-msg').textContent='';
  // Re-render highlighting for selected row
  if(polySelectedMarket) updatePolyOrderBooks();
}

function polyTogglePredPopup(){
  const popup = document.getElementById('poly-pred-popup');
  if(popup) popup.classList.toggle('hidden');
}

function polyTogglePredSettings(){
  const popup = document.getElementById('poly-pred-popup');
  if(!popup) return;
  const willShow = popup.classList.contains('hidden');
  if(willShow){
    popup.classList.remove('hidden');
    // Show templates section, hide results
    const analyse = document.getElementById('poly-pred-analyse-section');
    const divider = document.getElementById('poly-pred-divider');
    const tplList = document.getElementById('poly-tpl-list');
    const tplAdd = document.getElementById('poly-tpl-add-details');
    if(analyse) analyse.classList.add('hidden');
    if(divider) divider.classList.remove('hidden');
    if(tplList) tplList.style.display = '';
    if(tplAdd) tplAdd.style.display = '';
    polyLoadTemplates();
  } else {
    popup.classList.add('hidden');
  }
}

function polyTogglePredHistory(){
  const panel = document.getElementById('poly-pred-history-panel');
  if(!panel) return;
  if(panel.classList.contains('hidden')){
    panel.classList.remove('hidden');
    const lbl = document.getElementById('poly-hist-market-label');
    if(lbl) lbl.textContent = polySelectedMarket?.slug || '';
    polyRefreshPredHistory();
  } else {
    panel.classList.add('hidden');
  }
}

function polyClosePredHistoryPanel(){
  const panel = document.getElementById('poly-pred-history-panel');
  if(panel) panel.classList.add('hidden');
}

async function polyFetchPredRunsForChart(slug){
  if(!slug) return;
  try{
    const res = await fetch(API + '/api/poly/pred_runs/' + encodeURIComponent(slug) + '?limit=200');
    const runs = await res.json();
    if(Array.isArray(runs)) polyPredRunsCache = runs;
    // Redraw the ask price chart so markers appear
    if(typeof obDrawChart === 'function') obDrawChart();
  }catch(e){ /* ignore — markers just won't show */ }
}

async function polyRefreshPredHistory(){
  const slug = polySelectedMarket?.slug;
  const el = document.getElementById('poly-pred-history-content');
  if(!el) return;
  if(!slug){ el.innerHTML = '<div class="text-slate-500">No market selected.</div>'; return; }
  el.innerHTML = '<div class="text-slate-500">Loading...</div>';

  // Fetch prediction runs
  let runs = [];
  try{
    const res = await fetch(API + '/api/poly/pred_runs/' + encodeURIComponent(slug) + '?limit=200');
    runs = await res.json();
    if(!Array.isArray(runs)) runs = [];
  }catch(e){
    el.innerHTML = `<div class="text-red-400">Error loading runs: ${e.message}</div>`;
    return;
  }
  polyPredRunsCache = runs;

  if(runs.length === 0){
    el.innerHTML = '<div class="text-slate-500">No prediction history yet.</div>';
  } else {
    el.innerHTML = renderPredHistory(runs);
  }
}

async function polyLoadHistoryChart(slug, runs){
  const wrap = document.getElementById('poly-hist-chart-wrap');
  if(!wrap) return;
  wrap.innerHTML = '<div id="poly-hist-chart-scroll" style="overflow-x:auto"><span class="text-slate-400 text-xs">Loading chart...</span></div>';
  try{
    const ws = 1000, tail = 200;
    const res = await fetch(API+`/api/poly/prediction_candles/${encodeURIComponent(slug)}?window=${ws}&tail=${tail}`);
    const data = await res.json();
    if(data.error || !data.candles || !data.candles.length){
      wrap.innerHTML = `<div class="text-slate-500 text-xs py-2">${data.error || 'No candle data'}</div>`;
      return;
    }
    wrap.innerHTML = `<div class="text-xs text-slate-500 mb-1">${data.candles.length} candles · prediction markers shown</div>`
      + `<div id="poly-hist-chart-scroll" style="overflow-x:auto"><canvas id="poly-hist-chart-canvas" height="280"></canvas></div>`;
    const canvas = document.getElementById('poly-hist-chart-canvas');

    // Build prediction markers from runs: aggregate per market_ts
    const markers = buildPredMarkers(runs, data.candles);
    drawCandleChart(canvas, data.candles, data.market_ts, markers);

    // Scroll to market candle
    try{
      const sc = document.getElementById('poly-hist-chart-scroll');
      const mx = canvas?.dataset?.marketX ? Number(canvas.dataset.marketX) : null;
      if(sc && mx != null && Number.isFinite(mx)){
        sc.scrollLeft = Math.max(0, mx - (sc.clientWidth / 2));
      }
    }catch(e){}
  }catch(e){
    wrap.innerHTML = `<div class="text-red-400 text-xs">${e.message}</div>`;
  }
}

function buildPredMarkers(runs, candles){
  // Build a map: candle_ts → {up, down, unk, quantum}
  // Each run has market_ts (epoch seconds) — that's the candle ts for this market
  // For non-quantum runs, count votes; for quantum runs, mark separately
  const map = {};
  runs.forEach(r => {
    const ts = r.market_ts || (polySelectedMarket?.ts);
    if(!ts) return;
    if(!map[ts]) map[ts] = {up:0, down:0, unk:0, quantum:0, ts: ts};
    if(r.quantum){
      map[ts].quantum++;
    } else {
      if(r.prediction === 'UP') map[ts].up++;
      else if(r.prediction === 'DOWN') map[ts].down++;
      else map[ts].unk++;
    }
  });
  return Object.values(map);
}

function renderPredHistory(runs){
  // Group by batch_id preserving order
  const batches = [];
  const batchMap = {};
  runs.forEach(r => {
    if(!batchMap[r.batch_id]){
      batchMap[r.batch_id] = {batch_id: r.batch_id, started_at: r.started_at, quantum: r.quantum, rows: []};
      batches.push(batchMap[r.batch_id]);
    }
    batchMap[r.batch_id].rows.push(r);
  });

  let html = '<div style="display:flex;flex-direction:column;gap:8px">';
  batches.forEach(b => {
    const dt = b.started_at ? b.started_at.replace('T',' ').substring(0,19) : '—';
    const isQ = b.quantum;
    const qBadge = isQ ? '<span style="color:#8b5cf6;font-weight:700"> ⚛ QUANTUM</span>' : '';

    // Compute vote summary for non-quantum
    let voteSummary = '';
    if(!isQ){
      let up=0, dn=0, unk=0;
      b.rows.forEach(r => {
        if(r.prediction==='UP') up++;
        else if(r.prediction==='DOWN') dn++;
        else unk++;
      });
      const c = up > dn ? '#22c55e' : (dn > up ? '#ef4444' : '#94a3b8');
      voteSummary = ` <span style="color:${c};font-weight:700">`
        + (up > 0 ? `▲${up} ` : '') + (dn > 0 ? `▼${dn} ` : '') + (unk > 0 ? `?${unk}` : '')
        + `</span>`;
    }

    html += `<details style="background:#1e293b;border:1px solid #334155;border-radius:6px;padding:6px 8px">`;
    html += `<summary class="cursor-pointer" style="list-style:none;display:flex;align-items:center;justify-content:space-between">`;
    html += `<span><span class="text-slate-400">${dt}</span>${qBadge}${voteSummary}</span>`;
    html += `<span class="text-slate-600 text-xs">${b.rows.length} row(s)</span>`;
    html += `</summary>`;

    html += `<div style="margin-top:6px;display:flex;flex-direction:column;gap:4px">`;
    b.rows.forEach(r => {
      const predColor = r.prediction==='UP' ? '#22c55e' : (r.prediction==='DOWN' ? '#ef4444' : '#94a3b8');
      const predArrow = r.prediction==='UP' ? '▲' : (r.prediction==='DOWN' ? '▼' : '—');
      const prob = r.probability !== null ? Math.round(r.probability*100)+'%' : '';
      const dur = r.duration_ms !== null ? `${r.duration_ms}ms` : '';
      const scBadge = r.quantum_scenario ? `<span style="color:#8b5cf6;font-size:10px"> [${r.quantum_scenario}]</span>` : '';
      const errBadge = r.error ? `<span style="color:#ef4444"> ERR: ${r.error.substring(0,60)}</span>` : '';
      html += `<div style="display:flex;align-items:center;gap:6px;padding:3px 4px;border-radius:4px;background:#0f172a">`;
      html += `<span style="min-width:90px;color:#94a3b8;font-size:10px">${r.template_name||'?'}${scBadge}</span>`;
      html += `<span style="min-width:60px;font-size:10px;color:#64748b">${r.strategy||'—'} H${r.horizon||1}</span>`;
      html += `<span style="color:${predColor};font-weight:700;min-width:40px">${predArrow} ${r.prediction||'ERR'}</span>`;
      html += `<span style="color:#94a3b8;font-size:10px;min-width:34px">${prob}</span>`;
      html += `<span style="color:#475569;font-size:10px">${dur}</span>`;
      html += `${errBadge}</div>`;
    });
    html += `</div></details>`;
  });
  html += '</div>';
  return html;
}

function polyShowPredDetails(){
  const popup = document.getElementById('poly-pred-popup');
  const analyse = document.getElementById('poly-pred-analyse-section');
  const divider = document.getElementById('poly-pred-divider');
  const tplList = document.getElementById('poly-tpl-list');
  const tplAdd = document.getElementById('poly-tpl-add-details');
  if(popup) popup.classList.remove('hidden');
  if(divider) divider.classList.remove('hidden');
  if(tplList) tplList.style.display = '';
  if(tplAdd) tplAdd.style.display = '';
  if(analyse) analyse.classList.remove('hidden');
}

function polyHidePredDetails(){
  const popup = document.getElementById('poly-pred-popup');
  if(popup) popup.classList.add('hidden');
}

async function polyLoadPredictionDetails(slug){
  try{
    // Prefer persisted full payload (instant Analyse rendering)
    console.log('[polyLoadPredictionDetails] slug:', slug);
    let res = await fetch(API+'/api/poly/prediction/'+encodeURIComponent(slug));
    let data = await res.json();
    console.log('[polyLoadPredictionDetails] response status:', res.status, 'data:', data);
    if(!data || data.error){
      // Fallback to market summary
      res = await fetch(API+'/api/poly/market/'+encodeURIComponent(slug));
      data = await res.json();
      if(!data || data.error) return;
      const pred = (data.prediction_outcome||'');
      const predTs = (data.prediction_ts||null);
      if(!pred) return;
      const inlineEl = document.getElementById('poly-pred-result-inline');
      const resultEl = document.getElementById('poly-pred-result');
      if(!inlineEl || !resultEl) return;
      const color = pred === 'UP' ? '#22c55e' : (pred === 'DOWN' ? '#ef4444' : '#94a3b8');
      const arrow = pred === 'UP' ? '\u25b2' : (pred === 'DOWN' ? '\u25bc' : '\u2014');
      const t = predTs ? new Date(predTs*1000).toLocaleString('ru-RU',{timeZone:'UTC'}) : '';
      inlineEl.innerHTML=
        `<span style="color:${color};font-weight:700;font-size:14px">${arrow} ${pred}</span>`
        + (t ? ` <span class="text-slate-500 text-xs">@ ${t}</span>` : '')
        +`<button onclick="polyShowPredDetails()" class="btn btn-slate text-xs" style="margin-left:8px">Analyse</button>`;
      resultEl.innerHTML=`<div class="p-4 rounded-lg text-center" style="background:#1e293b;border:2px solid ${color}">`
        +`<div style="font-size:48px;font-weight:800;color:${color}">${arrow} ${pred}</div>`
        +`<div class="text-xs text-slate-400 mt-2">Saved prediction summary is available${t?` @ <b>${t}</b>`:''}.</div>`
        +`<div class="text-xs text-slate-500 mt-1">Detailed diagnostics are not available yet for this prediction.</div>`
        +`</div>`;
      return;
    }

    // Full payload (same shape as /api/poly/predict)
    const inlineEl = document.getElementById('poly-pred-result-inline');
    const resultEl = document.getElementById('poly-pred-result');
    if(!inlineEl || !resultEl) return;
    const pred = data.prediction || '';
    if(!pred) return;
    const color = pred === 'UP' ? '#22c55e' : (pred === 'DOWN' ? '#ef4444' : '#94a3b8');
    const arrow = pred === 'UP' ? '\u25b2' : (pred === 'DOWN' ? '\u25bc' : '\u2014');
    const prob = (typeof data.probability === 'number') ? Math.round(data.probability * 100) : null;
    inlineEl.innerHTML=
      `<span style="color:${color};font-weight:700;font-size:14px">${arrow} ${pred}</span> `
      + (prob !== null ? `<span class="text-slate-400 text-xs">${prob}%</span>` : '')
      +`<button onclick="polyShowPredDetails()" class="btn btn-slate text-xs" style="margin-left:8px">Analyse</button>`;

    const agoText = (typeof data.candles_ago === 'number' && data.candles_ago >= 0) ? `${data.candles_ago} candles ago` : 'no signal in tail';
    const sigDt = data.signal_candle_dt || '—';
    const sigRate = (typeof data.tail_size === 'number' && data.tail_size > 0) ? `${data.signals_in_tail}/${data.tail_size}` : '—';

    let diagHtml = '';
    if(data.diag){
      const d = data.diag;
      diagHtml += `<div class="mt-3 text-left" style="background:#0f172a;border:1px solid #334155;border-radius:6px;padding:8px">`;
      diagHtml += `<div class="text-xs text-slate-400 mb-1"><b>Diagnostics</b> (train: ${d.train_size}, tail: ${d.tail_size})</div>`;
      const baseOs = (d.base_oversold!==undefined && d.base_oversold!==null) ? d.base_oversold : '—';
      const baseOb = (d.base_overbought!==undefined && d.base_overbought!==null) ? d.base_overbought : '—';
      diagHtml += `<div class="text-xs text-slate-400">RSI base: &lt;${baseOs} / &gt;${baseOb} | effective: <b style="color:#22c55e">&lt;${d.effective_oversold}</b> / <b style="color:#ef4444">&gt;${d.effective_overbought}</b> (adaptive p10=${d.rsi_p10}, p90=${d.rsi_p90})</div>`;
      diagHtml += `<div class="text-xs text-slate-400">Tail RSI range: ${d.tail_rsi_min} — ${d.tail_rsi_max} | last: <b>${d.tail_rsi_last}</b></div>`;
      if(d.tail_detail && d.tail_detail.length){
        diagHtml += `<table class="mt-2 w-full text-xs"><thead><tr><th>Time</th><th>RSI</th><th>BB</th><th>Prob</th><th>Signal</th></tr></thead><tbody>`;
        d.tail_detail.forEach(r => {
          const sigLabel = r.pred === 1 ? '<span style="color:#22c55e;font-weight:700">UP</span>' : (r.pred === 0 ? '<span style="color:#ef4444;font-weight:700">DOWN</span>' : '<span style="color:#64748b">—</span>');
          const rsiColor = r.rsi < d.effective_oversold ? '#22c55e' : (r.rsi > d.effective_overbought ? '#ef4444' : '#94a3b8');
          diagHtml += `<tr><td>${r.dt}</td><td style="color:${rsiColor};font-weight:600">${r.rsi}</td><td>${r.bb}</td><td>${r.prob}</td><td>${sigLabel}</td></tr>`;
        });
        diagHtml += `</tbody></table>`;
      }
      diagHtml += `</div>`;
    }

    const ws = data.window_size || 1000;
    resultEl.innerHTML=`<div class="p-4 rounded-lg text-center" style="background:#1e293b;border:2px solid ${color}">`
      +`<div style="font-size:48px;font-weight:800;color:${color}">${arrow} ${pred}</div>`
      + (prob !== null ? `<div class="text-lg text-slate-300 mt-1">Probability: <b>${prob}%</b></div>` : '')
      +`<div class="text-xs text-slate-400 mt-2">Signal: <b>${agoText}</b> (${sigDt}) | Signals in tail: <b>${sigRate}</b></div>`
      +`<div class="text-xs text-slate-400 mt-1">Strategy: ${data.strategy||'—'} | Window: ${ws} | Last candle: ${data.last_candle_dt||'—'}</div>`
      +`<details class="mt-2 text-left">`
      +`<summary class="text-blue-400 cursor-pointer text-xs font-semibold">Check Json Params</summary>`
      +`<pre class="mt-2 text-xs text-slate-300" style="white-space:pre-wrap;word-break:break-word;background:#0f172a;border:1px solid #334155;border-radius:6px;padding:8px">${JSON.stringify(data.params||{},null,2)}</pre>`
      +`</details>`
      +diagHtml
      +`<div class="mt-3"><button onclick="loadPredictionCandles('${slug}',${ws})" class="btn btn-primary text-xs">Show Candles Chart</button></div>`
      +`</div>`
      +`<div id="poly-pred-chart-wrap" class="mt-3"></div>`;
  }catch(e){
    // ignore errors
  }
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
    const now = new Date().toLocaleTimeString('ru-RU',{timeZone:'UTC'});
    document.getElementById('poly-ob-status').innerHTML = `<span class="text-green-400">Updated ${now}</span>`;
  }catch(e){
    document.getElementById('poly-orderbook-up').innerHTML='<span class="text-red-400">Error.</span>';
    document.getElementById('poly-orderbook-down').innerHTML='<span class="text-red-400">Error.</span>';
  }
}

function renderAsks(data, outcomeName){
  if(!data || !data.asks || !data.asks.length) return `<div class="text-slate-400">${outcomeName}: no asks</div>`;
  const ts = new Date((data.ts||0)*1000);
  const timeStr = ts.toLocaleTimeString('ru-RU',{timeZone:'UTC'});
  const now = Date.now() / 1000;
  const delaySec = (now - (data.ts||0));
  const delayStr = delaySec >= 0 ? `+${delaySec.toFixed(1)} sec` : `${delaySec.toFixed(1)} sec`;
  const side = (outcomeName||'').toUpperCase().includes('DOWN') ? 'DOWN' : 'UP';
  // Sort ascending by price, take 5 lowest
  const sorted = [...data.asks].sort((a,b)=>parseFloat(a.price)-parseFloat(b.price));
  const top5 = sorted.slice(0,5);
  let html=`<div class="text-slate-500 text-xs mb-1">${side} · ${delayStr} · ${timeStr}</div>`;
  html+='<table class="text-xs"><thead><tr><th>Price (¢)</th><th>Size</th></tr></thead><tbody>';
  top5.forEach(a=>{
    const p = Math.round(parseFloat(a.price));
    const s = Math.round(parseFloat(a.size));
    const pVal = Number.isFinite(p)?p:parseFloat(a.price);
    const isSel = (polySelectedSide === side) && (polySelectedPriceCents !== null) && (Number(polySelectedPriceCents) === Number(pVal));
    const rowStyle = isSel ? (side==='UP' ? 'background:rgba(34,197,94,0.18);outline:1px solid #22c55e;' : 'background:rgba(239,68,68,0.18);outline:1px solid #ef4444;') : '';
    html+=`<tr class="text-red-300 cursor-pointer hover:bg-slate-700" style="${rowStyle}" onclick="polySetBuyPrice(${pVal},'${side}')"><td class="font-mono">${Number.isFinite(p)?p:a.price}</td><td>${Number.isFinite(s)?s:a.size}</td></tr>`;
  });
  html+='</tbody></table>';
  if(data.best_ask_cents!==null){
    html+=`<div class="mt-1 text-xs text-slate-500">Best ask: ${data.best_ask_cents}¢</div>`;
  }
  return html;
}

// ===== Prediction Templates =====

let polyTemplatesCache = [];
let polyCandleSyncPollInterval = null;

function polyStopCandleSyncPoll(){
  if(polyCandleSyncPollInterval){
    clearInterval(polyCandleSyncPollInterval);
    polyCandleSyncPollInterval = null;
  }
}

function polyStartCandleSyncPoll(){
  polyStopCandleSyncPoll();
  const resultEl = document.getElementById('poly-pred-result');
  if(!resultEl) return;
  polyCandleSyncPollInterval = setInterval(async () => {
    try{
      const res = await fetch(API + '/api/candles/sync_status');
      if(!res.ok) return;
      const st = await res.json();
      if(!st) return;
      const running = !!st.running;
      const expected = (typeof st.expected_candles === 'number') ? st.expected_candles : null;
      const downloaded = (typeof st.downloaded_candles === 'number') ? st.downloaded_candles : null;
      const inserted = (typeof st.inserted_rows === 'number') ? st.inserted_rows : null;
      const msg = st.message || '';
      if(running || (downloaded && expected)){
        const pct = (expected && downloaded !== null) ? Math.min(100, Math.floor((downloaded/expected)*100)) : null;
        const bar = (pct !== null)
          ? `<div style="margin-top:6px;width:220px;height:8px;background:#334155;border-radius:999px;overflow:hidden"><div style="height:100%;width:${pct}%;background:#f59e0b"></div></div>`
          : '';
        const lines = [];
        lines.push(`<div class="text-xs text-slate-300"><b>Syncing candles</b>...</div>`);
        if(msg) lines.push(`<div class="text-xs text-slate-400">${msg}</div>`);
        if(expected !== null && downloaded !== null) lines.push(`<div class="text-xs text-slate-400">${downloaded}/${expected} candles</div>`);
        if(inserted !== null) lines.push(`<div class="text-xs text-slate-500">inserted: ${inserted}</div>`);
        resultEl.innerHTML = `<div class="p-3 rounded-lg" style="background:#0f172a;border:1px solid #334155">${lines.join('')}${bar}</div>`;
      }
    }catch(e){}
  }, 350);
}

// --- Template CRUD ---

let polyTemplatesLastError = null;

async function polyLoadTemplates(){
  polyTemplatesLastError = null;
  try{
    const res = await fetch(API + '/api/poly/pred_templates');
    if(!res.ok){
      polyTemplatesLastError = `HTTP ${res.status}`;
      polyRenderTemplateList();
      return;
    }
    const data = await res.json();
    if(Array.isArray(data)){
      polyTemplatesCache = data;
    } else {
      polyTemplatesLastError = 'Invalid response';
    }
  }catch(e){
    const msg = (e && e.message) ? e.message : String(e || 'request failed');
    polyTemplatesLastError = msg;
    // Don't wipe cache on transient connection issues
  }
  polyRenderTemplateList();
}

function polyRenderTemplateList(){
  const el = document.getElementById('poly-tpl-list');
  if(!el) return;
  if(polyTemplatesLastError){
    const hasCached = Array.isArray(polyTemplatesCache) && polyTemplatesCache.length;
    el.innerHTML =
      `<div class="p-2 rounded" style="background:#1c1917;border:1px solid #ef4444">`
      + `<div class="text-xs text-red-300 font-semibold">Templates backend is unavailable</div>`
      + `<div class="text-xs text-red-200 mt-1">${polyTemplatesLastError}</div>`
      + `<div class="mt-2">`
      + `<button onclick="polyLoadTemplates()" class="btn btn-slate text-xs">Retry</button>`
      + (hasCached ? `<span class="text-slate-500 text-xs" style="margin-left:8px">Showing cached templates below.</span>` : ``)
      + `</div>`
      + `</div>`
      + (hasCached ? `<div class="mt-2">${polyTemplatesCache.length} cached template(s) loaded.</div>` : ``);
    if(!hasCached) return;
    // fall through to render cached templates table
  }
  if(!polyTemplatesCache.length){
    el.innerHTML = '<div class="text-slate-500 py-2">No templates yet. Add one below.</div>';
    return;
  }
  let html = '<table class="w-full text-xs"><thead><tr>'
    + '<th style="width:28px"></th><th>Name</th><th>Strategy</th><th>Win</th><th>H</th><th style="width:60px"></th>'
    + '</tr></thead><tbody>';
  polyTemplatesCache.forEach(t => {
    const checked = t.active ? 'checked' : '';
    const rowBg = t.active ? '' : 'opacity:0.45;';
    const hLabel = 'H' + t.horizon;
    html += `<tr style="${rowBg}">`;
    html += `<td><input type="checkbox" ${checked} onchange="polyToggleTemplate(${t.id})" title="Active"></td>`;
    html += `<td class="font-semibold text-slate-200">${t.name || '(unnamed)'}</td>`;
    html += `<td class="text-slate-400 font-mono">${t.strategy}</td>`;
    html += `<td class="text-slate-400">${t.window_size}</td>`;
    html += `<td><span class="font-mono font-bold" style="color:#60a5fa">${hLabel}</span></td>`;
    html += `<td class="text-right">`;
    html += `<button onclick="polyEditTemplate(${t.id})" class="text-blue-400 hover:text-blue-200 px-1" title="Edit">&#9998;</button>`;
    html += `<button onclick="polyDeleteTemplate(${t.id})" class="text-red-400 hover:text-red-200 px-1" title="Delete">&times;</button>`;
    html += `</td></tr>`;
  });
  html += '</tbody></table>';
  el.innerHTML = html;
}

async function polyToggleTemplate(id){
  try{ await fetch(API + `/api/poly/pred_templates/${id}/toggle`, {method:'POST'}); }catch(e){}
  await polyLoadTemplates();
}

async function polyDeleteTemplate(id){
  if(!confirm('Delete this template?')) return;
  try{ await fetch(API + `/api/poly/pred_templates/${id}`, {method:'DELETE'}); }catch(e){}
  await polyLoadTemplates();
}

async function polyCreateTemplate(){
  const name = document.getElementById('poly-tpl-name')?.value?.trim();
  const strategy = document.getElementById('poly-tpl-strategy')?.value || 'rsi_mean_reversion';
  const windowSize = parseInt(document.getElementById('poly-tpl-window')?.value) || 1000;
  const horizon = parseInt(document.getElementById('poly-tpl-horizon')?.value) || 1;
  const paramsText = document.getElementById('poly-tpl-params')?.value?.trim();
  if(!name){ alert('Template name is required'); return; }
  let params = null;
  if(paramsText){
    try{ params = JSON.parse(paramsText); }catch(e){ alert('Invalid JSON in params'); return; }
  }
  try{
    await fetch(API + '/api/poly/pred_templates', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ name, strategy, params, window_size: windowSize, horizon })
    });
    document.getElementById('poly-tpl-name').value = '';
    document.getElementById('poly-tpl-params').value = '';
    const det = document.getElementById('poly-tpl-add-details');
    if(det) det.removeAttribute('open');
  }catch(e){}
  await polyLoadTemplates();
}

async function polyEditTemplate(id){
  const tpl = polyTemplatesCache.find(t => t.id === id);
  if(!tpl) return;
  const newName = prompt('Template name:', tpl.name);
  if(newName === null) return;
  const newWindow = prompt('Window size:', tpl.window_size);
  if(newWindow === null) return;
  const newHorizon = prompt('Horizon (1, 2, or 3):', tpl.horizon);
  if(newHorizon === null) return;
  const newParams = prompt('Params JSON:', tpl.params ? JSON.stringify(tpl.params) : '');
  if(newParams === null) return;
  let parsedParams = null;
  if(newParams.trim()){
    try{ parsedParams = JSON.parse(newParams); }catch(e){ alert('Invalid JSON'); return; }
  }
  try{
    await fetch(API + `/api/poly/pred_templates/${id}`, {
      method:'PUT',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({
        name: newName.trim() || tpl.name,
        window_size: parseInt(newWindow) || tpl.window_size,
        horizon: Math.max(1, Math.min(3, parseInt(newHorizon) || tpl.horizon)),
        params: parsedParams,
      })
    });
  }catch(e){}
  await polyLoadTemplates();
}

async function polyPopulateStrategySelect(){
  const sel = document.getElementById('poly-tpl-strategy');
  if(!sel) return;
  try{
    const res = await fetch(API + '/api/strategies');
    const strats = await res.json();
    sel.innerHTML = '';
    (Array.isArray(strats) ? strats : []).forEach(s => {
      const opt = document.createElement('option');
      opt.value = s.name;
      opt.textContent = s.name;
      sel.appendChild(opt);
    });
  }catch(e){
    sel.innerHTML = '<option value="rsi_mean_reversion">rsi_mean_reversion</option>';
  }
}

// --- Batch Predict ---

async function runPolyBatchPredict(quantum){
  if(!polySelectedMarket || !polySelectedMarket.slug){
    const inlineEl = document.getElementById('poly-pred-result-inline');
    if(inlineEl) inlineEl.innerHTML='<span class="text-red-400 text-xs">Select a market first.</span>';
    return;
  }
  const activeCount = polyTemplatesCache.filter(t => t.active).length;
  if(!activeCount){
    const inlineEl = document.getElementById('poly-pred-result-inline');
    if(inlineEl) inlineEl.innerHTML='<span class="text-red-400 text-xs">No active templates. Open settings and create one.</span>';
    return;
  }

  const btnR = document.getElementById('poly-pred-run-btn');
  const btnQ = document.getElementById('poly-pred-quantum-btn');
  const resultEl = document.getElementById('poly-pred-result');
  const inlineEl = document.getElementById('poly-pred-result-inline');
  const analyseSection = document.getElementById('poly-pred-analyse-section');

  if(btnR) btnR.disabled = true;
  if(btnQ) btnQ.disabled = true;
  const activeBtn = quantum ? btnQ : btnR;
  const origHtml = activeBtn ? activeBtn.innerHTML : '';
  if(activeBtn) activeBtn.innerHTML = quantum
    ? '<span style="font-size:14px">⏳</span> QUANTUM...'
    : '<span style="font-size:14px">⏳</span> PREDICTING...';
  if(inlineEl) inlineEl.innerHTML = `<span class="text-slate-400 text-xs">Running ${activeCount} template(s)${quantum?' (quantum)':''}...</span>`;
  if(resultEl) resultEl.innerHTML = '<span class="text-slate-400">Running batch prediction...</span>';
  if(analyseSection) analyseSection.classList.remove('hidden');

  // Show popup with results section
  const popup = document.getElementById('poly-pred-popup');
  if(popup) popup.classList.remove('hidden');

  polyStartCandleSyncPoll();

  try{
    const res = await fetch(API + '/api/poly/batch_predict', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({
        slug: polySelectedMarket.slug,
        quantum: !!quantum,
        table: 'c_5m'
      })
    });
    const data = await res.json();
    polyStopCandleSyncPoll();

    if(data.error){
      if(resultEl) resultEl.innerHTML = `<div class="p-3 rounded-lg" style="background:#7f1d1d;border:1px solid #ef4444"><span class="text-red-300 font-semibold">Error:</span> <span class="text-red-200">${data.error}</span></div>`;
      if(inlineEl) inlineEl.innerHTML = `<span class="text-red-400 text-xs">${data.error}</span>`;
    } else if(data.results && data.results.length){
      if(quantum){
        renderQuantumResults(data.results, inlineEl, resultEl);
      } else {
        renderRegularResults(data.results, inlineEl, resultEl);
      }
    } else {
      if(resultEl) resultEl.innerHTML = '<span class="text-slate-400">No results returned.</span>';
    }
  }catch(e){
    polyStopCandleSyncPoll();
    if(resultEl) resultEl.innerHTML = `<span class="text-red-400">Request failed: ${e.message}</span>`;
    if(inlineEl) inlineEl.innerHTML = `<span class="text-red-400 text-xs">${e.message || 'Request failed'}</span>`;
  }

  if(btnR){ btnR.disabled = false; btnR.innerHTML = '<span style="font-size:14px">✨</span> PREDICT'; }
  if(btnQ){ btnQ.disabled = false; btnQ.innerHTML = '<span style="font-size:14px">⚛</span> QUANTUM'; }
}

function renderRegularResults(results, inlineEl, resultEl){
  // Inline summary: count UP vs DOWN
  let upCount = 0, downCount = 0, errCount = 0;
  results.forEach(r => {
    const pred = r.result?.prediction;
    if(pred === 'UP') upCount++;
    else if(pred === 'DOWN') downCount++;
    else errCount++;
  });
  const summaryColor = upCount > downCount ? '#22c55e' : (downCount > upCount ? '#ef4444' : '#94a3b8');
  const summaryArrow = upCount > downCount ? '\u25b2' : (downCount > upCount ? '\u25bc' : '\u2014');
  const summaryLabel = upCount > downCount ? 'UP' : (downCount > upCount ? 'DOWN' : 'MIXED');
  if(inlineEl) inlineEl.innerHTML =
    `<span style="color:${summaryColor};font-weight:700;font-size:14px">${summaryArrow} ${summaryLabel}</span> `
    + `<span class="text-slate-400 text-xs">(${upCount}UP/${downCount}DN${errCount?' +'+errCount+'err':''})</span>`
    + `<button onclick="polyShowPredDetails()" class="btn btn-slate text-xs" style="margin-left:8px">Details</button>`;

  // Update markets list: write pred_votes into cache so list shows ▲N ▼M immediately
  try{
    const slug = polySelectedMarket?.slug;
    if(slug && Array.isArray(polyMarketsCache)){
      const mm = polyMarketsCache.find(x => x && x.slug === slug);
      if(mm){
        mm.pred_votes = {up: upCount, down: downCount, unk: errCount, ts: Math.floor(Date.now()/1000)};
      }
      renderPolyMarkets();
    }
  }catch(e){}

  // Detailed results
  let html = '<div style="display:flex;flex-direction:column;gap:8px">';
  results.forEach(r => {
    const d = r.result;
    if(!d) return;
    if(d.error){
      html += `<div class="p-2 rounded" style="background:#1c1917;border:1px solid #ef4444">`;
      html += `<div class="text-xs font-semibold text-slate-300">${r.template_name} <span class="text-slate-500">H${r.horizon}</span></div>`;
      html += `<div class="text-xs text-red-400">${d.error}</div></div>`;
      return;
    }
    const color = d.prediction === 'UP' ? '#22c55e' : (d.prediction === 'DOWN' ? '#ef4444' : '#94a3b8');
    const arrow = d.prediction === 'UP' ? '\u25b2' : (d.prediction === 'DOWN' ? '\u25bc' : '\u2014');
    const prob = Math.round((d.probability || 0) * 100);
    html += `<div class="p-3 rounded-lg" style="background:#1e293b;border-left:4px solid ${color}">`;
    html += `<div class="flex items-center justify-between">`;
    html += `<div><span class="text-xs font-bold text-slate-200">${r.template_name}</span> <span class="text-xs text-slate-500 font-mono">H${r.horizon} · ${d.strategy}</span></div>`;
    html += `<div style="color:${color};font-weight:800;font-size:18px">${arrow} ${d.prediction} <span class="text-sm font-normal text-slate-400">${prob}%</span></div>`;
    html += `</div>`;
    if(d.shift_note){
      html += `<div class="text-xs mt-1 px-2 py-1 rounded" style="background:#422006;color:#fbbf24;border:1px solid #854d0e">${d.shift_note}</div>`;
    }
    if(d.signal_candle_dt){
      html += `<div class="text-xs text-slate-500 mt-1">Signal: ${d.signal_candle_dt} | Window: ${d.window_size} | Last: ${d.last_candle_dt}</div>`;
    }
    // Collapsible diagnostics
    if(d.diag){
      html += `<details class="mt-1"><summary class="text-blue-400 cursor-pointer text-xs">Diagnostics</summary>`;
      html += polyBuildDiagHtml(d);
      html += `</details>`;
    }
    html += `</div>`;
  });
  html += '</div>';
  if(resultEl) resultEl.innerHTML = html;
}

function renderQuantumResults(results, inlineEl, resultEl){
  // Inline summary
  if(inlineEl) inlineEl.innerHTML =
    `<span style="color:#8b5cf6;font-weight:700;font-size:14px">⚛ QUANTUM</span> `
    + `<span class="text-slate-400 text-xs">(${results.length} templates)</span>`
    + `<button onclick="polyShowPredDetails()" class="btn btn-slate text-xs" style="margin-left:8px">Details</button>`;

  let html = '<div style="display:flex;flex-direction:column;gap:10px">';
  results.forEach(r => {
    const d = r.result;
    if(!d) return;
    if(d.error){
      html += `<div class="p-2 rounded" style="background:#1c1917;border:1px solid #ef4444">`;
      html += `<div class="text-xs font-semibold text-slate-300">${r.template_name} <span class="text-slate-500">H${r.horizon}</span></div>`;
      html += `<div class="text-xs text-red-400">${d.error}</div></div>`;
      return;
    }
    const scenarios = d.scenarios || {};
    const green = scenarios.green || {};
    const red = scenarios.red || {};

    html += `<div class="p-3 rounded-lg" style="background:#1e293b;border:1px solid #6d28d9">`;
    html += `<div class="text-xs font-bold text-slate-200 mb-2">${r.template_name} <span class="text-slate-500 font-mono">H${r.horizon}</span></div>`;

    // Two-column layout: green scenario | red scenario
    html += `<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">`;

    // Green scenario
    const gColor = green.prediction === 'UP' ? '#22c55e' : (green.prediction === 'DOWN' ? '#ef4444' : '#94a3b8');
    const gArrow = green.prediction === 'UP' ? '\u25b2' : (green.prediction === 'DOWN' ? '\u25bc' : '\u2014');
    const gProb = Math.round((green.probability || 0) * 100);
    html += `<div class="p-2 rounded" style="background:#052e16;border:1px solid #22c55e50">`;
    html += `<div class="text-xs text-green-400 font-semibold mb-1">If next candle is GREEN \u25b2</div>`;
    html += `<div style="color:${gColor};font-weight:800;font-size:16px">${gArrow} ${green.prediction||'?'} <span class="text-sm font-normal text-slate-400">${gProb}%</span></div>`;
    html += `</div>`;

    // Red scenario
    const rColor = red.prediction === 'UP' ? '#22c55e' : (red.prediction === 'DOWN' ? '#ef4444' : '#94a3b8');
    const rArrow = red.prediction === 'UP' ? '\u25b2' : (red.prediction === 'DOWN' ? '\u25bc' : '\u2014');
    const rProb = Math.round((red.probability || 0) * 100);
    html += `<div class="p-2 rounded" style="background:#1c0a0a;border:1px solid #ef444450">`;
    html += `<div class="text-xs text-red-400 font-semibold mb-1">If next candle is RED \u25bc</div>`;
    html += `<div style="color:${rColor};font-weight:800;font-size:16px">${rArrow} ${red.prediction||'?'} <span class="text-sm font-normal text-slate-400">${rProb}%</span></div>`;
    html += `</div>`;

    html += `</div>`; // grid
    if(d.has_market_candle !== undefined){
      html += `<div class="text-xs text-slate-500 mt-1">Market candle ${d.has_market_candle ? 'exists' : 'missing (synthesized)'}</div>`;
    }
    html += `</div>`;
  });
  html += '</div>';
  if(resultEl) resultEl.innerHTML = html;
}

function polyBuildDiagHtml(data){
  let diagHtml = '';
  if(!data.diag) return diagHtml;
  const d = data.diag;
  diagHtml += `<div class="mt-1 text-left" style="background:#0f172a;border:1px solid #334155;border-radius:6px;padding:6px">`;
  diagHtml += `<div class="text-xs text-slate-400 mb-1"><b>Diagnostics</b> (train: ${d.train_size}, tail: ${d.tail_size})</div>`;
  const baseOs = d.base_oversold ?? '—';
  const baseOb = d.base_overbought ?? '—';
  diagHtml += `<div class="text-xs text-slate-400">RSI: &lt;${baseOs}/&gt;${baseOb} eff: <b style="color:#22c55e">&lt;${d.effective_oversold}</b>/<b style="color:#ef4444">&gt;${d.effective_overbought}</b></div>`;
  if(d.tail_detail && d.tail_detail.length){
    diagHtml += `<table class="mt-1 w-full text-xs"><thead><tr><th>Time</th><th>RSI</th><th>BB</th><th>Prob</th><th>Sig</th></tr></thead><tbody>`;
    d.tail_detail.forEach(r => {
      const sigLabel = r.pred === 1 ? '<span style="color:#22c55e;font-weight:700">UP</span>' : (r.pred === 0 ? '<span style="color:#ef4444;font-weight:700">DN</span>' : '<span style="color:#64748b">\u2014</span>');
      diagHtml += `<tr><td>${r.dt}</td><td>${r.rsi}</td><td>${r.bb}</td><td>${r.prob}</td><td>${sigLabel}</td></tr>`;
    });
    diagHtml += `</tbody></table>`;
  }
  diagHtml += `</div>`;
  return diagHtml;
}

// ===== Prediction candle chart =====

async function loadPredictionCandles(slug, windowSize){
  const wrap = document.getElementById('poly-pred-chart-wrap');
  if(!wrap) return;
  wrap.innerHTML='<span class="text-slate-400 text-xs">Loading candles...</span>';
  try{
    const tail = 200;
    const res = await fetch(API+`/api/poly/prediction_candles/${encodeURIComponent(slug)}?window=${windowSize}&tail=${tail}`);
    const data = await res.json();
    if(data.error){ wrap.innerHTML=`<span class="text-red-400 text-xs">${data.error}</span>`; return; }
    if(!data.candles || !data.candles.length){ wrap.innerHTML='<span class="text-slate-400 text-xs">No candles</span>'; return; }
    wrap.innerHTML=`<div class="text-xs text-slate-400 mb-1">${data.candles.length} candles shown (last ${tail} of ${windowSize} window)</div>`
      +`<div id="poly-pred-scroll" style="overflow-x:auto"><canvas id="poly-pred-canvas" height="320"></canvas></div>`;
    const canvas = document.getElementById('poly-pred-canvas');
    const markers = polyPredRunsCache.length ? buildPredMarkers(polyPredRunsCache, data.candles) : undefined;
    drawCandleChart(canvas, data.candles, data.market_ts, markers);
    // Scroll to market candle if present
    try{
      const sc = document.getElementById('poly-pred-scroll');
      const x = canvas?.dataset?.marketX ? Number(canvas.dataset.marketX) : null;
      if(sc && x != null && Number.isFinite(x)){
        const target = Math.max(0, x - (sc.clientWidth / 2));
        sc.scrollLeft = target;
      }
    }catch(e){/* ignore */}
  }catch(e){
    wrap.innerHTML=`<span class="text-red-400 text-xs">Error: ${e.message}</span>`;
  }
}

function drawCandleChart(canvas, candles, marketTs, markers){
  if(!canvas || !candles.length) return;
  const ctx = canvas.getContext('2d');
  const n = candles.length;
  const candleW = 6;
  const gap = 2;
  const step = candleW + gap;
  const padL = 70, padR = 20, padT = 20, padB = 50;
  const chartW = padL + n * step + padR;
  const chartH = canvas.height;
  canvas.width = Math.max(chartW, canvas.parentElement?.clientWidth || 800);
  const drawW = canvas.width - padL - padR;
  const drawH = chartH - padT - padB;

  // Price range
  let minP = Infinity, maxP = -Infinity;
  candles.forEach(c => { if(c.l < minP) minP = c.l; if(c.h > maxP) maxP = c.h; });
  const pRange = maxP - minP || 1;
  const yScale = drawH / pRange;
  const priceY = p => padT + (maxP - p) * yScale;
  const candleX = i => padL + i * step;

  // Build ts→index map for fast marker lookup
  const tsIdx = {};
  candles.forEach((c, i) => { tsIdx[c.t] = i; });

  // Background
  ctx.fillStyle = '#0f172a';
  ctx.fillRect(0, 0, canvas.width, chartH);

  // Grid lines
  ctx.strokeStyle = '#1e293b';
  ctx.lineWidth = 1;
  const gridSteps = 6;
  for(let i = 0; i <= gridSteps; i++){
    const p = minP + (pRange / gridSteps) * i;
    const y = priceY(p);
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(canvas.width - padR, y); ctx.stroke();
    ctx.fillStyle = '#64748b';
    ctx.font = '10px monospace';
    ctx.textAlign = 'right';
    ctx.fillText(p.toFixed(1), padL - 4, y + 3);
  }

  // Draw candles
  for(let i = 0; i < n; i++){
    const c = candles[i];
    const x = candleX(i);
    const isUp = c.c >= c.o;
    const color = isUp ? '#22c55e' : '#ef4444';

    // Wick
    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x + candleW/2, priceY(c.h));
    ctx.lineTo(x + candleW/2, priceY(c.l));
    ctx.stroke();

    // Body
    const bodyTop = priceY(Math.max(c.o, c.c));
    const bodyBot = priceY(Math.min(c.o, c.c));
    const bodyH = Math.max(bodyBot - bodyTop, 1);
    ctx.fillStyle = color;
    ctx.fillRect(x, bodyTop, candleW, bodyH);
  }

  // Market timestamp marker line
  if(marketTs){
    for(let i = 0; i < n; i++){
      if(candles[i].t === marketTs){
        const x = candleX(i) + candleW/2;
        // Expose for scroll-to-marker
        canvas.dataset.marketX = String(Math.max(0, x - padL));

        const c = candles[i];
        const yTop = padT + 2;
        const yTip = Math.max(padT + 10, priceY(c.h) - 6);

        // Dashed arrow stem
        ctx.strokeStyle = '#f59e0b';
        ctx.lineWidth = 2;
        ctx.setLineDash([4,3]);
        ctx.beginPath(); ctx.moveTo(x, yTop); ctx.lineTo(x, yTip); ctx.stroke();
        ctx.setLineDash([]);

        // Arrow head
        ctx.fillStyle = '#f59e0b';
        ctx.beginPath();
        ctx.moveTo(x, yTip);
        ctx.lineTo(x - 5, yTip - 8);
        ctx.lineTo(x + 5, yTip - 8);
        ctx.closePath();
        ctx.fill();

        // Label
        ctx.fillStyle = '#f59e0b';
        ctx.font = 'bold 10px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('MARKET', x, padT - 4);
        break;
      }
    }
  }

  // --- Prediction markers ---
  if(markers && markers.length){
    markers.forEach(m => {
      const idx = tsIdx[m.ts];
      if(idx === undefined) return;
      const cx = candleX(idx) + candleW / 2;
      const c = candles[idx];
      const yBot = priceY(c.l) + 6;  // below the candle low

      // Determine dominant direction
      const total = m.up + m.down + m.unk;
      let icon, iconColor;
      if(m.up > m.down && m.up > m.unk){
        icon = '\u25b2'; iconColor = '#22c55e'; // green triangle up
      } else if(m.down > m.up && m.down > m.unk){
        icon = '\u25bc'; iconColor = '#ef4444'; // red triangle down
      } else if(total === 0 && m.quantum > 0){
        icon = '\u269b'; iconColor = '#8b5cf6'; // purple atom for quantum-only
      } else {
        icon = '?'; iconColor = '#f59e0b'; // yellow question mark
      }

      // Draw small colored circle background
      ctx.fillStyle = iconColor;
      ctx.globalAlpha = 0.18;
      ctx.beginPath();
      ctx.arc(cx, yBot + 6, 7, 0, Math.PI * 2);
      ctx.fill();
      ctx.globalAlpha = 1.0;

      // Draw icon text
      ctx.fillStyle = iconColor;
      ctx.font = 'bold 10px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(icon, cx, yBot + 6);

      // Draw count label below icon
      if(total > 0){
        ctx.font = '8px sans-serif';
        ctx.fillStyle = '#94a3b8';
        ctx.textBaseline = 'top';
        ctx.fillText(String(total), cx, yBot + 14);
      }
      ctx.textBaseline = 'alphabetic';
    });
  }

  // X-axis timestamps (show every ~20 candles)
  ctx.fillStyle = '#64748b';
  ctx.font = '9px monospace';
  ctx.textAlign = 'center';
  const labelEvery = Math.max(1, Math.floor(n / 10));
  for(let i = 0; i < n; i += labelEvery){
    const c = candles[i];
    const x = candleX(i) + candleW/2;
    const d = new Date(c.t * 1000);
    const lbl = d.toLocaleDateString('ru-RU',{day:'2-digit',month:'2-digit',timeZone:'UTC'}) + ' ' + d.toLocaleTimeString('ru-RU',{hour:'2-digit',minute:'2-digit',timeZone:'UTC'});
    ctx.save();
    ctx.translate(x, chartH - padB + 8);
    ctx.rotate(Math.PI / 4);
    ctx.fillText(lbl, 0, 0);
    ctx.restore();
  }
}

// ===== Sim trades =====

async function submitSimTrade(){
  const msg=document.getElementById('poly-sim-msg');
  msg.textContent='';
  if(!polySelectedOutcome){msg.textContent='Select an outcome first';return}
  if(!polySelectedMarket || !polySelectedMarket.slug){msg.textContent='Select a market first';return}
  const priceVal = document.getElementById('poly-sim-price')?.value;
  if(!priceVal){msg.textContent='Select a price from order book';return}
  if(!polySelectedSide){msg.textContent='Select a price from order book';return}
  const price = parseFloat(priceVal);
  if(!price || price <= 0){msg.textContent='Invalid price';return}
  const qty=parseFloat(document.getElementById('poly-sim-qty').value)||0;
  if(qty <= 0){msg.textContent='Qty must be > 0';return}
  try{
    const res=await fetch(API+'/api/poly/sim/trade',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({
      slug: polySelectedMarket.slug,
      asset_id: polySelectedOutcome.asset_id,
      qty,
      outcome_side: polySelectedSide,
      price
    })});
    const data=await res.json();
    if(data.error){msg.innerHTML=`<span class="text-red-400">${data.error}</span>`;return}
    const fillNote = data.fill_price_cents < price ? ` (filled at better price ${data.fill_price_cents}¢)` : '';
    msg.innerHTML=`<span class="text-green-400">Trade #${data.id} filled @ ${data.fill_price_cents}¢${fillNote}</span>`;
    loadSimTrades();
    loadSimPositions();
    loadPolyMarkets();
  }catch(e){msg.textContent='Error submitting trade';}
}

async function loadSimTrades(){
  const el=document.getElementById('poly-trades');
  el.textContent='Loading...';
  try{
    const panel = document.getElementById('poly-trades-panel');
    if(!polySelectedMarketSlug){
      if(panel) panel.classList.add('hidden');
      el.innerHTML='';
      return;
    }
    const res=await fetch(API+'/api/poly/sim/trades?limit=200');
    const data=await res.json();
    let rows = Array.isArray(data) ? data : [];
    if(polySelectedMarketSlug){
      rows = rows.filter(t => t && t.slug === polySelectedMarketSlug);
    }
    if(!rows.length){
      if(panel) panel.classList.add('hidden');
      el.innerHTML='';
      return;
    }
    if(panel) panel.classList.remove('hidden');
    let html='<div class="max-h-56 overflow-y-auto"><table><thead><tr><th>Time</th><th>Direction</th><th>Side</th><th>Qty</th><th>Fill (¢)</th></tr></thead><tbody>';
    rows.forEach(t=>{
      const d=new Date((t.ts||0)*1000).toISOString().substring(11,19);
      const side=t.side==='BUY'?'<span class="badge badge-up">BUY</span>':'<span class="badge badge-down">SELL</span>';
      const os = t.outcome_side || '';
      const dirIcon = os === 'UP' ? '<span style="color:#22c55e;font-weight:700">▲ UP</span>' : (os === 'DOWN' ? '<span style="color:#ef4444;font-weight:700">▼ DOWN</span>' : '<span class="text-slate-500">—</span>');
      html+=`<tr><td class="text-xs text-slate-400">${d}</td><td>${dirIcon}</td><td>${side}</td><td>${t.qty}</td><td class="font-bold">${t.fill_price_cents}</td></tr>`;
    });
    html+='</tbody></table></div>';
    el.innerHTML=html;
  }catch(e){el.textContent='Error loading trades';}
}

async function loadSimPositions(){
  const el=document.getElementById('poly-positions');
  el.textContent='Loading...';
  try{
    const panel = document.getElementById('poly-positions-panel');
    const slug = polySelectedMarketSlug || '';
    if(!slug){
      if(panel) panel.classList.add('hidden');
      el.innerHTML='';
      return;
    }
    const url = slug ? (API+'/api/poly/sim/positions?slug='+encodeURIComponent(slug)) : (API+'/api/poly/sim/positions');
    const res=await fetch(url);
    const data=await res.json();
    const rows = Array.isArray(data) ? data : [];
    if(!rows.length){
      if(panel) panel.classList.add('hidden');
      el.innerHTML='';
      return;
    }
    if(panel) panel.classList.remove('hidden');
    let html='<table><thead><tr><th>Side</th><th>Pos</th><th>Mark</th><th>PnL (c)</th></tr></thead><tbody>';
    rows.forEach(p=>{
      const pnl=p.pnl_cents;
      const cls=pnl>0?'text-green-400':pnl<0?'text-red-400':'text-slate-300';
      // Determine side from asset_id by checking if it contains 'up' or 'down' (case-insensitive)
      const asset = (p.asset_id||'').toLowerCase();
      let side = '';
      let sideCls = '';
      if(asset.includes('up')){
        side = 'UP';
        sideCls = 'text-green-400 font-semibold';
      } else if(asset.includes('down')){
        side = 'DOWN';
        sideCls = 'text-red-400 font-semibold';
      } else {
        side = '—';
        sideCls = 'text-slate-400';
      }
      const posQty = typeof p.pos_qty === 'number' ? p.pos_qty.toFixed(1) : (p.pos_qty??'');
      const mark = typeof p.mark_cents === 'number' ? p.mark_cents.toFixed(1) : (p.mark_cents??'');
      const pnlVal = typeof pnl === 'number' ? pnl.toFixed(1) : (p.pnl_cents??'');
      html+=`<tr><td class="${sideCls}">${side}</td><td>${posQty}</td><td>${mark}</td><td class="font-bold ${cls}">${pnlVal}</td></tr>`;
    });
    html+='</tbody></table>';
    el.innerHTML=html;
  }catch(e){el.textContent='Error loading positions';}
}
