// ===== POLYMARKET =====
let polySelectedMarket=null;
let polySelectedOutcome=null;
let polySelectedMarketSlug=null;
let polySelectedOutcomeAssetId=null;
let polyOrderBookInterval=null;
let polyCountdownInterval=null;
let polySelectedSide=null;
let polySelectedPriceCents=null;

let polyLastBestAskCents = {UP: null, DOWN: null};

// Autopredict guards
let polyLastActiveTs = null;
let polyAutopredictLastTriggeredForEndedTs = null;

let polyAutopredictStateLoaded = false;

const POLY_MARKETS_PAGE_KEY = 'poly_markets_page_v1';
function _loadStoredMarketsPage(){
  try{
    const raw = localStorage.getItem(POLY_MARKETS_PAGE_KEY);
    const num = parseInt(raw || '1', 10);
    if(Number.isFinite(num) && num >= 1) return num;
  }catch(e){/* ignore */}
  return 1;
}

let polyMarketsPage = _loadStoredMarketsPage();
const POLY_MARKETS_PER_PAGE = 20;
let polyMarketsKnownMaxPage = polyMarketsPage;
let polyMarketsReachedLastPage = false;

let polyMarketsCache = null;
let polyMarketsWithPosCache = null;

let polyDetailTab = 'live';
let polyPredRunsCache = [];
let poly4sPredCache = []; // 4s-early predictions for current market

let polyLastPredBatchId = null;
let polyLastLiveOrderPlacedForBatchId = null;

// Store current prediction undefined reason for Why? button
let polyCurrentUndefinedReason = null;

let polyLastConfirmPromptKey = null;
let polyManualResolveCtx = { slug: null, btn: null, prevText: '' };

let polyNotifAudioCtx = null;

function polyPlayNotifSound(){
  try{
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if(!AudioCtx) return false;
    if(!polyNotifAudioCtx) polyNotifAudioCtx = new AudioCtx();
    if(polyNotifAudioCtx.state === 'suspended'){
      polyNotifAudioCtx.resume().catch(()=>{});
    }
    const o = polyNotifAudioCtx.createOscillator();
    const g = polyNotifAudioCtx.createGain();
    o.type = 'sine';
    o.frequency.setValueAtTime(880, polyNotifAudioCtx.currentTime);
    g.gain.setValueAtTime(0.0001, polyNotifAudioCtx.currentTime);
    g.gain.exponentialRampToValueAtTime(0.08, polyNotifAudioCtx.currentTime + 0.02);
    g.gain.exponentialRampToValueAtTime(0.0001, polyNotifAudioCtx.currentTime + 0.35);
    o.connect(g);
    g.connect(polyNotifAudioCtx.destination);
    o.start();
    o.stop(polyNotifAudioCtx.currentTime + 0.38);
    return true;
  }catch(e){
    return false;
  }
}

function polyDesktopNotify(title, body){
  try{
    if(!('Notification' in window)) return;
    if(Notification.permission === 'granted'){
      new Notification(title, { body });
    } else if(Notification.permission !== 'denied'){
      Notification.requestPermission().then(p => {
        if(p === 'granted') new Notification(title, { body });
      }).catch(()=>{});
    }
  }catch(e){}
}

const POLY_PRED_SETTINGS_KEY = 'poly_pred_settings_v1';

const POLY_LIVE_TRADE_SETTINGS_KEY = 'poly_live_trade_settings_v1';
const POLY_LIVE_TRADE_DEFAULTS = {
  bet_size_usd: 5.0,
  price_cap_cents: 52,
  auto_place: false,
};
const POLY_PRED_UPDATES_CURSOR_KEY = 'poly_pred_updates_cursor_v1';

let polyPredUpdatesPollTimer = null;
let polyPredUpdatesInFlight = false;

let polyEmulate = 'NONE';  // 'UP', 'DOWN', or 'NONE' (default)

let polyBetSizeRequestPollTimer = null;
let polyBetSizePendingState = null;
let polyBetSizePollInFlight = false;
let polyBetSizeCountdownTimer = null;

function polyGetPredUpdatesCursor(){
  try{
    const v = localStorage.getItem(POLY_PRED_UPDATES_CURSOR_KEY);
    const n = parseInt(v || '0', 10);
    return Number.isFinite(n) ? n : 0;
  }catch(e){
    return 0;
  }
}

function polySetPredUpdatesCursor(ts){
  try{ localStorage.setItem(POLY_PRED_UPDATES_CURSOR_KEY, String(parseInt(String(ts||0), 10) || 0)); }catch(e){}
}

async function polyPollPredUpdates(){
  if(polyPredUpdatesInFlight) return;
  polyPredUpdatesInFlight = true;
  try{
    const s = polyGetLiveTradeSettings();
    if(!s.auto_place){
      console.warn('[pred_updates] auto_place disabled, skipping poll tick');
      polyPredUpdatesInFlight = false;
      return;
    }
    const since = polyGetPredUpdatesCursor();
    const res = await fetch(API + `/api/poly/pred_updates?since=${encodeURIComponent(String(since))}&limit=20`);
    if(!res.ok){
      polyPredUpdatesInFlight = false;
      return;
    }
    const data = await res.json();
    const cursor = data?.cursor;
    if(cursor && Number.isFinite(Number(cursor))){
      polySetPredUpdatesCursor(Number(cursor));
    }

    const updates = Array.isArray(data?.updates) ? data.updates : [];
    if(!updates.length){
      console.log('[pred_updates] no updates available');
      polyPredUpdatesInFlight = false;
      return;
    }

    // Process updates in order; show one modal at a time.
    const nowUtc = Math.floor(Date.now()/1000);
    for(const u of updates){
      const slug = u?.slug;
      const rawPred = String(u?.prediction_outcome || '').toUpperCase();
      const pred = (rawPred === 'UNDEFINED' && (polyEmulate === 'UP' || polyEmulate === 'DOWN')) ? polyEmulate : rawPred;
      const pts = u?.prediction_ts ? Number(u.prediction_ts) : null;
      const ts = u?.ts ? Number(u.ts) : null;
      if(!slug || !(pred === 'UP' || pred === 'DOWN')) continue;
      if(!ts || !(nowUtc < ts)) continue; // future only
      if(!pts || (nowUtc - pts) > 60 || (nowUtc - pts) < 0) continue; // only within 60s

      const promptKey = `${slug}:${pts}:${pred}`;
      if(polyLastConfirmPromptKey === promptKey) continue;
      polyLastConfirmPromptKey = promptKey;

      console.log('[pred_updates] new prediction', u, {rawPred, pred, emulate: polyEmulate});
      try{ if(!polySelectedMarketSlug || polySelectedMarketSlug !== slug){ await selectPolyMarket(slug); } }catch(e){}
      await polyPlaceLiveOrderAfterPrediction(slug, pred, null);
      break;
    }
  }catch(e){
    // ignore
  }
  polyPredUpdatesInFlight = false;
}

function polyStartPredUpdatesPoll(){
  if(polyPredUpdatesPollTimer) return;
  // initial tick
  setTimeout(() => { try{ polyPollPredUpdates(); }catch(e){} }, 500);
  polyPredUpdatesPollTimer = setInterval(() => {
    try{ polyPollPredUpdates(); }catch(e){}
  }, 5000);
}

function polyNormalizeLiveTradeSettings(raw){
  const merged = {...POLY_LIVE_TRADE_DEFAULTS};
  if(raw && typeof raw === 'object'){
    const bet = raw.bet_size_usd ?? raw.betSizeUsd;
    const cap = raw.price_cap_cents ?? raw.priceCapCents;
    if(bet !== undefined) merged.bet_size_usd = Number(bet);
    if(cap !== undefined) merged.price_cap_cents = Number(cap);
    if(raw.auto_place !== undefined || raw.autoPlace !== undefined){
      merged.auto_place = !!(raw.auto_place ?? raw.autoPlace);
    }
  }
  merged.bet_size_usd = Number.isFinite(merged.bet_size_usd) ? Math.max(0, merged.bet_size_usd) : POLY_LIVE_TRADE_DEFAULTS.bet_size_usd;
  merged.price_cap_cents = Number.isFinite(merged.price_cap_cents) ? Math.min(Math.max(merged.price_cap_cents, 1), 53) : POLY_LIVE_TRADE_DEFAULTS.price_cap_cents;
  merged.auto_place = !!merged.auto_place;
  return merged;
}

function polyPersistLiveTradeSettings(settings){
  const normalized = polyNormalizeLiveTradeSettings(settings);
  try{ localStorage.setItem(POLY_LIVE_TRADE_SETTINGS_KEY, JSON.stringify(normalized)); }catch(e){}
  return normalized;
}

function polyGetLiveTradeSettings(){
  try{
    const raw = localStorage.getItem(POLY_LIVE_TRADE_SETTINGS_KEY);
    if(!raw) return POLY_LIVE_TRADE_DEFAULTS;
    return polyNormalizeLiveTradeSettings(JSON.parse(raw));
  }catch(e){
    return POLY_LIVE_TRADE_DEFAULTS;
  }
}

async function polyLoadLiveTradeSettings(){
  try{
    const res = await fetch(API + '/api/poly/live/trade_settings');
    if(!res.ok) throw new Error('live trade settings fetch failed');
    const data = await res.json();
    const normalized = polyPersistLiveTradeSettings(data);
    polyApplyLiveTradeSettingsToUI(normalized);
    setTimeout(() => { try{ polyFetchBetSizeRequestState(); }catch(e){} }, 0);
    return normalized;
  }catch(e){
    console.error('polyLoadLiveTradeSettings error:', e);
    polyApplyLiveTradeSettingsToUI();
    return null;
  }
}

// Legacy helpers removed: bet sizing is now fixed dollar amount only.

function polyApplyLiveTradeSettingsToUI(settings){
  const s = settings ? polyNormalizeLiveTradeSettings(settings) : polyGetLiveTradeSettings();
  const betEl = document.getElementById('poly-live-bet-size');
  const capEl = document.getElementById('poly-live-price-cap-cents');
  const autoEl = document.getElementById('poly-live-auto');
  if(betEl) betEl.value = String(s.bet_size_usd);
  if(capEl) capEl.value = String(s.price_cap_cents);
  if(autoEl) autoEl.checked = !!s.auto_place;
}

async function polySaveLiveTradeSettings(){
  const betEl = document.getElementById('poly-live-bet-size');
  const capEl = document.getElementById('poly-live-price-cap-cents');
  const autoEl = document.getElementById('poly-live-auto');
  const msgEl = document.getElementById('poly-live-settings-msg');

  const bet_size_usd = parseFloat(betEl?.value || '5');
  const price_cap_cents = parseFloat(capEl?.value || '52');
  const auto_place = !!autoEl?.checked;

  if(!(price_cap_cents > 0 && price_cap_cents <= 53)){
    if(msgEl) msgEl.innerHTML = '<span class="text-red-300">Price cap must be 1..53 cents</span>';
    return;
  }
  if(!(bet_size_usd >= 0)){
    if(msgEl) msgEl.innerHTML = '<span class="text-red-300">Bet size must be non-negative</span>';
    return;
  }

  const payload = {bet_size_usd, price_cap_cents, auto_place};
  try{
    const res = await fetch(API + '/api/poly/live/trade_settings', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify(payload),
    });
    if(!res.ok) throw new Error(`save failed (${res.status})`);
    const saved = await res.json();
    const status = (saved && saved.status) ? saved.status : 'saved';
    if(status === 'pending'){
      polyHandleBetSizeRequestState(saved);
      if(msgEl) msgEl.innerHTML = '<span class="text-amber-300">Ожидание подтверждения…</span>';
      return;
    }
    if(status === 'saved' && saved.settings){
      const normalized = polyPersistLiveTradeSettings(saved.settings);
      polyApplyLiveTradeSettingsToUI(normalized);
      polyHandleBetSizeRequestState({status:'none'});
      if(msgEl) msgEl.innerHTML = '<span class="text-green-300">Сохранено</span>';
      return;
    }
    if(status === 'pending_request' && saved.request){
      polyHandleBetSizeRequestState(saved);
      if(msgEl) msgEl.innerHTML = '<span class="text-amber-300">Запрос уже в ожидании</span>';
      return;
    }
    // fallback: treat as settings payload
    const normalized = polyPersistLiveTradeSettings(saved);
    polyApplyLiveTradeSettingsToUI(normalized);
    polyHandleBetSizeRequestState({status:'none'});
    if(msgEl) msgEl.innerHTML = '<span class="text-green-300">Обновлено</span>';
  }catch(e){
    console.error('polySaveLiveTradeSettings error:', e);
    const fallback = polyPersistLiveTradeSettings(payload);
    polyApplyLiveTradeSettingsToUI(fallback);
    if(msgEl) msgEl.innerHTML = '<span class="text-red-300">Save failed, cached locally</span>';
  }
}

function polySetBetSizePendingState(state){
  polyBetSizePendingState = state || null;
}

function polyStopBetSizeRequestPoll(){
  if(polyBetSizeRequestPollTimer){
    clearInterval(polyBetSizeRequestPollTimer);
    polyBetSizeRequestPollTimer = null;
  }
  polyBetSizePollInFlight = false;
}

function polyEnsureBetSizeRequestPoll(){
  if(polyBetSizeRequestPollTimer || !polyBetSizePendingState) return;
  polyBetSizeRequestPollTimer = setInterval(() => {
    try{ polyFetchBetSizeRequestState(); }catch(e){}
  }, 5000);
}

function polyStartBetSizeCountdown(ttl){
  const countdownEl = document.getElementById('poly-live-confirm-countdown');
  if(polyBetSizeCountdownTimer){
    clearInterval(polyBetSizeCountdownTimer);
    polyBetSizeCountdownTimer = null;
  }
  if(!countdownEl || !Number.isFinite(ttl) || ttl <= 0){
    if(countdownEl) countdownEl.textContent = '';
    return;
  }
  let remaining = Math.max(0, Math.floor(ttl));
  const render = () => {
    if(!countdownEl) return;
    countdownEl.textContent = `${remaining}s`;
  };
  render();
  polyBetSizeCountdownTimer = setInterval(() => {
    remaining -= 1;
    if(remaining <= 0){
      clearInterval(polyBetSizeCountdownTimer);
      polyBetSizeCountdownTimer = null;
      render();
      return;
    }
    render();
  }, 1000);
}

function polyRenderBetSizePendingMessage(state){
  const msgEl = document.getElementById('poly-live-settings-msg');
  const bannerEl = document.getElementById('poly-live-confirm-banner');
  const detailsEl = document.getElementById('poly-live-confirm-details');
  if(!msgEl) return;
  if(!state){
    msgEl.innerHTML = '';
    if(bannerEl) bannerEl.classList.add('hidden');
    if(detailsEl) detailsEl.textContent = '';
    polyStartBetSizeCountdown(0);
    return;
  }
  const status = state.status || 'pending';
  if(status === 'pending'){
    msgEl.innerHTML = '<span class="text-amber-300">Ожидание подтверждения (до 60 секунд)…</span>';
    if(bannerEl) bannerEl.classList.remove('hidden');
    if(detailsEl){
      const req = state.requested_bet_size ? Number(state.requested_bet_size) : null;
      const prev = state.previous_bet_size ? Number(state.previous_bet_size) : null;
      const delta = (req!==null && prev!==null) ? (req - prev) : null;
      const deltaStr = delta !== null ? ` (${delta>=0?'+':''}${delta.toFixed(2)})` : '';
      detailsEl.textContent = `Новый размер: ${req?.toFixed ? req.toFixed(2) : req || '?'} $${deltaStr}`;
    }
    polyStartBetSizeCountdown(state.expires_in_sec ?? 60);
  }else if(status === 'approved'){
    msgEl.innerHTML = '<span class="text-green-300">Размер ставки обновлён</span>';
    if(bannerEl) bannerEl.classList.add('hidden');
    if(detailsEl) detailsEl.textContent = '';
    polyStartBetSizeCountdown(0);
  }else if(status === 'rejected'){
    msgEl.innerHTML = '<span class="text-red-300">Запрос отклонён</span>';
    if(bannerEl) bannerEl.classList.add('hidden');
    if(detailsEl) detailsEl.textContent = '';
    polyStartBetSizeCountdown(0);
  }else if(status === 'expired'){
    msgEl.innerHTML = '<span class="text-red-300">Запрос отменён: нет подтверждения</span>';
    if(bannerEl) bannerEl.classList.add('hidden');
    if(detailsEl) detailsEl.textContent = '';
    polyStartBetSizeCountdown(0);
  }else{
    msgEl.innerHTML = '';
    if(bannerEl) bannerEl.classList.add('hidden');
    if(detailsEl) detailsEl.textContent = '';
    polyStartBetSizeCountdown(0);
  }
}

function polyHandleBetSizeRequestState(payload){
  if(!payload || payload.status === 'none' || !payload.request){
    polySetBetSizePendingState(null);
    polyStopBetSizeRequestPoll();
    polyRenderBetSizePendingMessage(null);
    return;
  }
  const req = payload.request || {};
  req.status = req.status || payload.status;
  const status = req.status || 'pending';
  polySetBetSizePendingState(req);
  polyRenderBetSizePendingMessage(req);
  if(status === 'pending'){
    polyEnsureBetSizeRequestPoll();
  }else{
    polyStopBetSizeRequestPoll();
    if(status === 'approved'){
      setTimeout(() => { try{ polyLoadLiveTradeSettings(); }catch(e){} }, 200);
    }
  }
}

async function polyFetchBetSizeRequestState(){
  if(polyBetSizePollInFlight) return;
  polyBetSizePollInFlight = true;
  try{
    const res = await fetch(API + '/api/poly/live/trade_settings/request');
    if(res.ok){
      const data = await res.json();
      polyHandleBetSizeRequestState(data);
    }
  }catch(e){
    // ignore
  }
  polyBetSizePollInFlight = false;
}

const POLY_LAST_MARKET_KEY = 'poly_last_selected_market_slug_v1';

function polyPersistMarketsPage(page){
  try{ localStorage.setItem(POLY_MARKETS_PAGE_KEY, String(page)); }catch(e){/* ignore */}
}

function polyGoToMarketsPage(page){
  const nextPage = Math.max(1, parseInt(page, 10) || 1);
  if(nextPage === polyMarketsPage) return;
  polyMarketsPage = nextPage;
  polyPersistMarketsPage(polyMarketsPage);
  loadPolyMarkets();
}

function polyMarketsPrevPage(){
  if(polyMarketsPage <= 1) return;
  polyGoToMarketsPage(polyMarketsPage - 1);
}

async function polyPlaceLiveOrderAfterPrediction(slug, prediction, batch_id){
  try{
    // Live trading is allowed only for FUTURE markets.
    const nowUtc = Math.floor(Date.now() / 1000);
    const mTs = polySelectedMarket?.ts ? Number(polySelectedMarket.ts) : null;
    if(!mTs || !(nowUtc < mTs)){
      console.warn('[live_buy] skip: market not future', {slug, prediction});
      return {success:false, error:'Live trading is allowed only for future markets'};
    }

    const s = polyGetLiveTradeSettings();
    const bet_size_usd = Number.isFinite(Number(s.bet_size_usd)) ? Number(s.bet_size_usd) : POLY_LIVE_TRADE_DEFAULTS.bet_size_usd;
    const price_cap_cents = Math.min(52, Math.max(1, Number(s.price_cap_cents||52)));
    const price_threshold = price_cap_cents / 100.0;
    console.log('[live_buy] settings', {bet_size_usd, price_cap_cents, price_threshold});

    // map prediction to outcome asset
    const pair = findUpDownOutcomes(polySelectedMarket);
    if(!pair){
      console.warn('[live_buy] skip: missing up/down outcomes', {slug, prediction, market: polySelectedMarket});
      return {success:false, error:'Need UP/DOWN outcomes'};
    }
    const pred = String(prediction||'').toUpperCase();
    const outcome_side = (pred === 'DOWN') ? 'DOWN' : 'UP';
    const o = (outcome_side === 'DOWN') ? pair.down : pair.up;
    if(!o || !o.asset_id){
      console.warn('[live_buy] skip: outcome asset_id not found', {slug, outcome_side, pair});
      return {success:false, error:'Outcome asset_id not found'};
    }

    // Use last synchronized best ask (prefer backend refreshed quote for confirm display)
    let snapCents = (polySelectedPriceCents !== null && Number.isFinite(Number(polySelectedPriceCents)))
      ? Number(polySelectedPriceCents)
      : null;
    if(snapCents === null){
      const cached = polyLastBestAskCents ? polyLastBestAskCents[outcome_side] : null;
      if(cached !== null && cached !== undefined && Number.isFinite(Number(cached))){
        snapCents = Number(cached);
      }
    }

    // For confirmation popup: re-fetch quote so displayed snapshot matches backend execution
    if(requireConfirm){
      try{
        const qRes = await fetch(API + `/api/poly/live/quote?slug=${encodeURIComponent(String(slug))}&asset_id=${encodeURIComponent(String(o.asset_id))}`);
        if(qRes.ok){
          const q = await qRes.json();
          const qC = (q && q.best_ask_cents !== undefined && q.best_ask_cents !== null) ? Number(q.best_ask_cents) : null;
          if(qC !== null && Number.isFinite(qC)){
            snapCents = qC;
            // keep cache fresh for later
            try{ if(polyLastBestAskCents) polyLastBestAskCents[outcome_side] = qC; }catch(e){}
          }
        } else {
          console.warn('[live_buy] quote fetch failed', qRes.status);
        }
      }catch(e){
        console.warn('[live_buy] quote fetch exception', e);
      }
    }

    const snapshot_price = snapCents !== null ? (snapCents/100.0) : price_threshold;
    if(snapCents === null){
      console.warn('[live_buy] snapshot_price_cents missing; falling back to threshold', {slug, outcome_side, polySelectedPriceCents, polyLastBestAskCents});
    }

    const payload = {
      slug,
      asset_id: o.asset_id,
      outcome_side,
      prediction_direction: outcome_side,
      amount_usd: bet_size_usd,
      snapshot_price,
      price_threshold,
      bet_size_usd,
      batch_id: batch_id || null,
    };
    console.log('[live_buy] request payload', payload);
    console.log('[live_buy] sending POST to', API + '/api/poly/live/buy');
    const res = await fetch(API + '/api/poly/live/buy', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify(payload),
    });
    console.log('[live_buy] response status', res.status, res.statusText);
    const data = await res.json();
    console.log('[live_buy] response body', data);
    // refresh live panels
    try{ loadLiveOrders(); loadLivePositions(); }catch(e){}
    return data;
  }catch(e){
    console.error('[live_buy] exception', e);
    return {success:false, error: (e && e.message) ? e.message : String(e||'failed')};
  }
}

// ===== LIVE TRADING =====
async function loadLivePositions(){
  const el = document.getElementById('poly-live-positions');
  const panel = document.getElementById('poly-live-positions-panel');
  if(!el) return;
  el.textContent = 'Loading...';
  const slug = polySelectedMarketSlug || '';
  if(!slug){
    if(panel) panel.classList.add('hidden');
    el.innerHTML = '';
    return;
  }
  try{
    const res = await fetch(API + '/api/poly/live/positions?status=open&slug=' + encodeURIComponent(slug));
    const data = await res.json();
    const rows = Array.isArray(data) ? data : [];
    if(!rows.length){
      if(panel) panel.classList.add('hidden');
      el.innerHTML = '';
      return;
    }
    if(panel) panel.classList.remove('hidden');
    let html = '<div class="max-h-56 overflow-y-auto"><table><thead><tr><th>Side</th><th>Shares</th><th>Avg</th><th>Cost</th><th>Opened</th></tr></thead><tbody>';
    rows.forEach(p=>{
      const side = p.outcome_side || '—';
      const sideCls = side === 'UP' ? 'text-green-400' : side === 'DOWN' ? 'text-red-400' : 'text-slate-400';
      const opened = p.opened_at ? String(p.opened_at).slice(11,19) : '';
      html += `<tr><td class="${sideCls}">${side}</td><td>${(p.shares||0).toFixed(4)}</td><td>${(p.avg_price||0).toFixed(4)}</td><td>$${(p.total_cost||0).toFixed(2)}</td><td class="text-slate-400 text-xs">${opened}</td></tr>`;
    });
    html += '</tbody></table></div>';
    el.innerHTML = html;
  }catch(e){
    el.textContent = 'Error loading live positions';
  }
}

async function loadLiveOrders(){
  const el = document.getElementById('poly-live-orders');
  const panel = document.getElementById('poly-live-orders-panel');
  if(!el) return;
  el.textContent = 'Loading...';
  const slug = polySelectedMarketSlug || '';
  if(!slug){
    if(panel) panel.classList.add('hidden');
    el.innerHTML = '';
    return;
  }
  try{
    const res = await fetch(API + '/api/poly/live/orders?limit=200&slug=' + encodeURIComponent(slug));
    const data = await res.json();
    const rows = Array.isArray(data) ? data : [];
    if(!rows.length){
      if(panel) panel.classList.add('hidden');
      el.innerHTML = '';
      return;
    }
    if(panel) panel.classList.remove('hidden');
    let html = '<div class="max-h-56 overflow-y-auto"><table><thead><tr><th>Side</th><th>Price</th><th>Amount</th><th>Status</th><th>Order</th></tr></thead><tbody>';
    rows.forEach(o=>{
      const side = o.side || '';
      const sideCls = side === 'BUY' ? 'text-green-400' : side === 'SELL' ? 'text-red-400' : 'text-slate-400';
      html += `<tr><td class="${sideCls}">${side}</td><td>${(o.price||0).toFixed(4)}</td><td>${(o.amount||0).toFixed(2)}</td><td class="text-slate-400 text-xs">${o.clob_status||''}</td><td class="text-xs font-mono">${o.clob_order_id||'-'}</td></tr>`;
    });
    html += '</tbody></table></div>';
    el.innerHTML = html;
  }catch(e){
    el.textContent = 'Error loading live orders';
  }
}

function polyMarketsNextPage(){
  if(polyMarketsReachedLastPage) return;
  polyGoToMarketsPage(polyMarketsPage + 1);
}

function polyRenderMarketsPageButtons(){
  const container = document.getElementById('poly-markets-page-buttons');
  if(!container){ return; }
  const MAX_BUTTONS = 6;
  let maxPage = polyMarketsKnownMaxPage;
  if(!polyMarketsReachedLastPage){
    maxPage = Math.max(maxPage, polyMarketsPage + 3);
  }
  maxPage = Math.max(maxPage, polyMarketsPage);

  let start = Math.max(1, polyMarketsPage - Math.floor(MAX_BUTTONS / 2));
  let end = start + MAX_BUTTONS - 1;
  if(end > maxPage){
    end = maxPage;
    start = Math.max(1, end - MAX_BUTTONS + 1);
  }

  let html = '';
  if(start > 1){
    html += `<button type="button" class="text-[11px] px-2 py-1 rounded bg-slate-800 text-slate-300 hover:bg-slate-700" onclick="polyGoToMarketsPage(1)">« 1</button>`;
  }
  for(let p = start; p <= end; p++){
    const active = p === polyMarketsPage;
    html += `<button type="button" class="text-[11px] px-2 py-1 rounded ${active ? 'bg-blue-600 text-white' : 'bg-slate-800 text-slate-300 hover:bg-slate-700'}" onclick="polyGoToMarketsPage(${p})">${p}</button>`;
  }
  if(end < maxPage){
    html += `<button type="button" class="text-[11px] px-2 py-1 rounded bg-slate-800 text-slate-300 hover:bg-slate-700" onclick="polyGoToMarketsPage(${maxPage})">${maxPage} »</button>`;
  }
  container.innerHTML = html;
}

async function renderPolyMarkets(){
  const el = document.getElementById('poly-markets');
  if(!el) return;
  const data = Array.isArray(polyMarketsCache) ? polyMarketsCache : [];

  const prevBtn = document.getElementById('poly-markets-prev');
  const nextBtn = document.getElementById('poly-markets-next');
  if(prevBtn) prevBtn.disabled = polyMarketsPage <= 1;
  if(nextBtn) nextBtn.disabled = polyMarketsReachedLastPage;

  const pgEl = document.getElementById('poly-markets-page');
  if(pgEl){
    const suffix = polyMarketsReachedLastPage ? ' (last page)' : '';
    pgEl.textContent = `Page ${polyMarketsPage}${suffix}`;
  }
  polyRenderMarketsPageButtons();

  if(!data.length){
    el.innerHTML = '<div class="text-slate-500 text-xs">No markets.</div>';
    return;
  }

  // Fetch orders for this page to detect successful buys
  let ordersBySlug = {};
  try{
    const slugs = data.map(m=>m.slug).filter(Boolean);
    if(slugs.length){
      const res = await fetch(API + '/api/poly/live/orders?limit=500');
      if(res.ok){
        const allOrders = await res.json();
        const matched = (Array.isArray(allOrders) ? allOrders : []).filter(o => o && o.slug && o.clob_status && (o.clob_status.toLowerCase() === 'matched' || o.clob_status.toLowerCase() === 'filled'));
        matched.forEach(o => { ordersBySlug[o.slug] = true; });
      }
    }
  }catch(e){/* ignore */}

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
    let undefCircle = '';
    const votes = (m && typeof m.pred_votes === 'object' && m.pred_votes) ? m.pred_votes : null;
    const hasDefinedVotes = !!(votes && ((Number(votes.up)||0) + (Number(votes.down)||0)) > 0);
    const hasPredDefined = !!(m && (m.has_pred_defined || hasDefinedVotes));
    const hasAnyPred = !!(m && m.has_pred);
    // Show gray circle if there are predictions but none are defined (UNDEFINED only)
    if(hasAnyPred && !hasPredDefined){
      undefCircle = '<span title="Has UNDEFINED prediction(s)" style="margin-left:8px;display:inline-block;width:10px;height:10px;border-radius:50%;background:#6b7280;border:1px solid rgba(51,65,85,0.8)"></span>';
    }
    if(hasPredDefined){
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
      if(!isResolved && hasPredDefined){
        fg = '#fbbf24'; // gold
        title = 'Has defined predictions (future market)';
      }
      predBadge = `<span title="${title}" style="margin-left:8px;display:inline-flex;align-items:center;justify-content:center;width:18px;height:18px;border-radius:6px;background:${bg};color:${fg};font-weight:900;font-size:11px;border:1px solid rgba(51,65,85,0.8)">P</span>`;
    }

    const isUnresolvedEnded = (status === 'ended' || status === 'done') && !resolved;
    const resolveBtn = isUnresolvedEnded
      ? `<button type="button" title="Set market outcome manually" onclick="polyResolveMarket('${m.slug}', event)" style="margin-left:8px;padding:2px 6px;font-size:10px;line-height:1;border:1px solid #334155;border-radius:6px;background:#0f172a;color:#cbd5e1">↻</button>`
      : '';
    const stHtml = `<span class="${statusClass}">${status}</span>${resolvedTri}${undefCircle}${predBadge}${resolveBtn}`;
    const isActive = (polyActiveTs!==null && (m.ts||0)===polyActiveTs && !m.closed);
    const dot = isActive ? '<span title="active" style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#ef4444;margin-right:6px"></span>' : '';
    const posDot = marketsWithPos.has(m.slug) ? '<span title="has position" style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#22c55e;margin-right:6px"></span>' : '';
    const isSelected = polySelectedMarketSlug === m.slug;
    const selectedClass = isSelected ? 'bg-blue-900' : '';

    const hasSuccessfulBuy = !!ordersBySlug[m.slug];
    const buyIndicator = hasSuccessfulBuy
      ? '<span title="has successful buy order" style="display:inline-block;width:6px;height:6px;background:#fbbf24;border:1px solid rgba(251,191,36,0.5);margin-right:6px;vertical-align:middle"></span>'
      : '';

    html += `<tr class="cursor-pointer ${selectedClass}" data-slug="${m.slug}" onclick="selectPolyMarket('${m.slug}')">`
      + `<td class="text-xs text-slate-400" style="white-space:nowrap">${dot}${posDot}${buyIndicator}${dateStr} <span class="font-mono text-blue-300">${slugSuffix}</span></td>`
      + `<td>${stHtml}</td>`
      + '</tr>';
  });
  html += '</tbody></table>';
  el.innerHTML = html;
}

function polyEnsureManualResolvePopup(){
  let popup = document.getElementById('poly-manual-resolve-popup');
  if(popup) return popup;

  popup = document.createElement('div');
  popup.id = 'poly-manual-resolve-popup';
  popup.className = 'hidden';
  popup.style.position = 'absolute';
  popup.style.zIndex = '9999';
  popup.style.padding = '10px';
  popup.style.border = '1px solid #334155';
  popup.style.borderRadius = '8px';
  popup.style.background = '#0f172a';
  popup.style.color = '#cbd5e1';
  popup.style.boxShadow = '0 4px 12px rgba(0,0,0,0.45)';
  popup.style.fontSize = '12px';
  popup.style.minWidth = '200px';
  popup.innerHTML = `
    <div style="font-size:11px;color:#94a3b8;margin-bottom:6px">Manual resolution</div>
    <div id="poly-manual-resolve-slug" style="font-size:10px;color:#64748b;word-break:break-all;margin-bottom:8px"></div>
    <div style="display:flex;gap:6px;justify-content:center;margin-bottom:8px">
      <button type="button" onclick="polySubmitManualResolve('UP')" style="padding:4px 8px;border:1px solid #14532d;border-radius:6px;background:rgba(34,197,94,0.12);color:#22c55e;font-weight:700;font-size:11px">UP</button>
      <button type="button" onclick="polySubmitManualResolve('DOWN')" style="padding:4px 8px;border:1px solid #7f1d1d;border-radius:6px;background:rgba(239,68,68,0.12);color:#ef4444;font-weight:700;font-size:11px">DOWN</button>
    </div>
    <div style="text-align:center">
      <button type="button" onclick="polyCloseManualResolvePopup()" style="padding:3px 8px;border:1px solid #334155;border-radius:6px;background:#111827;color:#cbd5e1;font-size:10px">Cancel</button>
    </div>
  `;
  document.body.appendChild(popup);
  return popup;
}

function polyCloseManualResolvePopup(){
  const popup = document.getElementById('poly-manual-resolve-popup');
  if(popup) popup.classList.add('hidden');
  polyManualResolveCtx.slug = null;
}

async function polySubmitManualResolve(outcome){
  const slug = polyManualResolveCtx.slug;
  if(!slug) return;
  const side = String(outcome || '').toUpperCase();
  if(side !== 'UP' && side !== 'DOWN') return;

  const btn = polyManualResolveCtx.btn || null;
  const prevText = polyManualResolveCtx.prevText || '↻';
  try{
    if(btn){
      btn.disabled = true;
      btn.textContent = '...';
    }
    const res = await fetch(API + '/api/poly/market/' + encodeURIComponent(slug) + '/resolve', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ outcome: side }),
    });
    const data = await res.json().catch(() => ({}));
    if(!res.ok || data?.error){
      throw new Error(data?.error || ('HTTP ' + res.status));
    }

    polyCloseManualResolvePopup();
    await loadPolyMarkets();
    if(polySelectedMarketSlug === slug){
      await showPolyMarket(slug);
    }
  }catch(e){
    console.error('polySubmitManualResolve error', e);
  }finally{
    if(btn){
      btn.disabled = false;
      btn.textContent = prevText;
    }
  }
}

async function polyResolveMarket(slug, ev){
  if(ev){
    ev.preventDefault();
    ev.stopPropagation();
  }
  if(!slug) return;
  const popup = polyEnsureManualResolvePopup();
  polyManualResolveCtx.slug = slug;
  polyManualResolveCtx.btn = ev && ev.currentTarget ? ev.currentTarget : null;
  polyManualResolveCtx.prevText = polyManualResolveCtx.btn ? polyManualResolveCtx.btn.textContent : '↻';
  const slugEl = document.getElementById('poly-manual-resolve-slug');
  if(slugEl) slugEl.textContent = slug;
  
  // Position popup near the clicked button
  if(polyManualResolveCtx.btn){
    const rect = polyManualResolveCtx.btn.getBoundingClientRect();
    popup.style.left = (rect.left + window.scrollX) + 'px';
    popup.style.top = (rect.bottom + window.scrollY + 2) + 'px';
  }
  
  popup.classList.remove('hidden');
}

let liveMarketPollInterval = null;
let polyAllowAutoSelectLiveMarket = true;
let polyMarketsRefreshTimer = null;

function polyScheduleMarketsRefresh(reason){
  try{
    if(polyMarketsRefreshTimer) clearTimeout(polyMarketsRefreshTimer);
    polyMarketsRefreshTimer = setTimeout(() => {
      try{
        console.log('[markets] auto refresh', {reason, active_ts: polyActiveTs, prev_active_ts: polyLastActiveTs});
        loadPolyMarkets();
      }catch(e){}
    }, 350);
  }catch(e){}
}

function startLiveMarketPoll(){
  stopLiveMarketPoll();
  liveMarketPollInterval = setInterval(async () => {
    try{
      const st = await fetch(API+'/api/poly/status');
      const s = await st.json();
      polyActiveTs = s.active_ts || null;
      // If active market changes, the previous one just ended -> trigger autopredict.
      // This is what makes the UI "know" that a new future market prediction should be run.
      if(polyLastActiveTs !== null && polyActiveTs !== null && polyActiveTs !== polyLastActiveTs){
        // Autopredict disabled on timers; keeping only list refresh.
        // try{ await polyAutopredictTrigger(polyLastActiveTs); }catch(e){}
        // Always refresh markets list so badges/icons update even if user is on another market.
        polyScheduleMarketsRefresh('active_ts_changed');
      }
      polyLastActiveTs = polyActiveTs;
      if(polyActiveTs === null) return;

      // Fetch first page to likely include the active market (sorted DESC)
      const res = await fetch(API+`/api/poly/markets?limit=${POLY_MARKETS_PER_PAGE}&offset=0`);
      const data = await res.json();
      if(!Array.isArray(data) || !data.length) return;

      if(polyMarketsPage === 1){
        polyMarketsCache = data;
        renderPolyMarkets();
      }

      const liveMarket = data.find(m => polyActiveTs !== null && (m.ts||0) === polyActiveTs && !m.closed);
      if(liveMarket){
        // Auto-select active market only if user hasn't manually selected a market.
        if(polyAllowAutoSelectLiveMarket && !polySelectedMarketSlug){
          stopLiveMarketPoll();
          await selectPolyMarket(liveMarket.slug);
        }
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
  await polyLoadLiveTradeSettings();
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
    const runs = Array.isArray(data?.results) ? data.results : [];
    const up = runs.filter(r => r.result?.prediction === 'UP').length;
    const dn = runs.filter(r => r.result?.prediction === 'DOWN').length;
    const unk = Math.max(0, runs.length - up - dn);
    console.log('[autopredict] response', {slug, batch_id: data?.batch_id || null, runs: runs.length, up, down: dn, undefined: unk, marketTs});

    if(data){
      // Auto-trade (ONLY for autopredict runs)
      try{
        // Only for FUTURE markets
        const nowUtc = Math.floor(Date.now() / 1000);
        const mTs = (typeof marketTs === 'number') ? marketTs : (data?.market_ts ? Number(data.market_ts) : null);
        const isFuture = !!(mTs && nowUtc < mTs);
        if(!isFuture){
          console.log('[auto-trade] skip: not future', {slug, nowUtc, mTs});
          throw new Error('not future');
        }

        const s = polyGetLiveTradeSettings();
        if(!s.auto_place){
          console.log('[auto-trade] skip: auto_place=false', {slug});
          throw new Error('auto_place=false');
        }
        if(runs.length){
          const summary = up > dn ? 'UP' : (dn > up ? 'DOWN' : null);
          if(!summary){
            console.log('[auto-trade] skip: no clear outcome (tie or empty)', {slug, up, dn, total: runs.length});
            throw new Error('no clear outcome');
          }
          const bid = data?.batch_id || null;
          if(summary && (!bid || polyLastLiveOrderPlacedForBatchId !== bid)){
            polyLastLiveOrderPlacedForBatchId = bid;
            // Ensure selected market context (for outcome mapping)
            try{ if(!polySelectedMarketSlug || polySelectedMarketSlug !== slug){ await selectPolyMarket(slug); } }catch(e){}
            console.log('[auto-trade] open confirm modal', {slug, summary, bid});
            await polyPlaceLiveOrderAfterPrediction(slug, summary, bid);
          } else {
            console.log('[auto-trade] skip: duplicate batch_id', {slug, summary, bid});
          }
        }
      }catch(e){/* ignore */}

      // Update markets cache so list re-renders with new P badge state
      if(Array.isArray(polyMarketsCache)){
        const mm = polyMarketsCache.find(x => x && x.slug === slug);
        if(mm){
          const defined = (up + dn) > 0;
          mm.has_pred = runs.length > 0;
          mm.has_pred_defined = defined;
          mm.pred_votes = {up, down: dn, unk, ts: Math.floor(Date.now()/1000)};
        }
      }
      renderPolyMarkets();
    }
    console.log(`[autopredict] ${slug} done`, runs.length, 'runs');
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
  const titleEl = document.getElementById('poly-market-title');
  if(titleEl) titleEl.textContent='';
  const emptyEl = document.getElementById('poly-market-empty');
  const secEl = document.getElementById('poly-market-sections');
  if(emptyEl) emptyEl.classList.remove('hidden');
  if(secEl) secEl.classList.add('hidden');
  const detailEl = document.getElementById('poly-market-detail');
  if(detailEl) detailEl.innerHTML='';
  const obUp = document.getElementById('poly-orderbook-up');
  const obDown = document.getElementById('poly-orderbook-down');
  const obStatus = document.getElementById('poly-ob-status');
  if(obUp) obUp.innerHTML='<span class="text-slate-400">Select a market.</span>';
  if(obDown) obDown.innerHTML='<span class="text-slate-400">Select a market.</span>';
  if(obStatus) obStatus.textContent='';
  const simMsg = document.getElementById('poly-sim-msg');
  if(simMsg) simMsg.textContent='';
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
  const titleEl = document.getElementById('poly-market-title');
  if(titleEl) titleEl.textContent='';
  const emptyEl = document.getElementById('poly-market-empty');
  const secEl = document.getElementById('poly-market-sections');
  if(emptyEl) emptyEl.classList.remove('hidden');
  if(secEl) secEl.classList.add('hidden');
  const detailEl = document.getElementById('poly-market-detail');
  if(detailEl) detailEl.innerHTML='';
  const obUp = document.getElementById('poly-orderbook-up');
  const obDown = document.getElementById('poly-orderbook-down');
  const obStatus = document.getElementById('poly-ob-status');
  if(obUp) obUp.innerHTML='<span class="text-slate-400">Select a market.</span>';
  if(obDown) obDown.innerHTML='<span class="text-slate-400">Select a market.</span>';
  if(obStatus) obStatus.textContent='';
  const simMsg = document.getElementById('poly-sim-msg');
  if(simMsg) simMsg.textContent='';
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
    // Trigger autopredict when the active market ends (so UI can predict upcoming future markets)
    // Autopredict disabled on timers.
    // try{
    //   const endedTs = Number(marketTs) || null;
    //   if(endedTs && endedTs !== polyAutopredictLastTriggeredForEndedTs){
    //     polyAutopredictTrigger(endedTs);
    //   }
    // }catch(e){}
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
  if(!polyAutopredictStateLoaded){
    polyAutopredictStateLoaded = true;
    loadAutopredictState();
  }
  polyStartPredUpdatesPoll();
  try{
    try{
      const st=await fetch(API+'/api/poly/status');
      const s=await st.json();
      polyActiveTs = s.active_ts||null;
      polyEmulate = (s.emulate || 'NONE').toUpperCase();
      polyNeedConfirmation = (s.need_confirmation === undefined || s.need_confirmation === null) ? true : !!s.need_confirmation;
      if(polyLastActiveTs !== null && polyActiveTs !== null && polyActiveTs !== polyLastActiveTs){
        // Autopredict disabled on timers.
        // try{ await polyAutopredictTrigger(polyLastActiveTs); }catch(e){}
      }
      polyLastActiveTs = polyActiveTs;
    }catch(e){polyActiveTs=null;}

    const prevCache = Array.isArray(polyMarketsCache) ? polyMarketsCache : null;
    const prevBySlug = new Map((prevCache || []).filter(x=>x&&x.slug).map(x=>[x.slug, x]));

    const res = await fetch(API+`/api/poly/markets?limit=${POLY_MARKETS_PER_PAGE}&offset=${(Math.max(1, polyMarketsPage)-1)*POLY_MARKETS_PER_PAGE}`);
    const data=await res.json();
    if(!Array.isArray(data)||!data.length){
      polyMarketsReachedLastPage = true;
      polyRenderMarketsPageButtons();
      el.textContent='No markets found';
      return;
    }
    polyMarketsReachedLastPage = data.length < POLY_MARKETS_PER_PAGE;
    polyMarketsKnownMaxPage = Math.max(polyMarketsKnownMaxPage, polyMarketsPage + (polyMarketsReachedLastPage ? 0 : 1));
    polyPersistMarketsPage(polyMarketsPage);
    polyMarketsCache=data;
    polyMarketsWithPosCache = new Set();
    renderPolyMarkets();
    polyRenderMarketsPageButtons();

    // Fallback: if a FUTURE market just gained a defined prediction (UP/DOWN) within the last 60s,
    // show confirmation even if the prediction was produced server-side (not by this UI tab).
    try{
      const s = polyGetLiveTradeSettings();
      const nowUtc = Math.floor(Date.now()/1000);
      if(s.auto_place && Array.isArray(data)){
        for(const m of data){
          if(!m || !m.slug) continue;
          const ts = m.ts ? Number(m.ts) : null;
          if(!ts || !(nowUtc < ts)) continue; // future only

          const pred = String(m.prediction_outcome || '').toUpperCase();
          if(!(pred === 'UP' || pred === 'DOWN')) continue;

          const votesTs = (m.prediction_ts || null);
          const age = votesTs ? (nowUtc - votesTs) : null;
          if(age === null || age < 0 || age > 60) continue; // only within 60s

          const prev = prevBySlug.get(m.slug);
          const prevDefined = !!(prev && (prev.has_pred_defined || ((prev.pred_votes?.up||0) + (prev.pred_votes?.down||0) > 0)));
          const nowDefined = !!(m.has_pred_defined || ((m.pred_votes?.up||0) + (m.pred_votes?.down||0) > 0));
          if(prevDefined || !nowDefined) continue; // only newly gained

          const promptKey = `${m.slug}:${votesTs}:${pred}`;
          if(polyLastConfirmPromptKey === promptKey) continue;
          polyLastConfirmPromptKey = promptKey;

          try{ if(!polySelectedMarketSlug || polySelectedMarketSlug !== m.slug){ await selectPolyMarket(m.slug); } }catch(e){}
          await polyPlaceLiveOrderAfterPrediction(m.slug, pred, null);
          break;
        }
      }
    }catch(e){/* ignore */}

    // Only restore last selected market if none is currently selected
    if(!polySelectedMarketSlug){
      let lastSlug = null;
      try{ lastSlug = localStorage.getItem(POLY_LAST_MARKET_KEY); }catch(e){ lastSlug = null; }
      const hasLast = !!(lastSlug && data.find(m => m && m.slug === lastSlug));
      if(hasLast){
        await selectPolyMarket(lastSlug);
      } else if(lastSlug){
        polySelectedMarketSlug = lastSlug;
        await showPolyMarket(lastSlug);
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
  // User manually selected a market. Keep polling running for list refresh,
  // but disable auto-select of the active market.
  polyAllowAutoSelectLiveMarket = false;
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
  poly4sPredCache = [];
  document.getElementById('poly-orderbook-up').innerHTML='<span class="text-slate-400">Loading...</span>';
  document.getElementById('poly-orderbook-down').innerHTML='<span class="text-slate-400">Loading...</span>';
  const simMsg = document.getElementById('poly-sim-msg');
  if(simMsg) simMsg.textContent='';
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
      loadLivePositions();
      loadLiveOrders();
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

function polySetBuyPrice(){
  // Sim trading removed
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
    polyApplyLiveTradeSettingsToUI();
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
  try{
    const res4 = await fetch(API + '/api/poly/predictions_4s/' + encodeURIComponent(slug));
    const p4 = await res4.json();
    poly4sPredCache = (p4 && !p4.error) ? [p4] : [];
  }catch(e){ poly4sPredCache = []; }
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
    const markers4s = poly4sPredCache.length ? build4sMarkers(poly4sPredCache) : undefined;
    drawCandleChart(canvas, data.candles, data.market_ts, markers, markers4s);

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
  // Build a map: candle_ts → aggregated counts + raw runs for hover tooltips
  const map = {};
  runs.forEach(r => {
    const ts = r.market_ts || (polySelectedMarket?.ts);
    if(!ts) return;
    if(!map[ts]) map[ts] = {up:0, down:0, unk:0, ts: ts, runs: []};
    if(r.prediction === 'UP') map[ts].up++;
    else if(r.prediction === 'DOWN') map[ts].down++;
    else map[ts].unk++;
    map[ts].runs.push(r);
  });
  return Object.values(map);
}

function build4sMarkers(preds4s){
  // Plot 4s markers strictly at prediction completion time (prediction_ts)
  return preds4s
    .filter(p => p && p.prediction && p.prediction_ts)
    .map(p => ({
      ts: Number(p.prediction_ts),
      market_ts: p.market_ts ? Number(p.market_ts) : null,
      signal: String(p.prediction).toUpperCase(),
      prob: typeof p.probability === 'number' ? p.probability : null,
      payload: p,
    }));
}

function renderPredHistory(runs){
  // Group by batch_id preserving order
  const batches = [];
  const batchMap = {};
  runs.forEach(r => {
    if(!batchMap[r.batch_id]){
      batchMap[r.batch_id] = {batch_id: r.batch_id, started_at: r.started_at, rows: []};
      batches.push(batchMap[r.batch_id]);
    }
    batchMap[r.batch_id].rows.push(r);
  });

  let html = '<div style="display:flex;flex-direction:column;gap:8px">';
  batches.forEach(b => {
    const dt = b.started_at ? b.started_at.replace('T',' ').substring(0,19) : '—';
    let voteSummary = '';
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

    html += `<details style="background:#1e293b;border:1px solid #334155;border-radius:6px;padding:6px 8px">`;
    html += `<summary class="cursor-pointer" style="list-style:none;display:flex;align-items:center;justify-content:space-between">`;
    html += `<span><span class="text-slate-400">${dt}</span>${voteSummary}</span>`;
    html += `<span class="text-slate-600 text-xs">${b.rows.length} row(s)</span>`;
    html += `</summary>`;

    html += `<div style="margin-top:6px;display:flex;flex-direction:column;gap:4px">`;
    b.rows.forEach(r => {
      const predColor = r.prediction==='UP' ? '#22c55e' : (r.prediction==='DOWN' ? '#ef4444' : '#94a3b8');
      const predArrow = r.prediction==='UP' ? '▲' : (r.prediction==='DOWN' ? '▼' : '—');
      const prob = r.probability !== null ? Math.round(r.probability*100)+'%' : '';
      const dur = r.duration_ms !== null ? `${r.duration_ms}ms` : '';
      const scBadge = '';
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
  // Clear bet size confirmation status when the Live Trade Settings panel is hidden
  polySetBetSizePendingState(null);
  polyStopBetSizeRequestPoll();
  polyRenderBetSizePendingMessage(null);
}

function polyShowUndefinedReason(){
  const reason = polyCurrentUndefinedReason;
  if(!reason){
    alert('No undefined reason details available.');
    return;
  }
  // Build detailed reasons list
  const reasonLabels = {
    volatility_filter: 'Volatility filter skipped this candle',
    rsi_neutral: 'RSI is within the neutral band',
    bb_not_low_enough: 'Bollinger Band confirmation not met (price not low enough)',
    bb_not_high_enough: 'Bollinger Band confirmation not met (price not high enough)',
    probability_threshold: 'Probability did not exceed decision threshold',
  };
  const code = reason.reason || 'unknown';
  const title = reasonLabels[code] || code.replace(/_/g, ' ');

  let checksHtml = '';
  if(Array.isArray(reason.checks) && reason.checks.length){
    checksHtml = `<table style="width:100%;border-collapse:collapse;margin-top:12px;font-size:12px">
      <thead>
        <tr style="border-bottom:1px solid #475569;color:#94a3b8;text-align:left">
          <th style="padding:6px">Check</th>
          <th style="padding:6px">Condition</th>
          <th style="padding:6px">Value</th>
          <th style="padding:6px">Status</th>
        </tr>
      </thead>
      <tbody>`;
    reason.checks.forEach(c => {
      const passed = !!c.passed;
      const statusColor = passed ? '#22c55e' : '#ef4444';
      const statusText = passed ? 'PASS' : 'FAIL';
      checksHtml += `<tr style="border-bottom:1px solid #334155;color:${passed ? '#e2e8f0' : '#fecaca'}">
        <td style="padding:6px">${c.name || '-'}</td>
        <td style="padding:6px;color:#94a3b8">${c.condition || '-'}</td>
        <td style="padding:6px">${c.value !== undefined ? c.value : '-'}</td>
        <td style="padding:6px;font-weight:700;color:${statusColor}">${statusText}</td>
      </tr>`;
    });
    checksHtml += `</tbody></table>`;
  }

  const messageHtml = reason.message ? `<div style="margin-top:10px;color:#fbbf24;font-size:13px">${reason.message}</div>` : '';
  const probHtml = reason.probability !== undefined ? `<div style="margin-top:8px;color:#94a3b8;font-size:12px">Probability: <b>${(reason.probability * 100).toFixed(2)}%</b></div>` : '';
  const thresholdHtml = reason.threshold !== undefined ? `<div style="color:#94a3b8;font-size:12px">Threshold: <b>${reason.threshold}</b></div>` : '';

  const html = `<div style="max-width:500px">
    <div style="font-size:16px;font-weight:700;color:#fbbf24;margin-bottom:8px">${title}</div>
    ${messageHtml}
    ${probHtml}
    ${thresholdHtml}
    ${checksHtml}
  </div>`;

  // Create or reuse modal
  let modal = document.getElementById('poly-undefined-reason-modal');
  if(!modal){
    modal = document.createElement('div');
    modal.id = 'poly-undefined-reason-modal';
    modal.style.cssText = 'position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,0.7);display:flex;align-items:center;justify-content:center;z-index:9999';
    document.body.appendChild(modal);
  }
  modal.innerHTML = `<div style="background:#0f172a;border:1px solid #334155;border-radius:8px;padding:20px;max-width:90vw;max-height:80vh;overflow:auto">
    ${html}
    <div style="margin-top:16px;text-align:right">
      <button onclick="document.getElementById('poly-undefined-reason-modal').style.display='none'" class="btn btn-slate">Close</button>
    </div>
  </div>`;
  modal.style.display = 'flex';
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
        +`</div>`
        +`<div id="poly-pred-chart-wrap" class="mt-3"></div>`;
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
    // Debug: log what we received
    console.log('[polyLoadPredictionDetails] pred:', pred, 'details_more:', data.details_more, 'diag:', data.diag);

    const undefinedReason = (pred === 'UNDEFINED')
      ? (data.details_more || (data.diag && data.diag.undefined_reason) || null)
      : null;
    // Store for Why? button
    polyCurrentUndefinedReason = undefinedReason;

    // Build shift note if present
    const shiftNoteHtml = data.shift_note
      ? `<div class="mt-2 p-2 rounded" style="background:#422006;color:#fbbf24;border:1px solid #854d0e;font-size:11px">${data.shift_note}</div>`
      : '';

    const renderUndefinedReasonBlock = (reason) => {
      if(!reason) return '';
      const reasonLabels = {
        volatility_filter: 'Volatility filter skipped candle',
        rsi_neutral: 'RSI within neutral band',
        bb_not_low_enough: 'BB confirmation not met (low)',
        bb_not_high_enough: 'BB confirmation not met (high)',
        probability_threshold: 'Probability below decision threshold',
      };
      const code = (reason.reason || 'unknown');
      const labelText = reasonLabels[code] || code.replace(/_/g,' ');
      const subLines = [];
      if(reason.message) subLines.push(reason.message);
      if(reason.candidate) subLines.push(`Candidate: ${reason.candidate}`);
      if(reason.rsi !== undefined) subLines.push(`RSI: ${reason.rsi}`);
      if(reason.bb_value !== undefined) subLines.push(`BB: ${reason.bb_value}`);
      if(reason.threshold !== undefined) subLines.push(`Threshold: ${reason.threshold}`);
      if(Array.isArray(reason.effective_band)){
        subLines.push(`Effective band: ${reason.effective_band[0]} - ${reason.effective_band[1]}`);
      }
      return `<div class="mt-3 p-3 rounded bg-slate-900/70 border border-amber-500/40">`
        + `<div class="text-xs text-amber-300 font-semibold uppercase tracking-wide">Undefined reason: ${labelText}</div>`
        + (subLines.length ? `<div class="text-xs text-slate-200 mt-1">${subLines.join('<br>')}</div>` : '')
        + `</div>`;
    };
    let undefinedReasonHtml = '';
    if(pred === 'UNDEFINED'){
      if(undefinedReason){
        undefinedReasonHtml = renderUndefinedReasonBlock(undefinedReason);
      } else {
        // Fallback when no reason details available
        undefinedReasonHtml = `<div class="mt-3 p-3 rounded bg-slate-900/70 border border-amber-500/40">`
          + `<div class="text-xs text-amber-300 font-semibold uppercase tracking-wide">Undefined reason: (not recorded)</div>`
          + `<div class="text-xs text-slate-200 mt-1">Probability: ${prob !== null ? prob + '%' : 'N/A'} | Threshold: ${data.params && data.params.threshold ? data.params.threshold : 'default'}</div>`
          + `</div>`;
      }
    }
    // Build Why? button for UNDEFINED predictions
    const whyBtnHtml = (pred === 'UNDEFINED' && undefinedReason)
      ? `<button onclick="polyShowUndefinedReason()" class="btn btn-amber text-xs" style="margin-left:6px">Why?</button>`
      : '';

    inlineEl.innerHTML=
      `<span style="color:${color};font-weight:700;font-size:14px">${arrow} ${pred}</span> `
      + (prob !== null ? `<span class="text-slate-400 text-xs">${prob}%</span>` : '')
      + whyBtnHtml
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
      if(undefinedReason){
        diagHtml += renderUndefinedReasonBlock(undefinedReason);
      }
      if(d.tail_detail && d.tail_detail.length){
        diagHtml += `<table class="mt-2 w-full text-xs"><thead><tr><th>Time</th><th>RSI</th><th>BB</th><th>Prob</th><th>Signal</th></tr></thead><tbody>`;
        d.tail_detail.forEach(r => {
          const sigLabel = r.pred === 1 ? '<span style="color:#22c55e;font-weight:700">UP</span>' : (r.pred === 0 ? '<span style="color:#ef4444;font-weight:700">DOWN</span>' : '<span style="color:#64748b">—</span>');
          const rsiColor = r.rsi < d.effective_oversold ? '#22c55e' : (r.rsi > d.effective_overbought ? '#ef4444' : '#94a3b8');
          diagHtml += `<tr><td>${r.dt}</td><td style="color:${rsiColor};font-weight:600">${r.rsi}</td><td>${r.bb}</td><td>${r.prob}</td><td>${sigLabel}</td></tr>`;
        });
        diagHtml += `</tbody></table>`;
      }
      if(Array.isArray(d.checks) && d.checks.length){
        diagHtml += `<div class="mt-3">`
          + `<div class="text-xs text-slate-300 font-semibold mb-1">Rule checks</div>`
          + `<table class="w-full text-[11px]">
              <thead>
                <tr class="text-slate-500"><th class="text-left">Check</th><th class="text-left">Condition</th><th>Value</th><th>Expected</th><th>Status</th></tr>
              </thead>
              <tbody>`;
        d.checks.forEach((c, idx) => {
          const passed = !!c.passed;
          const rowColor = passed ? '#22c55e' : '#ef4444';
          const status = passed ? 'PASS' : 'FAIL';
          const expected = c.expected !== undefined ? c.expected : '';
          diagHtml += `<tr style="color:${passed ? '#cbd5f5' : '#fecaca'}">
            <td class="py-0.5">${c.name || `check-${idx+1}`}</td>
            <td class="py-0.5 text-slate-300">${c.condition || '—'}</td>
            <td class="py-0.5 text-center">${c.value !== undefined ? c.value : '—'}</td>
            <td class="py-0.5 text-center">${expected}</td>
            <td class="py-0.5 text-center"><span style="color:${rowColor};font-weight:700">${status}</span></td>
          </tr>`;
        });
        diagHtml += `</tbody></table></div>`;
      }
      diagHtml += `</div>`;
    }

    const ws = data.window_size || 1000;
    resultEl.innerHTML=`<div class="p-4 rounded-lg text-center" style="background:#1e293b;border:2px solid ${color}">`
      +`<div style="font-size:48px;font-weight:800;color:${color}">${arrow} ${pred}</div>`
      + (prob !== null ? `<div class="text-lg text-slate-300 mt-1">Probability: <b>${prob}%</b></div>` : '')
      + undefinedReasonHtml
      + shiftNoteHtml
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

    // Cache best ask cents for snapshot-price usage (auto-trade path may not click a row)
    try{
      const upAsk = (upData && upData.best_ask_cents !== undefined && upData.best_ask_cents !== null) ? Number(upData.best_ask_cents) : null;
      const dnAsk = (downData && downData.best_ask_cents !== undefined && downData.best_ask_cents !== null) ? Number(downData.best_ask_cents) : null;
      polyLastBestAskCents.UP = (upAsk !== null && Number.isFinite(upAsk)) ? upAsk : null;
      polyLastBestAskCents.DOWN = (dnAsk !== null && Number.isFinite(dnAsk)) ? dnAsk : null;
    }catch(e){}

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

async function runPolyBatchPredict(){
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
  const resultEl = document.getElementById('poly-pred-result');
  const inlineEl = document.getElementById('poly-pred-result-inline');
  const analyseSection = document.getElementById('poly-pred-analyse-section');

  if(btnR) btnR.disabled = true;
  const activeBtn = btnR;
  const origHtml = activeBtn ? activeBtn.innerHTML : '';
  if(activeBtn) activeBtn.innerHTML = '<span style="font-size:14px">⏳</span> PREDICTING...';
  if(inlineEl) inlineEl.innerHTML = `<span class="text-slate-400 text-xs">Running ${activeCount} template(s)...</span>`;
  if(resultEl) resultEl.innerHTML = '<span class="text-slate-400">Running batch prediction...</span>';
  if(analyseSection) analyseSection.classList.remove('hidden');

  // Show popup with results section
  const popup = document.getElementById('poly-pred-popup');
  if(popup) popup.classList.remove('hidden');
  polyApplyLiveTradeSettingsToUI();

  polyStartCandleSyncPoll();

  try{
    const res = await fetch(API + '/api/poly/batch_predict', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({
        slug: polySelectedMarket.slug,
        table: 'c_5m'
      })
    });
    const data = await res.json();
    polyLastPredBatchId = data?.batch_id || null;
    polyStopCandleSyncPoll();

    if(data.error){
      if(resultEl) resultEl.innerHTML = `<div class="p-3 rounded-lg" style="background:#7f1d1d;border:1px solid #ef4444"><span class="text-red-300 font-semibold">Error:</span> <span class="text-red-200">${data.error}</span></div>`;
      if(inlineEl) inlineEl.innerHTML = `<span class="text-red-400 text-xs">${data.error}</span>`;
    } else if(data.results && data.results.length){
      renderRegularResults(data.results, inlineEl, resultEl, polyLastPredBatchId);
    } else {
      if(resultEl) resultEl.innerHTML = '<span class="text-slate-400">No results returned.</span>';
    }
  }catch(e){
    polyStopCandleSyncPoll();
    if(resultEl) resultEl.innerHTML = `<span class="text-red-400">Request failed: ${e.message}</span>`;
    if(inlineEl) inlineEl.innerHTML = `<span class="text-red-400 text-xs">${e.message || 'Request failed'}</span>`;
  }

  if(btnR){ btnR.disabled = false; btnR.innerHTML = '<span style="font-size:14px">✨</span> PREDICT'; }
}

function renderRegularResults(results, inlineEl, resultEl, batchId){
  // Inline summary: count UP vs DOWN
  let upCount = 0, downCount = 0, errCount = 0;
  let firstUndefinedReason = null;
  results.forEach(r => {
    const pred = r.result?.prediction;
    if(pred === 'UP') upCount++;
    else if(pred === 'DOWN') downCount++;
    else {
      errCount++;
      // Capture first undefined reason for Why? button
      if(!firstUndefinedReason){
        const reason = r.result?.details_more || (r.result?.diag && r.result?.diag.undefined_reason) || null;
        if(reason) firstUndefinedReason = reason;
      }
    }
  });
  // Store for Why? button
  polyCurrentUndefinedReason = firstUndefinedReason;

  const summaryColor = upCount > downCount ? '#22c55e' : (downCount > upCount ? '#ef4444' : '#94a3b8');
  const summaryArrow = upCount > downCount ? '\u25b2' : (downCount > upCount ? '\u25bc' : '\u2014');
  const summaryLabel = upCount > downCount ? 'UP' : (downCount > upCount ? 'DOWN' : 'MIXED');

  // Build Why? button if any undefined predictions exist
  const whyBtnHtml = (errCount > 0 && firstUndefinedReason)
    ? `<button onclick="polyShowUndefinedReason()" class="btn btn-amber text-xs" style="margin-left:6px">Why?</button>`
    : '';

  if(inlineEl) inlineEl.innerHTML =
    `<span style="color:${summaryColor};font-weight:700;font-size:14px">${summaryArrow} ${summaryLabel}</span> `
    + `<span class="text-slate-400 text-xs">(${upCount}UP/${downCount}DN${errCount?' +'+errCount+'err':''})</span>`
    + whyBtnHtml
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

  // Manual predictions (button click) MUST NOT place live orders.
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
    wrap.innerHTML=`<div class="text-xs text-slate-500 mb-1">${data.candles.length} candles shown (last ${tail} of ${windowSize} window)</div>`
      +`<div id="poly-pred-scroll" style="overflow-x:auto"><canvas id="poly-pred-canvas" height="320"></canvas></div>`;
    const canvas = document.getElementById('poly-pred-canvas');
    const markers = polyPredRunsCache.length ? buildPredMarkers(polyPredRunsCache, data.candles) : undefined;
    const markers4s = poly4sPredCache.length ? build4sMarkers(poly4sPredCache) : undefined;
    drawCandleChart(canvas, data.candles, data.market_ts, markers, markers4s);
    // Scroll to market candle if present
    try{
      const sc = document.getElementById('poly-pred-scroll');
      const x = canvas?.dataset?.marketX ? Number(canvas.dataset.marketX) : null;
      if(sc && x != null && Number.isFinite(x)){
        sc.scrollLeft = Math.max(0, x - (sc.clientWidth / 2));
      }
    }catch(e){/* ignore */}
  }catch(e){
    wrap.innerHTML=`<span class="text-red-400 text-xs">Error: ${e.message}</span>`;
  }
}

function drawCandleChart(canvas, candles, marketTs, markers, markers4s){
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
  const candleCenterX = i => candleX(i) + candleW / 2;

  // Build ts→index map for fast marker lookup
  const tsIdx = {};
  candles.forEach((c, i) => { tsIdx[c.t] = i; });
  const intervalSec = (() => {
    if(n >= 2){
      const d = Number(candles[1].t) - Number(candles[0].t);
      if(Number.isFinite(d) && d > 0) return d;
    }
    return 300;
  })();
  const markerXForTs = (ts) => {
    const t = Number(ts);
    if(!Number.isFinite(t)) return null;
    const exactIdx = tsIdx[t];
    if(exactIdx !== undefined) return candleCenterX(exactIdx);
    if(t < Number(candles[0].t) || t > Number(candles[n - 1].t) + intervalSec) return null;
    for(let i = 0; i < n; i++){
      const start = Number(candles[i].t);
      const end = start + intervalSec;
      if(t >= start && t <= end){
        const frac = Math.max(0, Math.min(1, (t - start) / intervalSec));
        return candleCenterX(i) + frac * step;
      }
    }
    return null;
  };

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

  const markerHits = [];
  const marker4sHits = [];

  // --- Prediction markers (triangles, normal candle-close predictions) ---
  if(markers && markers.length){
    markers.forEach(m => {
      const idx = tsIdx[m.ts];
      if(idx === undefined) return;
      const cx = candleCenterX(idx);
      const c = candles[idx];
      const yBot = priceY(c.l) + 1;  // below the candle low
      const markerRadius = 0.8;
      const triangleHalfWidth = 1.3;
      const triangleHeight = 1.8;

      // Determine dominant direction
      const total = m.up + m.down + m.unk;
      let icon, iconColor;
      if(m.up > m.down && m.up > m.unk){
        icon = '\u25b2'; iconColor = '#22c55e'; // green triangle up
      } else if(m.down > m.up && m.down > m.unk){
        icon = '\u25bc'; iconColor = '#ef4444'; // red triangle down
      } else {
        icon = '?'; iconColor = '#f59e0b'; // yellow question mark
      }

      // Draw small colored circle background
      ctx.fillStyle = iconColor;
      ctx.globalAlpha = 0.18;
      ctx.beginPath();
      ctx.arc(cx, yBot + markerRadius, markerRadius, 0, Math.PI * 2);
      ctx.fill();
      ctx.globalAlpha = 1.0;

      // Draw minimalist icon (custom triangles for up/down)
      ctx.fillStyle = iconColor;
      ctx.strokeStyle = iconColor;
      if(icon === '\u25b2'){ // up triangle
        ctx.beginPath();
        ctx.moveTo(cx, yBot - triangleHeight);
        ctx.lineTo(cx - triangleHalfWidth, yBot + triangleHalfWidth);
        ctx.lineTo(cx + triangleHalfWidth, yBot + triangleHalfWidth);
        ctx.closePath();
        ctx.fill();
      }else if(icon === '\u25bc'){ // down triangle
        ctx.beginPath();
        ctx.moveTo(cx, yBot + triangleHeight);
        ctx.lineTo(cx - triangleHalfWidth, yBot - triangleHalfWidth);
        ctx.lineTo(cx + triangleHalfWidth, yBot - triangleHalfWidth);
        ctx.closePath();
        ctx.fill();
      }else{
        ctx.font = '2.2px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(icon, cx, yBot);
      }

      // Draw count label below icon
      if(total > 0){
        ctx.font = '2px sans-serif';
        ctx.fillStyle = '#94a3b8';
        ctx.textBaseline = 'top';
        ctx.fillText(String(total), cx, yBot + 3);
      }
      ctx.textBaseline = 'alphabetic';

      markerHits.push({
        cx,
        cy: yBot,
        r: 10,
        color: iconColor,
        data: m,
      });
    });
  }

  // --- 4s-early prediction markers (diamonds, drawn above candle top) ---
  if(markers4s && markers4s.length){
    markers4s.forEach(m => {
      const cx = markerXForTs(m.ts);
      if(cx == null) return;
      let idx = tsIdx[m.ts];
      if(idx === undefined){
        for(let i = 0; i < n; i++){
          const start = Number(candles[i].t);
          const end = start + intervalSec;
          if(Number(m.ts) >= start && Number(m.ts) <= end){
            idx = i;
            break;
          }
        }
      }
      if(idx === undefined) return;
      const c = candles[idx];
      // Place diamond above the candle high, larger offset to avoid overlapping normal marker
      const yTop = priceY(c.h) - 12;
      const hw = 2.5; // half-width of diamond
      const hh = 3.5; // half-height of diamond

      let iconColor;
      if(m.signal === 'UP') iconColor = '#4ade80';        // lighter green
      else if(m.signal === 'DOWN') iconColor = '#f87171'; // lighter red
      else iconColor = '#fbbf24';                         // amber for UNDEFINED

      // Glow halo
      ctx.save();
      ctx.fillStyle = iconColor;
      ctx.globalAlpha = 0.20;
      ctx.beginPath();
      ctx.arc(cx, yTop, hw + 1.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.globalAlpha = 1.0;

      // Diamond shape (rotated square)
      ctx.fillStyle = iconColor;
      ctx.strokeStyle = '#0f172a';
      ctx.lineWidth = 0.5;
      ctx.beginPath();
      ctx.moveTo(cx,        yTop - hh); // top
      ctx.lineTo(cx + hw,   yTop);      // right
      ctx.lineTo(cx,        yTop + hh); // bottom
      ctx.lineTo(cx - hw,   yTop);      // left
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
      ctx.restore();

      // Small '4s' label above the diamond
      ctx.save();
      ctx.fillStyle = iconColor;
      ctx.font = 'bold 2.4px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'bottom';
      ctx.fillText('4s', cx, yTop - hh - 0.5);
      ctx.restore();

      marker4sHits.push({
        cx,
        cy: yTop,
        r: Math.max(hw + 6, 10),
        color: iconColor,
        payload: m.payload || {},
        signal: m.signal,
        prob: m.prob,
      });
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

  canvas._polyHoverMeta = {
    markerHits,
    marker4sHits,
  };
  bindPolyChartHover(canvas);
}

function polyGetChartTooltip(){
  let tip = document.getElementById('poly-chart-tooltip');
  if(!tip){
    tip = document.createElement('div');
    tip.id = 'poly-chart-tooltip';
    tip.style.position = 'fixed';
    tip.style.background = '#1e293b';
    tip.style.border = '1px solid #475569';
    tip.style.borderRadius = '6px';
    tip.style.padding = '8px 12px';
    tip.style.fontSize = '11px';
    tip.style.color = '#e2e8f0';
    tip.style.pointerEvents = 'none';
    tip.style.zIndex = '100000';
    tip.style.maxWidth = '320px';
    tip.style.display = 'none';
    document.body.appendChild(tip);
  }
  return tip;
}

function bindPolyChartHover(canvas){
  if(!canvas) return;
  if(!canvas._polyHoverHandler){
    canvas._polyHoverHandler = (evt) => polyChartHover(evt, canvas);
    canvas.addEventListener('mousemove', canvas._polyHoverHandler);
  }
  if(!canvas._polyLeaveHandler){
    canvas._polyLeaveHandler = () => polyHideChartTooltip();
    canvas.addEventListener('mouseleave', canvas._polyLeaveHandler);
  }
  const parent = canvas.parentElement;
  if(parent && !parent._polyHoverHandler){
    parent._polyHoverHandler = (evt) => polyChartHover(evt, canvas);
    parent.addEventListener('mousemove', parent._polyHoverHandler);
  }
  if(parent && !parent._polyLeaveHandler){
    parent._polyLeaveHandler = () => polyHideChartTooltip();
    parent.addEventListener('mouseleave', parent._polyLeaveHandler);
  }
}

function polyHideChartTooltip(){
  const tip = polyGetChartTooltip();
  if(tip) tip.style.display = 'none';
}

function polyChartHover(e, canvas){
  const meta = canvas?._polyHoverMeta;
  const tooltip = polyGetChartTooltip();
  if(!meta || !tooltip){
    polyHideChartTooltip();
    return;
  }
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const mx = (e.clientX - rect.left) * scaleX;
  const my = (e.clientY - rect.top) * scaleY;

  const hit4s = (meta.marker4sHits || []).find(m => {
    const dx = mx - m.cx;
    const dy = my - m.cy;
    return Math.sqrt(dx*dx + dy*dy) <= (m.r + 3);
  });
  if(hit4s){
    try{ canvas.style.cursor = 'pointer'; }catch(_){}
    polyRender4sTooltip(hit4s, e, tooltip);
    return;
  }

  const hitPred = (meta.markerHits || []).find(m => {
    const dx = mx - m.cx;
    const dy = my - m.cy;
    return Math.sqrt(dx*dx + dy*dy) <= (m.r + 3);
  });
  if(hitPred){
    try{ canvas.style.cursor = 'pointer'; }catch(_){}
    polyRenderPredTooltip(hitPred, e, tooltip);
    return;
  }

  polyHideChartTooltip();
  try{ canvas.style.cursor = 'default'; }catch(_){}
}

function polyRender4sTooltip(hit, evt, tooltip){
  const payload = hit.payload || {};
  const predTs = payload.prediction_ts ? new Date(payload.prediction_ts * 1000) : null;
  const marketTs = payload.market_ts ? new Date(payload.market_ts * 1000) : null;
  const signalTs = payload.signal_open_time ? new Date(payload.signal_open_time / 1000) : null;
  const prob = Number.isFinite(hit.prob) ? Math.round(hit.prob * 100) + '%' : '—';
  const rsi = payload.rsi !== undefined ? Number(payload.rsi).toFixed(1) : '—';
  const slug = payload.market_slug || payload.slug || payload._requested_slug || '—';
  const dirIcon = hit.signal === 'UP' ? '▲' : (hit.signal === 'DOWN' ? '▼' : '?');
  const dirColor = hit.signal === 'UP' ? '#4ade80' : (hit.signal === 'DOWN' ? '#f87171' : '#fbbf24');

  let html = `<div style="color:${dirColor};font-weight:700;font-size:13px">4s Early Prediction ${dirIcon}</div>`;
  if(predTs){
    html += `<div style="color:#94a3b8;font-size:10px;margin-bottom:4px">Predicted: ${predTs.toUTCString()}</div>`;
  }
  html += `<div style="font-size:11px;margin-bottom:4px">`;
  html += `<span style="color:${dirColor};font-weight:700">${hit.signal || 'UNDEFINED'}</span>`;
  html += ` · Prob: <span style="color:#94a3b8">${prob}</span>`;
  html += ` · RSI: <span style="color:#94a3b8">${rsi}</span>`;
  html += `</div>`;
  html += `<div style="font-size:11px;color:#cbd5e1">Market: <strong>${slug}</strong></div>`;
  if(marketTs){
    html += `<div style="font-size:10px;color:#94a3b8">Target market starts: ${marketTs.toUTCString()}</div>`;
  }
  if(signalTs){
    html += `<div style="font-size:10px;color:#94a3b8">Signal candle: ${signalTs.toUTCString()}</div>`;
  }

  tooltip.innerHTML = html;
  tooltip.style.display = 'block';
  tooltip.style.left = (evt.clientX + 14) + 'px';
  tooltip.style.top = (evt.clientY - 10) + 'px';
  const tr = tooltip.getBoundingClientRect();
  if(tr.right > window.innerWidth) tooltip.style.left = (evt.clientX - tr.width - 10) + 'px';
  if(tr.bottom > window.innerHeight) tooltip.style.top = (evt.clientY - tr.height - 10) + 'px';
}

function polyRenderPredTooltip(hit, evt, tooltip){
  const runs = hit.data?.runs || [];
  if(!runs.length){
    polyHideChartTooltip();
    return;
  }
  const dt = runs[0].started_at ? runs[0].started_at.replace('T',' ').substring(0,19) : '—';
  let html = `<div style="color:${hit.color};font-weight:700;font-size:13px">Prediction Batch</div>`;
  html += `<div style="color:#94a3b8;font-size:10px;margin-bottom:4px">${dt}</div>`;
  html += '<div style="border-top:1px solid #334155;padding-top:4px;display:flex;flex-direction:column;gap:2px">';
  runs.forEach(r => {
    const predColor = r.prediction==='UP' ? '#22c55e' : (r.prediction==='DOWN' ? '#ef4444' : '#f59e0b');
    const predIcon = r.prediction==='UP' ? '▲' : (r.prediction==='DOWN' ? '▼' : '?');
    const prob = r.probability !== null && r.probability !== undefined ? Math.round(r.probability*100)+'%' : '';
    html += `<div style="display:flex;align-items:center;gap:6px;font-size:10px">`;
    html += `<span style="color:#94a3b8;min-width:70px">${r.template_name||'?'}${r.quantum_scenario ? ' <span style=\'color:#8b5cf6\'>['+r.quantum_scenario+']</span>' : ''}</span>`;
    html += `<span style="color:${predColor};font-weight:700">${predIcon} ${r.prediction||'UNDEFINED'}</span>`;
    if(prob) html += `<span style="color:#94a3b8">${prob}</span>`;
    html += `</div>`;
  });
  html += '</div>';

  tooltip.innerHTML = html;
  tooltip.style.display = 'block';
  tooltip.style.left = (evt.clientX + 14) + 'px';
  tooltip.style.top = (evt.clientY - 10) + 'px';
  const tr = tooltip.getBoundingClientRect();
  if(tr.right > window.innerWidth) tooltip.style.left = (evt.clientX - tr.width - 10) + 'px';
  if(tr.bottom > window.innerHeight) tooltip.style.top = (evt.clientY - tr.height - 10) + 'px';
}

// ===== Sim trades =====

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
