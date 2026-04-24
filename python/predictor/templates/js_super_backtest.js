// ==================== SUPER BACKTEST ====================

let sbCurrentRunId = null;
let sbCurrentPredictions = [];

// Initialize Super Backtest tab
function initSuperBacktest() {
  // Load default config template
  const cfgEl = document.getElementById('sb-config');
  if (cfgEl && !cfgEl.value) {
    cfgEl.value = JSON.stringify({
      "strategy": "rsi_mean_reversion",
      "test_start": "2022-01-01",
      "test_end": "2025-12-31",
      "horizons": [2],
      "table": "c_5m",
      "window_size": 100,
      "retrain_every": 500,
      "params": {
        "rsi_period": 8,
        "rsi_oversold": 25,
        "rsi_overbought": 75,
        "bb_period": 20,
        "bb_std": 2,
        "bb_low": 0.05,
        "bb_high": 0.85,
        "min_vol": 0.0005,
        "max_vol": 0.01,
        "vol_ratio_max": 1.5,
        "vol_fast_window": 20,
        "vol_slow_window": 40,
        "vol_spike_multiplier": 1.5,
        "window_size": 100,
        "threshold": 0.54
      }
    }, null, 2);
  }
  loadSuperBacktestList();
}

async function runSuperBacktest() {
  const btn = document.getElementById('btn-sb-run');
  btn.disabled = true;
  btn.textContent = 'Running...';
  
  try {
    // Parse config from JSON textarea
    const raw = document.getElementById('sb-config').value.trim();
    const cfg = JSON.parse(raw);
    
    // Validate required fields
    if (!cfg.strategy || !cfg.test_start || !cfg.test_end) {
      alert('Config must include: strategy, test_start, test_end');
      return;
    }
    
    // Build request for single horizon (Super Backtest runs one horizon at a time)
    const horizon = (cfg.horizons && cfg.horizons[0]) || 1;
    
    // Calculate train period as 2x test period before test_start
    const testStart = new Date(cfg.test_start);
    const testEnd = new Date(cfg.test_end);
    const trainDays = Math.ceil((testEnd - testStart) / (1000 * 60 * 60 * 24)) * 2;
    const trainStart = new Date(testStart);
    trainStart.setDate(trainStart.getDate() - trainDays);
    
    const req = {
      strategy: cfg.strategy,
      params: cfg.params || {},
      train_start: trainStart.toISOString().split('T')[0],
      train_end: cfg.test_start,  // train ends when test starts
      test_start: cfg.test_start,
      test_end: cfg.test_end,
      horizon: horizon,
      table: cfg.table || 'c_5m',
      window_size: cfg.window_size || 100,
      retrain_every: cfg.retrain_every || 500,
    };
    
    const res = await fetch(API + '/api/super_backtest/run', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(req)
    });
    
    const data = await res.json();
    if (data.error) {
      alert('Error: ' + data.error);
    } else {
      alert('Super Backtest queued! Run #' + data.super_run_id + ', Task ID: ' + data.task_id);
    }
  } catch (e) {
    if (e instanceof SyntaxError) {
      alert('Invalid JSON in config: ' + e.message);
    } else {
      alert('Failed to start: ' + e.message);
    }
  } finally {
    btn.disabled = false;
    btn.textContent = 'Run Super Backtest';
  }
}

async function loadSuperBacktestList() {
  const el = document.getElementById('sb-list');
  el.innerHTML = '<p class="text-slate-500">Loading...</p>';
  
  try {
    const res = await fetch(API + '/api/super_backtest/list');
    const runs = await res.json();
    
    if (!Array.isArray(runs) || runs.length === 0) {
      el.innerHTML = '<p class="text-slate-500">No super backtest runs yet.</p>';
      return;
    }
    
    let html = '<table class="text-xs"><thead><tr>';
    html += '<th>ID</th><th>Strategy</th><th>Period</th><th>H</th>';
    html += '<th>Signals</th><th>Accuracy</th><th>HMM</th><th>Date</th><th></th>';
    html += '</tr></thead><tbody>';
    
    runs.forEach(r => {
      const accClass = r.accuracy_pct >= 55 ? 'text-green-400' : (r.accuracy_pct >= 50 ? 'text-yellow-400' : 'text-red-400');
      html += `<tr>`;
      html += `<td class="font-mono">#${r.id}</td>`;
      html += `<td>${r.strategy}</td>`;
      html += `<td>${r.train_start} → ${r.test_end}</td>`;
      html += `<td>H${r.horizon}</td>`;
      html += `<td>${r.signals}</td>`;
      html += `<td class="${accClass} font-bold">${r.accuracy_pct}%</td>`;
      html += `<td>${r.hmm_states || '-'}</td>`;
      html += `<td>${r.created_at?.substring(0, 10) || '-'}</td>`;
      html += `<td>`;
      html += `<button onclick="showSuperDetails(${r.id})" class="btn btn-blue text-xs py-0.5 px-2 mr-1">View</button>`;
      html += `<button onclick="deleteSuperBacktest(${r.id})" class="btn btn-red text-xs py-0.5 px-2" title="Delete run and all data">×</button>`;
      html += `</td>`;
      html += `</tr>`;
    });
    
    html += '</tbody></table>';
    el.innerHTML = html;
  } catch (e) {
    el.innerHTML = '<p class="text-red-400">Failed to load: ' + e.message + '</p>';
  }
}

async function showSuperDetails(runId) {
  sbCurrentRunId = runId;
  const detailsEl = document.getElementById('sb-details');
  const titleEl = document.getElementById('sb-detail-title');
  const statsEl = document.getElementById('sb-detail-stats');
  const hmmEl = document.getElementById('sb-hmm-results');
  
  detailsEl.classList.remove('hidden');
  titleEl.textContent = 'Run #' + runId + ' Details';
  statsEl.innerHTML = '<p class="text-slate-500">Loading...</p>';
  hmmEl.classList.add('hidden');
  
  try {
    // Load run details
    const res = await fetch(API + '/api/super_backtest/' + runId);
    const run = await res.json();
    
    if (run.error) {
      statsEl.innerHTML = '<p class="text-red-400">' + run.error + '</p>';
      return;
    }
    
    // Stats
    const accClass = run.accuracy_pct >= 55 ? 'text-green-400' : (run.accuracy_pct >= 50 ? 'text-yellow-400' : 'text-red-400');
    let html = '<div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">';
    html += `<div class="p-2 rounded bg-slate-900"><div class="text-slate-400 text-xs">Strategy</div><div class="font-semibold">${run.strategy}</div></div>`;
    html += `<div class="p-2 rounded bg-slate-900"><div class="text-slate-400 text-xs">Horizon</div><div class="font-semibold">H${run.horizon}</div></div>`;
    html += `<div class="p-2 rounded bg-slate-900"><div class="text-slate-400 text-xs">Signals</div><div class="font-semibold">${run.signals}</div></div>`;
    html += `<div class="p-2 rounded bg-slate-900"><div class="text-slate-400 text-xs">Accuracy</div><div class="font-semibold ${accClass}">${run.accuracy_pct}%</div></div>`;
    html += `<div class="p-2 rounded bg-slate-900"><div class="text-slate-400 text-xs">Correct</div><div class="font-semibold text-green-400">${run.correct}</div></div>`;
    html += `<div class="p-2 rounded bg-slate-900"><div class="text-slate-400 text-xs">Wrong</div><div class="font-semibold text-red-400">${run.wrong}</div></div>`;
    html += `<div class="p-2 rounded bg-slate-900"><div class="text-slate-400 text-xs">Train</div><div class="font-semibold">${run.train_start} → ${run.train_end}</div></div>`;
    html += `<div class="p-2 rounded bg-slate-900"><div class="text-slate-400 text-xs">Test</div><div class="font-semibold">${run.test_start} → ${run.test_end}</div></div>`;
    html += '</div>';
    
    if (run.hmm_model) {
      html += '<div class="text-purple-300 text-xs mb-2">✓ HMM model fitted with ' + run.hmm_model.n_states + ' states</div>';
      hmmEl.classList.remove('hidden');
      loadHmmResults(runId);
    }
    
    statsEl.innerHTML = html;
    
    // Load predictions
    loadSuperPredictions(runId);
    
  } catch (e) {
    statsEl.innerHTML = '<p class="text-red-400">Failed: ' + e.message + '</p>';
  }
}

async function loadSuperPredictions(runId) {
  const el = document.getElementById('sb-predictions');
  el.innerHTML = '<p class="text-slate-500">Loading predictions...</p>';
  
  try {
    const res = await fetch(API + '/api/super_backtest/' + runId + '/predictions');
    const preds = await res.json();
    sbCurrentPredictions = preds;
    
    if (!Array.isArray(preds) || preds.length === 0) {
      el.innerHTML = '<p class="text-slate-500">No predictions found.</p>';
      return;
    }
    
    let html = '<table class="text-xs"><thead><tr>';
    html += '<th>Idx</th><th>Time</th><th>Pred</th><th>Prob</th><th>Actual</th><th>Result</th>';
    html += '<th>RSI</th><th>BB Pos</th><th>Vol</th><th>Streak</th><th>HMM</th>';
    html += '</tr></thead><tbody>';
    
    // Show first 100
    preds.slice(0, 100).forEach(p => {
      const timeStr = new Date(p.open_time / 1000).toISOString().substring(0, 19).replace('T', ' ');
      const predClass = p.prediction === 1 ? 'text-green-400' : 'text-red-400';
      const resultClass = p.is_correct ? 'text-green-400' : 'text-red-400';
      const resultText = p.is_correct ? '✓' : '✗';
      const hmmState = p.hmm_state !== null ? `<span class="px-1 rounded bg-purple-900">S${p.hmm_state}</span>` : '-';
      
      html += `<tr>`;
      html += `<td class="font-mono">${p.candle_idx}</td>`;
      html += `<td>${timeStr}</td>`;
      html += `<td class="${predClass}">${p.prediction === 1 ? 'UP' : 'DOWN'}</td>`;
      html += `<td>${(p.probability * 100).toFixed(1)}%</td>`;
      html += `<td>${p.actual === 1 ? 'UP' : 'DOWN'}</td>`;
      html += `<td class="${resultClass} font-bold">${resultText}</td>`;
      html += `<td>${p.rsi ? p.rsi.toFixed(1) : '-'}</td>`;
      html += `<td>${p.bb_position ? p.bb_position.toFixed(2) : '-'}</td>`;
      html += `<td>${p.volatility_short ? (p.volatility_short * 100).toFixed(2) + '%' : '-'}</td>`;
      html += `<td>${p.prev_streak_len || '-'}</td>`;
      html += `<td>${hmmState}</td>`;
      html += `</tr>`;
    });
    
    html += '</tbody></table>';
    if (preds.length > 100) {
      html += `<p class="text-slate-500 text-xs mt-2">Showing first 100 of ${preds.length} predictions.</p>`;
    }
    el.innerHTML = html;
    
  } catch (e) {
    el.innerHTML = '<p class="text-red-400">Failed to load predictions: ' + e.message + '</p>';
  }
}

async function analyzeHmmForRun() {
  if (!sbCurrentRunId) return;
  
  const btn = document.getElementById('btn-sb-hmm');
  btn.disabled = true;
  btn.textContent = 'Analyzing...';
  
  try {
    const nStates = parseInt(document.getElementById('sb-hmm-states').value) || 2;
    const usePrev = document.getElementById('sb-use-prev')?.checked ?? true;
    const url = API + '/api/super_backtest/' + sbCurrentRunId + '/analyze_hmm?n_states=' + nStates + '&use_prev_result=' + usePrev;
    const res = await fetch(url, {
      method: 'POST'
    });
    
    const result = await res.json();
    
    if (result.error) {
      alert('HMM Analysis error: ' + result.error);
    } else {
      const featInfo = usePrev ? ' (with prev_result)' : ' (without prev_result)';
      alert('HMM Analysis complete! Detected ' + result.n_states + ' regimes.' + featInfo);
      document.getElementById('sb-hmm-results').classList.remove('hidden');
      displayHmmResults(result);
      loadSuperPredictions(sbCurrentRunId);
    }
  } catch (e) {
    alert('Failed: ' + e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Analyze with HMM';
  }
}

async function analyzeHmmWithThresholds() {
  if (!sbCurrentRunId) {
    alert('Please select a run first by clicking "View" on a backtest run');
    return;
  }
  
  const goodThreshold = parseFloat(document.getElementById('sb-good-threshold').value) || 55;
  const badThreshold = parseFloat(document.getElementById('sb-bad-threshold').value) || 45;
  const filterThreshold = parseFloat(document.getElementById('sb-filter-threshold').value) || 0.6;
  const nStates = parseInt(document.getElementById('sb-hmm-states').value) || 2;
  const usePrev = document.getElementById('sb-use-prev')?.checked ?? true;
  
  const url = API + '/api/super_backtest/' + sbCurrentRunId + '/analyze_hmm' +
    '?n_states=' + nStates +
    '&use_prev_result=' + usePrev +
    '&good_threshold=' + goodThreshold +
    '&bad_threshold=' + badThreshold +
    '&filter_threshold=' + filterThreshold;
  
  try {
    const res = await fetch(url, { method: 'POST' });
    const result = await res.json();
    
    console.log('[analyzeHmmWithThresholds] API response:', result);
    
    if (result.error) {
      alert('HMM Analysis error: ' + result.error);
    } else {
      document.getElementById('sb-details').classList.remove('hidden');
      document.getElementById('sb-hmm-results').classList.remove('hidden');
      displayHmmResults(result);
      alert('HMM Analysis complete! Good≥' + goodThreshold + '%, Bad≤' + badThreshold + '%. Check HMM Results section below.');
    }
  } catch (e) {
    console.error('[analyzeHmmWithThresholds] Error:', e);
    alert('Failed: ' + e.message);
  }
}

function displayHmmResults(result) {
  // Display HMM results with thresholds and monthly analysis
  const thresholdsEl = document.getElementById('sb-hmm-thresholds');
  const statesEl = document.getElementById('sb-hmm-states');
  const transitionEl = document.getElementById('sb-hmm-transition');
  const chartEl = document.getElementById('sb-hmm-chart');
  const monthlyEl = document.getElementById('sb-monthly-analysis');
  const monthlyTbody = document.getElementById('sb-monthly-tbody');
  
  // Show thresholds
  if (result.thresholds) {
    thresholdsEl.innerHTML = `Thresholds: Good ≥${result.thresholds.good_threshold}%, Bad ≤${result.thresholds.bad_threshold}%, Filter P(bad)>${result.thresholds.filter_threshold}`;
  }
  
  // Show state stats
  if (result.states && result.states.length > 0) {
    let html = '<div class="grid grid-cols-1 md:grid-cols-3 gap-3 mb-4">';
    
    result.states.forEach(stateStat => {
      const label = stateStat.label || 'neutral';
      const labelClass = label === 'good' ? 'text-green-400' : (label === 'bad' ? 'text-red-400' : 'text-yellow-400');
      const bgClass = label === 'good' ? 'bg-green-900/30' : (label === 'bad' ? 'bg-red-900/30' : 'bg-yellow-900/30');
      const featMeans = stateStat.feature_means || {};
      
      html += `<div class="p-3 rounded ${bgClass} border border-slate-700">`;
      html += `<div class="font-semibold ${labelClass}">State ${stateStat.state}: ${label.toUpperCase()}</div>`;
      html += `<div class="text-xs text-slate-400">${stateStat.predictions_count} predictions, ${stateStat.accuracy_pct}% accuracy</div>`;
      html += `<div class="text-xs text-slate-500">${stateStat.correct} correct, ${stateStat.wrong} wrong</div>`;
      
      // Feature means
      if (Object.keys(featMeans).length > 0) {
        html += `<div class="mt-2 text-[10px] text-slate-400 border-t border-slate-700 pt-1">`;
        html += `RSI:${featMeans.rsi_14?.toFixed(0)} `;
        html += `ΔRSI:${featMeans.delta_rsi?.toFixed(1)} `;
        html += `EMA:${(featMeans.ema_diff_20 * 100)?.toFixed(2)}% `;
        html += `ATR:${(featMeans.atr_14 * 100)?.toFixed(2)}%`;
        if (featMeans.prev_result !== undefined) {
          html += ` <span class="text-purple-300">Prev:${featMeans.prev_result.toFixed(2)}</span>`;
        }
        html += `</div>`;
      }
      
      // Filter effectiveness for bad state
      if (label === 'bad' && stateStat.trades_would_skip) {
        html += `<div class="mt-1 text-[10px] text-red-300">`;
        html += `Would skip: ${stateStat.trades_would_skip} (${stateStat.skip_accuracy}% correct skips)`;
        html += `</div>`;
      }
      
      html += `</div>`;
    });
    
    html += '</div>';
    statesEl.innerHTML = html;
  }
  
  // Transition matrix
  if (result.transition_matrix) {
    let tmHtml = '<h5 class="text-xs font-semibold mb-2">State Transition Matrix</h5>';
    tmHtml += '<table class="text-xs"><thead><tr><th></th>';
    for (let i = 0; i < result.n_states; i++) {
      tmHtml += `<th>To S${i}</th>`;
    }
    tmHtml += '</tr></thead><tbody>';
    
    result.transition_matrix.forEach((row, fromIdx) => {
      tmHtml += `<tr><td class="font-semibold">From S${fromIdx}</td>`;
      row.forEach(prob => {
        const probPct = (prob * 100).toFixed(1);
        const probClass = prob > 0.5 ? 'font-bold text-purple-300' : '';
        tmHtml += `<td class="${probClass}">${probPct}%</td>`;
      });
      tmHtml += '</tr>';
    });
    
    tmHtml += '</tbody></table>';
    transitionEl.innerHTML = tmHtml;
  }
  
  // Regime Strategy Summary
  if (result.regime_strategy) {
    const rs = result.regime_strategy;
    const improvement = rs.improvement;
    const improvementClass = improvement > 0 ? 'text-green-400' : (improvement < 0 ? 'text-red-400' : 'text-yellow-400');
    const improvementSign = improvement > 0 ? '+' : '';
    
    let rsHtml = '<div class="p-3 bg-slate-800 rounded border border-slate-700 mb-3">';
    rsHtml += '<h5 class="text-xs font-semibold mb-2 text-cyan-300">Regime Filter Strategy Performance</h5>';
    rsHtml += '<div class="grid grid-cols-2 gap-4">';
    
    // Baseline
    rsHtml += '<div class="text-center">';
    rsHtml += '<div class="text-xs text-slate-400">Baseline (All Trades)</div>';
    rsHtml += '<div class="text-lg font-bold">' + rs.baseline_winrate + '%</div>';
    rsHtml += '<div class="text-xs text-slate-500">' + rs.baseline_trades + ' trades</div>';
    rsHtml += '</div>';
    
    // Filtered
    rsHtml += '<div class="text-center border-l border-slate-600">';
    rsHtml += '<div class="text-xs text-cyan-400">With Regime Filter</div>';
    rsHtml += '<div class="text-lg font-bold ' + improvementClass + '">' + (rs.filtered_winrate || 'N/A') + '%</div>';
    rsHtml += '<div class="text-xs text-slate-500">' + rs.filtered_trades + ' trades taken</div>';
    rsHtml += '</div>';
    
    rsHtml += '</div>';
    
    // Stats row
    rsHtml += '<div class="mt-2 pt-2 border-t border-slate-700 flex justify-between text-xs">';
    rsHtml += '<span class="text-slate-400">Skipped: <span class="text-red-400">' + rs.skipped_trades + '</span></span>';
    if (improvement !== null) {
      rsHtml += '<span class="text-slate-400">Improvement: <span class="' + improvementClass + '">' + improvementSign + improvement + '%</span></span>';
    }
    rsHtml += '</div>';
    
    rsHtml += '</div>';
    
    // Insert before monthly analysis
    monthlyEl.insertAdjacentHTML('beforebegin', rsHtml);
  }
  
  // Monthly analysis table
  if (result.monthly_analysis && result.monthly_analysis.length > 0) {
    monthlyEl.classList.remove('hidden');
    let tbodyHtml = '';
    
    result.monthly_analysis.forEach(m => {
      const winRateClass = m.rsi_win_rate >= 55 ? 'text-green-400' : (m.rsi_win_rate >= 50 ? 'text-yellow-400' : 'text-red-400');
      const goodWinRateClass = m.good_regime_win_rate >= 55 ? 'text-green-400' : (m.good_regime_win_rate >= 50 ? 'text-yellow-400' : 'text-red-400');
      
      tbodyHtml += `<tr>`;
      tbodyHtml += `<td class="font-mono">${m.month}</td>`;
      tbodyHtml += `<td>${m.rsi_signals_count}</td>`;
      tbodyHtml += `<td class="${winRateClass} font-bold">${m.rsi_win_rate}%</td>`;
      tbodyHtml += `<td class="text-green-300">${m.good_regime_signals}</td>`;
      tbodyHtml += `<td class="${goodWinRateClass} font-bold">${m.good_regime_win_rate}%</td>`;
      tbodyHtml += `<td>${m.good_regime_pct}%</td>`;
      tbodyHtml += `</tr>`;
    });
    
    monthlyTbody.innerHTML = tbodyHtml;
  } else {
    monthlyEl.classList.add('hidden');
  }
  
  // Chart placeholder
  chartEl.innerHTML = '<p class="text-slate-500 text-xs">Regime chart not available in direct result view. Use "View" button for full chart.</p>';
}

async function loadHmmResults(runId) {
  try {
    // Get run details for HMM model
    const res = await fetch(API + '/api/super_backtest/' + runId);
    const run = await res.json();
    
    // Get regimes
    const regimesRes = await fetch(API + '/api/super_backtest/' + runId + '/regimes');
    const regimes = await regimesRes.json();
    
    const statesEl = document.getElementById('sb-hmm-states');
    const transitionEl = document.getElementById('sb-hmm-transition');
    const chartEl = document.getElementById('sb-hmm-chart');
    
    if (!Array.isArray(regimes) || regimes.length === 0) {
      statesEl.innerHTML = '<p class="text-slate-500">No regimes detected yet. Click "Analyze with HMM".</p>';
      return;
    }
    
    // Show state stats
    let html = '<div class="grid grid-cols-1 md:grid-cols-3 gap-3 mb-4">';
    
    // Group by state
    const stateMap = {};
    regimes.forEach(r => {
      if (!stateMap[r.state]) stateMap[r.state] = [];
      stateMap[r.state].push(r);
    });
    
    Object.keys(stateMap).forEach(stateNum => {
      const stateRegs = stateMap[stateNum];
      const totalPreds = stateRegs.reduce((s, r) => s + r.predictions_count, 0);
      const totalCorrect = stateRegs.reduce((s, r) => s + r.correct_count, 0);
      const avgAcc = totalPreds > 0 ? (totalCorrect / totalPreds * 100).toFixed(1) : 0;
      
      // Determine label from first regime
      const label = stateRegs[0].label || 'neutral';
      const labelClass = label === 'good' ? 'text-green-400' : (label === 'bad' ? 'text-red-400' : 'text-yellow-400');
      const bgClass = label === 'good' ? 'bg-green-900/30' : (label === 'bad' ? 'bg-red-900/30' : 'bg-yellow-900/30');
      
      // Get feature means from state stats if available
      const stateStat = run.states?.find(s => s.state == stateNum);
      const featMeans = stateStat?.feature_means || {};
      
      html += `<div class="p-3 rounded ${bgClass} border border-slate-700">`;
      html += `<div class="font-semibold ${labelClass}">State ${stateNum}: ${label.toUpperCase()}</div>`;
      html += `<div class="text-xs text-slate-400">${totalPreds} predictions, ${avgAcc}% accuracy</div>`;
      html += `<div class="text-xs text-slate-500">${stateRegs.length} regime segments</div>`;
      
      // Feature means
      if (Object.keys(featMeans).length > 0) {
        html += `<div class="mt-2 text-[10px] text-slate-400 border-t border-slate-700 pt-1">`;
        html += `RSI:${featMeans.rsi_14?.toFixed(0)} `;
        html += `ΔRSI:${featMeans.delta_rsi?.toFixed(1)} `;
        html += `EMA:${(featMeans.ema_diff_20 * 100)?.toFixed(2)}% `;
        html += `ATR:${(featMeans.atr_14 * 100)?.toFixed(2)}%`;
        if (featMeans.prev_result !== undefined) {
          html += ` <span class="text-purple-300">Prev:${featMeans.prev_result.toFixed(2)}</span>`;
        }
        html += `</div>`;
      }
      
      // Filter effectiveness for bad state
      if (label === 'bad' && stateStat?.trades_would_skip) {
        html += `<div class="mt-1 text-[10px] text-red-300">`;
        html += `Would skip: ${stateStat.trades_would_skip} (${stateStat.skip_accuracy}% correct skips)`;
        html += `</div>`;
      }
      
      html += `</div>`;
    });
    
    html += '</div>';
    statesEl.innerHTML = html;
    
    // Transition matrix
    if (run.hmm_model && run.hmm_model.transmat) {
      let tmHtml = '<h5 class="text-xs font-semibold mb-2">State Transition Matrix</h5>';
      tmHtml += '<table class="text-xs"><thead><tr><th></th>';
      for (let i = 0; i < run.hmm_model.n_states; i++) {
        tmHtml += `<th>To S${i}</th>`;
      }
      tmHtml += '</tr></thead><tbody>';
      
      run.hmm_model.transmat.forEach((row, fromIdx) => {
        tmHtml += `<tr><td class="font-semibold">From S${fromIdx}</td>`;
        row.forEach(prob => {
          const probPct = (prob * 100).toFixed(1);
          const probClass = prob > 0.5 ? 'font-bold text-purple-300' : '';
          tmHtml += `<td class="${probClass}">${probPct}%</td>`;
        });
        tmHtml += '</tr>';
      });
      
      tmHtml += '</tbody></table>';
      transitionEl.innerHTML = tmHtml;
    }
    
    // Regime timeline chart
    renderHmmRegimeChart(regimes, chartEl);
    
  } catch (e) {
    console.error('Failed to load HMM results:', e);
  }
}

function renderHmmRegimeChart(regimes, container) {
  if (!regimes.length) {
    container.innerHTML = '<p class="text-slate-500">No regime data.</p>';
    return;
  }
  
  const W = 900, H = 150, pad = {t: 20, r: 20, b: 40, l: 60};
  const plotW = W - pad.l - pad.r;
  const plotH = H - pad.t - pad.b;
  
  // Time range
  const minTime = Math.min(...regimes.map(r => r.start_time));
  const maxTime = Math.max(...regimes.map(r => r.end_time));
  const timeRange = maxTime - minTime || 1;
  
  // State colors
  const stateColors = ['#22c55e', '#ef4444', '#eab308', '#3b82f6', '#a855f7'];
  
  function timeToX(t) {
    return pad.l + ((t - minTime) / timeRange) * plotW;
  }
  
  let svg = `<svg viewBox="0 0 ${W} ${H}" style="width:100%;max-width:${W}px;height:auto">`;
  svg += `<rect width="${W}" height="${H}" fill="#0f172a" rx="6"/>`;
  
  // Draw regime segments
  regimes.forEach(r => {
    const x1 = timeToX(r.start_time);
    const x2 = timeToX(r.end_time);
    const color = stateColors[r.state % stateColors.length];
    const opacity = r.label === 'good' ? 0.6 : (r.label === 'bad' ? 0.4 : 0.5);
    
    svg += `<rect x="${x1}" y="${pad.t}" width="${x2 - x1}" height="${plotH}" fill="${color}" fill-opacity="${opacity}"/>`;
    
    // Accuracy indicator
    const accHeight = (r.accuracy_pct / 100) * plotH;
    svg += `<rect x="${x1}" y="${pad.t + plotH - accHeight}" width="${x2 - x1}" height="${accHeight}" fill="${color}" fill-opacity="0.8"/>`;
  });
  
  // Grid lines
  for (let i = 0; i <= 4; i++) {
    const y = pad.t + (plotH * i / 4);
    svg += `<line x1="${pad.l}" y1="${y}" x2="${W - pad.r}" y2="${y}" stroke="#334155" stroke-width="0.5"/>`;
  }
  
  // X axis labels (dates)
  const nLabels = 5;
  for (let i = 0; i < nLabels; i++) {
    const t = minTime + (timeRange * i / (nLabels - 1));
    const x = timeToX(t);
    const dateStr = new Date(t / 1000).toISOString().substring(0, 10);
    svg += `<text x="${x}" y="${H - 10}" text-anchor="middle" fill="#94a3b8" font-size="9">${dateStr}</text>`;
  }
  
  // Y axis label
  svg += `<text x="${pad.l - 10}" y="${pad.t}" text-anchor="end" fill="#94a3b8" font-size="9">100%</text>`;
  svg += `<text x="${pad.l - 10}" y="${pad.t + plotH}" text-anchor="end" fill="#94a3b8" font-size="9">0%</text>`;
  
  // Legend
  const legendY = H - 18;
  svg += `<text x="${pad.l}" y="${legendY}" fill="#22c55e" font-size="10">■ Good (trade)</text>`;
  svg += `<text x="${pad.l + 120}" y="${legendY}" fill="#ef4444" font-size="10">■ Bad (pause)</text>`;
  svg += `<text x="${pad.l + 240}" y="${legendY}" fill="#94a3b8" font-size="10">Height = accuracy in regime</text>`;
  
  svg += '</svg>';
  container.innerHTML = '<h5 class="text-xs font-semibold mb-2">Regime Timeline (height = accuracy)</h5>' + svg;
}

function closeSuperDetails() {
  document.getElementById('sb-details').classList.add('hidden');
  sbCurrentRunId = null;
  sbCurrentPredictions = [];
}

async function rescoreVolatility() {
  if (!sbCurrentRunId) {
    alert('Please select a run first by clicking "View" on a backtest run');
    return;
  }
  
  const btn = document.getElementById('btn-rescore-vol');
  const minVolStr = document.getElementById('sb-rescore-min').value;
  const maxVolStr = document.getElementById('sb-rescore-max').value;
  const ratioStr = document.getElementById('sb-rescore-ratio').value;
  
  const minVolValues = minVolStr.split(',').map(s => parseFloat(s.trim())).filter(v => !isNaN(v));
  const maxVolValues = maxVolStr.split(',').map(s => parseFloat(s.trim())).filter(v => !isNaN(v));
  const ratioValues = ratioStr.split(',').map(s => parseFloat(s.trim())).filter(v => !isNaN(v));
  
  if (minVolValues.length === 0 || maxVolValues.length === 0 || ratioValues.length === 0) {
    alert('Please enter valid volatility values');
    return;
  }
  
  btn.disabled = true;
  btn.textContent = 'Rescoring...';
  
  try {
    const url = API + '/api/super_backtest/' + sbCurrentRunId + '/rescore_volatility';
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        vol_min_values: minVolValues,
        vol_max_values: maxVolValues,
        vol_ratio_max_values: ratioValues,
      }),
    });
    
    const result = await res.json();
    
    console.log('Rescore response:', result);
    
    if (result.error) {
      alert('Rescore error: ' + result.error);
      return;
    }
    
    if (!result.results || !Array.isArray(result.results)) {
      alert('Invalid response: results not found');
      console.error('Invalid response:', result);
      return;
    }
    
    const resultsDiv = document.getElementById('rescore-results');
    const tbody = document.getElementById('rescore-tbody');
    
    let html = '';
    result.results.forEach((r, idx) => {
      const winClass = r.winrate >= 55 ? 'text-green-400' : (r.winrate >= 50 ? 'text-yellow-400' : 'text-red-400');
      html += '<tr>';
      html += '<td>' + (idx + 1) + '</td>';
      html += '<td>' + r.min_vol + '</td>';
      html += '<td>' + r.max_vol + '</td>';
      html += '<td>' + r.vol_ratio_max + '</td>';
      html += '<td class="' + winClass + ' font-bold">' + r.winrate + '%</td>';
      html += '<td>' + r.signals + '</td>';
      html += '</tr>';
    });
    
    tbody.innerHTML = html;
    resultsDiv.classList.remove('hidden');
    
    if (result.best) {
      alert('Best params: min=' + result.best.min_vol + ', max=' + result.best.max_vol + 
            ', ratio=' + result.best.vol_ratio_max + ' -> ' + result.best.winrate + '% winrate (' + result.best.signals + ' signals)');
    }
  } catch (e) {
    alert('Failed: ' + e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Rescore';
  }
}

async function deleteSuperBacktest(runId) {
  if (!confirm('Delete super backtest #' + runId + ' and all its predictions/regimes?')) {
    return;
  }
  
  try {
    const res = await fetch(API + '/api/super_backtest/' + runId, {
      method: 'DELETE'
    });
    
    const result = await res.json();
    
    if (result.success) {
      alert('Super backtest #' + runId + ' deleted');
      // Close details if viewing this run
      if (sbCurrentRunId === runId) {
        closeSuperDetails();
      }
      // Refresh list
      loadSuperBacktestList();
    } else {
      alert('Error: ' + (result.error || 'Failed to delete'));
    }
  } catch (e) {
    alert('Failed to delete: ' + e.message);
  }
}

async function runVolatilityBruteforce() {
  const btn = document.getElementById('btn-vol-brute');
  const configStr = document.getElementById('sb-config').value;
  
  if (!configStr) {
    alert('Please enter config JSON first');
    return;
  }
  
  let config;
  try {
    config = JSON.parse(configStr);
  } catch (e) {
    alert('Invalid JSON in config: ' + e.message);
    return;
  }
  
  // Parse volatility ranges
  const minVolStr = document.getElementById('sb-vol-min').value;
  const maxVolStr = document.getElementById('sb-vol-max').value;
  const ratioStr = document.getElementById('sb-vol-ratio').value;
  
  const minVolValues = minVolStr.split(',').map(s => parseFloat(s.trim())).filter(v => !isNaN(v));
  const maxVolValues = maxVolStr.split(',').map(s => parseFloat(s.trim())).filter(v => !isNaN(v));
  const ratioValues = ratioStr.split(',').map(s => parseFloat(s.trim())).filter(v => !isNaN(v));
  
  if (minVolValues.length === 0 || maxVolValues.length === 0 || ratioValues.length === 0) {
    alert('Please enter valid volatility values');
    return;
  }
  
  btn.disabled = true;
  btn.textContent = 'Running...';
  
  try {
    const params = {
      strategy_name: config.strategy || 'rsi_mean_reversion',
      strategy_params: config.params || {},
      train_start: config.train_start,
      train_end: config.train_end,
      test_start: config.test_start,
      test_end: config.test_end,
      symbol: config.symbol || 'BTCUSDT',
      timeframe: config.timeframe || '5m',
      horizon: config.horizon || 1,
      vol_min_values: minVolValues,
      vol_max_values: maxVolValues,
      vol_ratio_max_values: ratioValues,
    };
    
    const res = await fetch(API + '/api/super_backtest/volatility_bruteforce', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    
    const result = await res.json();
    
    if (result.error) {
      alert('Bruteforce error: ' + result.error);
    } else {
      // Show results table
      const resultsDiv = document.getElementById('vol-brute-results');
      const tbody = document.getElementById('vol-brute-tbody');
      
      let html = '';
      result.results.forEach((r, idx) => {
        const winClass = r.winrate >= 55 ? 'text-green-400' : (r.winrate >= 50 ? 'text-yellow-400' : 'text-red-400');
        html += '<tr>';
        html += '<td>' + (idx + 1) + '</td>';
        html += '<td>' + r.min_vol + '</td>';
        html += '<td>' + r.max_vol + '</td>';
        html += '<td>' + r.vol_ratio_max + '</td>';
        html += '<td class="' + winClass + ' font-bold">' + r.winrate + '%</td>';
        html += '<td>' + r.signals + '</td>';
        html += '</tr>';
      });
      
      tbody.innerHTML = html;
      resultsDiv.classList.remove('hidden');
      
      if (result.best) {
        alert('Best: min=' + result.best.min_vol + ', max=' + result.best.max_vol + 
              ', ratio=' + result.best.vol_ratio_max + ' -> ' + result.best.winrate + '% winrate');
      }
    }
  } catch (e) {
    alert('Failed: ' + e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Run Bruteforce';
  }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', initSuperBacktest);
