// ==================== SUPER BACKTEST ====================

let sbCurrentRunId = null;
let sbInitialized = false;

async function initSuperBacktest(force = false) {
  if (sbInitialized && !force) return;
  sbInitialized = true;

  // Load default config template (once per page load unless forced)
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
  loadHmmFeatures();
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

    // Load HMM v2 analyses for this run
    loadHmmAnalyses(runId);
    loadHmmSweeps(runId);
    
  } catch (e) {
    statsEl.innerHTML = '<p class="text-red-400">Failed: ' + e.message + '</p>';
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
  closeHmmSweepDetail();
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
  
  const statusEl = document.getElementById('sb-vol-brute-status');
  const combinations = minVolValues.length * maxVolValues.length * ratioValues.length;

  if (minVolValues.length === 0 || maxVolValues.length === 0 || ratioValues.length === 0) {
    alert('Please enter valid volatility values');
    return;
  }
  
  btn.disabled = true;
  btn.textContent = 'Running...';
  statusEl.textContent = `Testing ${combinations} combinations... (may take a while)`;
  statusEl.className = "text-[10px] text-slate-400 mt-3 self-center ml-2";

  try {
    // Only validate test dates; train dates are optional for some strategies
    const required = ['test_start', 'test_end'];
    for (const f of required) {
      if (!config[f]) {
        alert(`Missing required field in config: ${f}`);
        btn.disabled = false;
        btn.textContent = 'Run Bruteforce';
        return;
      }
    }

    const params = {
      strategy: config.strategy || 'rsi_mean_reversion',
      params: config.params || {},
      train_start: config.train_start || null,
      train_end: config.train_end || null,
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
    
    if (result.error || result.detail) {
      const msg = result.error || (Array.isArray(result.detail) ? result.detail.map(d => d.msg).join(', ') : JSON.stringify(result.detail));
      alert('Error: ' + msg);
      statusEl.textContent = 'Error during bruteforce';
      statusEl.className = "text-[10px] text-red-400 mt-3 self-center ml-2";
      return;
    } else {
      statusEl.textContent = `Completed ${combinations} combinations.`;
      statusEl.className = "text-[10px] text-green-400 mt-3 self-center ml-2";
      
      // Show results table
      const resultsDiv = document.getElementById('vol-brute-results');
      const tbody = document.getElementById('vol-brute-tbody');
      
      if (!result.results) {
        alert('No results returned from server');
        return;
      }

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

// ==================== HMM ANALYSES v2 ====================

let hmmFeatureCache = null;
let hmmCurrentTimeline = null;
let hmmCurrentSweep = null;

function parseFloatList(str) {
  if (!str) return [];
  const seen = new Set();
  const values = [];
  str.split(',').map(s => parseFloat(s.trim())).forEach(v => {
    if (Number.isFinite(v)) {
      const key = v.toFixed(6);
      if (!seen.has(key)) {
        seen.add(key);
        values.push(v);
      }
    }
  });
  return values;
}

async function loadHmmFeatures() {
  if (hmmFeatureCache && hmmFeatureCache.length > 0) return hmmFeatureCache;
  try {
    const res = await fetch(API + '/api/super_backtest/hmm/features');
    const data = await res.json();
    hmmFeatureCache = data.features || [];
    console.log("HMM Features loaded:", hmmFeatureCache.length);
  } catch (e) {
    console.error("Failed to load HMM features:", e);
    hmmFeatureCache = [];
  }
  return hmmFeatureCache;
}

async function openHmmAnalysisForm() {
  const form = document.getElementById('sb-hmm-form');
  form.classList.remove('hidden');
  document.getElementById('sb-hmm-status').textContent = '';

  const wrap = document.getElementById('sb-hmm-features');
  wrap.innerHTML = '<p class="text-slate-500 col-span-full">Loading features...</p>';

  // Render features checkboxes (default: a sensible regime-detection set)
  const features = await loadHmmFeatures();
  
  if (!features || features.length === 0) {
    wrap.innerHTML = '<p class="text-red-400 col-span-full">Failed to load features from API</p>';
    return;
  }

  const defaultPicks = new Set(['rsi_14', 'bb_pos', 'volatility_5', 'atr_14', 'ema_diff_20']);
  wrap.innerHTML = features.map(f => {
    const checked = defaultPicks.has(f) ? 'checked' : '';
    return `<label class="flex items-center gap-1 text-slate-300 cursor-pointer hover:text-white"><input type="checkbox" data-feat="${f}" ${checked} class="hmm-feat-cb"/> <span>${f}</span></label>`;
  }).join('');

  // Show walk-forward params only when chosen
  const fitSel = document.getElementById('sb-hmm-fitmode');
  fitSel.onchange = () => {
    const wf = document.getElementById('sb-hmm-walkparams');
    if (fitSel.value === 'walk_forward') wf.classList.remove('hidden');
    else wf.classList.add('hidden');
  };
  fitSel.onchange();
}

function closeHmmAnalysisForm() {
  document.getElementById('sb-hmm-form').classList.add('hidden');
}

async function openHmmSweepForm() {
  const form = document.getElementById('sb-hmm-sweep-form');
  if (!form) return;
  form.classList.remove('hidden');
  document.getElementById('sb-hmm-sweep-status').textContent = '';

  const wrap = document.getElementById('sb-hmm-sweep-features');
  wrap.innerHTML = '<p class="text-slate-500 col-span-full">Loading features...</p>';

  const features = await loadHmmFeatures();
  if (!features || features.length === 0) {
    wrap.innerHTML = '<p class="text-red-400 col-span-full">Failed to load features from API</p>';
    return;
  }

  const defaultPicks = new Set(['rsi_14', 'bb_pos', 'volatility_5', 'atr_14', 'ema_diff_20']);
  wrap.innerHTML = features.map(f => {
    const checked = defaultPicks.has(f) ? 'checked' : '';
    return `<label class="flex items-center gap-1 text-slate-300 cursor-pointer hover:text-white"><input type="checkbox" data-feat="${f}" ${checked} class="hmm-sweep-feat-cb"/> <span>${f}</span></label>`;
  }).join('');

  const fitSel = document.getElementById('sb-hmm-sweep-fitmode');
  const toggleWalk = () => {
    const wf = document.getElementById('sb-hmm-sweep-walk');
    if (!wf) return;
    if (fitSel.value === 'walk_forward') wf.classList.remove('hidden');
    else wf.classList.add('hidden');
  };
  fitSel.onchange = toggleWalk;
  toggleWalk();
}

function closeHmmSweepForm() {
  const form = document.getElementById('sb-hmm-sweep-form');
  if (form) form.classList.add('hidden');
}

async function submitHmmAnalysis() {
  if (!sbCurrentRunId) { alert('No run selected'); return; }
  const features = Array.from(document.querySelectorAll('.hmm-feat-cb'))
    .filter(cb => cb.checked).map(cb => cb.dataset.feat);
  if (features.length < 2) { alert('Pick at least 2 features'); return; }
  if (features.length > 8) { alert('Max 8 features'); return; }

  const body = {
    name: document.getElementById('sb-hmm-name').value || null,
    n_states: parseInt(document.getElementById('sb-hmm-nstates').value) || 2,
    features: features,
    fit_mode: document.getElementById('sb-hmm-fitmode').value,
    walk_train_len: parseInt(document.getElementById('sb-hmm-walktrain').value) || null,
    walk_step: parseInt(document.getElementById('sb-hmm-walkstep').value) || null,
    good_threshold: parseFloat(document.getElementById('sb-hmm-good').value) || 55,
    bad_threshold: parseFloat(document.getElementById('sb-hmm-bad').value) || 45,
    filter_threshold: parseFloat(document.getElementById('sb-hmm-filter').value) || 0.6,
    min_regime_len: parseInt(document.getElementById('sb-hmm-minlen').value) || 1,
  };

  const btn = document.getElementById('btn-hmm-submit');
  const status = document.getElementById('sb-hmm-status');
  btn.disabled = true;
  status.textContent = 'Fitting HMM... (may take 5-60s)';

  try {
    const res = await fetch(API + '/api/super_backtest/' + sbCurrentRunId + '/hmm_analyses', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (data.error) {
      status.textContent = 'Error: ' + data.error;
      status.className = 'text-xs text-red-400 self-center';
    } else {
      status.textContent = 'Done in ' + data.time_sec + 's';
      status.className = 'text-xs text-green-400 self-center';
      closeHmmAnalysisForm();
      await loadHmmAnalyses(sbCurrentRunId);
      // Auto-open the new analysis
      if (data.id) showHmmAnalysisDetail(data.id);
    }
  } catch (e) {
    status.textContent = 'Failed: ' + e.message;
  } finally {
    btn.disabled = false;
  }
}

async function submitHmmSweep() {
  if (!sbCurrentRunId) { alert('No run selected'); return; }
  const features = Array.from(document.querySelectorAll('.hmm-sweep-feat-cb'))
    .filter(cb => cb.checked).map(cb => cb.dataset.feat);
  if (features.length < 2) { alert('Pick at least 2 features'); return; }
  if (features.length > 8) { alert('Max 8 features'); return; }

  const goodVals = parseFloatList(document.getElementById('sb-hmm-sweep-good').value);
  const badVals = parseFloatList(document.getElementById('sb-hmm-sweep-bad').value);
  const filterVals = parseFloatList(document.getElementById('sb-hmm-sweep-filter').value);
  if (goodVals.length === 0 || badVals.length === 0 || filterVals.length === 0) {
    alert('Enter at least one value for each threshold list');
    return;
  }
  const combos = goodVals.length * badVals.length * filterVals.length;
  if (combos > 75) {
    alert('Too many combinations. Please limit to 75 or fewer.');
    return;
  }

  const body = {
    name: document.getElementById('sb-hmm-sweep-name').value || null,
    n_states: parseInt(document.getElementById('sb-hmm-sweep-nstates').value) || 3,
    features,
    fit_mode: document.getElementById('sb-hmm-sweep-fitmode').value,
    walk_train_len: parseInt(document.getElementById('sb-hmm-sweep-walktrain').value) || null,
    walk_step: parseInt(document.getElementById('sb-hmm-sweep-walkstep').value) || null,
    min_regime_len: parseInt(document.getElementById('sb-hmm-sweep-minlen').value) || 1,
    good_thresholds: goodVals,
    bad_thresholds: badVals,
    filter_thresholds: filterVals,
  };

  const btn = document.getElementById('btn-hmm-sweep');
  const status = document.getElementById('sb-hmm-sweep-status');
  btn.disabled = true;
  status.textContent = `Running ${combos} combos...`;
  status.className = 'text-xs text-slate-400 self-center';

  try {
    const res = await fetch(API + '/api/super_backtest/' + sbCurrentRunId + '/hmm_sweeps', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (data.error) {
      status.textContent = 'Error: ' + data.error;
      status.className = 'text-xs text-red-400 self-center';
    } else {
      status.textContent = 'Sweep completed in ' + data.time_sec + 's';
      status.className = 'text-xs text-green-400 self-center';
      closeHmmSweepForm();
      await loadHmmSweeps(sbCurrentRunId);
      if (data.id) showHmmSweepDetail(data.id);
    }
  } catch (e) {
    status.textContent = 'Failed: ' + e.message;
    status.className = 'text-xs text-red-400 self-center';
  } finally {
    btn.disabled = false;
  }
}

async function loadHmmAnalyses(runId) {
  const el = document.getElementById('sb-hmm-list');
  if (!el) return;
  el.innerHTML = '<p class="text-slate-500 text-xs">Loading...</p>';
  try {
    const res = await fetch(API + '/api/super_backtest/' + runId + '/hmm_analyses');
    const list = await res.json();
    if (!Array.isArray(list) || list.length === 0) {
      el.innerHTML = '<p class="text-slate-500 text-xs">No analyses yet. Click "+ New Analysis".</p>';
      return;
    }
    let html = '<table class="text-xs w-full"><thead><tr>'
      + '<th>ID</th><th>Name</th><th>States</th><th>Fit</th><th>Features</th>'
      + '<th>Baseline</th><th>Filtered</th><th>Δ</th><th>Skipped</th><th>Actions</th></tr></thead><tbody>';
    for (const a of list) {
      const baseline = (a.baseline_winrate ?? 0).toFixed(2);
      const filtered = a.filtered_winrate != null ? a.filtered_winrate.toFixed(2) : '—';
      const imp = a.improvement;
      const impStr = imp == null ? '—' : (imp >= 0 ? '+' : '') + imp.toFixed(2) + '%';
      const impClass = imp == null ? 'text-slate-400' : (imp >= 0 ? 'text-green-400' : 'text-red-400');
      const featsShort = (a.features || []).slice(0, 3).join(',') + (a.features.length > 3 ? '+' + (a.features.length - 3) : '');
      html += `<tr class="border-b border-slate-800">
        <td>#${a.id}</td>
        <td>${a.name || '—'}</td>
        <td>${a.n_states}</td>
        <td><span class="text-[10px] px-1 py-0.5 rounded bg-slate-700">${a.fit_mode}</span></td>
        <td title="${(a.features||[]).join(', ')}" class="font-mono text-[10px]">${featsShort}</td>
        <td>${baseline}%</td>
        <td>${filtered}${filtered !== '—' ? '%' : ''}</td>
        <td class="${impClass} font-bold">${impStr}</td>
        <td>${a.trades_skipped ?? '—'}</td>
        <td>
          <button onclick="showHmmAnalysisDetail(${a.id})" class="btn btn-purple text-[10px] py-0.5 px-2">View</button>
          <button onclick="deleteHmmAnalysis(${a.id})" class="btn btn-slate text-[10px] py-0.5 px-2">×</button>
        </td>
      </tr>`;
    }
    html += '</tbody></table>';
    el.innerHTML = html;
  } catch (e) {
    el.innerHTML = '<p class="text-red-400 text-xs">Failed: ' + e.message + '</p>';
  }
}

async function deleteHmmAnalysis(id) {
  if (!confirm('Delete analysis #' + id + '?')) return;
  await fetch(API + '/api/super_backtest/hmm/' + id, {method: 'DELETE'});
  closeHmmDetail();
  if (sbCurrentRunId) loadHmmAnalyses(sbCurrentRunId);
}

async function showHmmAnalysisDetail(analysisId) {
  const wrap = document.getElementById('sb-hmm-detail');
  const summary = document.getElementById('sb-hmm-detail-summary');
  const statesEl = document.getElementById('sb-hmm-detail-states');
  const transEl = document.getElementById('sb-hmm-detail-transition');
  const titleEl = document.getElementById('sb-hmm-detail-title');

  wrap.classList.remove('hidden');
  summary.innerHTML = '<p class="text-slate-500">Loading...</p>';
  statesEl.innerHTML = '';
  transEl.innerHTML = '';

  try {
    const [detailRes, timelineRes] = await Promise.all([
      fetch(API + '/api/super_backtest/hmm/' + analysisId).then(r => r.json()),
      fetch(API + '/api/super_backtest/hmm/' + analysisId + '/timeline?max_points=10000').then(r => r.json()),
    ]);

    if (detailRes.error) {
      summary.innerHTML = '<p class="text-red-400">' + detailRes.error + '</p>';
      return;
    }

    titleEl.textContent = 'Analysis #' + detailRes.id + (detailRes.name ? ' — ' + detailRes.name : '');

    // Summary metrics
    const baseline = (detailRes.baseline_winrate ?? 0).toFixed(2);
    const filtered = detailRes.filtered_winrate != null ? detailRes.filtered_winrate.toFixed(2) : '—';
    const imp = detailRes.improvement;
    const impStr = imp == null ? '—' : (imp >= 0 ? '+' : '') + imp.toFixed(2) + '%';
    const impClass = imp == null ? 'text-slate-400' : (imp >= 0 ? 'text-green-400' : 'text-red-400');
    summary.innerHTML = `
      <div class="grid grid-cols-2 md:grid-cols-5 gap-2">
        <div class="p-2 rounded bg-slate-800"><div class="text-slate-400">N States</div><div class="font-semibold">${detailRes.n_states}</div></div>
        <div class="p-2 rounded bg-slate-800"><div class="text-slate-400">Fit Mode</div><div class="font-semibold">${detailRes.fit_mode}</div></div>
        <div class="p-2 rounded bg-slate-800"><div class="text-slate-400">Baseline WR</div><div class="font-semibold">${baseline}%</div></div>
        <div class="p-2 rounded bg-slate-800"><div class="text-slate-400">Filtered WR</div><div class="font-semibold">${filtered}${filtered !== '—' ? '%' : ''}</div></div>
        <div class="p-2 rounded bg-slate-800"><div class="text-slate-400">Improvement</div><div class="font-semibold ${impClass}">${impStr}</div></div>
        <div class="p-2 rounded bg-slate-800"><div class="text-slate-400">Trades Total</div><div class="font-semibold">${detailRes.trades_total ?? '—'}</div></div>
        <div class="p-2 rounded bg-slate-800"><div class="text-slate-400">Trades Taken</div><div class="font-semibold text-green-400">${detailRes.trades_taken ?? '—'}</div></div>
        <div class="p-2 rounded bg-slate-800"><div class="text-slate-400">Trades Skipped</div><div class="font-semibold text-orange-400">${detailRes.trades_skipped ?? '—'}</div></div>
        <div class="p-2 rounded bg-slate-800"><div class="text-slate-400">Candles Analyzed</div><div class="font-semibold">${detailRes.candles_analyzed ?? '—'}</div></div>
        <div class="p-2 rounded bg-slate-800 col-span-2"><div class="text-slate-400">Features</div><div class="font-mono text-[10px]">${(detailRes.features || []).join(', ')}</div></div>
      </div>`;

    // States table
    let stHtml = '<h5 class="text-xs font-semibold text-slate-300 mb-1">States</h5>';
    stHtml += '<table class="text-xs w-full"><thead><tr><th>State</th><th>Label</th><th>Candles</th><th>Signals</th><th>Correct</th><th>Wrong</th><th>Accuracy</th><th>Signal Rate</th><th>Feature Means</th></tr></thead><tbody>';
    for (const s of detailRes.states || []) {
      const labelColor = s.label === 'good' ? 'bg-green-700' : (s.label === 'bad' ? 'bg-red-700' : 'bg-slate-700');
      const accColor = s.accuracy_pct >= 55 ? 'text-green-400' : (s.accuracy_pct >= 50 ? 'text-yellow-400' : 'text-red-400');
      const fm = Object.entries(s.feature_means || {}).map(([k, v]) => `${k}=${v}`).join(' | ');
      stHtml += `<tr class="border-b border-slate-800">
        <td>${s.state}</td>
        <td><span class="text-[10px] px-1 py-0.5 rounded ${labelColor}">${s.label}</span></td>
        <td>${s.candles}</td>
        <td>${s.signals}</td>
        <td class="text-green-400">${s.correct}</td>
        <td class="text-red-400">${s.wrong}</td>
        <td class="${accColor} font-bold">${s.accuracy_pct}%</td>
        <td>${s.signal_rate_pct}%</td>
        <td class="font-mono text-[10px] text-slate-400">${fm}</td>
      </tr>`;
    }
    stHtml += '</tbody></table>';
    statesEl.innerHTML = stHtml;

    // Transition matrix
    if (detailRes.transition_matrix) {
      let tm = '<h5 class="text-xs font-semibold text-slate-300 mb-1">Transition Matrix (rows = from, cols = to)</h5>';
      tm += '<table class="text-xs"><thead><tr><th></th>';
      for (let j = 0; j < detailRes.n_states; j++) tm += `<th>→${j}</th>`;
      tm += '</tr></thead><tbody>';
      for (let i = 0; i < detailRes.transition_matrix.length; i++) {
        tm += `<tr><th>${i}</th>`;
        for (const v of detailRes.transition_matrix[i]) {
          const intensity = Math.min(1, v);
          const bg = `rgba(168,85,247,${intensity * 0.4})`;
          tm += `<td style="background:${bg}">${(v * 100).toFixed(1)}%</td>`;
        }
        tm += '</tr>';
      }
      tm += '</tbody></table>';
      transEl.innerHTML = tm;
    }

    renderHmmMonthlyTable(timelineRes.monthly || {});

  } catch (e) {
    summary.innerHTML = '<p class="text-red-400">Failed: ' + e.message + '</p>';
  }
}

function closeHmmDetail() {
  document.getElementById('sb-hmm-detail').classList.add('hidden');
  hmmCurrentTimeline = null;
}

function getHmmSweepBankrollSettings() {
  const startBank = parseFloat(document.getElementById('sb-hmm-sweep-bank')?.value || '1000') || 1000;
  const buyPriceCents = parseFloat(document.getElementById('sb-hmm-sweep-buyprice')?.value || '52') || 52;
  const maxBet = parseFloat(document.getElementById('sb-hmm-sweep-maxbet')?.value || '500') || 500;
  const kellyPct = parseFloat(document.getElementById('sb-hmm-sweep-kelly')?.value || '3.34') || 3.34;
  const feePct = parseFloat(document.getElementById('sb-hmm-sweep-fee')?.value || '1.56');
  return {
    startBank,
    buyPriceCents,
    maxBet,
    kellyPct,
    betFeeRate: Number.isFinite(feePct) && feePct >= 0 ? (feePct / 100) : DEFAULT_BET_FEE_RATE,
  };
}

function simulateHmmSweepCombo(monthlyData, settings) {
  const startBank = settings.startBank;
  const cost = settings.buyPriceCents / 100;
  const profitPerShare = 1.0 - cost;
  const b = profitPerShare / cost;
  const kellyApplied = settings.kellyPct / 100;
  const months = Object.entries(monthlyData || {}).sort((a, b2) => a[0].localeCompare(b2[0]));

  function simMonth(bank, nSignals, winProb, betPct) {
    const evPerDollar = (winProb * b) - (1 - winProb);
    let avgStake = 0;
    let maxStakeUsed = 0;
    let edgeSignals = 0;
    if (!(Math.abs(betPct) > 0) || nSignals <= 0) {
      return { bank, avgStake: 0, maxStakeUsed: 0, edgeSignals: 0 };
    }
    for (let j = 0; j < nSignals; j++) {
      const rawStake = bank * Math.abs(betPct);
      const stake = Math.max(0, Math.min(rawStake, settings.maxBet || rawStake, bank));
      if (stake <= 0) break;
      bank += stake * evPerDollar;
      bank -= stake * settings.betFeeRate;
      if (bank < 0.01) bank = 0.01;
      avgStake += stake;
      if (stake > maxStakeUsed) maxStakeUsed = stake;
      edgeSignals += 1;
    }
    avgStake = edgeSignals ? (avgStake / edgeSignals) : 0;
    return { bank, avgStake, maxStakeUsed, edgeSignals };
  }

  let bank = startBank;
  const monthEntries = [];
  for (const [month, data] of months) {
    const signals = Number(data && data.taken ? data.taken : 0);
    const filteredCorrect = Number(data && data.filtered_correct ? data.filtered_correct : 0);
    const accuracyPct = signals > 0 ? ((filteredCorrect / signals) * 100) : 0;
    const winProb = accuracyPct / 100;
    const res = simMonth(bank, signals, winProb, kellyApplied);
    bank = res.bank;
    monthEntries.push({
      month,
      bank: Math.round(bank * 100) / 100,
      signals,
      edge_signals: res.edgeSignals,
      avg_stake: res.avgStake,
      max_stake: res.maxStakeUsed,
      accuracy: accuracyPct,
    });
  }
  return {
    finalBank: bank,
    roiPct: startBank > 0 ? ((bank - startBank) / startBank * 100) : 0,
    profit: bank - startBank,
    monthEntries,
  };
}

function recalcHmmSweepProfit() {
  if (!hmmCurrentSweep || !Array.isArray(hmmCurrentSweep.results)) return;
  const summary = document.getElementById('sb-hmm-sweep-summary');
  const combosBody = document.getElementById('sb-hmm-sweep-combos');
  const profitSummary = document.getElementById('sb-hmm-sweep-profit-summary');
  if (!summary || !combosBody || !profitSummary) return;
  const settings = getHmmSweepBankrollSettings();
  const results = hmmCurrentSweep.results.map(r => {
    const sim = simulateHmmSweepCombo(r.monthly || {}, settings);
    return {...r, sim};
  }).sort((a, b) => (b.sim?.finalBank || 0) - (a.sim?.finalBank || 0));

  const best = results[0] || null;
  const avgFinalBank = results.length ? (results.reduce((sum, r) => sum + (r.sim?.finalBank || 0), 0) / results.length) : settings.startBank;
  const breakeven = settings.buyPriceCents.toFixed(1) + '%';
  profitSummary.innerHTML = `
    <div class="p-2 rounded bg-slate-800 border border-slate-700"><div class="text-[10px] text-slate-400 uppercase">Best Final Bank</div><div class="text-lg font-semibold text-emerald-300">$${best ? best.sim.finalBank.toFixed(2) : settings.startBank.toFixed(2)}</div></div>
    <div class="p-2 rounded bg-slate-800 border border-slate-700"><div class="text-[10px] text-slate-400 uppercase">Best Profit</div><div class="text-lg font-semibold ${(best && best.sim.profit >= 0) ? 'text-emerald-300' : 'text-rose-300'}">${best ? ((best.sim.profit >= 0 ? '+' : '') + '$' + best.sim.profit.toFixed(2)) : '$0.00'}</div></div>
    <div class="p-2 rounded bg-slate-800 border border-slate-700"><div class="text-[10px] text-slate-400 uppercase">Average Final Bank</div><div class="text-lg font-semibold text-indigo-200">$${avgFinalBank.toFixed(2)}</div></div>
    <div class="p-2 rounded bg-slate-800 border border-slate-700"><div class="text-[10px] text-slate-400 uppercase">Breakeven WR</div><div class="text-lg font-semibold text-amber-300">${breakeven}</div></div>`;

  const baseline = hmmCurrentSweep.baseline_winrate != null ? hmmCurrentSweep.baseline_winrate.toFixed(2) + '%' : '—';
  summary.innerHTML = `
    <div class="grid grid-cols-2 md:grid-cols-5 gap-2">
      <div class="p-2 rounded bg-slate-800"><div class="text-slate-400 text-[11px]">States</div><div class="font-semibold">${hmmCurrentSweep.n_states}</div></div>
      <div class="p-2 rounded bg-slate-800"><div class="text-slate-400 text-[11px]">Fit Mode</div><div class="font-semibold">${hmmCurrentSweep.fit_mode}</div></div>
      <div class="p-2 rounded bg-slate-800"><div class="text-slate-400 text-[11px]">Baseline WR</div><div class="font-semibold">${baseline}</div></div>
      <div class="p-2 rounded bg-slate-800"><div class="text-slate-400 text-[11px]">Combos</div><div class="font-semibold">${hmmCurrentSweep.combos_total}</div></div>
      <div class="p-2 rounded bg-slate-800"><div class="text-slate-400 text-[11px]">Features</div><div class="font-mono text-[10px]">${(hmmCurrentSweep.features || []).join(', ')}</div></div>
    </div>`;

  combosBody.innerHTML = results.map(r => {
    const baseWR = r.baseline_winrate != null ? r.baseline_winrate.toFixed(2) + '%' : '—';
    const filtWR = r.filtered_winrate != null ? r.filtered_winrate.toFixed(2) + '%' : '—';
    const imp = r.improvement == null ? null : r.improvement;
    const impStr = imp == null ? '—' : (imp >= 0 ? '+' : '') + imp.toFixed(2) + '%';
    const impClass = imp == null ? 'text-slate-400' : (imp >= 0 ? 'text-green-400' : 'text-red-400');
    const roiClass = r.sim.profit >= 0 ? 'text-emerald-300' : 'text-rose-300';
    return `<tr class="border-b border-slate-800">
      <td class="p-2">${r.combo_index}</td>
      <td class="p-2">${r.good_threshold}</td>
      <td class="p-2">${r.bad_threshold}</td>
      <td class="p-2">${r.filter_threshold}</td>
      <td class="p-2 text-center">${baseWR}</td>
      <td class="p-2 text-center font-bold text-green-400">${filtWR}</td>
      <td class="p-2 text-center">${r.filtered_trades ?? '—'}</td>
      <td class="p-2 text-center text-orange-300">${r.trades_skipped ?? '—'}</td>
      <td class="p-2 text-center font-bold ${impClass}">${impStr}</td>
      <td class="p-2 text-center font-mono ${roiClass}">$${r.sim.finalBank.toFixed(2)}</td>
      <td class="p-2 text-center font-bold ${roiClass}">${r.sim.roiPct >= 0 ? '+' : ''}${r.sim.roiPct.toFixed(2)}%</td>
      <td class="p-2 text-center font-mono ${roiClass}">${r.sim.profit >= 0 ? '+' : ''}$${r.sim.profit.toFixed(2)}</td>
    </tr>`;
  }).join('');
}

async function loadHmmSweeps(runId) {
  const wrap = document.getElementById('sb-hmm-sweep-list');
  if (!wrap) return;
  if (!runId) {
    wrap.innerHTML = '<p class="text-slate-500 text-xs">Select a run to view sweeps.</p>';
    return;
  }
  wrap.innerHTML = '<p class="text-slate-500 text-xs">Loading sweeps...</p>';
  try {
    const res = await fetch(API + '/api/super_backtest/' + runId + '/hmm_sweeps');
    const sweeps = await res.json();
    if (!Array.isArray(sweeps) || sweeps.length === 0) {
      wrap.innerHTML = '<p class="text-slate-500 text-xs">No sweeps yet. Click "+ Threshold Sweep".</p>';
      return;
    }
    let html = '<table class="text-[11px] w-full"><thead><tr>' +
      '<th>ID</th><th>Name</th><th>States</th><th>Fit</th><th>Features</th>' +
      '<th>Baseline</th><th>Best Filtered</th><th>Δ</th><th>Combos</th><th>Actions</th></tr></thead><tbody>';
    sweeps.forEach(s => {
      const feats = Array.isArray(s.features) ? s.features : [];
      const featsShort = feats.slice(0, 3).join(',') + (feats.length > 3 ? '+' + (feats.length - 3) : '');
      const baseline = s.baseline_winrate != null ? s.baseline_winrate.toFixed(2) + '%' : '—';
      const best = s.best_result;
      const bestWr = best?.filtered_winrate != null ? best.filtered_winrate.toFixed(2) + '%' : '—';
      const imp = best?.improvement;
      const impStr = imp == null ? '—' : (imp >= 0 ? '+' : '') + imp.toFixed(2) + '%';
      const impClass = imp == null ? 'text-slate-400' : (imp >= 0 ? 'text-green-400' : 'text-red-400');
      html += `<tr class="border-b border-slate-800">
        <td>#${s.id}</td>
        <td>${s.name || '—'}</td>
        <td>${s.n_states}</td>
        <td><span class="text-[10px] px-1 py-0.5 rounded bg-slate-700">${s.fit_mode}</span></td>
        <td title="${feats.join(', ')}" class="font-mono text-[10px]">${featsShort}</td>
        <td>${baseline}</td>
        <td>${bestWr}</td>
        <td class="${impClass} font-bold">${impStr}</td>
        <td>${s.combos_total}</td>
        <td>
          <button onclick="showHmmSweepDetail(${s.id})" class="btn btn-amber text-[10px] py-0.5 px-2">View</button>
          <button onclick="deleteHmmSweep(${s.id})" class="btn btn-slate text-[10px] py-0.5 px-2">×</button>
        </td>
      </tr>`;
    });
    html += '</tbody></table>';
    wrap.innerHTML = html;
  } catch (e) {
    wrap.innerHTML = '<p class="text-red-400 text-xs">Failed: ' + e.message + '</p>';
  }
}

async function deleteHmmSweep(id) {
  if (!confirm('Delete sweep #' + id + '?')) return;
  await fetch(API + '/api/super_backtest/hmm_sweeps/' + id, {method: 'DELETE'});
  closeHmmSweepDetail();
  if (sbCurrentRunId) loadHmmSweeps(sbCurrentRunId);
}

async function showHmmSweepDetail(sweepId) {
  const wrap = document.getElementById('sb-hmm-sweep-detail');
  const summary = document.getElementById('sb-hmm-sweep-summary');
  const combosBody = document.getElementById('sb-hmm-sweep-combos');
  if (!wrap || !summary || !combosBody) return;
  wrap.classList.remove('hidden');
  summary.innerHTML = '<p class="text-slate-500">Loading sweep...</p>';
  combosBody.innerHTML = '';

  try {
    const res = await fetch(API + '/api/super_backtest/hmm_sweeps/' + sweepId);
    const data = await res.json();
    if (data.error) {
      summary.innerHTML = '<p class="text-red-400">' + data.error + '</p>';
      return;
    }
    hmmCurrentSweep = data;
    document.getElementById('sb-hmm-sweep-title').textContent = 'Sweep #' + data.id + (data.name ? ' — ' + data.name : '');
    if (!Array.isArray(data.results) || data.results.length === 0) {
      combosBody.innerHTML = '<tr><td colspan="12" class="p-3 text-slate-500 text-center">No combo results saved.</td></tr>';
      return;
    }
    recalcHmmSweepProfit();
  } catch (e) {
    summary.innerHTML = '<p class="text-red-400">Failed: ' + e.message + '</p>';
  }
}

function closeHmmSweepDetail() {
  const wrap = document.getElementById('sb-hmm-sweep-detail');
  if (!wrap) return;
  wrap.classList.add('hidden');
  const summary = document.getElementById('sb-hmm-sweep-summary');
  const combosBody = document.getElementById('sb-hmm-sweep-combos');
  const profitSummary = document.getElementById('sb-hmm-sweep-profit-summary');
  if (summary) summary.innerHTML = '';
  if (combosBody) combosBody.innerHTML = '';
  if (profitSummary) profitSummary.innerHTML = '';
  hmmCurrentSweep = null;
}

function renderHmmMonthlyTable(monthlyData) {
  const tbody = document.getElementById('sb-hmm-monthly-tbody');
  if (!tbody) return;
  tbody.innerHTML = '';

  const entries = Object.entries(monthlyData);
  if (entries.length === 0) {
    tbody.innerHTML = '<tr><td class="p-2 text-slate-500" colspan="7">No signal data</td></tr>';
    return;
  }

  const sorted = entries.sort((a, b) => (a[0] < b[0] ? 1 : -1));
  let html = '';
  for (const [m, d] of sorted) {
    const baseWR = d.total > 0 ? (d.baseline_correct / d.total * 100).toFixed(2) : '0.00';
    const filtWR = d.taken > 0 ? (d.filtered_correct / d.taken * 100).toFixed(2) : '0.00';
    const imp = (parseFloat(filtWR) - parseFloat(baseWR)) || 0;
    const impClass = imp >= 0 ? 'text-green-400' : 'text-red-400';

    html += `<tr class="border-b border-slate-800 hover:bg-slate-800/50">
      <td class="p-2 font-mono">${m}</td>
      <td class="p-2 text-center">${d.total}</td>
      <td class="p-2 text-center text-green-400">${d.taken}</td>
      <td class="p-2 text-center text-orange-400">${d.skipped}</td>
      <td class="p-2 text-center text-slate-400">${baseWR}%</td>
      <td class="p-2 text-center font-bold text-green-400">${filtWR}%</td>
      <td class="p-2 text-center font-bold ${impClass}">${imp >= 0 ? '+' : ''}${imp.toFixed(2)}%</td>
    </tr>`;
  }
  tbody.innerHTML = html;
}

