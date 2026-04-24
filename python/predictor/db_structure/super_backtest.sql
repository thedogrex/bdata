-- Super Backtest: Detailed per-prediction data for regime analysis
-- Used for Hidden Markov Model and streak pattern analysis

-- --------------------------------------------------------

--
-- Table structure for table `super_backtest_runs`
--

CREATE TABLE IF NOT EXISTS `super_backtest_runs` (
  `id` int NOT NULL AUTO_INCREMENT,
  `strategy` varchar(100) NOT NULL,
  `params_json` json DEFAULT NULL,
  `train_start` varchar(20) NOT NULL,
  `train_end` varchar(20) NOT NULL,
  `test_start` varchar(20) NOT NULL,
  `test_end` varchar(20) NOT NULL,
  `table_name` varchar(50) NOT NULL DEFAULT 'c_5m',
  `window_size` int NOT NULL DEFAULT '0',
  `horizon` int NOT NULL DEFAULT '1',
  `total_candles` int NOT NULL DEFAULT '0',
  `signals` int NOT NULL DEFAULT '0',
  `correct` int NOT NULL DEFAULT '0',
  `wrong` int NOT NULL DEFAULT '0',
  `accuracy_pct` float NOT NULL DEFAULT '0',
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `hmm_states` int NOT NULL DEFAULT '2' COMMENT 'Number of HMM hidden states used',
  `hmm_model_json` json DEFAULT NULL COMMENT 'Serialized HMM model parameters',
  PRIMARY KEY (`id`),
  KEY `idx_strategy` (`strategy`),
  KEY `idx_created` (`created_at`),
  KEY `idx_accuracy` (`accuracy_pct` DESC)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- --------------------------------------------------------

--
-- Table structure for table `super_backtest_predictions`
-- Stores individual prediction results with features for HMM analysis
--

CREATE TABLE IF NOT EXISTS `super_backtest_predictions` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `super_run_id` int NOT NULL,
  `candle_idx` int NOT NULL COMMENT 'Index in the candle dataframe',
  `open_time` bigint NOT NULL COMMENT 'Microseconds timestamp',
  `prediction` tinyint NOT NULL COMMENT '-1=skip, 0=down, 1=up',
  `probability` float NOT NULL DEFAULT '0.5',
  `actual` tinyint NOT NULL COMMENT '0=down, 1=up',
  `is_correct` tinyint(1) NOT NULL DEFAULT '0',
  `is_signal` tinyint(1) NOT NULL DEFAULT '0' COMMENT 'Was not skipped',
  `vol_skip` tinyint(1) NOT NULL DEFAULT '0' COMMENT 'Skipped due to volatility',
  
  -- Technical features at prediction time
  `rsi` float DEFAULT NULL,
  `rsi_percentile` float DEFAULT NULL,
  `bb_position` float DEFAULT NULL,
  `volatility_short` float DEFAULT NULL,
  `volatility_long` float DEFAULT NULL,
  `trend_3c` float DEFAULT NULL,
  `trend_10c` float DEFAULT NULL,
  `volume_ratio` float DEFAULT NULL,
  
  -- Streak features (computed in post-processing)
  `prev_streak_type` tinyint DEFAULT NULL COMMENT '0=lose, 1=win',
  `prev_streak_len` int DEFAULT NULL,
  
  -- HMM state (computed after model fitting)
  `hmm_state` tinyint DEFAULT NULL,
  `hmm_state_prob` float DEFAULT NULL,
  
  PRIMARY KEY (`id`),
  UNIQUE KEY `uq_run_candle` (`super_run_id`, `candle_idx`),
  KEY `idx_run_id` (`super_run_id`),
  KEY `idx_open_time` (`open_time`),
  KEY `idx_prediction` (`prediction`),
  KEY `idx_is_correct` (`is_correct`),
  KEY `idx_hmm_state` (`hmm_state`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- --------------------------------------------------------

--
-- Table structure for table `super_backtest_regimes`
-- Stores detected regime segments from HMM analysis
--

CREATE TABLE IF NOT EXISTS `super_backtest_regimes` (
  `id` int NOT NULL AUTO_INCREMENT,
  `super_run_id` int NOT NULL,
  `state` tinyint NOT NULL COMMENT 'HMM hidden state number',
  `state_label` varchar(20) DEFAULT NULL COMMENT 'good/bad/neutral',
  `start_idx` int NOT NULL,
  `end_idx` int NOT NULL,
  `start_time` bigint NOT NULL,
  `end_time` bigint NOT NULL,
  `predictions_count` int NOT NULL DEFAULT '0',
  `signals_count` int NOT NULL DEFAULT '0',
  `correct_count` int NOT NULL DEFAULT '0',
  `wrong_count` int NOT NULL DEFAULT '0',
  `accuracy_pct` float NOT NULL DEFAULT '0',
  `avg_probability` float NOT NULL DEFAULT '0',
  PRIMARY KEY (`id`),
  KEY `idx_run_state` (`super_run_id`, `state`),
  KEY `idx_run_time` (`super_run_id`, `start_time`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

