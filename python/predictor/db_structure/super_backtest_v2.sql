-- Super Backtest v2: Multiple HMM analyses per run + timeline support
-- Run this AFTER super_backtest.sql
-- Safe to run multiple times (IF NOT EXISTS / IF EXISTS guards).

-- --------------------------------------------------------
-- Table: super_backtest_hmm_analyses
-- Stores metadata for each HMM configuration applied to a run.
-- One super_backtest_run can have N analyses (n_states, feature_set, fit_mode ...).
-- --------------------------------------------------------

CREATE TABLE IF NOT EXISTS `super_backtest_hmm_analyses` (
  `id` int NOT NULL AUTO_INCREMENT,
  `super_run_id` int NOT NULL,
  `name` varchar(100) DEFAULT NULL COMMENT 'User label, e.g. "2-state rsi+vol"',
  `n_states` int NOT NULL DEFAULT 2,
  `feature_set` text NOT NULL COMMENT 'JSON array of feature names',
  `fit_mode` varchar(20) NOT NULL DEFAULT 'all_candles'
      COMMENT 'all_candles | signals_only | walk_forward',
  `walk_train_len` int DEFAULT NULL COMMENT 'For walk_forward mode',
  `walk_step` int DEFAULT NULL COMMENT 'For walk_forward mode',
  `good_threshold` float NOT NULL DEFAULT 55.0,
  `bad_threshold` float NOT NULL DEFAULT 45.0,
  `filter_threshold` float NOT NULL DEFAULT 0.6,
  `min_regime_len` int NOT NULL DEFAULT 1 COMMENT 'Min contiguous candles to keep regime',
  `states_json` longtext DEFAULT NULL COMMENT 'Per-state stats (accuracy, feature means, label)',
  `transition_matrix_json` text DEFAULT NULL,
  `model_json` longtext DEFAULT NULL COMMENT 'HMM means/covars/transmat',
  `baseline_winrate` float DEFAULT NULL,
  `filtered_winrate` float DEFAULT NULL,
  `improvement` float DEFAULT NULL,
  `trades_total` int DEFAULT NULL,
  `trades_taken` int DEFAULT NULL,
  `trades_skipped` int DEFAULT NULL,
  `candles_analyzed` int DEFAULT NULL COMMENT 'Total candles fed to HMM',
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `idx_run_id` (`super_run_id`),
  KEY `idx_created` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- --------------------------------------------------------
-- Table: super_backtest_prediction_states
-- Per-analysis HMM state labels for every candle (including non-signals).
-- Enables comparing multiple analyses on same run without overwriting.
-- --------------------------------------------------------

CREATE TABLE IF NOT EXISTS `super_backtest_prediction_states` (
  `hmm_analysis_id` int NOT NULL,
  `candle_idx` int NOT NULL,
  `open_time` bigint NOT NULL,
  `hmm_state` tinyint NOT NULL,
  `hmm_state_prob` float NOT NULL DEFAULT 0,
  PRIMARY KEY (`hmm_analysis_id`, `candle_idx`),
  KEY `idx_analysis` (`hmm_analysis_id`),
  KEY `idx_open_time` (`open_time`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
