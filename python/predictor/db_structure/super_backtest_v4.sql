-- Super Backtest v4 incremental migration
-- Adds storage for HMM threshold sweep requests and their combo results.

CREATE TABLE IF NOT EXISTS `super_backtest_hmm_sweeps` (
  `id` int NOT NULL AUTO_INCREMENT,
  `super_run_id` int NOT NULL,
  `name` varchar(120) DEFAULT NULL,
  `n_states` int NOT NULL DEFAULT 2,
  `feature_set` text NOT NULL COMMENT 'JSON array of feature names',
  `fit_mode` varchar(20) NOT NULL DEFAULT 'all_candles'
      COMMENT 'all_candles | signals_only | walk_forward',
  `walk_train_len` int DEFAULT NULL,
  `walk_step` int DEFAULT NULL,
  `min_regime_len` int NOT NULL DEFAULT 1,
  `combos_total` int NOT NULL DEFAULT 0 COMMENT 'How many threshold combinations were evaluated',
  `baseline_trades` int DEFAULT NULL,
  `baseline_correct` int DEFAULT NULL,
  `baseline_winrate` float DEFAULT NULL,
  `candles_analyzed` int DEFAULT NULL,
  `states_json` longtext DEFAULT NULL,
  `transition_matrix_json` text DEFAULT NULL,
  `model_json` longtext DEFAULT NULL,
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `idx_hmm_sweeps_run` (`super_run_id`),
  KEY `idx_hmm_sweeps_created` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS `super_backtest_hmm_sweep_results` (
  `id` int NOT NULL AUTO_INCREMENT,
  `sweep_id` int NOT NULL,
  `super_run_id` int NOT NULL,
  `combo_index` int NOT NULL COMMENT '1-based index of the evaluated combo',
  `good_threshold` float NOT NULL,
  `bad_threshold` float NOT NULL,
  `filter_threshold` float NOT NULL,
  `baseline_winrate` float DEFAULT NULL,
  `baseline_trades` int DEFAULT NULL,
  `baseline_correct` int DEFAULT NULL,
  `filtered_winrate` float DEFAULT NULL,
  `filtered_trades` int DEFAULT NULL,
  `filtered_correct` int DEFAULT NULL,
  `trades_skipped` int DEFAULT NULL,
  `improvement` float DEFAULT NULL,
  `state_labels_json` text DEFAULT NULL,
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `idx_hmm_sweep_id` (`sweep_id`),
  KEY `idx_hmm_sweep_run` (`super_run_id`),
  KEY `idx_hmm_sweep_winrate` (`filtered_winrate`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
