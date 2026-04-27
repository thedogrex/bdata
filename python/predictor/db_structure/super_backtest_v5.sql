ALTER TABLE `super_backtest_hmm_sweep_results`
ADD COLUMN `monthly_json` longtext DEFAULT NULL AFTER `state_labels_json`;
