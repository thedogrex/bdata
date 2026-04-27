-- Super Backtest v3 incremental migration
-- Adds is_skipped flag to prediction states so UI can display filtered vs skipped trades.

ALTER TABLE `super_backtest_prediction_states`
  ADD COLUMN IF NOT EXISTS `is_skipped` tinyint(1) NOT NULL DEFAULT 0
    COMMENT '1 if trade filtered out by HMM' AFTER `hmm_state_prob`;
