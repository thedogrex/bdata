-- ============================================
-- v2: Brute-force checkpoint persistence
-- Allows pause → shutdown → restart → resume
-- Run this against the 'trading' database
-- ============================================

-- Add retrain_every column
ALTER TABLE bruteforce_sessions
    ADD COLUMN retrain_every INT NOT NULL DEFAULT 500 AFTER window_size;

-- Add combos_json column
ALTER TABLE bruteforce_sessions
    ADD COLUMN combos_json LONGTEXT AFTER total_combos;

-- Add elapsed_before_pause column
ALTER TABLE bruteforce_sessions
    ADD COLUMN elapsed_before_pause FLOAT NOT NULL DEFAULT 0 AFTER total_time_sec;
