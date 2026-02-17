-- ============================================
-- Candle Predictor: MySQL Migration
-- Database: trading
-- ============================================

-- Backtest runs (one row per backtest execution)
CREATE TABLE IF NOT EXISTS backtest_runs (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    strategy        VARCHAR(64)     NOT NULL,
    params_json     TEXT            NOT NULL,
    train_start     VARCHAR(16)     NOT NULL,
    train_end       VARCHAR(16)     NOT NULL,
    test_start      VARCHAR(16)     NOT NULL,
    test_end        VARCHAR(16)     NOT NULL,
    tbl             VARCHAR(32)     NOT NULL DEFAULT 'c_5m',
    window_size     INT             NOT NULL DEFAULT 5000,
    train_candles   INT             NOT NULL DEFAULT 0,
    test_candles    INT             NOT NULL DEFAULT 0,
    total_time_sec  FLOAT           NOT NULL DEFAULT 0,
    is_bruteforce   TINYINT(1)      NOT NULL DEFAULT 0,
    bruteforce_id   INT             DEFAULT NULL,
    created_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_strategy (strategy),
    INDEX idx_created (created_at),
    INDEX idx_bruteforce (bruteforce_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Per-horizon results (one row per horizon per backtest run)
CREATE TABLE IF NOT EXISTS backtest_horizons (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    run_id          INT             NOT NULL,
    horizon         INT             NOT NULL,
    accuracy        FLOAT           NOT NULL DEFAULT 0,
    accuracy_pct    FLOAT           NOT NULL DEFAULT 0,
    total_candles   INT             NOT NULL DEFAULT 0,
    signals         INT             NOT NULL DEFAULT 0,
    skipped         INT             NOT NULL DEFAULT 0,
    correct         INT             NOT NULL DEFAULT 0,
    wrong           INT             NOT NULL DEFAULT 0,
    up_predictions  INT             NOT NULL DEFAULT 0,
    up_correct      INT             NOT NULL DEFAULT 0,
    up_accuracy     FLOAT           NOT NULL DEFAULT 0,
    down_predictions INT            NOT NULL DEFAULT 0,
    down_correct    INT             NOT NULL DEFAULT 0,
    down_accuracy   FLOAT           NOT NULL DEFAULT 0,
    max_win_streak  INT             NOT NULL DEFAULT 0,
    max_lose_streak INT             NOT NULL DEFAULT 0,
    fit_time_sec    FLOAT           NOT NULL DEFAULT 0,
    predict_time_sec FLOAT          NOT NULL DEFAULT 0,
    monthly_json    TEXT,
    daily_json      TEXT,
    confidence_json TEXT,
    FOREIGN KEY (run_id) REFERENCES backtest_runs(id) ON DELETE CASCADE,
    INDEX idx_run (run_id),
    INDEX idx_accuracy (accuracy_pct DESC)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Brute-force sessions
CREATE TABLE IF NOT EXISTS bruteforce_sessions (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    strategy        VARCHAR(64)     NOT NULL,
    param_grid_json TEXT            NOT NULL,
    train_start     VARCHAR(16)     NOT NULL,
    train_end       VARCHAR(16)     NOT NULL,
    test_start      VARCHAR(16)     NOT NULL,
    test_end        VARCHAR(16)     NOT NULL,
    tbl             VARCHAR(32)     NOT NULL DEFAULT 'c_5m',
    horizon         INT             NOT NULL DEFAULT 1,
    window_size     INT             NOT NULL DEFAULT 5000,
    total_combos    INT             NOT NULL DEFAULT 0,
    completed       INT             NOT NULL DEFAULT 0,
    best_accuracy   FLOAT           NOT NULL DEFAULT 0,
    best_params_json TEXT,
    status          VARCHAR(16)     NOT NULL DEFAULT 'pending',
    total_time_sec  FLOAT           NOT NULL DEFAULT 0,
    created_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_status (status),
    INDEX idx_best (best_accuracy DESC)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ============================================
-- v2: Brute-force checkpoint persistence
-- Allows pause → shutdown → restart → resume
-- ============================================

ALTER TABLE bruteforce_sessions
    ADD COLUMN IF NOT EXISTS retrain_every INT NOT NULL DEFAULT 500 AFTER window_size,
    ADD COLUMN IF NOT EXISTS combos_json LONGTEXT AFTER total_combos,
    ADD COLUMN IF NOT EXISTS elapsed_before_pause FLOAT NOT NULL DEFAULT 0 AFTER total_time_sec;
