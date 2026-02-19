import asyncio
from db import DbProvider

db = DbProvider()

STATEMENTS = [
    """
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
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
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
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
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
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,

    """
    CREATE TABLE IF NOT EXISTS poly_markets (
        slug        VARCHAR(255) PRIMARY KEY,
        ts          INT NOT NULL,
        end_date    VARCHAR(64),
        question    TEXT,
        description TEXT,
        closed      TINYINT(1) NOT NULL DEFAULT 0,
        resolved_outcome VARCHAR(16) NULL,
        last_resolution_check_ts INT NULL,
        prediction_outcome VARCHAR(16) NULL,
        prediction_ts INT NULL,
        created_at  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        INDEX idx_ts (ts),
        INDEX idx_closed (closed)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,

    """
    ALTER TABLE poly_markets
        ADD COLUMN resolved_outcome VARCHAR(16) NULL;
    """,

    """
    ALTER TABLE poly_markets
        ADD COLUMN last_resolution_check_ts INT NULL;
    """,

    """
    ALTER TABLE poly_markets
        ADD COLUMN prediction_outcome VARCHAR(16) NULL;
    """,

    """
    ALTER TABLE poly_markets
        ADD COLUMN prediction_ts INT NULL;
    """,

    """
    CREATE TABLE IF NOT EXISTS poly_outcomes (
        id         INT AUTO_INCREMENT PRIMARY KEY,
        slug       VARCHAR(255) NOT NULL,
        asset_id   VARCHAR(128) NOT NULL,
        name       VARCHAR(255) NOT NULL,
        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        UNIQUE KEY uq_outcome (slug, asset_id),
        INDEX idx_asset (asset_id),
        FOREIGN KEY (slug) REFERENCES poly_markets(slug) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,

    """
    CREATE TABLE IF NOT EXISTS poly_orderbook_snapshots (
        id              BIGINT AUTO_INCREMENT PRIMARY KEY,
        slug            VARCHAR(255) NOT NULL,
        asset_id        VARCHAR(128) NOT NULL,
        ts              INT NOT NULL,
        best_bid_cents  DOUBLE NULL,
        best_ask_cents  DOUBLE NULL,
        mid_cents       DOUBLE NULL,
        bids_json       JSON,
        asks_json       JSON,
        created_at      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_asset_ts (asset_id, ts),
        INDEX idx_slug_ts (slug, ts),
        FOREIGN KEY (slug) REFERENCES poly_markets(slug) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,

    """
    CREATE TABLE IF NOT EXISTS poly_sim_trades (
        id               BIGINT AUTO_INCREMENT PRIMARY KEY,
        ts               INT NOT NULL,
        slug             VARCHAR(255) NOT NULL,
        asset_id         VARCHAR(128) NOT NULL,
        side             VARCHAR(8) NOT NULL,
        qty              DOUBLE NOT NULL,
        fill_price_cents DOUBLE NOT NULL,
        snapshot_ts      INT NULL,
        created_at       DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_ts (ts),
        INDEX idx_asset (asset_id),
        INDEX idx_slug (slug)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
]


async def main():
    for i, stmt in enumerate(STATEMENTS):
        print(f"Running statement {i+1}/{len(STATEMENTS)}...")
        try:
            await db.execute(stmt)
            print(f"  OK")
        except Exception as e:
            msg = str(e)
            # Old MySQL doesn't support IF NOT EXISTS for ADD COLUMN;
            # make migrations idempotent by ignoring duplicate-column errors.
            if "Duplicate column name" in msg or "1060" in msg:
                print(f"  SKIP (already applied)")
                continue
            raise
    print("Migration complete!")


if __name__ == "__main__":
    asyncio.run(main())
