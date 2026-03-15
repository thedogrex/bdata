-- phpMyAdmin SQL Dump
-- version 5.2.3
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1:3306
-- Generation Time: Mar 15, 2026 at 08:33 AM
-- Server version: 8.4.7
-- PHP Version: 8.3.28

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `trading`
--

-- --------------------------------------------------------

--
-- Table structure for table `backtest_horizons`
--

DROP TABLE IF EXISTS `backtest_horizons`;
CREATE TABLE IF NOT EXISTS `backtest_horizons` (
  `id` int NOT NULL AUTO_INCREMENT,
  `run_id` int NOT NULL,
  `horizon` int NOT NULL,
  `accuracy` float NOT NULL DEFAULT '0',
  `accuracy_pct` float NOT NULL DEFAULT '0',
  `total_candles` int NOT NULL DEFAULT '0',
  `signals` int NOT NULL DEFAULT '0',
  `skipped` int NOT NULL DEFAULT '0',
  `correct` int NOT NULL DEFAULT '0',
  `wrong` int NOT NULL DEFAULT '0',
  `up_predictions` int NOT NULL DEFAULT '0',
  `up_correct` int NOT NULL DEFAULT '0',
  `up_accuracy` float NOT NULL DEFAULT '0',
  `down_predictions` int NOT NULL DEFAULT '0',
  `down_correct` int NOT NULL DEFAULT '0',
  `down_accuracy` float NOT NULL DEFAULT '0',
  `max_win_streak` int NOT NULL DEFAULT '0',
  `max_lose_streak` int NOT NULL DEFAULT '0',
  `fit_time_sec` float NOT NULL DEFAULT '0',
  `predict_time_sec` float NOT NULL DEFAULT '0',
  `monthly_json` text,
  `daily_json` longtext,
  `confidence_json` text,
  PRIMARY KEY (`id`),
  KEY `idx_run` (`run_id`),
  KEY `idx_accuracy` (`accuracy_pct` DESC)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- --------------------------------------------------------

--
-- Table structure for table `backtest_runs`
--

DROP TABLE IF EXISTS `backtest_runs`;
CREATE TABLE IF NOT EXISTS `backtest_runs` (
  `id` int NOT NULL AUTO_INCREMENT,
  `strategy` varchar(64) NOT NULL,
  `params_json` text NOT NULL,
  `train_start` varchar(16) NOT NULL,
  `train_end` varchar(16) NOT NULL,
  `test_start` varchar(16) NOT NULL,
  `test_end` varchar(16) NOT NULL,
  `tbl` varchar(32) NOT NULL DEFAULT 'c_5m',
  `window_size` int NOT NULL DEFAULT '5000',
  `train_candles` int NOT NULL DEFAULT '0',
  `test_candles` int NOT NULL DEFAULT '0',
  `total_time_sec` float NOT NULL DEFAULT '0',
  `is_bruteforce` tinyint(1) NOT NULL DEFAULT '0',
  `bruteforce_id` int DEFAULT NULL,
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `idx_strategy` (`strategy`),
  KEY `idx_created` (`created_at`),
  KEY `idx_bruteforce` (`bruteforce_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- --------------------------------------------------------

--
-- Table structure for table `bruteforce_sessions`
--

DROP TABLE IF EXISTS `bruteforce_sessions`;
CREATE TABLE IF NOT EXISTS `bruteforce_sessions` (
  `id` int NOT NULL AUTO_INCREMENT,
  `strategy` varchar(64) NOT NULL,
  `param_grid_json` text NOT NULL,
  `train_start` varchar(16) NOT NULL,
  `train_end` varchar(16) NOT NULL,
  `test_start` varchar(16) NOT NULL,
  `test_end` varchar(16) NOT NULL,
  `tbl` varchar(32) NOT NULL DEFAULT 'c_5m',
  `horizon` int NOT NULL DEFAULT '1',
  `window_size` int NOT NULL DEFAULT '5000',
  `retrain_every` int NOT NULL DEFAULT '500',
  `total_combos` int NOT NULL DEFAULT '0',
  `combos_json` longtext,
  `completed` int NOT NULL DEFAULT '0',
  `best_accuracy` float NOT NULL DEFAULT '0',
  `best_params_json` text,
  `status` varchar(16) NOT NULL DEFAULT 'pending',
  `total_time_sec` float NOT NULL DEFAULT '0',
  `elapsed_before_pause` float NOT NULL DEFAULT '0',
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `idx_status` (`status`),
  KEY `idx_best` (`best_accuracy` DESC)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- --------------------------------------------------------

--
-- Table structure for table `c_5m`
--

DROP TABLE IF EXISTS `c_5m`;
CREATE TABLE IF NOT EXISTS `c_5m` (
  `id` int NOT NULL AUTO_INCREMENT,
  `open_time` bigint NOT NULL,
  `open` float NOT NULL,
  `high` float NOT NULL,
  `low` float NOT NULL,
  `close` float NOT NULL,
  `volume` float NOT NULL,
  `close_time` bigint NOT NULL,
  `quota_volume` float NOT NULL,
  `trades` int NOT NULL,
  `taker_base_volume` float NOT NULL,
  `taker_quota_volume` float NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `saqx` (`open_time`)
) ENGINE=MyISAM DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `poly_daily_balance_digest`
--

DROP TABLE IF EXISTS `poly_daily_balance_digest`;
CREATE TABLE IF NOT EXISTS `poly_daily_balance_digest` (
  `digest_date` date NOT NULL,
  `start_balance_usd` double DEFAULT NULL,
  `report_sent_at` datetime DEFAULT NULL,
  `digest_sent` tinyint(1) NOT NULL DEFAULT '0',
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`digest_date`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- --------------------------------------------------------

--
-- Table structure for table `poly_live_orders`
--

DROP TABLE IF EXISTS `poly_live_orders`;
CREATE TABLE IF NOT EXISTS `poly_live_orders` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `slug` varchar(255) NOT NULL COMMENT 'Market slug',
  `asset_id` varchar(128) NOT NULL COMMENT 'CLOB token ID',
  `outcome_side` varchar(8) DEFAULT NULL COMMENT 'UP or DOWN',
  `side` varchar(8) NOT NULL COMMENT 'BUY or SELL',
  `order_type` varchar(16) NOT NULL DEFAULT 'FOK' COMMENT 'FOK, GTC, GTD, FAK',
  `price` double NOT NULL COMMENT 'Price per share (0..1)',
  `amount` double NOT NULL COMMENT 'Dollar amount (BUY) or shares (SELL)',
  `fill_shares` double DEFAULT NULL,
  `fill_total_spent_usd` double DEFAULT NULL,
  `fill_avg_price_cents` double DEFAULT NULL,
  `clob_order_id` varchar(128) DEFAULT NULL COMMENT 'Order ID returned by CLOB',
  `clob_status` varchar(32) DEFAULT NULL COMMENT 'live, matched, delayed, unmatched, error',
  `clob_error_msg` text COMMENT 'Error message if any',
  `clob_response_json` json DEFAULT NULL COMMENT 'Full CLOB API response',
  `prediction_batch_id` varchar(64) DEFAULT NULL COMMENT 'Link to poly_pred_runs batch',
  `template_id` int DEFAULT NULL COMMENT 'Prediction template that triggered this order',
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `idx_slug` (`slug`),
  KEY `idx_asset` (`asset_id`),
  KEY `idx_clob_order` (`clob_order_id`),
  KEY `idx_batch` (`prediction_batch_id`),
  KEY `idx_created` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- --------------------------------------------------------

--
-- Table structure for table `poly_live_positions`
--

DROP TABLE IF EXISTS `poly_live_positions`;
CREATE TABLE IF NOT EXISTS `poly_live_positions` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `slug` varchar(255) NOT NULL COMMENT 'Market slug',
  `asset_id` varchar(128) NOT NULL COMMENT 'CLOB token ID',
  `outcome_side` varchar(8) DEFAULT NULL COMMENT 'UP or DOWN',
  `shares` double NOT NULL DEFAULT '0' COMMENT 'Current share count',
  `avg_price` double NOT NULL DEFAULT '0' COMMENT 'Volume-weighted avg entry price',
  `total_cost` double NOT NULL DEFAULT '0' COMMENT 'Total USD spent (cost basis)',
  `status` varchar(16) NOT NULL DEFAULT 'open' COMMENT 'open, closed, resolved',
  `resolved_outcome` varchar(16) DEFAULT NULL COMMENT 'Final market outcome if resolved',
  `pnl` double DEFAULT NULL COMMENT 'Realised P&L once closed/resolved',
  `snapshot_price_cents` double DEFAULT NULL COMMENT 'Snapshot ask price used for the buy',
  `prediction_direction` varchar(8) DEFAULT NULL COMMENT 'UP or DOWN — what the model predicted',
  `prediction_batch_id` varchar(64) DEFAULT NULL COMMENT 'Link to poly_pred_runs batch',
  `template_id` int DEFAULT NULL COMMENT 'Prediction template that triggered this position',
  `opened_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `closed_at` datetime DEFAULT NULL,
  `updated_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uq_position` (`slug`,`asset_id`,`status`),
  KEY `idx_slug` (`slug`),
  KEY `idx_status` (`status`),
  KEY `idx_opened` (`opened_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- --------------------------------------------------------

--
-- Table structure for table `poly_live_trade_settings`
--

DROP TABLE IF EXISTS `poly_live_trade_settings`;
CREATE TABLE IF NOT EXISTS `poly_live_trade_settings` (
  `id` varchar(32) NOT NULL DEFAULT 'default',
  `auto_place` tinyint(1) NOT NULL DEFAULT '0',
  `bet_size_usd` double NOT NULL DEFAULT '5',
  `price_cap_cents` int NOT NULL DEFAULT '52',
  `updated_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- --------------------------------------------------------

--
-- Table structure for table `poly_markets`
--

DROP TABLE IF EXISTS `poly_markets`;
CREATE TABLE IF NOT EXISTS `poly_markets` (
  `slug` varchar(255) NOT NULL,
  `condition_id` varchar(66) DEFAULT NULL,
  `parent_collection_id` varchar(66) DEFAULT NULL,
  `ts` int NOT NULL,
  `end_date` varchar(64) DEFAULT NULL,
  `question` text,
  `description` text,
  `closed` tinyint(1) NOT NULL DEFAULT '0',
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  `resolved_outcome` varchar(16) DEFAULT NULL,
  `last_resolution_check_ts` int DEFAULT NULL,
  `prediction_outcome` varchar(16) DEFAULT NULL,
  `prediction_ts` int DEFAULT NULL,
  `pred_votes` longtext,
  PRIMARY KEY (`slug`),
  KEY `idx_ts` (`ts`),
  KEY `idx_closed` (`closed`),
  KEY `idx_condition` (`condition_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- --------------------------------------------------------

--
-- Table structure for table `poly_orderbook_snapshots`
--

DROP TABLE IF EXISTS `poly_orderbook_snapshots`;
CREATE TABLE IF NOT EXISTS `poly_orderbook_snapshots` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `slug` varchar(255) NOT NULL,
  `asset_id` varchar(128) NOT NULL,
  `ts` int NOT NULL,
  `best_bid_cents` double DEFAULT NULL,
  `best_ask_cents` double DEFAULT NULL,
  `mid_cents` double DEFAULT NULL,
  `bids_json` json DEFAULT NULL,
  `asks_json` json DEFAULT NULL,
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `idx_asset_ts` (`asset_id`,`ts`),
  KEY `idx_slug_ts` (`slug`,`ts`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- --------------------------------------------------------

--
-- Table structure for table `poly_outcomes`
--

DROP TABLE IF EXISTS `poly_outcomes`;
CREATE TABLE IF NOT EXISTS `poly_outcomes` (
  `id` int NOT NULL AUTO_INCREMENT,
  `slug` varchar(255) NOT NULL,
  `asset_id` varchar(128) NOT NULL,
  `index_set` int UNSIGNED NOT NULL DEFAULT '0',
  `payout_vector` tinyint DEFAULT NULL,
  `name` varchar(255) NOT NULL,
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uq_outcome` (`slug`,`asset_id`),
  KEY `idx_asset` (`asset_id`),
  KEY `idx_slug_index` (`slug`,`index_set`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- --------------------------------------------------------

--
-- Table structure for table `poly_predictions`
--

DROP TABLE IF EXISTS `poly_predictions`;
CREATE TABLE IF NOT EXISTS `poly_predictions` (
  `slug` varchar(255) NOT NULL,
  `prediction_ts` int NOT NULL,
  `payload_json` longtext NOT NULL,
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`slug`),
  KEY `idx_prediction_ts` (`prediction_ts`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- --------------------------------------------------------

--
-- Table structure for table `poly_predictions_past15`
--

DROP TABLE IF EXISTS `poly_predictions_past15`;
CREATE TABLE IF NOT EXISTS `poly_predictions_past15` (
  `id` bigint UNSIGNED NOT NULL AUTO_INCREMENT,
  `slug` varchar(255) NOT NULL,
  `prediction_ts` int NOT NULL,
  `payload_json` longtext NOT NULL,
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `idx_past15_slug_ts` (`slug`,`prediction_ts`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- --------------------------------------------------------

--
-- Table structure for table `poly_pred_runs`
--

DROP TABLE IF EXISTS `poly_pred_runs`;
CREATE TABLE IF NOT EXISTS `poly_pred_runs` (
  `id` bigint UNSIGNED NOT NULL AUTO_INCREMENT,
  `slug` varchar(255) NOT NULL,
  `batch_id` char(36) NOT NULL,
  `template_id` int UNSIGNED DEFAULT NULL,
  `template_name` varchar(100) DEFAULT NULL,
  `strategy` varchar(50) DEFAULT NULL,
  `params_json` longtext,
  `window_size` int DEFAULT NULL,
  `horizon` tinyint UNSIGNED DEFAULT NULL,
  `quantum` tinyint UNSIGNED NOT NULL DEFAULT '0',
  `quantum_scenario` varchar(10) DEFAULT NULL,
  `prediction` varchar(10) DEFAULT NULL,
  `probability` double DEFAULT NULL,
  `started_at` datetime(3) NOT NULL,
  `finished_at` datetime(3) DEFAULT NULL,
  `duration_ms` int UNSIGNED DEFAULT NULL,
  `error` text,
  `result_json` longtext,
  PRIMARY KEY (`id`),
  KEY `idx_slug` (`slug`),
  KEY `idx_batch` (`batch_id`),
  KEY `idx_slug_started` (`slug`,`started_at`),
  KEY `idx_template` (`template_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- --------------------------------------------------------

--
-- Table structure for table `poly_pred_templates`
--

DROP TABLE IF EXISTS `poly_pred_templates`;
CREATE TABLE IF NOT EXISTS `poly_pred_templates` (
  `id` int NOT NULL AUTO_INCREMENT,
  `name` varchar(100) COLLATE utf8mb4_general_ci NOT NULL,
  `strategy` varchar(50) COLLATE utf8mb4_general_ci NOT NULL DEFAULT 'rsi_mean_reversion',
  `params_json` text COLLATE utf8mb4_general_ci,
  `window_size` int NOT NULL DEFAULT '1000',
  `horizon` int NOT NULL DEFAULT '1',
  `active` tinyint NOT NULL DEFAULT '1',
  `sort_order` int NOT NULL DEFAULT '0',
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`)
) ENGINE=MyISAM DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `poly_settings`
--

DROP TABLE IF EXISTS `poly_settings`;
CREATE TABLE IF NOT EXISTS `poly_settings` (
  `id` varchar(32) NOT NULL DEFAULT 'default',
  `autopredict` tinyint(1) NOT NULL DEFAULT '0',
  `strategy` varchar(64) NOT NULL DEFAULT 'rsi_mean_reversion',
  `params_json` text,
  `window_size` int NOT NULL DEFAULT '1000',
  `updated_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- --------------------------------------------------------

--
-- Table structure for table `poly_sim_trades`
--

DROP TABLE IF EXISTS `poly_sim_trades`;
CREATE TABLE IF NOT EXISTS `poly_sim_trades` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `ts` int NOT NULL,
  `slug` varchar(255) NOT NULL,
  `asset_id` varchar(128) NOT NULL,
  `side` varchar(8) NOT NULL,
  `outcome_side` varchar(8) NOT NULL DEFAULT 'NONE',
  `qty` double NOT NULL,
  `fill_price_cents` double NOT NULL,
  `snapshot_ts` int DEFAULT NULL,
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `idx_ts` (`ts`),
  KEY `idx_asset` (`asset_id`),
  KEY `idx_slug` (`slug`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

--
-- Constraints for dumped tables
--

--
-- Constraints for table `poly_orderbook_snapshots`
--
ALTER TABLE `poly_orderbook_snapshots`
  ADD CONSTRAINT `poly_orderbook_snapshots_ibfk_1` FOREIGN KEY (`slug`) REFERENCES `poly_markets` (`slug`) ON DELETE CASCADE;

--
-- Constraints for table `poly_outcomes`
--
ALTER TABLE `poly_outcomes`
  ADD CONSTRAINT `poly_outcomes_ibfk_1` FOREIGN KEY (`slug`) REFERENCES `poly_markets` (`slug`) ON DELETE CASCADE;

--
-- Constraints for table `poly_predictions`
--
ALTER TABLE `poly_predictions`
  ADD CONSTRAINT `fk_poly_predictions_slug` FOREIGN KEY (`slug`) REFERENCES `poly_markets` (`slug`) ON DELETE CASCADE;

--
-- Constraints for table `poly_predictions_past15`
--
ALTER TABLE `poly_predictions_past15`
  ADD CONSTRAINT `fk_poly_predictions_past15_slug` FOREIGN KEY (`slug`) REFERENCES `poly_markets` (`slug`) ON DELETE CASCADE;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
