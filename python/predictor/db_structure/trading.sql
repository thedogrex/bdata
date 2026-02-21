-- phpMyAdmin SQL Dump
-- version 5.2.1
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1:3306
-- Generation Time: Feb 21, 2026 at 02:21 AM
-- Server version: 9.1.0
-- PHP Version: 8.3.14

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
  `daily_json` text,
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
-- Table structure for table `poly_markets`
--

DROP TABLE IF EXISTS `poly_markets`;
CREATE TABLE IF NOT EXISTS `poly_markets` (
  `slug` varchar(255) NOT NULL,
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
  PRIMARY KEY (`slug`),
  KEY `idx_ts` (`ts`),
  KEY `idx_closed` (`closed`)
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
  `name` varchar(255) NOT NULL,
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uq_outcome` (`slug`,`asset_id`),
  KEY `idx_asset` (`asset_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

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
  `outcome_side` varchar(8) DEFAULT NULL,
  `qty` double NOT NULL,
  `requested_price_cents` double DEFAULT NULL,
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
-- Constraints for table `backtest_horizons`
--
ALTER TABLE `backtest_horizons`
  ADD CONSTRAINT `backtest_horizons_ibfk_1` FOREIGN KEY (`run_id`) REFERENCES `backtest_runs` (`id`) ON DELETE CASCADE;

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
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
