"""Collect ~8s-before-close Binance 5m candle snapshots into c_5m_8s.

Uses the Binance USDS-M Futures REST Kline endpoint so that we can query the
current 5-minute candle repeatedly and store a snapshot shortly before it
closes. Can run standalone (``python -m predictor.binance_snapshot``) or as a
background task launched from the FastAPI server startup hook.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Dict, Optional, Set

import requests
import websockets

import app.config as config

from db import DbProvider

LOGGER = logging.getLogger("binance_snapshot")
LOGGER.setLevel(logging.DEBUG)
if not LOGGER.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter(
            "[%(name)s %(levelname)s %(asctime)s] %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    LOGGER.addHandler(_handler)
LOGGER.propagate = False

BINANCE_FUTURES_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"
BINANCE_WS_BASE_URL = "wss://stream.binance.com:9443/ws"
SYMBOL = "BTCUSDT"
INTERVAL = "5m"
SNAPSHOT_TARGETS = {
    8_000: "c_5m_8s",
    7_000: "c_5m_7s",
    5_000: "c_5m_5s",
    4_000: "c_5m_4s",
    3_000: "c_5m_3s",
}
POLL_INTERVAL_SEC = 1.0
RETRY_DELAY_SEC = 5.0
CACHE_LIMIT = 1500


_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS `{table}` (
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
  UNIQUE KEY `{unique_key}` (`open_time`)
) ENGINE=MyISAM DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
"""


class SnapshotCollector:
    """Polls Binance REST klines and stores a near-close snapshot once per candle."""

    def __init__(self, symbol: str = SYMBOL, interval: str = INTERVAL):
        self.symbol = symbol.upper()
        self.interval = interval
        self.db = DbProvider()
        self._stop_event = asyncio.Event()
        self._seen_snapshots: Dict[int, Set[int]] = {
            lead_ms: set() for lead_ms in SNAPSHOT_TARGETS
        }
        self._debug_price = bool(getattr(config, "DEBUG_BINANCE_PRICE", False))
        self._last_price_log_ms = 0
        self._lat_sum_ms = 0.0
        self._lat_count = 0
        self._lat_window_started_s = int(time.time())

    async def run_forever(self) -> None:
        await self._ensure_tables()
        LOGGER.info(
            "Starting Binance REST snapshot collector for %s %s", self.symbol, self.interval
        )
        while not self._stop_event.is_set():
            try:
                await self._run_ws_loop()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - network errors
                LOGGER.warning("Snapshot loop error: %s", exc)
                await asyncio.sleep(RETRY_DELAY_SEC)

    async def stop(self) -> None:
        self._stop_event.set()

    async def _ensure_tables(self) -> None:
        for lead_ms, table in SNAPSHOT_TARGETS.items():
            unique_key = f"uniq_snapshot_{lead_ms // 1000}s"
            await self.db.execute(
                _CREATE_TABLE_SQL.format(table=table, unique_key=unique_key)
            )

    async def _run_ws_loop(self) -> None:
        stream = f"{BINANCE_WS_BASE_URL}/{self.symbol.lower()}@kline_{self.interval}"
        LOGGER.info("Connecting to Binance websocket stream %s", stream)
        async with websockets.connect(stream, ping_interval=20, ping_timeout=10) as ws:
            LOGGER.info("Binance websocket connected")
            while not self._stop_event.is_set():
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=30)
                except asyncio.TimeoutError:
                    await ws.ping()
                    continue

                data = json.loads(msg)
                kline = data.get("k")
                if not kline:
                    continue

                event_time = int(data.get("E", 0))
                if event_time:
                    latency_ms = self._now_ms() - event_time
                    self._lat_sum_ms += float(latency_ms)
                    self._lat_count += 1
                    now_s = int(time.time())
                    if now_s - self._lat_window_started_s >= 60:
                        avg_ms = (self._lat_sum_ms / self._lat_count) if self._lat_count > 0 else 0.0
                        LOGGER.info(
                            "[binance_snapshot] websocket event latency avg=%.1f ms (%d events/60s)",
                            avg_ms,
                            int(self._lat_count),
                        )
                        self._lat_sum_ms = 0.0
                        self._lat_count = 0
                        self._lat_window_started_s = now_s

                await self._process_kline(kline)

    async def _process_kline(self, kline: dict) -> None:
        open_time_ms = int(kline["t"])
        close_time_ms = int(kline["T"])
        now_ms = self._now_ms()
        time_to_close_ms = close_time_ms - now_ms

        if self._debug_price and now_ms - self._last_price_log_ms >= 1000:
            price = float(kline["c"])
            LOGGER.info(
                "[binance_snapshot] price=%.2f open_time=%s closes_in=%.1fs",
                price,
                time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(open_time_ms / 1000)),
                max(time_to_close_ms / 1000, 0),
            )
            self._last_price_log_ms = now_ms

        for lead_ms, table in SNAPSHOT_TARGETS.items():
            seen = self._seen_snapshots[lead_ms]
            if open_time_ms in seen:
                continue
            if time_to_close_ms > lead_ms:
                continue

            fields = self._map_fields(kline)
            await self._store_snapshot(fields, table)
            seen.add(open_time_ms)
            if len(seen) > CACHE_LIMIT:
                self._seen_snapshots[lead_ms] = set(sorted(seen)[-CACHE_LIMIT:])

    async def _store_snapshot(self, fields: dict, table: str) -> None:
        result = await self.db.insert_one(table, fields=fields, ignore=True, print_query=False)
        if result >= 0:
            LOGGER.info(
                "Stored %s snapshot for candle %s (%s) close=%.2f",
                table,
                fields["open_time"],
                time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(fields["open_time"] / 1_000_000)),
                fields["close"],
            )
            if table == "c_5m_5s":
                asyncio.create_task(self._trigger_4s_predict(fields["open_time"], fields))

    async def _trigger_4s_predict(self, open_time_us: int, live_fields: dict) -> None:
        """Fire 5s-early autopredict for the latest possible market (horizon=2).

        With horizon=2 the signal candle predicts 2 intervals ahead, so the
        single target market is at  open_time + 2 * interval.
        live_fields is passed directly so predict_4s skips the DB offset query."""
        try:
            from predictor.predict_4s import try_autopredict_4s
            interval_us = 5 * 60 * 1_000_000
            horizon = 2
            target_ts_us = open_time_us + horizon * interval_us
            target_ts = target_ts_us // 1_000_000
            now_utc = int(time.time())
            if target_ts <= now_utc:
                return
            rows = await self.db.fetchall(
                "SELECT slug FROM poly_markets WHERE ts=%s AND closed=0 LIMIT 10",
                (target_ts,),
            )
            for (slug,) in (rows or []):
                LOGGER.info("[4s_trigger] firing autopredict_4s for %s (horizon=%d, starts in %ds)",
                            slug, horizon, target_ts - now_utc)
                asyncio.create_task(try_autopredict_4s(slug, live_fields))
        except Exception as exc:
            LOGGER.warning("[4s_trigger] error: %s", exc)

    @staticmethod
    def _map_fields(kline: dict) -> dict:
        open_time_ms = int(kline["t"])
        close_time_ms = int(kline["T"])
        return {
            "open_time": open_time_ms * 1000,  # store in microseconds
            "open": float(kline["o"]),
            "high": float(kline["h"]),
            "low": float(kline["l"]),
            "close": float(kline["c"]),
            "volume": float(kline["v"]),
            "close_time": close_time_ms,
            "quota_volume": float(kline["q"]),
            "trades": int(kline["n"]),
            "taker_base_volume": float(kline["V"]),
            "taker_quota_volume": float(kline["Q"]),
        }

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)


_SNAPSHOT_COLLECTOR: Optional[SnapshotCollector] = None
_SNAPSHOT_TASK: Optional[asyncio.Task] = None


def start_snapshot_collector() -> None:
    """Launch the background snapshot collector if it is not already running."""

    global _SNAPSHOT_COLLECTOR, _SNAPSHOT_TASK
    if _SNAPSHOT_TASK and not _SNAPSHOT_TASK.done():
        return

    _SNAPSHOT_COLLECTOR = SnapshotCollector()
    _SNAPSHOT_TASK = asyncio.create_task(_SNAPSHOT_COLLECTOR.run_forever())
    LOGGER.info("Started Binance snapshot background task")


async def stop_snapshot_collector() -> None:
    """Stop the background snapshot collector if running."""

    global _SNAPSHOT_COLLECTOR, _SNAPSHOT_TASK
    if _SNAPSHOT_COLLECTOR:
        await _SNAPSHOT_COLLECTOR.stop()
    if _SNAPSHOT_TASK:
        _SNAPSHOT_TASK.cancel()
        try:
            await _SNAPSHOT_TASK
        except asyncio.CancelledError:
            pass
    _SNAPSHOT_COLLECTOR = None
    _SNAPSHOT_TASK = None


async def main() -> None:
    collector = SnapshotCollector()
    try:
        await collector.run_forever()
    finally:
        await collector.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        LOGGER.info("Snapshot collector stopped by user")
