"""Collect ~8s-before-close Binance 5m candle snapshots into c_5m_8s.

Uses the Binance USDS-M Futures REST Kline endpoint so that we can query the
current 5-minute candle repeatedly and store a snapshot shortly before it
closes. Can run standalone (``python -m predictor.binance_snapshot``) or as a
background task launched from the FastAPI server startup hook.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional, Set

import requests

from db import DbProvider

LOGGER = logging.getLogger("binance_snapshot")
LOGGER.setLevel(logging.DEBUG)

BINANCE_FUTURES_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"
SYMBOL = "BTCUSDT"
INTERVAL = "5m"
SNAPSHOT_LEAD_MS = 8_000  # grab data ~8 seconds before candle close
POLL_INTERVAL_SEC = 1.0
RETRY_DELAY_SEC = 5.0
CACHE_LIMIT = 1500


_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS `c_5m_8s` (
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
  UNIQUE KEY `saqx_snapshot` (`open_time`)
) ENGINE=MyISAM DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
"""


class SnapshotCollector:
    """Polls Binance REST klines and stores a near-close snapshot once per candle."""

    def __init__(self, symbol: str = SYMBOL, interval: str = INTERVAL):
        self.symbol = symbol.upper()
        self.interval = interval
        self.db = DbProvider()
        self._stop_event = asyncio.Event()
        self._seen_snapshots: Set[int] = set()

    async def run_forever(self) -> None:
        await self.db.execute(_CREATE_TABLE_SQL)
        LOGGER.info(
            "Starting Binance REST snapshot collector for %s %s", self.symbol, self.interval
        )
        while not self._stop_event.is_set():
            try:
                await self._maybe_collect_snapshot()
                await asyncio.wait_for(self._stop_event.wait(), timeout=POLL_INTERVAL_SEC)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - network errors
                LOGGER.warning("Snapshot loop error: %s", exc)
                await asyncio.sleep(RETRY_DELAY_SEC)

    async def stop(self) -> None:
        self._stop_event.set()

    async def _maybe_collect_snapshot(self) -> None:
        kline = await self._fetch_latest_kline()
        if not kline:
            return

        open_time_ms = int(kline[0])
        close_time_ms = int(kline[6])
        close_price = float(kline[4])
        snapshot_at_ms = close_time_ms - SNAPSHOT_LEAD_MS
        now_ms = self._now_ms()

        LOGGER.debug(
            "[binance_snapshot] candle open_time=%s close_price=%s snapshot_at=%s now=%s",
            open_time_ms,
            close_price,
            snapshot_at_ms,
            now_ms,
        )

        if open_time_ms in self._seen_snapshots:
            return

        if now_ms < snapshot_at_ms:
            return

        await self._store_snapshot(kline)
        self._seen_snapshots.add(open_time_ms)
        if len(self._seen_snapshots) > CACHE_LIMIT:
            # keep the set from growing indefinitely
            self._seen_snapshots = set(sorted(self._seen_snapshots)[-CACHE_LIMIT:])

    async def _fetch_latest_kline(self) -> Optional[list]:
        params = {
            "symbol": self.symbol,
            "interval": self.interval,
            "limit": 1,
        }
        resp = await asyncio.to_thread(
            requests.get,
            BINANCE_FUTURES_KLINES_URL,
            params=params,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        return data[-1] if data else None

    async def _store_snapshot(self, kline: list) -> None:
        fields = self._map_fields(kline)
        result = await self.db.insert_one("c_5m_8s", fields=fields, ignore=True, print_query=False)
        if result >= 0:
            LOGGER.info(
                "Stored snapshot for candle %s (%s) close=%.2f",
                fields["open_time"],
                time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(fields["open_time"] / 1_000_000)),
                fields["close"],
            )

    @staticmethod
    def _map_fields(kline: list) -> dict:
        open_time_ms = int(kline[0])
        close_time_ms = int(kline[6])
        return {
            "open_time": open_time_ms * 1000,  # store in microseconds
            "open": float(kline[1]),
            "high": float(kline[2]),
            "low": float(kline[3]),
            "close": float(kline[4]),
            "volume": float(kline[5]),
            "close_time": close_time_ms,
            "quota_volume": float(kline[7]),
            "trades": int(kline[8]),
            "taker_base_volume": float(kline[9]),
            "taker_quota_volume": float(kline[10]),
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
