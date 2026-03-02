"""
Binance REST API candle downloader.

Downloads 5m BTCUSDT klines from Binance public API and inserts them into
the `c_5m` table.  Designed to fill gaps that the daily-zip downloader
(download-kline.py) cannot cover — especially candles from the current day.

Binance API docs: https://binance-docs.github.io/apidocs/spot/en/#kline-candlestick-data
  GET /api/v3/klines
  - symbol: BTCUSDT
  - interval: 5m
  - startTime / endTime: milliseconds
  - limit: max 1000

open_time in our DB is stored in **microseconds** (16-digit int).
Binance returns open_time in **milliseconds** (13-digit int).
"""

import time
import asyncio
from typing import List, Tuple, Optional

import requests

from db import DbProvider

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
SYMBOL = "BTCUSDT"
INTERVAL = "5m"
INTERVAL_MS = 5 * 60 * 1000          # 300 000 ms
INTERVAL_US = 5 * 60 * 1_000_000     # 300 000 000 us
MAX_LIMIT = 1000                      # Binance max per request


_SYNC_STATUS = {
    "running": False,
    "phase": None,
    "message": "",
    "start_ms": None,
    "end_ms": None,
    "cursor_ms": None,
    "expected_candles": None,
    "downloaded_candles": 0,
    "inserted_rows": 0,
    "updated_ts": 0,
}

db = DbProvider()


def get_candle_sync_status() -> dict:
    return dict(_SYNC_STATUS)


def reset_candle_sync_status() -> None:
    _SYNC_STATUS.update({
        "running": False,
        "phase": None,
        "message": "",
        "start_ms": None,
        "end_ms": None,
        "cursor_ms": None,
        "expected_candles": None,
        "downloaded_candles": 0,
        "inserted_rows": 0,
        "updated_ts": int(time.time()),
    })


async def fetch_and_store_klines(
    start_ms: int,
    end_ms: int,
    symbol: str = SYMBOL,
    interval: str = INTERVAL,
    limit: int = 100,
) -> int:
    """Download klines from Binance REST API and INSERT IGNORE into c_5m.

    Args:
        start_ms: start time in milliseconds (inclusive).
        end_ms:   end time in milliseconds (inclusive).

    Returns:
        Number of candles inserted.
    """
    total_inserted = 0
    cursor_ms = start_ms

    while cursor_ms <= end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cursor_ms,
            "endTime": end_ms,
            "limit": int(limit) if int(limit) > 0 else 100,
        }

        try:
            resp = await asyncio.to_thread(
                requests.get, BINANCE_KLINES_URL, params=params, timeout=15
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"[candle_sync] Binance API error: {e}")
            _SYNC_STATUS.update({
                "running": False,
                "phase": "error",
                "message": f"Binance API error: {e}",
                "updated_ts": int(time.time()),
            })
            break

        if not data:
            break

        _SYNC_STATUS.update({
            "running": True,
            "phase": "downloading",
            "cursor_ms": int(cursor_ms),
            "downloaded_candles": int(_SYNC_STATUS.get("downloaded_candles", 0)) + int(len(data)),
            "message": f"Downloading candles... (+{len(data)})",
            "updated_ts": int(time.time()),
        })

        for k in data:
            # Binance kline format:
            # [0] Open time (ms), [1] Open, [2] High, [3] Low, [4] Close,
            # [5] Volume, [6] Close time (ms), [7] Quote asset volume,
            # [8] Number of trades, [9] Taker buy base vol, [10] Taker buy quote vol, [11] Ignore
            open_time_ms = int(k[0])
            open_time_us = open_time_ms * 1000  # convert ms -> us

            await db.insert_one(
                "c_5m",
                fields={
                    "open_time": open_time_us,
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                    "close_time": int(k[6]),
                    "quota_volume": float(k[7]),
                    "trades": int(k[8]),
                    "taker_base_volume": float(k[9]),
                    "taker_quota_volume": float(k[10]),
                },
                ignore=True,
                print_query=False,
            )
            total_inserted += 1

        _SYNC_STATUS.update({
            "inserted_rows": int(_SYNC_STATUS.get("inserted_rows", 0)) + int(len(data)),
            "updated_ts": int(time.time()),
        })

        # Advance cursor past the last candle we received
        last_open_ms = int(data[-1][0])
        if last_open_ms <= cursor_ms:
            # No progress — avoid infinite loop
            break
        cursor_ms = last_open_ms + INTERVAL_MS

        # Small delay to respect rate limits
        await asyncio.sleep(0.2)

    return total_inserted


async def sync_candles_up_to(
    target_ts_sec: int,
    window_candles: int = 1100,
    table: str = "c_5m",
) -> dict:
    """Ensure candles exist in DB up to (and including) the 5m candle
    whose open_time aligns with *target_ts_sec*.

    1. Find the latest candle in DB.
    2. If it's already >= target, nothing to do.
    3. Otherwise download from Binance API to fill the gap.

    Returns dict with sync stats.
    """
    target_us = target_ts_sec * 1_000_000

    # Find latest candle in DB
    row = await db.fetchone(
        f"SELECT MAX(open_time) FROM {table}"
    )
    latest_us = int(row[0]) if row and row[0] else 0

    if latest_us >= target_us:
        return {"status": "ok", "message": "Candles already up to date", "downloaded": 0}

    # We need candles from latest_us + interval up to target_us
    start_ms = (latest_us // 1000) + INTERVAL_MS if latest_us > 0 else (target_us // 1000) - window_candles * INTERVAL_MS
    end_ms = target_us // 1000

    if start_ms > end_ms:
        return {"status": "ok", "message": "No gap to fill", "downloaded": 0}

    print(f"[candle_sync] Downloading candles from {start_ms} to {end_ms} (ms)")
    reset_candle_sync_status()
    expected = int(max(0, (end_ms - start_ms) // INTERVAL_MS + 1))
    _SYNC_STATUS.update({
        "running": True,
        "phase": "sync",
        "message": "Starting candle synchronization...",
        "start_ms": int(start_ms),
        "end_ms": int(end_ms),
        "cursor_ms": int(start_ms),
        "expected_candles": expected,
        "updated_ts": int(time.time()),
    })
    count = await fetch_and_store_klines(start_ms, end_ms, limit=100)

    _SYNC_STATUS.update({
        "running": False,
        "phase": "done",
        "message": f"Downloaded {count} candles",
        "updated_ts": int(time.time()),
    })

    return {
        "status": "ok",
        "message": f"Downloaded {count} candles from Binance API",
        "downloaded": count,
        "start_ms": start_ms,
        "end_ms": end_ms,
    }


async def check_and_fill_gaps(
    target_ts_sec: int,
    window_candles: int = 1000,
    table: str = "c_5m",
) -> dict:
    """Check for gaps in the last *window_candles* candles before target_ts_sec.
    If gaps found, try to fill them from Binance API.

    Returns dict with gap info and fill results.
    """
    import numpy as np

    target_us = target_ts_sec * 1_000_000
    need = window_candles + 1

    rows = await db.fetchall(
        f"""
        SELECT open_time FROM {table}
        WHERE open_time <= %s
        ORDER BY open_time DESC
        LIMIT %s
        """,
        (target_us, need),
    )

    if not rows or len(rows) < 2:
        # Not enough candles at all — do a bulk download
        start_ms = (target_us // 1000) - need * INTERVAL_MS
        end_ms = target_us // 1000
        count = await fetch_and_store_klines(start_ms, end_ms)
        return {
            "status": "downloaded",
            "message": f"Bulk downloaded {count} candles (had {len(rows) if rows else 0})",
            "downloaded": count,
        }

    # Check for gaps
    times = sorted([int(r[0]) for r in rows])
    arr = np.array(times, dtype=np.int64)
    diffs = np.diff(arr)
    bad_idx = np.where(diffs != INTERVAL_US)[0]

    if len(bad_idx) == 0:
        return {"status": "ok", "message": "No gaps found", "downloaded": 0, "candles_available": len(times)}

    # Fill each gap
    total_downloaded = 0
    for idx in bad_idx:
        gap_start_us = int(arr[idx])
        gap_end_us = int(arr[idx + 1])
        # Download candles between gap_start and gap_end
        start_ms = (gap_start_us // 1000) + INTERVAL_MS
        end_ms = (gap_end_us // 1000) - INTERVAL_MS
        if start_ms <= end_ms:
            count = await fetch_and_store_klines(start_ms, end_ms)
            total_downloaded += count

    return {
        "status": "filled",
        "message": f"Found {len(bad_idx)} gap(s), downloaded {total_downloaded} candles",
        "gaps_found": int(len(bad_idx)),
        "downloaded": total_downloaded,
    }
