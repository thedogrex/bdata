import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional
from db import DbProvider

db = DbProvider()


def date_to_us(date_str: str, end_of_day: bool = False) -> int:
    time_part = "23:59:59" if end_of_day else "00:00:00"
    try:
        dt = datetime.strptime(f"{date_str} {time_part}", "%Y-%m-%d %H:%M:%S")
    except ValueError as exc:
        raise ValueError(
            f"Invalid date '{date_str}'. Expected format YYYY-MM-DD and a real calendar date."
        ) from exc
    return int(pd.Timestamp(dt).timestamp() * 1_000_000)


async def load_candles(
    table: str = "c_5m",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    where = "1=1"
    if start_date:
        where += f" AND open_time >= {date_to_us(start_date)}"
    if end_date:
        where += f" AND open_time <= {date_to_us(end_date, True)}"

    query = f"""
        SELECT open_time, open, high, low, close, volume,
               close_time, quota_volume, trades, taker_base_volume, taker_quota_volume
        FROM {table}
        WHERE {where}
        ORDER BY open_time ASC
    """
    rows = await db.fetchall(query)
    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quota_volume", "trades", "taker_base_volume", "taker_quota_volume"
    ])
    for c in ["open", "high", "low", "close", "volume", "quota_volume",
              "taker_base_volume", "taker_quota_volume"]:
        df[c] = df[c].astype(float)
    df["trades"] = df["trades"].astype(int)
    return df


def add_direction(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["direction"] = (df["close"] > df["open"]).astype(np.int8)  # 1=UP, 0=DOWN
    return df


def add_future_directions(df: pd.DataFrame, horizons: list[int] = None) -> pd.DataFrame:
    if horizons is None:
        horizons = [1, 2, 3, 4, 5]
    df = df.copy()
    for h in horizons:
        df[f"future_dir_{h}"] = (df["close"].shift(-h) > df["open"].shift(-h)).astype(np.int8)
    return df
