import pandas as pd
import numpy as np


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Price-based ---
    df["body"] = df["close"] - df["open"]
    df["body_pct"] = df["body"] / df["open"]
    df["upper_shadow"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_shadow"] = df[["open", "close"]].min(axis=1) - df["low"]
    df["range"] = df["high"] - df["low"]

    # --- Moving averages ---
    for w in [5, 10, 20, 50]:
        df[f"sma_{w}"] = df["close"].rolling(w).mean()
        df[f"ema_{w}"] = df["close"].ewm(span=w, adjust=False).mean()
        df[f"dist_sma_{w}"] = (df["close"] - df[f"sma_{w}"]) / df[f"sma_{w}"]

    # --- RSI ---
    for period in [6, 14]:
        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        df[f"rsi_{period}"] = 100 - (100 / (1 + rs))

    # --- MACD ---
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # --- Bollinger Bands ---
    sma20 = df["close"].rolling(20).mean()
    std20 = df["close"].rolling(20).std()
    df["bb_upper"] = sma20 + 2 * std20
    df["bb_lower"] = sma20 - 2 * std20
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / sma20
    df["bb_pos"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    # --- Volatility ---
    df["volatility_5"] = df["close"].rolling(5).std()
    df["volatility_20"] = df["close"].rolling(20).std()
    df["atr_14"] = _atr(df, 14)

    # --- Momentum ---
    for lag in [1, 3, 6, 12]:
        df[f"momentum_{lag}"] = df["close"] - df["close"].shift(lag)
        df[f"roc_{lag}"] = df["close"].pct_change(lag)

    # --- Volume features ---
    df["volume_sma_10"] = df["volume"].rolling(10).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma_10"]
    df["taker_ratio"] = df["taker_base_volume"] / df["volume"].replace(0, np.nan)

    # --- Candle patterns ---
    df["direction"] = (df["close"] > df["open"]).astype(np.int8)
    for lag in range(1, 11):
        df[f"dir_lag_{lag}"] = df["direction"].shift(lag)

    # --- Streak features ---
    df["up_streak"] = _streak(df["direction"], 1)
    df["down_streak"] = _streak(df["direction"], 0)

    # --- Hour / day of week (from microsecond timestamp) ---
    dt = pd.to_datetime(df["open_time"], unit="us")
    df["hour"] = dt.dt.hour
    df["dow"] = dt.dt.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    return df


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _streak(series: pd.Series, value: int) -> pd.Series:
    groups = (series != value).cumsum()
    streaks = series.groupby(groups).cumsum()
    return streaks


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {
        "open_time", "close_time", "direction",
    }
    # Also exclude future_dir_* columns
    cols = [c for c in df.columns if c not in exclude and not c.startswith("future_dir_")]
    return cols
