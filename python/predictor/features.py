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
    for w in [5, 10, 20, 50, 100]:
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

    # --- Stochastic Oscillator ---
    low14 = df["low"].rolling(14).min()
    high14 = df["high"].rolling(14).max()
    df["stoch_k"] = 100 * (df["close"] - low14) / (high14 - low14).replace(0, np.nan)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    # --- ADX / DMI ---
    df["adx"] = _adx(df, 14)
    plus_dm = df["high"].diff().clip(lower=0)
    minus_dm = (-df["low"].diff()).clip(lower=0)
    tr14 = _true_range(df).rolling(14).sum()
    df["di_plus"] = 100 * plus_dm.rolling(14).sum() / tr14.replace(0, np.nan)
    df["di_minus"] = 100 * minus_dm.rolling(14).sum() / tr14.replace(0, np.nan)

    # --- Candle patterns ---
    df["direction"] = (df["close"] > df["open"]).astype(np.int8)
    for lag in range(1, 11):
        df[f"dir_lag_{lag}"] = df["direction"].shift(lag)

    # --- Streak features ---
    df["up_streak"] = _streak(df["direction"], 1)
    df["down_streak"] = _streak(df["direction"], 0)

    # --- Candlestick pattern recognition ---
    body_abs = df["body"].abs()
    avg_body = body_abs.rolling(10).mean()
    df["is_doji"] = (body_abs < avg_body * 0.1).astype(np.int8)
    df["is_hammer"] = ((df["lower_shadow"] > body_abs * 2) & (df["upper_shadow"] < body_abs * 0.5)).astype(np.int8)
    df["is_inv_hammer"] = ((df["upper_shadow"] > body_abs * 2) & (df["lower_shadow"] < body_abs * 0.5)).astype(np.int8)
    prev_body = df["body"].shift(1)
    df["is_engulfing_bull"] = ((df["body"] > 0) & (prev_body < 0) & (body_abs > prev_body.abs())).astype(np.int8)
    df["is_engulfing_bear"] = ((df["body"] < 0) & (prev_body > 0) & (body_abs > prev_body.abs())).astype(np.int8)
    df["is_morning_star"] = (
        (df["body"].shift(2) < 0) &
        (body_abs.shift(1) < avg_body.shift(1) * 0.3) &
        (df["body"] > 0)
    ).astype(np.int8)
    df["is_evening_star"] = (
        (df["body"].shift(2) > 0) &
        (body_abs.shift(1) < avg_body.shift(1) * 0.3) &
        (df["body"] < 0)
    ).astype(np.int8)

    # --- Rolling stats ---
    df["avg_body_5"] = body_abs.rolling(5).mean()
    df["avg_body_10"] = body_abs.rolling(10).mean()
    df["std_close_5"] = df["close"].rolling(5).std()
    df["std_close_10"] = df["close"].rolling(10).std()
    df["avg_range_5"] = df["range"].rolling(5).mean()

    # --- Lag deltas for indicators ---
    df["delta_rsi_14"] = df["rsi_14"].diff()
    df["delta_macd_hist"] = df["macd_hist"].diff()
    df["delta_volume"] = df["volume"].diff()
    df["delta_close"] = df["close"].diff()
    df["delta_bb_pos"] = df["bb_pos"].diff()

    # --- Indicator interaction signals ---
    df["rsi_overbought_macd_cross"] = ((df["rsi_14"] > 70) & (df["macd_hist"] < 0)).astype(np.int8)
    df["rsi_oversold_macd_cross"] = ((df["rsi_14"] < 30) & (df["macd_hist"] > 0)).astype(np.int8)
    df["ema_cross_up"] = ((df["ema_5"] > df["ema_20"]) & (df["ema_5"].shift(1) <= df["ema_20"].shift(1))).astype(np.int8)
    df["ema_cross_down"] = ((df["ema_5"] < df["ema_20"]) & (df["ema_5"].shift(1) >= df["ema_20"].shift(1))).astype(np.int8)
    df["vol_surge_up"] = ((df["volume_ratio"] > 1.5) & (df["momentum_1"] > 0)).astype(np.int8)
    df["vol_surge_down"] = ((df["volume_ratio"] > 1.5) & (df["momentum_1"] < 0)).astype(np.int8)

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


def _true_range(df: pd.DataFrame) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    return pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)


def _adx(df: pd.DataFrame, period: int) -> pd.Series:
    tr = _true_range(df)
    plus_dm = df["high"].diff().clip(lower=0)
    minus_dm = (-df["low"].diff()).clip(lower=0)
    atr = tr.rolling(period).mean()
    di_plus = 100 * plus_dm.rolling(period).mean() / atr.replace(0, np.nan)
    di_minus = 100 * minus_dm.rolling(period).mean() / atr.replace(0, np.nan)
    dx = (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan) * 100
    return dx.rolling(period).mean()


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {
        "open_time", "close_time", "direction",
    }
    # Also exclude future_dir_* columns
    cols = [c for c in df.columns if c not in exclude and not c.startswith("future_dir_")]
    return cols
