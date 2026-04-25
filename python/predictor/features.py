import pandas as pd
import numpy as np


# Toggle feature groups on/off to balance accuracy vs speed.
# Set any entry to False to skip computing that entire block.
FEATURE_USAGE: dict[str, bool] = {
    "moving_averages": False,
    "rsi": True,
    "macd": False,
    "bollinger": True,
    "volatility": True,
    "momentum": False,
    "volume": False,
    "stochastic": False,
    "adx_dmi": False,
    "direction_lags": False,
    "streaks": False,
    "candlestick_patterns": False,
    "rolling_stats": False,
    "lag_deltas": False,
    "indicator_interactions": False,
    "returns": False,
    "return_magnitude": False,
    "ema_diff": False,
    "volatility_extra": False,
    "volume_zscore": False,
    "orderflow": False,
    "higher_moments": False,
    "time_features": False,
}


def set_feature_usage(**overrides: bool) -> None:
    """Convenience helper for toggling feature blocks from callers/tests."""
    for key, value in overrides.items():
        if key in FEATURE_USAGE:
            FEATURE_USAGE[key] = bool(value)
        else:
            raise KeyError(f"Unknown feature block '{key}'. Available: {list(FEATURE_USAGE)}")


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Price-based ---
    df["body"] = df["close"] - df["open"]
    df["body_pct"] = df["body"] / df["open"]
    df["upper_shadow"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_shadow"] = df[["open", "close"]].min(axis=1) - df["low"]
    df["range"] = df["high"] - df["low"]

    # --- Moving averages ---
    if FEATURE_USAGE["moving_averages"]:
        for w in [5, 10, 20, 50, 100]:
            df[f"sma_{w}"] = df["close"].rolling(w).mean()
            df[f"ema_{w}"] = df["close"].ewm(span=w, adjust=False).mean()
            df[f"dist_sma_{w}"] = (df["close"] - df[f"sma_{w}"]) / df[f"sma_{w}"]

    # --- RSI ---
    if FEATURE_USAGE["rsi"]:
        for period in [6, 14]:
            delta = df["close"].diff()
            gain = delta.clip(lower=0).rolling(period).mean()
            loss = (-delta.clip(upper=0)).rolling(period).mean()
            rs = gain / loss.replace(0, np.nan)
            df[f"rsi_{period}"] = 100 - (100 / (1 + rs))

    # --- MACD ---
    if FEATURE_USAGE["macd"]:
        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

    # --- Bollinger Bands ---
    if FEATURE_USAGE["bollinger"]:
        bb_upper, bb_lower, bb_width, bb_pos = bollinger_components(
            df["close"].astype(float), period=20, std_mult=2.0
        )
        df["bb_upper"] = bb_upper
        df["bb_lower"] = bb_lower
        df["bb_width"] = bb_width
        df["bb_pos"] = bb_pos

    # --- Volatility ---
    if FEATURE_USAGE["volatility"]:
        df["volatility_5"] = df["close"].rolling(5).std()
        df["volatility_20"] = df["close"].rolling(20).std()
        df["atr_14"] = _atr(df, 14)

    # --- Momentum ---
    if FEATURE_USAGE["momentum"]:
        for lag in [1, 3, 6, 12]:
            df[f"momentum_{lag}"] = df["close"] - df["close"].shift(lag)
            df[f"roc_{lag}"] = df["close"].pct_change(lag)

    # --- Volume features ---
    if FEATURE_USAGE["volume"]:
        df["volume_sma_10"] = df["volume"].rolling(10).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma_10"]
        df["taker_ratio"] = df["taker_base_volume"] / df["volume"].replace(0, np.nan)

    # --- Stochastic Oscillator ---
    if FEATURE_USAGE["stochastic"]:
        low14 = df["low"].rolling(14).min()
        high14 = df["high"].rolling(14).max()
        df["stoch_k"] = 100 * (df["close"] - low14) / (high14 - low14).replace(0, np.nan)
        df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    # --- ADX / DMI ---
    if FEATURE_USAGE["adx_dmi"]:
        df["adx"] = _adx(df, 14)
        plus_dm = df["high"].diff().clip(lower=0)
        minus_dm = (-df["low"].diff()).clip(lower=0)
        tr14 = _true_range(df).rolling(14).sum()
        df["di_plus"] = 100 * plus_dm.rolling(14).sum() / tr14.replace(0, np.nan)
        df["di_minus"] = 100 * minus_dm.rolling(14).sum() / tr14.replace(0, np.nan)

    # --- Candle patterns & direction lags ---
    if FEATURE_USAGE["direction_lags"] or FEATURE_USAGE["streaks"] or FEATURE_USAGE["candlestick_patterns"]:
        df["direction"] = (df["close"] > df["open"]).astype(np.int8)

    if FEATURE_USAGE["direction_lags"]:
        for lag in range(1, 11):
            df[f"dir_lag_{lag}"] = df["direction"].shift(lag)

    if FEATURE_USAGE["streaks"]:
        df["up_streak"] = _streak(df["direction"], 1)
        df["down_streak"] = _streak(df["direction"], 0)

    body_abs = df["body"].abs()
    avg_body = body_abs.rolling(10).mean()

    if FEATURE_USAGE["candlestick_patterns"]:
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
    if FEATURE_USAGE["rolling_stats"]:
        df["avg_body_5"] = body_abs.rolling(5).mean()
        df["avg_body_10"] = body_abs.rolling(10).mean()
        df["std_close_5"] = df["close"].rolling(5).std()
        df["std_close_10"] = df["close"].rolling(10).std()
        df["avg_range_5"] = df["range"].rolling(5).mean()

    # --- Lag deltas for indicators ---
    if FEATURE_USAGE["lag_deltas"]:
        if "rsi_14" in df:
            df["delta_rsi_14"] = df["rsi_14"].diff()
        if "macd_hist" in df:
            df["delta_macd_hist"] = df["macd_hist"].diff()
        df["delta_volume"] = df["volume"].diff()
        df["delta_close"] = df["close"].diff()
        if "bb_pos" in df:
            df["delta_bb_pos"] = df["bb_pos"].diff()

    # --- Indicator interaction signals ---
    if FEATURE_USAGE["indicator_interactions"]:
        required = ["rsi_14", "macd_hist", "ema_5", "ema_20", "volume_ratio", "momentum_1"]
        if all(col in df for col in required):
            df["rsi_overbought_macd_cross"] = ((df["rsi_14"] > 70) & (df["macd_hist"] < 0)).astype(np.int8)
            df["rsi_oversold_macd_cross"] = ((df["rsi_14"] < 30) & (df["macd_hist"] > 0)).astype(np.int8)
            df["ema_cross_up"] = ((df["ema_5"] > df["ema_20"]) & (df["ema_5"].shift(1) <= df["ema_20"].shift(1))).astype(np.int8)
            df["ema_cross_down"] = ((df["ema_5"] < df["ema_20"]) & (df["ema_5"].shift(1) >= df["ema_20"].shift(1))).astype(np.int8)
            df["vol_surge_up"] = ((df["volume_ratio"] > 1.5) & (df["momentum_1"] > 0)).astype(np.int8)
            df["vol_surge_down"] = ((df["volume_ratio"] > 1.5) & (df["momentum_1"] < 0)).astype(np.int8)

    # --- Returns (explicit) ---
    if FEATURE_USAGE["returns"]:
        for lag in [1, 2, 3, 5, 10]:
            df[f"returns_{lag}"] = df["close"].pct_change(lag)

    # --- Return magnitude (close-to-close) for multiple horizons ---
    if FEATURE_USAGE["return_magnitude"]:
        df["return_mag_1"] = (df["close"] - df["close"].shift(1)) / df["close"].shift(1)
        df["return_mag_3"] = (df["close"] - df["close"].shift(3)) / df["close"].shift(3)

    # --- EMA diff (close minus EMA, normalised) ---
    if FEATURE_USAGE["ema_diff"]:
        needed = [f"ema_{w}" for w in (5, 20, 50)]
        if all(col in df for col in needed):
            df["ema_diff_5"] = (df["close"] - df["ema_5"]) / df["ema_5"]
            df["ema_diff_20"] = (df["close"] - df["ema_20"]) / df["ema_20"]
            df["ema_diff_50"] = (df["close"] - df["ema_50"]) / df["ema_50"]

    # --- Volatility 10 ---
    if FEATURE_USAGE["volatility_extra"]:
        df["volatility_10"] = df["close"].rolling(10).std()

    # --- Volume z-score ---
    if FEATURE_USAGE["volume_zscore"]:
        vol_mean = df["volume"].rolling(20).mean()
        vol_std = df["volume"].rolling(20).std()
        df["volume_zscore"] = (df["volume"] - vol_mean) / vol_std.replace(0, np.nan)

    # --- Orderflow proxy (taker buy pressure) ---
    if FEATURE_USAGE["orderflow"]:
        df["orderflow_proxy"] = (
            (df["taker_base_volume"] - (df["volume"] - df["taker_base_volume"]))
            / df["volume"].replace(0, np.nan)
        )

    # --- Rolling skew & kurtosis of returns ---
    if FEATURE_USAGE["higher_moments"]:
        ret = df["close"].pct_change()
        df["rolling_skew_20"] = ret.rolling(20).skew()
        df["rolling_kurtosis_20"] = ret.rolling(20).kurt()
        df["rolling_skew_50"] = ret.rolling(50).skew()
        df["rolling_kurtosis_50"] = ret.rolling(50).kurt()

    # --- Hour / day of week (from microsecond timestamp) ---
    if FEATURE_USAGE["time_features"] and "open_time" in df:
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


def bollinger_components(series: pd.Series, period: int, std_mult: float) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    period = max(1, int(period))
    std_mult = float(std_mult)
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    width = (upper - lower) / sma.replace(0, np.nan)
    denom = (upper - lower).replace(0, np.nan)
    pos = (series - lower) / denom
    return upper, lower, width, pos


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
