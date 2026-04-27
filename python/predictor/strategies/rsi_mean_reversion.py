import asyncio
import logging
import time

import numpy as np
import pandas as pd

from predictor.strategies.base import BaseStrategy
from predictor.features import add_technical_features, bollinger_components
from predictor.utils.prediction_thresholds import classify_probability, resolve_probability_threshold
from app.config import DEBUG_EMA_FEATURE


LOGGER = logging.getLogger(__name__)


class RSIMeanReversionStrategy(BaseStrategy):

    @staticmethod
    def description() -> str:
        return (
            "RSI mean-reversion strategy. Predicts UP when RSI is oversold "
            "(below lower threshold) and DOWN when RSI is overbought (above upper threshold). "
            "Combines RSI with Bollinger Band position and optional EMA-distance confirmation."
        )

    @staticmethod
    def default_params() -> dict:
        return {
            "rsi_period": 14,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "use_bb_confirm": True,
            "require_bb_confirm": False,  # If True, signal only if BOTH RSI and BB conditions met
            "bb_low": 0.2,
            "bb_high": 0.8,
            "bb_period": 20,
            "bb_std": 2.0,
            "use_ema_filter": False,
            "ema_period": 20,
            "ema_diff_threshold": 0.0,
            "use_ema_trend_strength_filter": False,
            "ema_fast_period": 20,
            "ema_slow_period": 50,
            "ema_trend_strength_threshold": 0.005,
            "use_ema_direction_filter": False,
            "ema_direction_period": 50,
            "min_vol": 0.0005,
            "max_vol": 0.02,
            "vol_ratio_max": 1.5,
            "vol_fast_window": 20,
            "vol_slow_window": 50,
            "vol_spike_multiplier": 1.5,
            "adaptive_rsi": True,
        }

    @staticmethod
    def param_docs() -> dict:
        return {
            "rsi_period": "RSI lookback period. Shorter = more sensitive. Typical: 6-21.",
            "rsi_oversold": "RSI level considered oversold (buy signal). Lower = rarer signals. Typical: 20-35.",
            "rsi_overbought": "RSI level considered overbought (sell signal). Higher = rarer signals. Typical: 65-80.",
            "use_bb_confirm": "Whether to use Bollinger Band position confirmation. If False, BB is ignored entirely.",
            "require_bb_confirm": "If True, signal ONLY if both RSI AND BB conditions met (strict). If False (default), RSI can signal alone with BB as bonus.",
            "bb_low": "Lower Bollinger Band threshold for confirmation (0-1). Price must be below this to buy. Typical: 0.15-0.25.",
            "bb_high": "Upper Bollinger Band threshold for confirmation (0-1). Price must be above this to sell. Typical: 0.75-0.85.",
            "bb_period": "Bollinger Band lookback period for confirmation. Typical: 10-50.",
            "bb_std": "Bollinger Band standard deviation multiplier. Typical: 1.0-3.0.",
            "use_ema_filter": "If True, keep long signals only when price is below the chosen EMA and short signals only when price is above it.",
            "ema_period": "EMA period used by the EMA-distance filter. Typical: 20 or 50.",
            "ema_diff_threshold": "Minimum normalized distance from EMA required for confirmation. Example 0.005 means price must be 0.5% away from EMA.",
            "use_ema_trend_strength_filter": "If True, skip trades when EMA fast and EMA slow diverge too much relative to price, treating that as a strong trend.",
            "ema_fast_period": "Fast EMA period for trend-strength filter. Typical: 20.",
            "ema_slow_period": "Slow EMA period for trend-strength filter. Typical: 50.",
            "ema_trend_strength_threshold": "Skip trade when abs(ema_fast - ema_slow) / price exceeds this threshold. Example 0.005 means skip when EMA spread exceeds 0.5% of price.",
            "use_ema_direction_filter": "If True, ignore longs below the chosen EMA and shorts above the chosen EMA, filtering trades that go against trend direction.",
            "ema_direction_period": "EMA period used for directional trend filter. Typical: 50.",
            "min_vol": "Minimum fast volatility (log-return std) required to take trades.",
            "max_vol": "Maximum fast volatility allowed before pausing trades.",
            "vol_ratio_max": "Maximum fast/slow volatility ratio (regime) to allow trades.",
            "vol_fast_window": "Rolling window (5m candles) for fast volatility.",
            "vol_slow_window": "Rolling window for slow regime volatility.",
            "vol_spike_multiplier": "Reject trades when vol_fast exceeds vol_slow * multiplier.",
            "adaptive_rsi": "Blend fixed RSI thresholds with training percentiles (p10/p90). Default true; set false for fixed thresholds.",
        }

    def fit(self, df: pd.DataFrame, horizon: int = 1) -> None:
        # Learn adaptive thresholds from training window
        if "rsi_14" not in df.columns:
            df = add_technical_features(df)

        self._last_vol_skip_count = 0
        self._last_vol_skip_flags = np.array([], dtype=bool)

        period = self.params["rsi_period"]
        rsi_col = f"rsi_{period}" if f"rsi_{period}" in df.columns else "rsi_14"
        rsi = df[rsi_col].dropna().values

        if len(rsi) > 100:
            self._rsi_median = float(np.median(rsi))
            self._rsi_p10 = float(np.percentile(rsi, 10))
            self._rsi_p90 = float(np.percentile(rsi, 90))
            bb_vals = self._compute_bb_pos(df)
            valid_bb = bb_vals[~np.isnan(bb_vals)] if bb_vals is not None else None
            if valid_bb is not None and len(valid_bb) > 100:
                self._bb_median = float(np.median(valid_bb))
            else:
                self._bb_median = 0.5
        else:
            self._rsi_median = 50.0
            self._rsi_p10 = None
            self._rsi_p90 = None
            self._bb_median = 0.5

    async def predict_proba(self, df: pd.DataFrame, horizon: int = 1) -> np.ndarray:
        # Use pre-computed features if available, else compute
        if "rsi_14" not in df.columns:
            df = add_technical_features(df)

        period = self.params["rsi_period"]
        rsi_col = f"rsi_{period}"
        if rsi_col not in df.columns:
            rsi_col = "rsi_14"

        rsi = df[rsi_col].values.copy()
        rsi = np.nan_to_num(rsi, nan=50.0)
        bb_pos = self._compute_bb_pos(df)

        # Blend fixed thresholds with adaptive percentiles from fit()
        base_oversold = self.params["rsi_oversold"]
        base_overbought = self.params["rsi_overbought"]
        use_adaptive = self.params.get("adaptive_rsi", True)
        if use_adaptive and hasattr(self, '_rsi_p10') and self._rsi_p10 is not None:
            oversold = (base_oversold + self._rsi_p10) / 2
            overbought = (base_overbought + self._rsi_p90) / 2
        else:
            oversold = base_oversold
            overbought = base_overbought

        length = len(rsi)
        skip_breakdown = {
            "ema_distance": 0,
            "ema_trend_strength": 0,
            "ema_direction": 0,
            "vol_min": 0,
            "vol_max": 0,
            "vol_spike": 0,
            "vol_ratio": 0,
            "vol_nan": 0,
        }

        if length == 0:
            self._last_vol_skip_count = 0
            self._last_vol_skip_flags = np.array([], dtype=bool)
            self._last_skip_breakdown = skip_breakdown
            return np.array([], dtype=float)

        proba = np.full(length, 0.5)
        bb_arr = None if bb_pos is None else np.asarray(bb_pos, dtype=float)
        ema_diff_arr = self._compute_ema_diff(df)
        ema_direction_arr = self._compute_price_vs_ema(df, int(self.params.get("ema_direction_period", 50)))
        ema_fast_arr = self._compute_price_vs_ema(df, int(self.params.get("ema_fast_period", 20)), value_col="ema")
        ema_slow_arr = self._compute_price_vs_ema(df, int(self.params.get("ema_slow_period", 50)), value_col="ema")
        if DEBUG_EMA_FEATURE:
            self._log_ema_debug_stats(df, ema_diff_arr, ema_direction_arr, ema_fast_arr, ema_slow_arr)
        vol_metrics = self._get_vol_metrics(df)
        vol_skip_flags = np.zeros(length, dtype=bool)

        use_ema_filter = bool(self.params.get("use_ema_filter", False))
        ema_diff_threshold = max(0.0, float(self.params.get("ema_diff_threshold", 0.0)))
        use_ema_trend_strength_filter = bool(self.params.get("use_ema_trend_strength_filter", False))
        ema_trend_strength_threshold = max(0.0, float(self.params.get("ema_trend_strength_threshold", 0.005)))
        use_ema_direction_filter = bool(self.params.get("use_ema_direction_filter", False))
        min_vol = float(self.params.get("min_vol", 0.0))
        max_vol = float(self.params.get("max_vol", 1.0))
        ratio_max = float(self.params.get("vol_ratio_max", 1.5))
        spike_mult = float(self.params.get("vol_spike_multiplier", 1.5))
        if max_vol <= min_vol:
            max_vol = min_vol + 1e-6

        vol_fast = vol_slow = vol_ratio = None
        if vol_metrics is not None:
            vol_fast = np.asarray(vol_metrics["fast"], dtype=float)
            vol_slow = np.asarray(vol_metrics["slow"], dtype=float)
            vol_ratio = np.asarray(vol_metrics["ratio"], dtype=float)

        chunk_size = self._determine_chunk_size(length)
        last_yield = time.monotonic()

        async def maybe_yield(force: bool = False) -> None:
            nonlocal last_yield
            now = time.monotonic()
            if force or (now - last_yield) >= 1.0:
                await asyncio.sleep(0)
                last_yield = time.monotonic()

        total_vol_skips = 0

        for start in range(0, length, chunk_size):
            end = min(start + chunk_size, length)
            chunk = slice(start, end)
            rsi_chunk = rsi[chunk]
            proba_chunk = proba[chunk]

            os_mask = rsi_chunk < oversold
            ob_mask = rsi_chunk > overbought
            if oversold > 0:
                proba_chunk[os_mask] = 0.5 + ((oversold - rsi_chunk[os_mask]) / oversold) * 0.3
            if overbought < 100:
                proba_chunk[ob_mask] = 0.5 - ((rsi_chunk[ob_mask] - overbought) / (100 - overbought)) * 0.3

            if self.params["use_bb_confirm"] and bb_arr is not None:
                bb_chunk = np.nan_to_num(bb_arr[chunk], nan=0.5)
                
                if self.params.get("require_bb_confirm", False):
                    # STRICT mode: signal ONLY if both RSI AND BB conditions met
                    # First reset ALL proba to 0.5 (no signal)
                    proba_chunk[:] = 0.5
                    # Then set signals ONLY for valid combinations
                    up_valid = os_mask & (bb_chunk < self.params["bb_low"])
                    down_valid = ob_mask & (bb_chunk > self.params["bb_high"])
                    # Apply RSI-based probability ONLY for valid combinations
                    if oversold > 0:
                        rsi_up = rsi_chunk[up_valid]
                        proba_chunk[up_valid] = 0.5 + ((oversold - rsi_up) / oversold) * 0.3 + 0.05
                    if overbought < 100:
                        rsi_down = rsi_chunk[down_valid]
                        proba_chunk[down_valid] = 0.5 - ((rsi_down - overbought) / (100 - overbought)) * 0.3 - 0.05
                else:
                    # LENIENT mode (default): RSI can signal alone, BB adds bonus
                    proba_chunk[bb_chunk < self.params["bb_low"]] += 0.05
                    proba_chunk[bb_chunk > self.params["bb_high"]] -= 0.05

            if use_ema_filter and ema_diff_arr is not None:
                ema_chunk = np.nan_to_num(ema_diff_arr[chunk], nan=0.0)
                long_mask = proba_chunk > 0.5
                short_mask = proba_chunk < 0.5
                long_valid = ema_chunk <= (-ema_diff_threshold)
                short_valid = ema_chunk >= ema_diff_threshold
                if np.any(long_mask):
                    long_invalid = long_mask & ~long_valid
                    skipped = int(np.count_nonzero(long_invalid))
                    if skipped:
                        skip_breakdown["ema_distance"] += skipped
                        proba_chunk[long_invalid] = 0.5
                    proba_chunk[long_mask & long_valid] += 0.03
                if np.any(short_mask):
                    short_invalid = short_mask & ~short_valid
                    skipped = int(np.count_nonzero(short_invalid))
                    if skipped:
                        skip_breakdown["ema_distance"] += skipped
                        proba_chunk[short_invalid] = 0.5
                    proba_chunk[short_mask & short_valid] -= 0.03

            if use_ema_trend_strength_filter and ema_fast_arr is not None and ema_slow_arr is not None:
                fast_chunk = np.nan_to_num(ema_fast_arr[chunk], nan=0.0)
                slow_chunk = np.nan_to_num(ema_slow_arr[chunk], nan=0.0)
                price_chunk = np.nan_to_num(df["close"].astype(float).values[chunk], nan=0.0)
                safe_price = np.where(np.abs(price_chunk) > 1e-12, price_chunk, np.nan)
                trend_strength = np.abs(fast_chunk - slow_chunk) / safe_price
                strong_trend_mask = np.where(np.isfinite(trend_strength), trend_strength > ema_trend_strength_threshold, False)
                if np.any(strong_trend_mask):
                    skipped = int(np.count_nonzero(strong_trend_mask))
                    skip_breakdown["ema_trend_strength"] += skipped
                    proba_chunk[strong_trend_mask] = 0.5

            if use_ema_direction_filter and ema_direction_arr is not None:
                direction_chunk = np.nan_to_num(ema_direction_arr[chunk], nan=0.0)
                long_mask = proba_chunk > 0.5
                short_mask = proba_chunk < 0.5
                if np.any(long_mask):
                    invalid_longs = long_mask & (direction_chunk < 0.0)
                    skipped = int(np.count_nonzero(invalid_longs))
                    if skipped:
                        skip_breakdown["ema_direction"] += skipped
                        proba_chunk[invalid_longs] = 0.5
                if np.any(short_mask):
                    invalid_shorts = short_mask & (direction_chunk > 0.0)
                    skipped = int(np.count_nonzero(invalid_shorts))
                    if skipped:
                        skip_breakdown["ema_direction"] += skipped
                        proba_chunk[invalid_shorts] = 0.5

            if vol_fast is not None and vol_slow is not None and vol_ratio is not None:
                vf = np.asarray(vol_fast[chunk], dtype=float)
                vs = np.asarray(vol_slow[chunk], dtype=float)
                ratio = np.asarray(vol_ratio[chunk], dtype=float)
                finite_mask = np.isfinite(vf) & np.isfinite(vs)
                valid = finite_mask.copy()
                nan_mask = ~finite_mask
                if np.any(nan_mask):
                    skip_breakdown["vol_nan"] += int(np.count_nonzero(nan_mask))
                valid &= finite_mask

                min_mask = vf > min_vol
                invalid_min = valid & ~min_mask
                if np.any(invalid_min):
                    skip_breakdown["vol_min"] += int(np.count_nonzero(invalid_min))
                valid &= min_mask

                max_mask = vf < max_vol
                invalid_max = valid & ~max_mask
                if np.any(invalid_max):
                    skip_breakdown["vol_max"] += int(np.count_nonzero(invalid_max))
                valid &= max_mask

                spike_mask = vf < (vs * spike_mult)
                invalid_spike = valid & ~spike_mask
                if np.any(invalid_spike):
                    skip_breakdown["vol_spike"] += int(np.count_nonzero(invalid_spike))
                valid &= spike_mask

                if ratio_max > 0:
                    ratio_mask = np.where(np.isfinite(ratio), ratio, np.inf)
                    ratio_valid = ratio_mask < ratio_max
                    invalid_ratio = valid & ~ratio_valid
                    if np.any(invalid_ratio):
                        skip_breakdown["vol_ratio"] += int(np.count_nonzero(invalid_ratio))
                    valid &= ratio_valid

                skip_mask = ~valid
                if np.any(skip_mask):
                    total_vol_skips += int(np.count_nonzero(skip_mask))
                    chunk_flags = vol_skip_flags[start:end]
                    chunk_flags[skip_mask] = True
                    vol_skip_flags[start:end] = chunk_flags
                    proba_chunk[skip_mask] = 0.5

            await maybe_yield()

        self._last_vol_skip_count = total_vol_skips
        self._last_vol_skip_flags = vol_skip_flags
        self._last_skip_breakdown = skip_breakdown
        await maybe_yield(force=False)

        np.clip(proba, 0, 1, out=proba)
        if DEBUG_EMA_FEATURE and any(skip_breakdown.values()):
            LOGGER.info(
                "[DEBUG_EMA_FEATURE] Skip breakdown: %s",
                skip_breakdown,
            )
        return proba

    def _log_ema_debug_stats(
        self,
        df: pd.DataFrame,
        ema_diff_arr: np.ndarray | None,
        ema_direction_arr: np.ndarray | None,
        ema_fast_arr: np.ndarray | None,
        ema_slow_arr: np.ndarray | None,
    ) -> None:
        if not DEBUG_EMA_FEATURE:
            return
        if df is None or df.empty or "open_time" not in df.columns or "close" not in df.columns:
            LOGGER.info("[DEBUG_EMA_FEATURE] Skipping EMA stats (missing open_time/close columns)")
            return

        length = len(df)

        def _prepare_array(arr: np.ndarray | None) -> np.ndarray:
            out = np.full(length, np.nan)
            if arr is None:
                return out
            arr = np.asarray(arr, dtype=float)
            if arr.size == length:
                return arr
            n = min(length, arr.size)
            out[:n] = arr[:n]
            return out

        ema_diff = _prepare_array(ema_diff_arr)
        ema_direction = _prepare_array(ema_direction_arr)
        ema_fast = _prepare_array(ema_fast_arr)
        ema_slow = _prepare_array(ema_slow_arr)
        price = df["close"].astype(float).to_numpy()

        try:
            index = pd.to_datetime(df["open_time"], unit="us", errors="coerce")
            if index.isna().all():
                index = pd.to_datetime(df["open_time"], unit="ms", errors="coerce")
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("[DEBUG_EMA_FEATURE] Failed to parse open_time: %s", exc)
            return

        stats_df = pd.DataFrame(
            {
                "ema_diff_norm": ema_diff,
                "ema_direction_diff": ema_direction,
                "ema_fast": ema_fast,
                "ema_slow": ema_slow,
                "price": price,
            },
            index=index,
        )
        stats_df = stats_df.loc[stats_df.index.notna()]
        if stats_df.empty:
            LOGGER.info("[DEBUG_EMA_FEATURE] No valid timestamps to compute EMA stats")
            return

        with np.errstate(divide="ignore", invalid="ignore"):
            trend_strength = np.abs(ema_fast - ema_slow) / np.where(np.abs(price) > 1e-12, price, np.nan)
        stats_df["ema_trend_strength"] = trend_strength

        monthly = stats_df.resample("M").mean().dropna(how="all")
        if monthly.empty:
            LOGGER.info("[DEBUG_EMA_FEATURE] EMA stats dataframe empty after monthly resample")
            return

        LOGGER.info(
            "[DEBUG_EMA_FEATURE] Monthly EMA stats (%d rows -> %d months):\n%s",
            len(stats_df),
            len(monthly),
            monthly.round(6).to_string(),
        )

    async def predict(self, df: pd.DataFrame, horizon: int = 1) -> np.ndarray:
        proba = await self.predict_proba(df, horizon)
        preds = np.full(len(proba), -1, dtype=np.int8)
        threshold = resolve_probability_threshold(self.params)
        up_mask = proba > threshold
        down_mask = proba < (1.0 - threshold)
        preds[up_mask] = 1
        preds[down_mask] = 0
        return preds

    def _determine_chunk_size(self, length: int) -> int:
        if length <= 5000:
            return max(512, length)
        return min(20000, max(2048, length // 8))

    def _compute_bb_pos(self, df: pd.DataFrame) -> np.ndarray:
        period = int(self.params.get("bb_period", 20))
        std_mult = float(self.params.get("bb_std", 2.0))
        if period == 20 and abs(std_mult - 2.0) < 1e-9 and "bb_pos" in df.columns:
            return df["bb_pos"].values.astype(float, copy=True)

        close = df["close"].astype(float)
        _, _, _, pos = bollinger_components(close, period=period, std_mult=std_mult)
        return pos.values if hasattr(pos, "values") else np.array(pos, dtype=float)

    def _compute_ema_diff(self, df: pd.DataFrame) -> np.ndarray | None:
        if "close" not in df.columns:
            return None

        period = max(2, int(self.params.get("ema_period", 20)))
        precomputed_col = f"ema_diff_{period}"
        if precomputed_col in df.columns:
            return df[precomputed_col].astype(float).values

        ema_col = f"ema_{period}"
        if ema_col in df.columns:
            ema = df[ema_col].astype(float)
        else:
            close = df["close"].astype(float)
            ema = close.ewm(span=period, adjust=False).mean()

        close = df["close"].astype(float)
        ema_safe = ema.replace(0, np.nan) if hasattr(ema, "replace") else ema
        diff = (close - ema_safe) / ema_safe
        return diff.values if hasattr(diff, "values") else np.array(diff, dtype=float)

    def _compute_price_vs_ema(self, df: pd.DataFrame, period: int, value_col: str = "diff") -> np.ndarray | None:
        if "close" not in df.columns:
            return None

        period = max(2, int(period))
        ema_col = f"ema_{period}"
        if ema_col in df.columns:
            ema = df[ema_col].astype(float)
        else:
            close = df["close"].astype(float)
            ema = close.ewm(span=period, adjust=False).mean()

        if value_col == "ema":
            return ema.values if hasattr(ema, "values") else np.array(ema, dtype=float)

        close = df["close"].astype(float)
        diff = close - ema
        return diff.values if hasattr(diff, "values") else np.array(diff, dtype=float)

    def _get_vol_metrics(self, df: pd.DataFrame) -> dict | None:
        if "close" not in df.columns:
            return None

        close = df["close"].astype(float)
        
        # Use pre-computed volatility features if available (for single-candle predictions)
        if "volatility_5" in df.columns and "volatility_20" in df.columns and len(df) == 1:
            # These are now returns-based volatility (not price-based)
            vol_fast = df["volatility_5"].astype(float).values
            vol_slow = df["volatility_20"].astype(float).values
            # Compute ratio, handling zeros
            vol_ratio = np.where(
                (vol_slow != 0) & np.isfinite(vol_slow),
                vol_fast / vol_slow,
                np.nan
            )
            return {
                "fast": vol_fast,
                "slow": vol_slow,
                "ratio": vol_ratio,
            }
        
        returns = np.log(close / close.shift(1))

        fast_window = max(2, int(self.params.get("vol_fast_window", 20)))
        slow_window = max(fast_window + 1, int(self.params.get("vol_slow_window", 50)))

        vol_fast = returns.rolling(fast_window).std()
        vol_slow = returns.rolling(slow_window).std()
        vol_ratio = vol_fast / vol_slow.replace(0, np.nan)
        vol_ratio = vol_ratio.replace([np.inf, -np.inf], np.nan)

        return {
            "fast": vol_fast.values,
            "slow": vol_slow.values,
            "ratio": vol_ratio.values,
        }
