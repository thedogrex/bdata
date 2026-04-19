import asyncio
import time

import numpy as np
import pandas as pd

from predictor.strategies.base import BaseStrategy
from predictor.features import add_technical_features, bollinger_components
from predictor.utils.prediction_thresholds import classify_probability, resolve_probability_threshold


class RSIMeanReversionStrategy(BaseStrategy):

    @staticmethod
    def description() -> str:
        return (
            "RSI mean-reversion strategy. Predicts UP when RSI is oversold "
            "(below lower threshold) and DOWN when RSI is overbought (above upper threshold). "
            "Combines RSI-6 and RSI-14 with Bollinger Band position for confirmation."
        )

    @staticmethod
    def default_params() -> dict:
        return {
            "rsi_period": 14,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "use_bb_confirm": True,
            "bb_low": 0.2,
            "bb_high": 0.8,
            "bb_period": 20,
            "bb_std": 2.0,
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
            "use_bb_confirm": "Whether to require Bollinger Band position confirmation. Reduces false signals.",
            "bb_low": "Lower Bollinger Band threshold for confirmation (0-1). Price must be below this to buy. Typical: 0.15-0.25.",
            "bb_high": "Upper Bollinger Band threshold for confirmation (0-1). Price must be above this to sell. Typical: 0.75-0.85.",
            "bb_period": "Bollinger Band lookback period for confirmation. Typical: 10-50.",
            "bb_std": "Bollinger Band standard deviation multiplier. Typical: 1.0-3.0.",
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
        if length == 0:
            self._last_vol_skip_count = 0
            self._last_vol_skip_flags = np.array([], dtype=bool)
            return np.array([], dtype=float)

        proba = np.full(length, 0.5)
        bb_arr = None if bb_pos is None else np.asarray(bb_pos, dtype=float)
        vol_metrics = self._get_vol_metrics(df)
        vol_skip_flags = np.zeros(length, dtype=bool)

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
                proba_chunk[bb_chunk < self.params["bb_low"]] += 0.05
                proba_chunk[bb_chunk > self.params["bb_high"]] -= 0.05

            if vol_fast is not None and vol_slow is not None and vol_ratio is not None:
                vf = vol_fast[chunk]
                vs = vol_slow[chunk]
                ratio = vol_ratio[chunk]
                valid = np.isfinite(vf) & np.isfinite(vs)
                valid &= (vf > min_vol) & (vf < max_vol)
                valid &= vf < (vs * spike_mult)
                if ratio_max > 0:
                    ratio_mask = np.where(np.isfinite(ratio), ratio, np.inf)
                    valid &= ratio_mask < ratio_max

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
        await maybe_yield(force=False)

        np.clip(proba, 0, 1, out=proba)
        return proba

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

    def _get_vol_metrics(self, df: pd.DataFrame) -> dict | None:
        if "close" not in df.columns:
            return None

        close = df["close"].astype(float)
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
