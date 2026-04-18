import numpy as np
import pandas as pd

from predictor.strategies.base import BaseStrategy
from predictor.features import add_technical_features, bollinger_components


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
        }

    def fit(self, df: pd.DataFrame, horizon: int = 1) -> None:
        # Learn adaptive thresholds from training window
        if "rsi_14" not in df.columns:
            df = add_technical_features(df)

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

    def predict_proba(self, df: pd.DataFrame, horizon: int = 1) -> np.ndarray:
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
        if hasattr(self, '_rsi_p10') and self._rsi_p10 is not None:
            oversold = (base_oversold + self._rsi_p10) / 2
            overbought = (base_overbought + self._rsi_p90) / 2
        else:
            oversold = base_oversold
            overbought = base_overbought

        # Vectorized RSI signal
        proba = np.full(len(rsi), 0.5)
        os_mask = rsi < oversold
        ob_mask = rsi > overbought
        proba[os_mask] = 0.5 + ((oversold - rsi[os_mask]) / oversold) * 0.3
        proba[ob_mask] = 0.5 - ((rsi[ob_mask] - overbought) / (100 - overbought)) * 0.3

        # Vectorized BB confirmation
        if self.params["use_bb_confirm"]:
            bb = np.nan_to_num(bb_pos, nan=0.5)
            proba[bb < self.params["bb_low"]] += 0.05
            proba[bb > self.params["bb_high"]] -= 0.05

        # Volatility filter — skip signals outside desired band
        vol_metrics = self._get_vol_metrics(df)
        if vol_metrics is not None:
            vol_fast = vol_metrics["fast"]
            vol_slow = vol_metrics["slow"]
            vol_ratio = vol_metrics["ratio"]

            min_vol = float(self.params.get("min_vol", 0.0))
            max_vol = float(self.params.get("max_vol", 1.0))
            ratio_max = float(self.params.get("vol_ratio_max", 1.5))
            spike_mult = float(self.params.get("vol_spike_multiplier", 1.5))
            if max_vol <= min_vol:
                max_vol = min_vol + 1e-6

            valid = np.isfinite(vol_fast) & np.isfinite(vol_slow)
            valid &= (vol_fast > min_vol) & (vol_fast < max_vol)
            valid &= vol_fast < (vol_slow * spike_mult)
            if ratio_max > 0:
                ratio_safe = np.where(np.isfinite(vol_ratio), vol_ratio, np.inf)
                valid &= ratio_safe < ratio_max

            proba[~valid] = 0.5

        return np.clip(proba, 0, 1)

    def predict(self, df: pd.DataFrame, horizon: int = 1) -> np.ndarray:
        proba = self.predict_proba(df, horizon)
        preds = np.full(len(proba), -1, dtype=np.int8)
        preds[proba > 0.55] = 1
        preds[proba < 0.45] = 0
        return preds

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
