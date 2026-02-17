import numpy as np
import pandas as pd

from predictor.strategies.base import BaseStrategy
from predictor.features import add_technical_features


class MomentumStrategy(BaseStrategy):

    @staticmethod
    def description() -> str:
        return (
            "Multi-timeframe momentum strategy. Combines MACD histogram direction, "
            "EMA crossovers (5/20), volume surge detection, and price momentum "
            "to predict continuation or reversal. Trend-following approach."
        )

    @staticmethod
    def default_params() -> dict:
        return {
            "ema_fast": 5,
            "ema_slow": 20,
            "macd_weight": 0.35,
            "ema_weight": 0.3,
            "volume_weight": 0.2,
            "momentum_weight": 0.15,
            "volume_surge_threshold": 1.5,
        }

    @staticmethod
    def param_docs() -> dict:
        return {
            "ema_fast": "Fast EMA period for trend detection. Lower = more sensitive. Typical: 3-8.",
            "ema_slow": "Slow EMA period for trend detection. Higher = smoother trend. Typical: 15-30.",
            "macd_weight": "Weight of MACD histogram signal in final decision (0-1). Typical: 0.25-0.45.",
            "ema_weight": "Weight of EMA crossover signal in final decision (0-1). Typical: 0.2-0.4.",
            "volume_weight": "Weight of volume surge signal in final decision (0-1). Typical: 0.1-0.3.",
            "momentum_weight": "Weight of price momentum signal in final decision (0-1). Typical: 0.1-0.2.",
            "volume_surge_threshold": "Volume surge multiplier vs recent average. Higher = rarer signals. Typical: 1.3-2.0.",
        }

    def fit(self, df: pd.DataFrame, horizon: int = 1) -> None:
        # Learn adaptive baselines from training window
        if "macd_hist" not in df.columns:
            df = add_technical_features(df)
        df = df.fillna(0)

        macd = df["macd_hist"].values if "macd_hist" in df.columns else np.zeros(len(df))
        vol_r = df["volume_ratio"].values if "volume_ratio" in df.columns else np.ones(len(df))
        mom = df["momentum_3"].values if "momentum_3" in df.columns else np.zeros(len(df))

        if len(df) > 100:
            self._macd_mean = float(np.mean(macd))
            self._macd_std = float(np.std(macd)) or 1.0
            self._vol_median = float(np.median(vol_r))
            self._mom_mean = float(np.mean(mom))
        else:
            self._macd_mean = 0.0
            self._macd_std = 1.0
            self._vol_median = 1.0
            self._mom_mean = 0.0

    def predict_proba(self, df: pd.DataFrame, horizon: int = 1) -> np.ndarray:
        # Use pre-computed features if available, else compute
        if "macd_hist" not in df.columns:
            df = add_technical_features(df)
        df = df.fillna(0)

        fast_key = f"ema_{self.params['ema_fast']}"
        slow_key = f"ema_{self.params['ema_slow']}"

        n = len(df)
        macd_hist = df["macd_hist"].values if "macd_hist" in df.columns else np.zeros(n)
        ema_fast = df[fast_key].values if fast_key in df.columns else df["close"].values
        ema_slow = df[slow_key].values if slow_key in df.columns else df["close"].values
        vol_ratio = df["volume_ratio"].values if "volume_ratio" in df.columns else np.ones(n)
        mom = df["momentum_3"].values if "momentum_3" in df.columns else np.zeros(n)

        w_macd = self.params["macd_weight"]
        w_ema = self.params["ema_weight"]
        w_vol = self.params["volume_weight"]
        w_mom = self.params["momentum_weight"]

        # Adaptive baselines from fit()
        macd_baseline = getattr(self, '_macd_mean', 0.0)
        macd_scale = getattr(self, '_macd_std', 1.0)
        vol_baseline = getattr(self, '_vol_median', 1.0)
        mom_baseline = getattr(self, '_mom_mean', 0.0)

        # Normalize MACD relative to training baseline
        macd_norm = (macd_hist - macd_baseline) / macd_scale

        # Vectorized scoring
        score = np.zeros(n)

        # MACD: use normalized values relative to training window
        macd_prev = np.roll(macd_norm, 1)
        macd_prev[0] = 0
        score += np.where((macd_norm > 0) & (macd_norm > macd_prev), w_macd,
                 np.where((macd_norm < 0) & (macd_norm < macd_prev), -w_macd, 0.0))

        # EMA crossover
        score += np.where(ema_fast > ema_slow, w_ema, -w_ema)

        # Volume surge: relative to training median instead of fixed threshold
        adaptive_surge_thresh = self.params["volume_surge_threshold"] * vol_baseline
        surge = vol_ratio > adaptive_surge_thresh
        mom_adj = mom - mom_baseline  # momentum relative to training bias
        score += np.where(surge & (mom_adj > 0), w_vol,
                 np.where(surge & (mom_adj < 0), -w_vol, 0.0))

        # Momentum relative to training bias
        score += np.where(mom_adj > 0, w_mom, np.where(mom_adj < 0, -w_mom, 0.0))

        # First element stays at 0.5
        score[0] = 0.0

        proba = 0.5 + score * 0.5
        return np.clip(proba, 0, 1)

    def predict(self, df: pd.DataFrame, horizon: int = 1) -> np.ndarray:
        proba = self.predict_proba(df, horizon)
        preds = np.full(len(proba), -1, dtype=np.int8)
        preds[proba > 0.55] = 1
        preds[proba < 0.45] = 0
        return preds
