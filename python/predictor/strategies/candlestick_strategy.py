import numpy as np
import pandas as pd

from predictor.strategies.base import BaseStrategy
from predictor.features import add_technical_features


class CandlestickPatternStrategy(BaseStrategy):
    """
    Pure candlestick pattern strategy. Combines multiple pattern signals
    (hammer, engulfing, doji, morning/evening star) with volume confirmation
    and trend context (EMA). Adaptive: fit() learns pattern hit rates from
    training window to weight patterns by historical reliability.
    """

    @staticmethod
    def description() -> str:
        return (
            "Candlestick pattern recognition strategy. Detects Hammer, Engulfing, "
            "Doji, Morning/Evening Star patterns and combines them with volume "
            "confirmation and EMA trend context. Learns pattern reliability from "
            "training data to weight signals adaptively."
        )

    @staticmethod
    def default_params() -> dict:
        return {
            "pattern_weight": 0.5,
            "volume_confirm_weight": 0.2,
            "trend_context_weight": 0.3,
            "volume_surge_threshold": 1.3,
            "ema_trend_period": 20,
            "min_pattern_score": 0.1,
        }

    @staticmethod
    def param_docs() -> dict:
        return {
            "pattern_weight": "Weight of candlestick pattern signals (0-1). Typical: 0.3-0.6.",
            "volume_confirm_weight": "Weight of volume confirmation (0-1). Typical: 0.1-0.3.",
            "trend_context_weight": "Weight of EMA trend context (0-1). Typical: 0.2-0.4.",
            "volume_surge_threshold": "Volume ratio above which confirms pattern. Typical: 1.2-2.0.",
            "ema_trend_period": "EMA period for trend context. Typical: 10-50.",
            "min_pattern_score": "Minimum pattern score to generate signal. Typical: 0.05-0.2.",
        }

    def fit(self, df: pd.DataFrame, horizon: int = 1) -> None:
        if "is_hammer" not in df.columns:
            df = add_technical_features(df)

        target_col = f"future_dir_{horizon}"
        if target_col not in df.columns:
            df[target_col] = (df["close"].shift(-horizon) > df["open"].shift(-horizon)).astype(np.int8)

        df_clean = df.dropna(subset=[target_col])
        y = df_clean[target_col].values

        # Learn pattern reliability: P(up | pattern) for bullish, P(down | pattern) for bearish
        self._pattern_scores = {}
        bullish_patterns = ["is_hammer", "is_engulfing_bull", "is_morning_star"]
        bearish_patterns = ["is_inv_hammer", "is_engulfing_bear", "is_evening_star"]

        for pat in bullish_patterns:
            if pat in df_clean.columns:
                mask = df_clean[pat].values == 1
                if mask.sum() > 5:
                    hit_rate = float(y[mask].mean())  # P(up | pattern)
                    self._pattern_scores[pat] = hit_rate - 0.5  # centered
                else:
                    self._pattern_scores[pat] = 0.05  # small default bullish

        for pat in bearish_patterns:
            if pat in df_clean.columns:
                mask = df_clean[pat].values == 1
                if mask.sum() > 5:
                    hit_rate = float((1 - y[mask]).mean())  # P(down | pattern)
                    self._pattern_scores[pat] = -(hit_rate - 0.5)  # centered, negative = bearish
                else:
                    self._pattern_scores[pat] = -0.05

        # Learn volume baseline
        vol_r = df_clean["volume_ratio"].dropna().values if "volume_ratio" in df_clean.columns else None
        self._vol_median = float(np.median(vol_r)) if vol_r is not None and len(vol_r) > 100 else 1.0

    def predict_proba(self, df: pd.DataFrame, horizon: int = 1) -> np.ndarray:
        if "is_hammer" not in df.columns:
            df = add_technical_features(df)

        n = len(df)
        w_pat = self.params["pattern_weight"]
        w_vol = self.params["volume_confirm_weight"]
        w_trend = self.params["trend_context_weight"]
        vol_thresh = self.params["volume_surge_threshold"]
        min_score = self.params["min_pattern_score"]

        # Pattern score
        pat_score = np.zeros(n)
        pattern_scores = getattr(self, '_pattern_scores', {})
        for pat, weight in pattern_scores.items():
            if pat in df.columns:
                pat_score += df[pat].fillna(0).values * weight

        # Volume confirmation
        vol_ratio = df["volume_ratio"].fillna(1.0).values if "volume_ratio" in df.columns else np.ones(n)
        vol_baseline = getattr(self, '_vol_median', 1.0)
        vol_confirm = (vol_ratio > vol_thresh * vol_baseline).astype(float)

        # Trend context from EMA
        ema_col = f"ema_{self.params['ema_trend_period']}"
        if ema_col in df.columns:
            close = df["close"].to_numpy()
            ema_raw = df[ema_col].to_numpy()
            ema = np.where(np.isnan(ema_raw), close, ema_raw)
            trend = np.where(close > ema, 1.0, -1.0)
        else:
            trend = np.zeros(n)

        # Combine
        score = np.zeros(n)
        score += w_pat * pat_score
        # Volume amplifies pattern signal direction
        score += w_vol * vol_confirm * np.sign(pat_score) * 0.3
        # Trend context: adds confidence when pattern aligns with trend
        aligned = np.sign(pat_score) == np.sign(trend)
        score[aligned] += w_trend * 0.2
        score[~aligned] -= w_trend * 0.1

        # Only emit signal if pattern score is meaningful
        weak = np.abs(pat_score) < min_score
        score[weak] = 0.0

        proba = 0.5 + np.clip(score, -0.5, 0.5)
        return np.clip(proba, 0, 1)

    def predict(self, df: pd.DataFrame, horizon: int = 1) -> np.ndarray:
        proba = self.predict_proba(df, horizon)
        preds = np.full(len(proba), -1, dtype=np.int8)
        preds[proba > 0.55] = 1
        preds[proba < 0.45] = 0
        return preds
