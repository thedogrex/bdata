import numpy as np
import pandas as pd

from predictor.strategies.base import BaseStrategy
from predictor.features import add_technical_features


class StochasticADXStrategy(BaseStrategy):
    """
    Combines Stochastic Oscillator (%K/%D) with ADX trend strength.
    - Strong trend (ADX > threshold) + Stochastic extreme → high-confidence signal
    - Weak trend → reduce confidence or skip
    Adaptive: fit() learns ADX/Stochastic baselines from training window.
    """

    @staticmethod
    def description() -> str:
        return (
            "Stochastic Oscillator + ADX trend filter strategy. Uses %K/%D crossovers "
            "for entry signals, filtered by ADX trend strength. Strong trends amplify "
            "signals, weak trends reduce confidence. Adaptive baselines from training window."
        )

    @staticmethod
    def default_params() -> dict:
        return {
            "stoch_oversold": 20,
            "stoch_overbought": 80,
            "adx_strong": 25,
            "adx_weak": 15,
            "trend_weight": 0.4,
            "stoch_weight": 0.4,
            "crossover_weight": 0.2,
        }

    @staticmethod
    def param_docs() -> dict:
        return {
            "stoch_oversold": "Stochastic %K level considered oversold. Typical: 15-30.",
            "stoch_overbought": "Stochastic %K level considered overbought. Typical: 70-85.",
            "adx_strong": "ADX level indicating strong trend. Typical: 20-35.",
            "adx_weak": "ADX level indicating weak/no trend. Typical: 10-20.",
            "trend_weight": "Weight of ADX trend signal (0-1). Typical: 0.2-0.5.",
            "stoch_weight": "Weight of Stochastic signal (0-1). Typical: 0.3-0.5.",
            "crossover_weight": "Weight of %K/%D crossover signal (0-1). Typical: 0.1-0.3.",
        }

    def fit(self, df: pd.DataFrame, horizon: int = 1) -> None:
        if "stoch_k" not in df.columns:
            df = add_technical_features(df)

        stoch = df["stoch_k"].dropna().values
        adx = df["adx"].dropna().values

        if len(stoch) > 100:
            self._stoch_median = float(np.median(stoch))
            self._stoch_p15 = float(np.percentile(stoch, 15))
            self._stoch_p85 = float(np.percentile(stoch, 85))
            self._adx_median = float(np.median(adx)) if len(adx) > 100 else 20.0
        else:
            self._stoch_median = 50.0
            self._stoch_p15 = None
            self._stoch_p85 = None
            self._adx_median = 20.0

    def predict_proba(self, df: pd.DataFrame, horizon: int = 1) -> np.ndarray:
        if "stoch_k" not in df.columns:
            df = add_technical_features(df)

        stoch_k = np.nan_to_num(df["stoch_k"].values, nan=50.0)
        stoch_d = np.nan_to_num(df["stoch_d"].values, nan=50.0)
        adx = np.nan_to_num(df["adx"].values, nan=20.0)
        di_plus = np.nan_to_num(df["di_plus"].values, nan=25.0) if "di_plus" in df.columns else np.full(len(df), 25.0)
        di_minus = np.nan_to_num(df["di_minus"].values, nan=25.0) if "di_minus" in df.columns else np.full(len(df), 25.0)

        # Adaptive thresholds
        if hasattr(self, '_stoch_p15') and self._stoch_p15 is not None:
            oversold = (self.params["stoch_oversold"] + self._stoch_p15) / 2
            overbought = (self.params["stoch_overbought"] + self._stoch_p85) / 2
        else:
            oversold = self.params["stoch_oversold"]
            overbought = self.params["stoch_overbought"]

        adx_strong = self.params["adx_strong"]
        adx_weak = self.params["adx_weak"]
        w_trend = self.params["trend_weight"]
        w_stoch = self.params["stoch_weight"]
        w_cross = self.params["crossover_weight"]

        n = len(stoch_k)
        score = np.zeros(n)

        # Stochastic signal: oversold = bullish, overbought = bearish
        os_mask = stoch_k < oversold
        ob_mask = stoch_k > overbought
        score[os_mask] += w_stoch * (oversold - stoch_k[os_mask]) / oversold
        score[ob_mask] -= w_stoch * (stoch_k[ob_mask] - overbought) / (100 - overbought)

        # %K/%D crossover
        cross_up = (stoch_k > stoch_d) & (np.roll(stoch_k, 1) <= np.roll(stoch_d, 1))
        cross_down = (stoch_k < stoch_d) & (np.roll(stoch_k, 1) >= np.roll(stoch_d, 1))
        score[cross_up] += w_cross
        score[cross_down] -= w_cross

        # ADX trend strength + DI direction
        strong_trend = adx > adx_strong
        weak_trend = adx < adx_weak
        bullish_trend = di_plus > di_minus
        bearish_trend = di_minus > di_plus

        score[strong_trend & bullish_trend] += w_trend
        score[strong_trend & bearish_trend] -= w_trend
        score[weak_trend] *= 0.5  # reduce confidence in weak trends

        # First element neutral
        score[0] = 0.0

        proba = 0.5 + np.clip(score, -0.5, 0.5)
        return np.clip(proba, 0, 1)

    def predict(self, df: pd.DataFrame, horizon: int = 1) -> np.ndarray:
        proba = self.predict_proba(df, horizon)
        preds = np.full(len(proba), -1, dtype=np.int8)
        preds[proba > 0.55] = 1
        preds[proba < 0.45] = 0
        return preds
