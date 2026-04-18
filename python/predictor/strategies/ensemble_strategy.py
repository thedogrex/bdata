import numpy as np
import pandas as pd

from predictor.strategies.base import BaseStrategy
from predictor.strategies.xgboost_strategy import XGBoostStrategy
from predictor.strategies.rsi_mean_reversion import RSIMeanReversionStrategy
from predictor.strategies.momentum_strategy import MomentumStrategy
from predictor.strategies.pattern_sequence import PatternSequenceStrategy
from predictor.utils.async_utils import resolve_awaitable


class EnsembleStrategy(BaseStrategy):

    def __init__(self, params: dict):
        super().__init__(params)
        self.sub_strategies: list[BaseStrategy] = []

    @staticmethod
    def description() -> str:
        return (
            "Ensemble strategy combining XGBoost, RSI Mean Reversion, Momentum, "
            "and Pattern Sequence strategies via weighted soft voting. "
            "Each sub-strategy produces a probability, and the ensemble averages them "
            "with configurable weights. Best for achieving stable 54-55% accuracy."
        )

    @staticmethod
    def default_params() -> dict:
        return {
            "xgboost_weight": 0.4,
            "rsi_weight": 0.15,
            "momentum_weight": 0.2,
            "pattern_weight": 0.25,
            "threshold": 0.53,
            "xgboost_params": {
                "n_estimators": 300,
                "max_depth": 4,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "threshold": 0.53,
            },
            "rsi_params": {
                "rsi_period": 14,
                "rsi_oversold": 30,
                "rsi_overbought": 70,
                "use_bb_confirm": True,
                "bb_low": 0.2,
                "bb_high": 0.8,
            },
            "momentum_params": {
                "ema_fast": 5,
                "ema_slow": 20,
                "macd_weight": 0.35,
                "ema_weight": 0.3,
                "volume_weight": 0.2,
                "momentum_weight": 0.15,
                "volume_surge_threshold": 1.5,
            },
            "pattern_params": {
                "lookback_lengths": [3, 4, 5, 6, 7],
                "weights": [0.1, 0.15, 0.25, 0.25, 0.25],
                "min_occurrences": 5,
            },
        }

    @staticmethod
    def param_docs() -> dict:
        return {
            "xgboost_weight": "Weight of XGBoost sub-strategy in ensemble (0-1). Higher = more ML influence. Typical: 0.3-0.5.",
            "rsi_weight": "Weight of RSI Mean Reversion sub-strategy in ensemble (0-1). Higher = more mean-reversion bias. Typical: 0.1-0.3.",
            "momentum_weight": "Weight of Momentum sub-strategy in ensemble (0-1). Higher = more trend-following bias. Typical: 0.1-0.3.",
            "pattern_weight": "Weight of Pattern Sequence sub-strategy in ensemble (0-1). Higher = more pattern-based bias. Typical: 0.2-0.4.",
            "threshold": "Minimum ensemble probability to make a prediction. Higher = fewer but more confident signals. Typical: 0.50-0.55.",
            "xgboost_params": "Nested parameters for XGBoost sub-strategy. See XGBoost docs.",
            "rsi_params": "Nested parameters for RSI Mean Reversion sub-strategy. See RSI docs.",
            "momentum_params": "Nested parameters for Momentum sub-strategy. See Momentum docs.",
            "pattern_params": "Nested parameters for Pattern Sequence sub-strategy. See Pattern docs.",
        }

    def fit(self, df: pd.DataFrame, horizon: int = 1) -> None:
        self.sub_strategies = [
            XGBoostStrategy(self.params.get("xgboost_params", {})),
            RSIMeanReversionStrategy(self.params.get("rsi_params", {})),
            MomentumStrategy(self.params.get("momentum_params", {})),
            PatternSequenceStrategy(self.params.get("pattern_params", {})),
        ]
        for s in self.sub_strategies:
            s.fit(df, horizon)

    async def predict_proba(self, df: pd.DataFrame, horizon: int = 1) -> np.ndarray:
        weights = [
            self.params["xgboost_weight"],
            self.params["rsi_weight"],
            self.params["momentum_weight"],
            self.params["pattern_weight"],
        ]
        total_w = sum(weights)
        weights = [w / total_w for w in weights]

        probas = []
        for s in self.sub_strategies:
            probas.append(await resolve_awaitable(s.predict_proba(df, horizon)))

        combined = np.zeros(len(df))
        for w, p in zip(weights, probas):
            combined += w * p

        return combined

    async def predict(self, df: pd.DataFrame, horizon: int = 1) -> np.ndarray:
        proba = await self.predict_proba(df, horizon)
        threshold = self.params["threshold"]
        preds = np.full(len(proba), -1, dtype=np.int8)
        preds[proba > threshold] = 1
        preds[proba < (1 - threshold)] = 0
        return preds
