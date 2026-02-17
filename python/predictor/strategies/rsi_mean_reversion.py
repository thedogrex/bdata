import numpy as np
import pandas as pd

from predictor.strategies.base import BaseStrategy
from predictor.features import add_technical_features


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
        }

    def fit(self, df: pd.DataFrame, horizon: int = 1) -> None:
        # Rule-based strategy — no training needed
        pass

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
        bb_pos = df["bb_pos"].values.copy() if "bb_pos" in df.columns else np.full(len(df), 0.5)

        oversold = self.params["rsi_oversold"]
        overbought = self.params["rsi_overbought"]

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

        return np.clip(proba, 0, 1)

    def predict(self, df: pd.DataFrame, horizon: int = 1) -> np.ndarray:
        proba = self.predict_proba(df, horizon)
        preds = np.full(len(proba), -1, dtype=np.int8)
        preds[proba > 0.55] = 1
        preds[proba < 0.45] = 0
        return preds
