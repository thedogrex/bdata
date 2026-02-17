import numpy as np
import pandas as pd
from collections import Counter

from predictor.strategies.base import BaseStrategy
from predictor.features import add_technical_features


class PatternSequenceStrategy(BaseStrategy):

    def __init__(self, params: dict):
        super().__init__(params)
        self.pattern_probs: dict[tuple, float] = {}

    @staticmethod
    def description() -> str:
        return (
            "Pattern sequence strategy. Analyzes historical sequences of N candle directions "
            "(UP/DOWN) and calculates the conditional probability of the next candle being UP "
            "given each observed pattern. Uses Laplace smoothing. "
            "Can combine multiple lookback lengths for a weighted vote."
        )

    @staticmethod
    def default_params() -> dict:
        return {
            "lookback_lengths": [3, 4, 5, 6, 7],
            "weights": [0.1, 0.15, 0.25, 0.25, 0.25],
            "min_occurrences": 5,
        }

    @staticmethod
    def param_docs() -> dict:
        return {
            "lookback_lengths": "Pattern lengths to consider (list). Longer patterns capture more context but are rarer. Typical: [3,4,5,6,7].",
            "weights": "Voting weights for each lookback length (list). Must sum to 1. Higher weight = more influence. Typical: [0.1,0.15,0.25,0.25,0.25].",
            "min_occurrences": "Minimum times a pattern must have appeared historically to trust its probability. Higher = fewer but more reliable signals. Typical: 3-20.",
        }

    def fit(self, df: pd.DataFrame, horizon: int = 1) -> None:
        # Only need direction column - skip full feature computation if available
        if "direction" not in df.columns:
            df = add_technical_features(df).fillna(0)
        directions = df["direction"].values

        target = np.zeros(len(directions), dtype=np.int8)
        if horizon <= len(directions) - 1:
            for i in range(len(directions) - horizon):
                target[i] = directions[i + horizon]

        self.pattern_probs = {}
        min_occ = self.params["min_occurrences"]

        for length in self.params["lookback_lengths"]:
            up_counts: dict[tuple, int] = Counter()
            total_counts: dict[tuple, int] = Counter()

            for i in range(length, len(directions) - horizon):
                pattern = tuple(directions[i - length: i])
                total_counts[pattern] += 1
                if target[i] == 1:
                    up_counts[pattern] += 1

            for pattern, total in total_counts.items():
                if total >= min_occ:
                    # Laplace smoothing
                    prob = (up_counts.get(pattern, 0) + 1) / (total + 2)
                    self.pattern_probs[(length, pattern)] = prob

    def predict_proba(self, df: pd.DataFrame, horizon: int = 1) -> np.ndarray:
        if "direction" not in df.columns:
            df = add_technical_features(df).fillna(0)
        directions = df["direction"].values
        proba = np.full(len(df), 0.5)

        lengths = self.params["lookback_lengths"]
        weights = self.params["weights"]
        if len(weights) != len(lengths):
            weights = [1.0 / len(lengths)] * len(lengths)

        max_len = max(lengths)

        for i in range(max_len, len(directions)):
            weighted_prob = 0.0
            total_weight = 0.0

            for length, w in zip(lengths, weights):
                pattern = tuple(directions[i - length: i])
                key = (length, pattern)
                if key in self.pattern_probs:
                    weighted_prob += w * self.pattern_probs[key]
                    total_weight += w

            if total_weight > 0:
                proba[i] = weighted_prob / total_weight

        return proba

    def predict(self, df: pd.DataFrame, horizon: int = 1) -> np.ndarray:
        proba = self.predict_proba(df, horizon)
        preds = np.full(len(proba), -1, dtype=np.int8)
        preds[proba > 0.53] = 1
        preds[proba < 0.47] = 0
        return preds
