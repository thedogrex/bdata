import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from predictor.strategies.base import BaseStrategy
from predictor.features import add_technical_features, get_feature_columns


class RandomForestStrategy(BaseStrategy):

    def __init__(self, params: dict):
        super().__init__(params)
        self.model = None
        self.feature_cols = []

    @staticmethod
    def description() -> str:
        return (
            "Random Forest classifier — robust ensemble of decision trees. "
            "Less prone to overfitting than single boosting models. Uses all "
            "technical indicators, candlestick patterns, and interaction features. "
            "Good baseline model with interpretable feature importances."
        )

    @staticmethod
    def default_params() -> dict:
        return {
            "n_estimators": 200,
            "max_depth": 8,
            "min_samples_leaf": 20,
            "max_features": "sqrt",
            "threshold": 0.52,
        }

    @staticmethod
    def param_docs() -> dict:
        return {
            "n_estimators": "Number of trees in the forest. More = more stable but slower. Typical: 100-500.",
            "max_depth": "Maximum tree depth. None for unlimited. Controls complexity. Typical: 5-15.",
            "min_samples_leaf": "Minimum samples in a leaf node. Higher = more conservative. Typical: 10-50.",
            "max_features": "Features considered per split. 'sqrt' or float (0-1). Typical: 'sqrt' or 0.5-0.8.",
            "threshold": "Minimum probability to emit a signal. Lower = more trades. Typical: 0.50-0.55.",
        }

    def fit(self, df: pd.DataFrame, horizon: int = 1) -> None:
        if "rsi_14" not in df.columns:
            df = add_technical_features(df)
        target_col = f"future_dir_{horizon}"
        if target_col not in df.columns:
            df[target_col] = (df["close"].shift(-horizon) > df["open"].shift(-horizon)).astype(np.int8)

        df = df.dropna()
        self.feature_cols = get_feature_columns(df)

        X = df[self.feature_cols].to_numpy(dtype=np.float32, copy=False)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        y = df[target_col].to_numpy(copy=False)

        max_features = self.params["max_features"]
        if isinstance(max_features, str) and max_features not in ("sqrt", "log2"):
            try:
                max_features = float(max_features)
            except (ValueError, TypeError):
                max_features = "sqrt"

        self.model = RandomForestClassifier(
            n_estimators=self.params["n_estimators"],
            max_depth=self.params["max_depth"],
            min_samples_leaf=self.params["min_samples_leaf"],
            max_features=max_features,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X, y)

    def predict_proba_row(self, x_row: np.ndarray) -> float:
        """Fast path: predict probability for a single candle feature row."""
        if self.model is None or not self.feature_cols:
            return 0.5
        x_row = np.nan_to_num(x_row, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        return float(self.model.predict_proba(x_row.reshape(1, -1))[0, 1])

    def predict_row(self, x_row: np.ndarray) -> int:
        """Fast path: predict class for a single candle feature row."""
        p = self.predict_proba_row(x_row)
        threshold = self.params["threshold"]
        if p > threshold:
            return 1
        if p < (1 - threshold):
            return 0
        return -1

    def predict_proba(self, df: pd.DataFrame, horizon: int = 1) -> np.ndarray:
        if "rsi_14" not in df.columns:
            df = add_technical_features(df)
        df = df.fillna(0)
        X = df[self.feature_cols].to_numpy(dtype=np.float32, copy=False)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        return self.model.predict_proba(X)[:, 1]

    def predict(self, df: pd.DataFrame, horizon: int = 1) -> np.ndarray:
        proba = self.predict_proba(df, horizon)
        threshold = self.params["threshold"]
        preds = np.full(len(proba), -1, dtype=np.int8)
        preds[proba > threshold] = 1
        preds[proba < (1 - threshold)] = 0
        return preds
