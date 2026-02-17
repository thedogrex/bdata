import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from predictor.strategies.base import BaseStrategy
from predictor.features import add_technical_features, get_feature_columns


class XGBoostStrategy(BaseStrategy):

    def __init__(self, params: dict):
        super().__init__(params)
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = []

    @staticmethod
    def description() -> str:
        return (
            "XGBoost gradient boosting classifier with technical indicators. "
            "Uses RSI, MACD, Bollinger Bands, volume ratios, candle patterns, "
            "and lagged direction features. Supports confidence threshold filtering."
        )

    @staticmethod
    def default_params() -> dict:
        return {
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "threshold": 0.53,
        }

    @staticmethod
    def param_docs() -> dict:
        return {
            "n_estimators": "Number of trees in the ensemble. Higher values increase model capacity but training time. Typical: 100-500.",
            "max_depth": "Maximum tree depth. Controls model complexity. Higher values can overfit. Typical: 3-6.",
            "learning_rate": "Step size shrinkage. Lower values require more trees but often generalize better. Typical: 0.01-0.1.",
            "subsample": "Fraction of samples used per tree. Prevents overfitting. Typical: 0.6-1.0.",
            "colsample_bytree": "Fraction of features used per tree. Prevents overfitting. Typical: 0.6-1.0.",
            "threshold": "Minimum probability to make a prediction. Higher = fewer but more confident signals. Typical: 0.50-0.55.",
        }

    def fit(self, df: pd.DataFrame, horizon: int = 1) -> None:
        if "rsi_14" not in df.columns:
            df = add_technical_features(df)
        target_col = f"future_dir_{horizon}"
        if target_col not in df.columns:
            df[target_col] = (df["close"].shift(-horizon) > df["open"].shift(-horizon)).astype(np.int8)

        df = df.dropna()
        self.feature_cols = get_feature_columns(df)

        X = df[self.feature_cols].values
        y = df[target_col].values

        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)

        self.model = xgb.XGBClassifier(
            n_estimators=self.params["n_estimators"],
            max_depth=self.params["max_depth"],
            learning_rate=self.params["learning_rate"],
            subsample=self.params["subsample"],
            colsample_bytree=self.params["colsample_bytree"],
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X_scaled, y)

    def predict_proba(self, df: pd.DataFrame, horizon: int = 1) -> np.ndarray:
        if "rsi_14" not in df.columns:
            df = add_technical_features(df)
        df = df.fillna(0)
        X = df[self.feature_cols].values
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def predict(self, df: pd.DataFrame, horizon: int = 1) -> np.ndarray:
        proba = self.predict_proba(df, horizon)
        threshold = self.params["threshold"]
        preds = np.full(len(proba), -1, dtype=np.int8)
        preds[proba > threshold] = 1
        preds[proba < (1 - threshold)] = 0
        return preds
