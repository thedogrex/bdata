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
            "n_estimators": 200,
            "max_depth": 3,
            "learning_rate": 0.08,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "min_child_weight": 5,
            "reg_lambda": 1.0,
            "reg_alpha": 0.0,
            "gamma": 0.0,
            "tree_method": "hist",
            "max_bin": 256,
            "n_jobs": 4,
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
            "min_child_weight": "Minimum sum of instance weight (hessian) needed in a child. Higher values reduce overfitting and can speed up training. Typical: 1-10.",
            "reg_lambda": "L2 regularization term on weights. Typical: 1-10.",
            "reg_alpha": "L1 regularization term on weights. Typical: 0-1.",
            "gamma": "Minimum loss reduction required to make a split. Higher values make the model more conservative. Typical: 0-2.",
            "tree_method": "Tree construction algorithm. Use 'hist' for fastest CPU training on tabular data.",
            "max_bin": "Max number of bins for histogram algorithm (tree_method='hist'). Lower can be faster. Typical: 128-512.",
            "n_jobs": "CPU threads for training/prediction. Limit this to keep realtime market polling responsive. Typical: 2-8.",
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
            n_estimators=int(self.params.get("n_estimators", 200)),
            max_depth=int(self.params.get("max_depth", 3)),
            learning_rate=float(self.params.get("learning_rate", 0.08)),
            subsample=float(self.params.get("subsample", 0.85)),
            colsample_bytree=float(self.params.get("colsample_bytree", 0.85)),
            min_child_weight=float(self.params.get("min_child_weight", 5)),
            reg_lambda=float(self.params.get("reg_lambda", 1.0)),
            reg_alpha=float(self.params.get("reg_alpha", 0.0)),
            gamma=float(self.params.get("gamma", 0.0)),
            tree_method=str(self.params.get("tree_method", "hist")),
            max_bin=int(self.params.get("max_bin", 256)),
            eval_metric="logloss",
            random_state=42,
            n_jobs=int(self.params.get("n_jobs", 4)),
            verbosity=0,
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
