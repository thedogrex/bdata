import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

from predictor.strategies.base import BaseStrategy
from predictor.features import add_technical_features, get_feature_columns


class LightGBMStrategy(BaseStrategy):

    def __init__(self, params: dict):
        super().__init__(params)
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = []
        self.feature_importances_: dict[str, float] | None = None

    @staticmethod
    def description() -> str:
        return (
            "LightGBM gradient boosting classifier optimized for speed and accuracy. "
            "Uses all technical indicators, candlestick patterns, Stochastic, ADX/DMI, "
            "rolling stats, and indicator interaction features. Faster training than "
            "XGBoost with comparable or better accuracy on tabular data."
        )

    @staticmethod
    def default_params() -> dict:
        return {
            "n_estimators": 300,
            "max_depth": 5,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 20,
            "lambda_l1": 0.0,
            "lambda_l2": 0.0,
            "threshold": 0.52,
            "deterministic": True,
        }

    @staticmethod
    def param_docs() -> dict:
        return {
            "n_estimators": "Number of boosting rounds. Higher = more capacity but slower. Typical: 100-500.",
            "max_depth": "Maximum tree depth. -1 for no limit. Controls complexity. Typical: 3-7.",
            "learning_rate": "Step size shrinkage. Lower = more trees needed but better generalization. Typical: 0.01-0.1.",
            "num_leaves": "Max number of leaves per tree. Higher = more complex. Typical: 15-63.",
            "subsample": "Fraction of samples per tree. Prevents overfitting. Typical: 0.6-1.0.",
            "colsample_bytree": "Fraction of features per tree. Prevents overfitting. Typical: 0.6-1.0.",
            "min_child_samples": "Minimum samples in a leaf. Higher = more conservative. Typical: 10-50.",
            "lambda_l1": "L1 regularization. Encourages sparsity in leaf weights. Helps with noisy features. Typical: 0-1.",
            "lambda_l2": "L2 regularization. Smooths leaf weights. Reduces overfitting on noisy data. Typical: 0-1.",
            "threshold": "Minimum probability to emit a signal. Lower = more trades. Typical: 0.50-0.55.",
            "deterministic": "If True, disables subsampling and forces single-threaded training for reproducible results.",
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

        # Keep feature names consistent for sklearn validation + debugging
        X_scaled_df = pd.DataFrame(X_scaled, columns=self.feature_cols)

        deterministic = bool(self.params.get("deterministic", True))
        effective_subsample = 1.0 if deterministic else self.params["subsample"]
        effective_colsample = 1.0 if deterministic else self.params["colsample_bytree"]

        self.model = lgb.LGBMClassifier(
            n_estimators=self.params["n_estimators"],
            max_depth=self.params["max_depth"],
            learning_rate=self.params["learning_rate"],
            num_leaves=self.params["num_leaves"],
            subsample=effective_subsample,
            colsample_bytree=effective_colsample,
            min_child_samples=self.params["min_child_samples"],
            reg_alpha=self.params.get("lambda_l1", 0.0),
            reg_lambda=self.params.get("lambda_l2", 0.0),
            bagging_freq=0 if deterministic else 1,
            bagging_seed=42,
            feature_fraction_seed=42,
            random_state=42,
            n_jobs=1 if deterministic else -1,
            verbose=-1,
        )
        self.model.fit(X_scaled_df, y)
        booster = getattr(self.model, "booster_", None)
        if booster is not None:
            gains = booster.feature_importance(importance_type="gain")
        else:
            gains = self.model.feature_importances_
        self.feature_importances_ = {
            col: float(imp)
            for col, imp in zip(self.feature_cols, gains)
        }

    def get_feature_importance(self, top_n: int | None = None, normalize: bool = True) -> list[tuple[str, float]]:
        """Return sorted feature importances using gain metric."""
        if not self.feature_importances_:
            return []
        items = list(self.feature_importances_.items())
        total = sum(val for _, val in items) or 1.0
        if normalize:
            items = [(name, val / total) for name, val in items]
        items.sort(key=lambda x: x[1], reverse=True)
        if top_n is not None:
            items = items[:top_n]
        return items

    def predict_proba(self, df: pd.DataFrame, horizon: int = 1) -> np.ndarray:
        if "rsi_14" not in df.columns:
            df = add_technical_features(df)
        df = df.fillna(0)
        X = df[self.feature_cols].values
        X_scaled = self.scaler.transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=self.feature_cols)
        return self.model.predict_proba(X_scaled_df)[:, 1]

    def predict(self, df: pd.DataFrame, horizon: int = 1) -> np.ndarray:
        proba = self.predict_proba(df, horizon)
        threshold = self.params["threshold"]
        preds = np.full(len(proba), -1, dtype=np.int8)
        preds[proba > threshold] = 1
        preds[proba < (1 - threshold)] = 0
        return preds
