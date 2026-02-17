from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class BaseStrategy(ABC):

    def __init__(self, params: dict):
        merged = {**self.default_params(), **(params or {})}
        self.params = merged

    @staticmethod
    @abstractmethod
    def description() -> str:
        ...

    @staticmethod
    @abstractmethod
    def default_params() -> dict:
        ...

    @staticmethod
    @abstractmethod
    def param_docs() -> dict:
        """Return documentation for each parameter."""

    @abstractmethod
    def fit(self, df: pd.DataFrame, horizon: int = 1) -> None:
        """Train the strategy on historical data."""
        ...

    @abstractmethod
    def predict(self, df: pd.DataFrame, horizon: int = 1) -> np.ndarray:
        """Return array of predictions: 1=UP, 0=DOWN, -1=SKIP."""
        ...

    @abstractmethod
    def predict_proba(self, df: pd.DataFrame, horizon: int = 1) -> np.ndarray:
        """Return array of probabilities for UP direction [0..1]."""
        ...
