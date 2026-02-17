from predictor.strategies.base import BaseStrategy
from predictor.strategies.xgboost_strategy import XGBoostStrategy
from predictor.strategies.rsi_mean_reversion import RSIMeanReversionStrategy
from predictor.strategies.momentum_strategy import MomentumStrategy
from predictor.strategies.pattern_sequence import PatternSequenceStrategy
from predictor.strategies.ensemble_strategy import EnsembleStrategy

STRATEGY_REGISTRY: dict[str, type[BaseStrategy]] = {
    "xgboost": XGBoostStrategy,
    "rsi_mean_reversion": RSIMeanReversionStrategy,
    "momentum": MomentumStrategy,
    "pattern_sequence": PatternSequenceStrategy,
    "ensemble": EnsembleStrategy,
}


def get_strategy(name: str, params: dict | None = None) -> BaseStrategy:
    cls = STRATEGY_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGY_REGISTRY.keys())}")
    return cls(params or {})


def list_strategies() -> list[dict]:
    result = []
    for name, cls in STRATEGY_REGISTRY.items():
        info = {
            "name": name,
            "description": cls.description(),
            "default_params": cls.default_params(),
            "param_docs": cls.param_docs(),
            "recommended": RECOMMENDED_PARAMS.get(name, {}),
            "needs_training": name in ("xgboost", "pattern_sequence", "ensemble"),
        }
        result.append(info)
    return result


# Recommended param ranges and notes for each strategy
RECOMMENDED_PARAMS: dict[str, dict] = {
    "xgboost": {
        "notes": "ML-based strategy. Training time scales with n_estimators. Use n_estimators=100-200 for faster brute-force. Higher max_depth risks overfitting on noisy candle data.",
        "fast_preset": {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.08, "subsample": 0.8, "colsample_bytree": 0.8, "threshold": 0.53},
        "balanced_preset": {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "threshold": 0.53},
        "thorough_preset": {"n_estimators": 500, "max_depth": 5, "learning_rate": 0.03, "subsample": 0.85, "colsample_bytree": 0.85, "threshold": 0.52},
        "brute_force_include": ["n_estimators", "max_depth", "learning_rate", "threshold"],
    },
    "rsi_mean_reversion": {
        "notes": "Rule-based, no training needed. Very fast execution. Best for mean-reversion markets. Works well with BB confirmation enabled.",
        "fast_preset": {"rsi_period": 14, "rsi_oversold": 30, "rsi_overbought": 70, "use_bb_confirm": True, "bb_low": 0.2, "bb_high": 0.8},
        "aggressive_preset": {"rsi_period": 6, "rsi_oversold": 25, "rsi_overbought": 75, "use_bb_confirm": True, "bb_low": 0.15, "bb_high": 0.85},
        "conservative_preset": {"rsi_period": 14, "rsi_oversold": 20, "rsi_overbought": 80, "use_bb_confirm": True, "bb_low": 0.15, "bb_high": 0.85},
        "brute_force_include": ["rsi_period", "rsi_oversold", "rsi_overbought", "bb_low", "bb_high"],
    },
    "momentum": {
        "notes": "Rule-based, no training needed. Very fast execution. Trend-following approach. Weight params must be tuned together (should roughly sum to 1).",
        "fast_preset": {"ema_fast": 5, "ema_slow": 20, "macd_weight": 0.35, "ema_weight": 0.3, "volume_weight": 0.2, "momentum_weight": 0.15, "volume_surge_threshold": 1.5},
        "trend_heavy_preset": {"ema_fast": 3, "ema_slow": 15, "macd_weight": 0.45, "ema_weight": 0.35, "volume_weight": 0.1, "momentum_weight": 0.1, "volume_surge_threshold": 1.3},
        "volume_heavy_preset": {"ema_fast": 5, "ema_slow": 20, "macd_weight": 0.25, "ema_weight": 0.2, "volume_weight": 0.35, "momentum_weight": 0.2, "volume_surge_threshold": 2.0},
        "brute_force_include": ["macd_weight", "ema_weight", "volume_weight", "volume_surge_threshold"],
    },
    "pattern_sequence": {
        "notes": "Trains on historical candle direction patterns. Fast training. min_occurrences controls signal reliability vs frequency tradeoff.",
        "fast_preset": {"lookback_lengths": [3, 4, 5], "weights": [0.3, 0.35, 0.35], "min_occurrences": 5},
        "balanced_preset": {"lookback_lengths": [3, 4, 5, 6, 7], "weights": [0.1, 0.15, 0.25, 0.25, 0.25], "min_occurrences": 5},
        "deep_preset": {"lookback_lengths": [4, 5, 6, 7, 8], "weights": [0.1, 0.15, 0.25, 0.25, 0.25], "min_occurrences": 10},
        "brute_force_include": ["lookback_lengths", "min_occurrences"],
    },
    "ensemble": {
        "notes": "Combines all 4 strategies with weighted voting. Slowest but most stable. Weight tuning is key — higher xgboost_weight for ML-heavy, higher pattern_weight for pattern-heavy.",
        "fast_preset": {"xgboost_weight": 0.4, "rsi_weight": 0.15, "momentum_weight": 0.2, "pattern_weight": 0.25, "threshold": 0.53, "xgboost_params": {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.08, "subsample": 0.8, "colsample_bytree": 0.8, "threshold": 0.53}, "rsi_params": {"rsi_period": 14, "rsi_oversold": 30, "rsi_overbought": 70, "use_bb_confirm": True, "bb_low": 0.2, "bb_high": 0.8}, "momentum_params": {"ema_fast": 5, "ema_slow": 20, "macd_weight": 0.35, "ema_weight": 0.3, "volume_weight": 0.2, "momentum_weight": 0.15, "volume_surge_threshold": 1.5}, "pattern_params": {"lookback_lengths": [3, 4, 5, 6, 7], "weights": [0.1, 0.15, 0.25, 0.25, 0.25], "min_occurrences": 5}},
        "brute_force_include": ["xgboost_weight", "rsi_weight", "momentum_weight", "pattern_weight", "threshold"],
    },
}
