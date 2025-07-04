# Libraries
import os
from datetime import datetime, timedelta

from dotenv import load_dotenv
load_dotenv()

# API Configuration
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "your_api_key_here")
POLYGON_BASE_URL = "https://api.polygon.io"

# Trading Pairs Configuration
SUPPORTED_PAIRS = [
    "USD:GBP", "GBP:USD", "EUR:USD", "USD:EUR",
    "GBP:EUR", "EUR:GBP", "USD:JPY", "EUR:JPY"
]

# Model Parameters
LOOKBACK_PERIODS = {
    "short": 20,
    "medium": 50,
    "long": 200
}

TECHNICAL_INDICATORS = {
    "sma_periods": [10, 20, 50],
    "ema_periods": [12, 26],
    "rsi_period": 14,
    "bb_period": 20,
    "bb_std": 2,
    "atr_period": 14,
    "stoch_period": 14,
    "williams_period": 14,
    "cci_period": 20,
    "obv_momentum_period": 10
}

# Enhanced Prediction Thresholds
PREDICTION_THRESHOLDS = {
    "strong_bullish": 0.75,
    "bullish": 0.60,
    "weak_bullish": 0.52,
    "neutral_high": 0.52,
    "neutral_low": 0.48,
    "weak_bearish": 0.48,
    "bearish": 0.40,
    "strong_bearish": 0.25
}

# ML Model Parameters
ML_CONFIG = {
    "min_training_samples": 100,
    "feature_selection_k": 10,
    "ensemble_models": ["random_forest", "gradient_boosting"],
    "class_thresholds": {
        "strong_bullish": 0.01,    # > 1% return
        "bullish": 0.002,          # 0.2% to 1%
        "neutral": 0.002,          # -0.2% to 0.2%
        "bearish": -0.002,         # -1% to -0.2%
        "strong_bearish": -0.01    # < -1%
    }
}

# Data Storage
DATA_PATH = "forex_data"
CACHE_EXPIRY_HOURS = 24

# Enhanced ML Configuration
ENHANCED_ML_CONFIG = {
    "ensemble_models": ["random_forest", "xgboost", "lightgbm"],
    "feature_selection_method": "mutual_info",
    "max_features": 30,
    "time_series_cv_splits": 5,
    "validation_gap": 5,
    "regime_aware_training": True,
    "dynamic_feature_importance": True
}

# Regime Detection Configuration
REGIME_CONFIG = {
    "volatility_lookback": 252,
    "trend_lookback": 50,
    "momentum_lookback": 20,
    "regime_stability_window": 20,
    "stress_detection_threshold": 0.7
}

# Enhanced Prediction Configuration
ENHANCED_PREDICTION_CONFIG = {
    "dynamic_thresholds": True,
    "regime_adaptive_weights": True,
    "stress_filtering": True,
    "multi_model_consensus": True,
    "confidence_boosting": True
}

# Advanced Feature Configuration
ADVANCED_FEATURES = {
    "market_microstructure": True,
    "time_based_features": True,
    "interaction_features": True,
    "regime_features": True,
    "lag_features": [1, 2, 3, 5, 10],
    "advanced_volatility": True,
    "pattern_recognition": True
}