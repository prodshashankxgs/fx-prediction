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
    "atr_period": 14
}

# Prediction Thresholds
PREDICTION_THRESHOLDS = {
    "strong_bullish": 0.7,
    "bullish": 0.55,
    "neutral": 0.45,
    "bearish": 0.3,
    "strong_bearish": 0.0
}

# Data Storage
DATA_PATH = "forex_data"
CACHE_EXPIRY_HOURS = 24