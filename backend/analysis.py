# Libraries
import polars as pl
import numpy as np
from typing import Dict
from scipy import stats

from backend.ml_analyzer import MLAnalyzer


class MarketAnalyzer:
    """Perform statistical analysis and pattern recognition."""

    def __init__(self):
        self.analysis_results = {}
        self.ml_analyzer = MLAnalyzer()

    def analyze(self, df: pl.DataFrame) -> Dict:
        """Run complete analysis suite including ML."""
        self.analysis_results = {
            "trend": self.analyze_trend(df),
            "momentum": self.analyze_momentum(df),
            "volatility": self.analyze_volatility(df),
            "mean_reversion": self.analyze_mean_reversion(df),
            "technical_signals": self.analyze_technical_signals(df),
            "ml_analysis": self.ml_analyzer.analyze(df)  # Make sure this line is here
        }

        # Debug: print ML analysis results
        print("ML Analysis Results:", self.analysis_results.get("ml_analysis", {}))

        return self.analysis_results

    def analyze_trend(self, df: pl.DataFrame) -> Dict:
        """Analyze market trend using multiple methods."""
        latest = df.tail(1)

        # SMA trend analysis
        sma_trend = {
            "short_term": 1 if latest["close"][0] > latest["sma_20"][0] else -1,
            "medium_term": 1 if latest["close"][0] > latest["sma_50"][0] else -1,
            "sma_alignment": 1 if latest["sma_20"][0] > latest["sma_50"][0] else -1
        }

        # Price momentum
        returns = df["return_20d"].tail(20).to_numpy()
        trend_strength = np.mean(returns) / (np.std(returns) + 1e-6)

        return {
            "sma_signals": sma_trend,
            "trend_strength": trend_strength,
            "direction": 1 if trend_strength > 0 else -1
        }

    def analyze_momentum(self, df: pl.DataFrame) -> Dict:
        """Analyze momentum indicators."""
        latest = df.tail(1)

        # RSI analysis
        rsi_value = latest["rsi"][0]
        rsi_signal = self._interpret_rsi(rsi_value)

        # MACD analysis
        macd_signal = 1 if latest["macd"][0] > latest["macd_signal"][0] else -1
        macd_strength = abs(latest["macd_histogram"][0])

        # Recent momentum
        recent_momentum = latest["momentum_20d"][0]

        return {
            "rsi": {"value": rsi_value, "signal": rsi_signal},
            "macd": {"signal": macd_signal, "strength": macd_strength},
            "recent_momentum": recent_momentum
        }

    def analyze_volatility(self, df: pl.DataFrame) -> Dict:
        """Analyze market volatility."""
        latest = df.tail(1)

        # Current volatility vs historical
        current_vol = latest["volatility_20d"][0]
        historical_vol = df["volatility_20d"].tail(252).mean()
        vol_percentile = stats.percentileofscore(
            df["volatility_20d"].tail(252).to_numpy(),
            current_vol
        )

        # Bollinger Band position
        bb_position = latest["bb_position"][0]

        return {
            "current": current_vol,
            "historical_avg": historical_vol,
            "percentile": vol_percentile,
            "bb_position": bb_position,
            "regime": "high" if vol_percentile > 70 else "normal" if vol_percentile > 30 else "low"
        }

    def analyze_mean_reversion(self, df: pl.DataFrame) -> Dict:
        """Analyze mean reversion potential."""
        latest = df.tail(1)

        # Distance from moving averages
        distance_sma20 = latest["price_to_sma20"][0]
        distance_sma50 = latest["price_to_sma50"][0]

        # Z-score calculation
        returns = df["return_1d"].tail(50).to_numpy()
        z_score = (returns[-1] - np.mean(returns)) / (np.std(returns) + 1e-6)

        # Mean reversion signal
        reversion_signal = 0
        if abs(z_score) > 2:
            reversion_signal = -1 if z_score > 0 else 1

        return {
            "distance_sma20": distance_sma20,
            "distance_sma50": distance_sma50,
            "z_score": z_score,
            "signal": reversion_signal
        }

    def analyze_technical_signals(self, df: pl.DataFrame) -> Dict:
        """Compile technical indicator signals."""
        latest = df.tail(1)

        signals = []

        # RSI signals
        rsi = latest["rsi"][0]
        if rsi > 70:
            signals.append(("RSI Overbought", -1, 0.7))
        elif rsi < 30:
            signals.append(("RSI Oversold", 1, 0.7))

        # Bollinger Band signals
        bb_pos = latest["bb_position"][0]
        if bb_pos > 0.95:
            signals.append(("BB Upper Touch", -1, 0.6))
        elif bb_pos < 0.05:
            signals.append(("BB Lower Touch", 1, 0.6))

        # MACD signals
        if latest["macd"][0] > latest["macd_signal"][0]:
            signals.append(("MACD Bullish Cross", 1, 0.8))
        else:
            signals.append(("MACD Bearish Cross", -1, 0.8))

        # Moving average signals
        if latest["close"][0] > latest["sma_20"][0] > latest["sma_50"][0]:
            signals.append(("MA Alignment Bullish", 1, 0.9))
        elif latest["close"][0] < latest["sma_20"][0] < latest["sma_50"][0]:
            signals.append(("MA Alignment Bearish", -1, 0.9))

        return {"signals": signals}

    def _interpret_rsi(self, rsi: float) -> int:
        """Interpret RSI value into signal."""
        if rsi > 70:
            return -1  # Overbought
        elif rsi < 30:
            return 1  # Oversold
        else:
            return 0  # Neutral