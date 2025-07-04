# Libraries
import polars as pl
import numpy as np
from typing import Dict, List
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
            "volume_analysis": self.analyze_volume(df),
            "pattern_recognition": self.analyze_patterns(df),
            "multi_timeframe": self.analyze_multi_timeframe(df),
            "ml_analysis": self.ml_analyzer.analyze(df)
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
        
        # Trend consistency score
        if "trend_consistency" in df.columns:
            trend_consistency = latest["trend_consistency"][0]
        else:
            trend_consistency = 0

        # ADX-like trend strength (using price range)
        high_low_range = df.select((pl.col("high") - pl.col("low")) / pl.col("close")).tail(20).mean().item()
        adx_strength = min(high_low_range * 100, 100)

        return {
            "sma_signals": sma_trend,
            "trend_strength": trend_strength,
            "direction": 1 if trend_strength > 0 else -1,
            "trend_consistency": trend_consistency,
            "adx_strength": adx_strength
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
        
        # Stochastic analysis
        stoch_signal = 0
        if "stoch_k" in df.columns and "stoch_d" in df.columns:
            stoch_k = latest["stoch_k"][0]
            stoch_d = latest["stoch_d"][0]
            if stoch_k > 80 and stoch_d > 80:
                stoch_signal = -1  # Overbought
            elif stoch_k < 20 and stoch_d < 20:
                stoch_signal = 1   # Oversold
            elif stoch_k > stoch_d:
                stoch_signal = 0.5  # Bullish crossover
            else:
                stoch_signal = -0.5  # Bearish crossover
        
        # Williams %R
        williams_signal = 0
        if "williams_r" in df.columns:
            williams_r = latest["williams_r"][0]
            if williams_r > -20:
                williams_signal = -1  # Overbought
            elif williams_r < -80:
                williams_signal = 1   # Oversold
        
        # CCI analysis
        cci_signal = 0
        if "cci" in df.columns:
            cci = latest["cci"][0]
            if cci > 100:
                cci_signal = 1  # Bullish
            elif cci < -100:
                cci_signal = -1  # Bearish

        return {
            "rsi": {"value": rsi_value, "signal": rsi_signal},
            "macd": {"signal": macd_signal, "strength": macd_strength},
            "recent_momentum": recent_momentum,
            "stochastic": {"k": stoch_k if "stoch_k" in df.columns else 50, 
                          "d": stoch_d if "stoch_d" in df.columns else 50, 
                          "signal": stoch_signal},
            "williams_r": {"value": williams_r if "williams_r" in df.columns else -50, 
                          "signal": williams_signal},
            "cci": {"value": cci if "cci" in df.columns else 0, 
                   "signal": cci_signal}
        }

    def analyze_volatility(self, df: pl.DataFrame) -> Dict:
        """Analyze market volatility."""
        latest = df.tail(1)
        
        # Check if volatility columns exist
        if "volatility_20d" not in df.columns:
            return {
                "current": 0,
                "historical_avg": 0,
                "percentile": 50,
                "bb_position": 0.5,
                "regime": "normal",
                "error": "Insufficient data for volatility calculation"
            }

        # Current volatility vs historical
        current_vol = latest["volatility_20d"][0]
        
        # Check if we have enough data for historical volatility
        if len(df) >= 252:
            historical_vol = df["volatility_20d"].tail(252).mean()
            vol_percentile = stats.percentileofscore(
                df["volatility_20d"].tail(252).to_numpy(),
                current_vol
            )
        else:
            # Use what data we have
            historical_vol = df["volatility_20d"].mean()
            vol_percentile = 50  # Default to middle percentile

        # Bollinger Band position
        bb_position = latest["bb_position"][0] if "bb_position" in df.columns else 0.5

        # Advanced volatility measures
        volatility_analysis = {
            "current": current_vol,
            "historical_avg": historical_vol,
            "percentile": vol_percentile,
            "bb_position": bb_position,
            "regime": "high" if vol_percentile > 70 else "normal" if vol_percentile > 30 else "low"
        }

        # Add advanced volatility if available
        if "parkinson_vol" in df.columns:
            volatility_analysis["parkinson_vol"] = latest["parkinson_vol"][0]
        if "gk_vol" in df.columns:
            volatility_analysis["gk_vol"] = latest["gk_vol"][0]
        if "vol_ratio" in df.columns:
            volatility_analysis["vol_ratio"] = latest["vol_ratio"][0]
            # Volatility expansion/contraction
            if volatility_analysis["vol_ratio"] > 1.2:
                volatility_analysis["vol_trend"] = "expanding"
            elif volatility_analysis["vol_ratio"] < 0.8:
                volatility_analysis["vol_trend"] = "contracting"
            else:
                volatility_analysis["vol_trend"] = "stable"

        return volatility_analysis

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

    def analyze_volume(self, df: pl.DataFrame) -> Dict:
        """Analyze volume patterns."""
        latest = df.tail(1)
        volume_analysis = {}
        
        if "obv" in df.columns and "obv_momentum" in df.columns:
            # OBV trend
            obv_slope = df["obv"].tail(20).to_numpy()
            obv_trend = np.polyfit(range(len(obv_slope)), obv_slope, 1)[0]
            
            volume_analysis = {
                "obv_momentum": latest["obv_momentum"][0],
                "obv_trend": 1 if obv_trend > 0 else -1,
                "volume_price_divergence": self._check_divergence(df, "close", "obv")
            }
        
        # Volume analysis
        if "volume" in df.columns:
            recent_vol_data = df["volume"].tail(20).to_numpy()
            historical_vol_data = df["volume"].tail(100).to_numpy()
            
            if len(recent_vol_data) > 0 and len(historical_vol_data) > 0:
                recent_vol = float(np.mean(recent_vol_data))
                historical_vol = float(np.mean(historical_vol_data))
                
                volume_analysis["volume_ratio"] = recent_vol / historical_vol if historical_vol > 0 else 1.0
                volume_analysis["volume_trend"] = "increasing" if recent_vol > historical_vol * 1.2 else "decreasing" if recent_vol < historical_vol * 0.8 else "stable"
            else:
                volume_analysis["volume_ratio"] = 1.0
                volume_analysis["volume_trend"] = "stable"
        
        return volume_analysis
    
    def analyze_patterns(self, df: pl.DataFrame) -> Dict:
        """Analyze price patterns."""
        pattern_signals = []
        
        # Higher highs and lower lows pattern
        if "higher_high" in df.columns and "lower_low" in df.columns:
            recent_hh = df["higher_high"].tail(10).sum()
            recent_ll = df["lower_low"].tail(10).sum()
            
            if recent_hh > recent_ll:
                pattern_signals.append(("Uptrend Pattern (HH/HL)", 1, 0.7))
            elif recent_ll > recent_hh:
                pattern_signals.append(("Downtrend Pattern (LL/LH)", -1, 0.7))
        
        # Price acceleration
        if "price_acceleration" in df.columns:
            accel = df["price_acceleration"].tail(5).mean()
            if abs(accel) > 0.001:
                signal = 1 if accel > 0 else -1
                pattern_signals.append((f"Price {'Accelerating' if accel > 0 else 'Decelerating'}", signal, 0.6))
        
        # Support/Resistance levels
        support_resistance = self._identify_support_resistance(df)
        pattern_signals.extend(support_resistance)
        
        return {"pattern_signals": pattern_signals}
    
    def analyze_multi_timeframe(self, df: pl.DataFrame) -> Dict:
        """Analyze multiple timeframes for confluence."""
        timeframes = {}
        
        # Short-term (5 days)
        short_term = df.tail(5)
        timeframes["short"] = {
            "trend": 1 if short_term["close"].tail(1).item() > short_term["close"].head(1).item() else -1,
            "momentum": short_term["return_1d"].mean() if "return_1d" in df.columns else 0
        }
        
        # Medium-term (20 days)
        medium_term = df.tail(20)
        timeframes["medium"] = {
            "trend": 1 if medium_term["close"].tail(1).item() > medium_term["close"].head(1).item() else -1,
            "momentum": medium_term["return_1d"].mean() if "return_1d" in df.columns else 0
        }
        
        # Long-term (50 days)
        if len(df) >= 50:
            long_term = df.tail(50)
            timeframes["long"] = {
                "trend": 1 if long_term["close"].tail(1).item() > long_term["close"].head(1).item() else -1,
                "momentum": long_term["return_1d"].mean() if "return_1d" in df.columns else 0
            }
        
        # Calculate alignment
        trends = [tf["trend"] for tf in timeframes.values()]
        alignment = sum(trends) / len(trends) if trends else 0
        
        return {
            "timeframes": timeframes,
            "alignment": alignment,
            "signal": 1 if alignment > 0.5 else -1 if alignment < -0.5 else 0
        }
    
    def _check_divergence(self, df: pl.DataFrame, price_col: str, indicator_col: str, periods: int = 20) -> int:
        """Check for divergence between price and indicator."""
        if len(df) < periods:
            return 0
        
        price_data = df[price_col].tail(periods).to_numpy()
        indicator_data = df[indicator_col].tail(periods).to_numpy()
        
        # Calculate trends
        price_trend = np.polyfit(range(len(price_data)), price_data, 1)[0]
        indicator_trend = np.polyfit(range(len(indicator_data)), indicator_data, 1)[0]
        
        # Check for divergence
        if price_trend > 0 and indicator_trend < 0:
            return -1  # Bearish divergence
        elif price_trend < 0 and indicator_trend > 0:
            return 1   # Bullish divergence
        else:
            return 0   # No divergence
    
    def _identify_support_resistance(self, df: pl.DataFrame) -> List:
        """Identify support and resistance levels."""
        signals = []
        
        if len(df) < 50:
            return signals
        
        # Get recent price data
        prices = df["close"].tail(50).to_numpy()
        current_price = prices[-1]
        
        # Find local maxima and minima
        highs = []
        lows = []
        
        for i in range(2, len(prices) - 2):
            if prices[i] > prices[i-1] and prices[i] > prices[i-2] and prices[i] > prices[i+1] and prices[i] > prices[i+2]:
                highs.append(prices[i])
            elif prices[i] < prices[i-1] and prices[i] < prices[i-2] and prices[i] < prices[i+1] and prices[i] < prices[i+2]:
                lows.append(prices[i])
        
        # Check proximity to support/resistance
        if highs:
            nearest_resistance = min(highs, key=lambda x: abs(x - current_price) if x > current_price else float('inf'))
            if nearest_resistance != float('inf') and (nearest_resistance - current_price) / current_price < 0.01:
                signals.append(("Near Resistance Level", -0.5, 0.6))
        
        if lows:
            nearest_support = min(lows, key=lambda x: abs(x - current_price) if x < current_price else float('inf'))
            if nearest_support != float('inf') and (current_price - nearest_support) / current_price < 0.01:
                signals.append(("Near Support Level", 0.5, 0.6))
        
        return signals