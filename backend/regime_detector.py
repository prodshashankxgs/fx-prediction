# Libraries
import numpy as np
import polars as pl
from typing import Dict, List, Tuple, Optional
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class RegimeDetector:
    """Detect and classify market regimes for adaptive strategy parameters."""
    
    def __init__(self):
        self.volatility_regimes = {}
        self.trend_regimes = {}
        self.momentum_regimes = {}
        self.regime_history = []
        
    def detect_volatility_regime(self, df: pl.DataFrame) -> Dict:
        """Detect volatility regime using multiple methods."""
        if len(df) < 50:
            return {"regime": "normal", "percentile": 50, "z_score": 0}
        
        # Get volatility data
        vol_data = df.select("volatility_20d").drop_nulls().to_numpy().flatten()
        if len(vol_data) < 20:
            return {"regime": "normal", "percentile": 50, "z_score": 0}
        
        current_vol = vol_data[-1]
        
        # Method 1: Percentile-based regime
        if len(vol_data) >= 252:  # Need at least 1 year of data
            historical_vol = vol_data[-252:]
        else:
            historical_vol = vol_data
        
        percentile = stats.percentileofscore(historical_vol, current_vol)
        
        # Method 2: Z-score based regime
        vol_mean = np.mean(historical_vol)
        vol_std = np.std(historical_vol)
        z_score = (current_vol - vol_mean) / (vol_std + 1e-8)
        
        # Method 3: Rolling regime detection
        if len(vol_data) >= 60:
            recent_vol = np.mean(vol_data[-20:])
            long_term_vol = np.mean(vol_data[-60:])
            vol_ratio = recent_vol / (long_term_vol + 1e-8)
        else:
            vol_ratio = 1.0
        
        # Combine methods to determine regime
        if percentile > 80 or z_score > 1.5 or vol_ratio > 1.3:
            regime = "high"
        elif percentile < 20 or z_score < -1.0 or vol_ratio < 0.7:
            regime = "low"
        else:
            regime = "normal"
        
        # Regime strength
        strength = min(abs(z_score), 3.0) / 3.0  # Normalize to 0-1
        
        return {
            "regime": regime,
            "percentile": percentile,
            "z_score": z_score,
            "vol_ratio": vol_ratio,
            "strength": strength,
            "current_vol": current_vol,
            "historical_avg": vol_mean
        }
    
    def detect_trend_regime(self, df: pl.DataFrame) -> Dict:
        """Detect trend regime using multiple timeframe analysis."""
        if len(df) < 50:
            return {"regime": "sideways", "strength": 0, "direction": 0}
        
        # Multiple timeframe trend analysis
        timeframes = [5, 10, 20, 50]
        trend_signals = []
        trend_strengths = []
        
        for tf in timeframes:
            if len(df) >= tf:
                # Price change over timeframe
                price_data = df.select("close").tail(tf + 1).to_numpy().flatten()
                if len(price_data) >= 2:
                    price_change = (price_data[-1] - price_data[0]) / price_data[0]
                    trend_signals.append(1 if price_change > 0 else -1)
                    trend_strengths.append(abs(price_change))
        
        if not trend_signals:
            return {"regime": "sideways", "strength": 0, "direction": 0}
        
        # Calculate trend consistency
        trend_direction = np.mean(trend_signals)
        trend_strength = np.mean(trend_strengths)
        
        # SMA-based trend confirmation
        if "sma_20" in df.columns and "sma_50" in df.columns and len(df) >= 50:
            latest = df.tail(1)
            close_price = latest.select("close").item()
            sma_20 = latest.select("sma_20").item()
            sma_50 = latest.select("sma_50").item()
            
            # Price vs MA relationship
            price_vs_sma20 = (close_price - sma_20) / sma_20
            price_vs_sma50 = (close_price - sma_50) / sma_50
            
            # MA alignment
            ma_alignment = 1 if sma_20 > sma_50 else -1
            
            # Combine signals
            ma_signal = np.sign(price_vs_sma20 + price_vs_sma50 + ma_alignment * 0.5)
        else:
            ma_signal = 0
            price_vs_sma20 = 0
            price_vs_sma50 = 0
        
        # Final trend regime determination
        combined_signal = (trend_direction + ma_signal) / 2
        combined_strength = trend_strength
        
        if combined_strength > 0.02 and abs(combined_signal) > 0.5:
            if combined_signal > 0:
                regime = "uptrend"
            else:
                regime = "downtrend"
        else:
            regime = "sideways"
        
        return {
            "regime": regime,
            "direction": combined_signal,
            "strength": combined_strength,
            "trend_consistency": abs(trend_direction),
            "price_vs_sma20": price_vs_sma20,
            "price_vs_sma50": price_vs_sma50,
            "timeframe_agreement": len([s for s in trend_signals if s == np.sign(trend_direction)]) / len(trend_signals)
        }
    
    def detect_momentum_regime(self, df: pl.DataFrame) -> Dict:
        """Detect momentum regime using multiple momentum indicators."""
        if len(df) < 20:
            return {"regime": "neutral", "strength": 0, "direction": 0}
        
        momentum_signals = []
        latest = df.tail(1)
        
        # RSI-based momentum
        if "rsi" in df.columns:
            rsi = latest.select("rsi").item()
            if rsi > 65:
                momentum_signals.append(("rsi", 1, (rsi - 50) / 50))
            elif rsi < 35:
                momentum_signals.append(("rsi", -1, (50 - rsi) / 50))
            else:
                momentum_signals.append(("rsi", 0, 0))
        
        # MACD-based momentum
        if "macd" in df.columns and "macd_signal" in df.columns:
            macd = latest.select("macd").item()
            macd_signal = latest.select("macd_signal").item()
            macd_diff = macd - macd_signal
            macd_direction = 1 if macd_diff > 0 else -1
            macd_strength = min(abs(macd_diff) * 1000, 1.0)  # Scale appropriately
            momentum_signals.append(("macd", macd_direction, macd_strength))
        
        # Price momentum
        if "momentum_20d" in df.columns:
            momentum_20d = latest.select("momentum_20d").item()
            momentum_direction = 1 if momentum_20d > 0 else -1
            momentum_strength = min(abs(momentum_20d) * 10, 1.0)
            momentum_signals.append(("momentum", momentum_direction, momentum_strength))
        
        # Stochastic momentum
        if "stoch_k" in df.columns and "stoch_d" in df.columns:
            stoch_k = latest.select("stoch_k").item()
            stoch_d = latest.select("stoch_d").item()
            if stoch_k > 75 and stoch_d > 75:
                momentum_signals.append(("stochastic", -1, 0.8))  # Overbought
            elif stoch_k < 25 and stoch_d < 25:
                momentum_signals.append(("stochastic", 1, 0.8))   # Oversold
            else:
                stoch_signal = 1 if stoch_k > stoch_d else -1
                momentum_signals.append(("stochastic", stoch_signal, 0.3))
        
        if not momentum_signals:
            return {"regime": "neutral", "strength": 0, "direction": 0}
        
        # Calculate weighted momentum
        total_weight = sum(signal[2] for signal in momentum_signals)
        if total_weight > 0:
            weighted_direction = sum(signal[1] * signal[2] for signal in momentum_signals) / total_weight
            avg_strength = total_weight / len(momentum_signals)
        else:
            weighted_direction = 0
            avg_strength = 0
        
        # Determine regime
        if avg_strength > 0.6 and abs(weighted_direction) > 0.5:
            if weighted_direction > 0:
                regime = "bullish"
            else:
                regime = "bearish"
        else:
            regime = "neutral"
        
        return {
            "regime": regime,
            "direction": weighted_direction,
            "strength": avg_strength,
            "signal_count": len(momentum_signals),
            "signal_details": {signal[0]: {"direction": signal[1], "strength": signal[2]} 
                             for signal in momentum_signals}
        }
    
    def detect_market_stress(self, df: pl.DataFrame) -> Dict:
        """Detect market stress conditions."""
        if len(df) < 20:
            return {"stress_level": "normal", "stress_score": 0}
        
        stress_indicators = []
        
        # Volatility stress
        vol_regime = self.detect_volatility_regime(df)
        if vol_regime["regime"] == "high":
            stress_indicators.append(("volatility", vol_regime["strength"]))
        
        # Price gap stress (large overnight moves)
        if "open" in df.columns and "close" in df.columns and len(df) >= 5:
            recent_data = df.tail(5)
            gaps = []
            for i in range(1, len(recent_data)):
                prev_close = recent_data.select("close").slice(i-1, 1).item()
                curr_open = recent_data.select("open").slice(i, 1).item()
                gap = abs(curr_open - prev_close) / prev_close
                gaps.append(gap)
            
            if gaps:
                avg_gap = np.mean(gaps)
                if avg_gap > 0.01:  # 1% gap threshold
                    stress_indicators.append(("gaps", min(avg_gap * 100, 1.0)))
        
        # Volume stress (unusual volume patterns)
        if "volume" in df.columns and len(df) >= 20:
            recent_volume = df.select("volume").tail(5).mean().item()
            historical_volume = df.select("volume").tail(20).mean().item()
            volume_ratio = recent_volume / (historical_volume + 1e-8)
            
            if volume_ratio > 2.0 or volume_ratio < 0.3:
                stress_indicators.append(("volume", min(abs(volume_ratio - 1), 1.0)))
        
        # Price range stress (unusual price ranges)
        if len(df) >= 10:
            recent_ranges = []
            for i in range(max(0, len(df) - 10), len(df)):
                row = df.slice(i, 1)
                if "high" in df.columns and "low" in df.columns:
                    high_val = row.select("high").item()
                    low_val = row.select("low").item()
                    close_val = row.select("close").item()
                    daily_range = (high_val - low_val) / close_val
                    recent_ranges.append(daily_range)
            
            if recent_ranges:
                avg_range = np.mean(recent_ranges)
                if avg_range > 0.03:  # 3% daily range threshold
                    stress_indicators.append(("range", min(avg_range * 20, 1.0)))
        
        # Calculate overall stress score
        if stress_indicators:
            stress_score = np.mean([indicator[1] for indicator in stress_indicators])
        else:
            stress_score = 0
        
        # Determine stress level
        if stress_score > 0.7:
            stress_level = "high"
        elif stress_score > 0.4:
            stress_level = "moderate"
        else:
            stress_level = "normal"
        
        return {
            "stress_level": stress_level,
            "stress_score": stress_score,
            "stress_indicators": {indicator[0]: indicator[1] for indicator in stress_indicators}
        }
    
    def get_regime_adaptive_parameters(self, df: pl.DataFrame) -> Dict:
        """Get adaptive parameters based on current market regime."""
        vol_regime = self.detect_volatility_regime(df)
        trend_regime = self.detect_trend_regime(df)
        momentum_regime = self.detect_momentum_regime(df)
        stress_conditions = self.detect_market_stress(df)
        
        # Base parameters
        params = {
            "position_size_multiplier": 1.0,
            "confidence_threshold": 0.6,
            "hold_period_adjustment": 1.0,
            "risk_multiplier": 1.0
        }
        
        # Volatility adjustments
        if vol_regime["regime"] == "high":
            params["position_size_multiplier"] *= 0.7  # Reduce position size
            params["confidence_threshold"] += 0.1      # Require higher confidence
            params["risk_multiplier"] *= 1.5          # Increase risk awareness
        elif vol_regime["regime"] == "low":
            params["position_size_multiplier"] *= 1.2  # Increase position size
            params["confidence_threshold"] -= 0.05     # Allow lower confidence
            params["risk_multiplier"] *= 0.8          # Reduce risk premium
        
        # Trend adjustments
        if trend_regime["regime"] in ["uptrend", "downtrend"]:
            if trend_regime["strength"] > 0.03:
                params["hold_period_adjustment"] *= 1.5  # Hold longer in strong trends
                params["confidence_threshold"] -= 0.05   # Lower threshold in trending markets
        elif trend_regime["regime"] == "sideways":
            params["hold_period_adjustment"] *= 0.7     # Shorter holds in sideways markets
            params["confidence_threshold"] += 0.05      # Higher threshold needed
        
        # Momentum adjustments
        if momentum_regime["strength"] > 0.7:
            params["position_size_multiplier"] *= 1.1   # Slightly larger positions
        
        # Stress adjustments
        if stress_conditions["stress_level"] == "high":
            params["position_size_multiplier"] *= 0.5   # Significantly reduce positions
            params["confidence_threshold"] += 0.15      # Much higher confidence needed
            params["risk_multiplier"] *= 2.0           # Double risk awareness
        elif stress_conditions["stress_level"] == "moderate":
            params["position_size_multiplier"] *= 0.8
            params["confidence_threshold"] += 0.1
            params["risk_multiplier"] *= 1.3
        
        # Ensure parameters stay within reasonable bounds
        params["position_size_multiplier"] = max(0.1, min(2.0, params["position_size_multiplier"]))
        params["confidence_threshold"] = max(0.5, min(0.9, params["confidence_threshold"]))
        params["hold_period_adjustment"] = max(0.5, min(2.0, params["hold_period_adjustment"]))
        params["risk_multiplier"] = max(0.5, min(3.0, params["risk_multiplier"]))
        
        return {
            "parameters": params,
            "volatility_regime": vol_regime,
            "trend_regime": trend_regime,
            "momentum_regime": momentum_regime,
            "stress_conditions": stress_conditions
        }
    
    def analyze_regime_stability(self, df: pl.DataFrame, lookback: int = 10) -> Dict:
        """Analyze how stable the current regime has been."""
        if len(df) < lookback + 10:
            return {"stability": "unknown", "regime_changes": 0}
        
        regime_history = []
        
        # Check regime over last N periods
        for i in range(lookback):
            subset_df = df.slice(0, len(df) - lookback + i + 1)
            if len(subset_df) >= 20:
                vol_regime = self.detect_volatility_regime(subset_df)["regime"]
                trend_regime = self.detect_trend_regime(subset_df)["regime"]
                regime_history.append((vol_regime, trend_regime))
        
        if not regime_history:
            return {"stability": "unknown", "regime_changes": 0}
        
        # Count regime changes
        vol_changes = sum(1 for i in range(1, len(regime_history)) 
                         if regime_history[i][0] != regime_history[i-1][0])
        trend_changes = sum(1 for i in range(1, len(regime_history)) 
                           if regime_history[i][1] != regime_history[i-1][1])
        
        total_changes = vol_changes + trend_changes
        stability_score = 1.0 - (total_changes / (2 * (len(regime_history) - 1)))
        
        if stability_score > 0.8:
            stability = "high"
        elif stability_score > 0.5:
            stability = "moderate"
        else:
            stability = "low"
        
        return {
            "stability": stability,
            "stability_score": stability_score,
            "regime_changes": total_changes,
            "vol_changes": vol_changes,
            "trend_changes": trend_changes,
            "regime_history": regime_history
        }