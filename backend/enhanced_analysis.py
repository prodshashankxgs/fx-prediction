# Libraries
import polars as pl
import numpy as np
from typing import Dict, List
from scipy import stats

from backend.enhanced_ml_analyzer import EnhancedMLAnalyzer
from backend.regime_detector import RegimeDetector


class EnhancedMarketAnalyzer:
    """Enhanced market analyzer with regime detection and advanced ML."""

    def __init__(self, config: Dict = None):
        self.analysis_results = {}
        self.ml_analyzer = EnhancedMLAnalyzer(config)
        self.regime_detector = RegimeDetector()
        self.config = config or {}

    def analyze(self, df: pl.DataFrame) -> Dict:
        """Run complete enhanced analysis suite."""
        
        # Core technical analysis
        core_analysis = {
            "trend": self.analyze_trend(df),
            "momentum": self.analyze_momentum(df),
            "volatility": self.analyze_volatility(df),
            "mean_reversion": self.analyze_mean_reversion(df),
            "technical_signals": self.analyze_technical_signals(df),
            "volume_analysis": self.analyze_volume(df),
            "pattern_recognition": self.analyze_patterns(df),
            "multi_timeframe": self.analyze_multi_timeframe(df)
        }
        
        # Enhanced regime analysis
        regime_analysis = self.regime_detector.get_regime_adaptive_parameters(df)
        regime_stability = self.regime_detector.analyze_regime_stability(df)
        regime_analysis["stability_analysis"] = regime_stability
        
        # Enhanced ML analysis
        ml_analysis = self.ml_analyzer.analyze_with_regime(df)
        
        # Combine all analyses
        self.analysis_results = {
            **core_analysis,
            "regime_analysis": regime_analysis,
            "ml_analysis": ml_analysis
        }

        return self.analysis_results

    def analyze_trend(self, df: pl.DataFrame) -> Dict:
        """Enhanced trend analysis with regime awareness."""
        if len(df) < 20:
            return {"error": "Insufficient data for trend analysis"}
        
        latest = df.tail(1)

        # SMA trend analysis with multiple timeframes
        sma_signals = {}
        if all(col in df.columns for col in ["close", "sma_20", "sma_50"]):
            close_price = latest["close"][0]
            sma_20 = latest["sma_20"][0]
            sma_50 = latest["sma_50"][0]
            
            sma_signals = {
                "short_term": 1 if close_price > sma_20 else -1,
                "medium_term": 1 if close_price > sma_50 else -1,
                "sma_alignment": 1 if sma_20 > sma_50 else -1,
                "price_vs_sma20": (close_price - sma_20) / sma_20,
                "price_vs_sma50": (close_price - sma_50) / sma_50
            }

        # Enhanced trend strength calculation
        returns = df["return_20d"].tail(20).to_numpy()
        trend_strength = np.mean(returns) / (np.std(returns) + 1e-6)
        
        # Trend consistency with multiple methods
        price_changes = df["close"].diff().tail(20).to_numpy()
        trend_consistency = np.mean(np.sign(price_changes))
        
        # ADX-like trend strength
        if "atr" in df.columns:
            atr_data = df["atr"].tail(20).mean()
            close_data = latest["close"][0]
            adx_strength = min((atr_data / close_data) * 100, 100)
        else:
            high_low_range = df.select((pl.col("high") - pl.col("low")) / pl.col("close")).tail(20).mean().item()
            adx_strength = min(high_low_range * 100, 100)

        # Trend momentum
        short_trend = df["close"].tail(5).mean() - df["close"].tail(10).head(5).mean()
        long_trend = df["close"].tail(20).mean() - df["close"].tail(40).head(20).mean()
        trend_acceleration = short_trend - long_trend

        return {
            "sma_signals": sma_signals,
            "trend_strength": trend_strength,
            "direction": 1 if trend_strength > 0 else -1,
            "trend_consistency": trend_consistency,
            "adx_strength": adx_strength,
            "trend_acceleration": trend_acceleration,
            "trend_momentum": {
                "short_term": short_trend,
                "long_term": long_trend
            }
        }

    def analyze_momentum(self, df: pl.DataFrame) -> Dict:
        """Enhanced momentum analysis."""
        if len(df) < 14:
            return {"error": "Insufficient data for momentum analysis"}
        
        latest = df.tail(1)
        momentum_analysis = {}

        # RSI analysis with dynamic levels
        if "rsi" in df.columns:
            rsi_value = latest["rsi"][0]
            rsi_signal = self._interpret_rsi_dynamic(rsi_value, df)
            
            # RSI divergence
            rsi_trend = df["rsi"].tail(10).to_numpy()
            price_trend = df["close"].tail(10).to_numpy()
            rsi_divergence = self._check_divergence_advanced(price_trend, rsi_trend)
            
            momentum_analysis["rsi"] = {
                "value": rsi_value,
                "signal": rsi_signal,
                "divergence": rsi_divergence
            }

        # Enhanced MACD analysis
        if all(col in df.columns for col in ["macd", "macd_signal", "macd_histogram"]):
            macd = latest["macd"][0]
            macd_signal = latest["macd_signal"][0]
            macd_histogram = latest["macd_histogram"][0]
            
            macd_direction = 1 if macd > macd_signal else -1
            macd_strength = abs(macd_histogram)
            
            # MACD momentum
            macd_trend = df["macd"].tail(5).mean() - df["macd"].tail(10).head(5).mean()
            
            momentum_analysis["macd"] = {
                "signal": macd_direction,
                "strength": macd_strength,
                "histogram": macd_histogram,
                "momentum": macd_trend,
                "crossover_strength": abs(macd - macd_signal)
            }

        # Multi-oscillator momentum
        oscillator_signals = []
        
        # Stochastic
        if all(col in df.columns for col in ["stoch_k", "stoch_d"]):
            stoch_k = latest["stoch_k"][0]
            stoch_d = latest["stoch_d"][0]
            stoch_signal = self._analyze_stochastic_advanced(stoch_k, stoch_d)
            momentum_analysis["stochastic"] = stoch_signal
            oscillator_signals.append(stoch_signal["signal"])
        
        # Williams %R
        if "williams_r" in df.columns:
            williams_r = latest["williams_r"][0]
            williams_signal = self._analyze_williams_advanced(williams_r)
            momentum_analysis["williams_r"] = williams_signal
            oscillator_signals.append(williams_signal["signal"])
        
        # CCI
        if "cci" in df.columns:
            cci = latest["cci"][0]
            cci_signal = self._analyze_cci_advanced(cci)
            momentum_analysis["cci"] = cci_signal
            oscillator_signals.append(cci_signal["signal"])

        # Composite momentum score
        if oscillator_signals:
            momentum_analysis["composite_momentum"] = {
                "signal": np.mean(oscillator_signals),
                "agreement": len([s for s in oscillator_signals if abs(s) > 0.3]) / len(oscillator_signals),
                "strength": np.std(oscillator_signals)  # Lower std = more agreement
            }

        # Recent momentum with multiple timeframes
        if "return_1d" in df.columns:
            momentum_1d = df["return_1d"].tail(1).item()
            momentum_5d = df["return_1d"].tail(5).sum()
            momentum_20d = df["return_1d"].tail(20).sum()
            
            momentum_analysis["recent_momentum"] = momentum_20d
            momentum_analysis["momentum_breakdown"] = {
                "1d": momentum_1d,
                "5d": momentum_5d,
                "20d": momentum_20d
            }

        return momentum_analysis

    def analyze_volatility(self, df: pl.DataFrame) -> Dict:
        """Enhanced volatility analysis with regime detection."""
        if len(df) < 20:
            return {"error": "Insufficient data for volatility analysis"}
        
        latest = df.tail(1)
        
        # Get regime-based volatility analysis
        vol_regime = self.regime_detector.detect_volatility_regime(df)
        
        # Enhanced volatility measures
        volatility_analysis = {
            "regime": vol_regime["regime"],
            "percentile": vol_regime["percentile"],
            "z_score": vol_regime["z_score"],
            "current": vol_regime["current_vol"],
            "historical_avg": vol_regime["historical_avg"]
        }
        
        # Bollinger Band analysis
        if "bb_position" in df.columns:
            bb_position = latest["bb_position"][0]
            volatility_analysis["bb_position"] = bb_position
            
            # BB squeeze detection
            if all(col in df.columns for col in ["bb_upper", "bb_lower"]):
                bb_width = (latest["bb_upper"][0] - latest["bb_lower"][0]) / latest["close"][0]
                historical_bb_width = df.select(
                    (pl.col("bb_upper") - pl.col("bb_lower")) / pl.col("close")
                ).tail(20).mean().item()
                
                bb_squeeze = bb_width < historical_bb_width * 0.8
                volatility_analysis["bb_squeeze"] = bb_squeeze
                volatility_analysis["bb_width"] = bb_width

        # Advanced volatility measures
        if "parkinson_vol" in df.columns:
            volatility_analysis["parkinson_vol"] = latest["parkinson_vol"][0]
        
        if "gk_vol" in df.columns:
            volatility_analysis["gk_vol"] = latest["gk_vol"][0]
        
        # Volatility clustering
        if "volatility_20d" in df.columns and len(df) >= 40:
            vol_data = df["volatility_20d"].tail(40).to_numpy()
            vol_clustering = self._detect_volatility_clustering(vol_data)
            volatility_analysis["clustering"] = vol_clustering

        return volatility_analysis

    def analyze_mean_reversion(self, df: pl.DataFrame) -> Dict:
        """Enhanced mean reversion analysis."""
        if len(df) < 50:
            return {"error": "Insufficient data for mean reversion analysis"}
        
        latest = df.tail(1)

        # Multiple mean reversion indicators
        mr_analysis = {}

        # Price vs moving averages
        if all(col in df.columns for col in ["close", "sma_20", "sma_50"]):
            close_price = latest["close"][0]
            sma_20 = latest["sma_20"][0]
            sma_50 = latest["sma_50"][0]
            
            distance_sma20 = (close_price - sma_20) / sma_20
            distance_sma50 = (close_price - sma_50) / sma_50
            
            mr_analysis.update({
                "distance_sma20": distance_sma20,
                "distance_sma50": distance_sma50
            })

        # Z-score based mean reversion
        returns = df["return_1d"].tail(50).to_numpy()
        current_return = returns[-1]
        z_score = (current_return - np.mean(returns)) / (np.std(returns) + 1e-6)
        
        # Enhanced z-score with different windows
        z_scores = {}
        for window in [20, 50]:
            if len(df) >= window:
                window_returns = df["return_1d"].tail(window).to_numpy()
                window_zscore = (current_return - np.mean(window_returns)) / (np.std(window_returns) + 1e-6)
                z_scores[f"{window}d"] = window_zscore

        # Mean reversion signal strength
        mr_signal = 0
        signal_strength = 0
        
        if abs(z_score) > 2:
            mr_signal = -1 if z_score > 0 else 1
            signal_strength = min(abs(z_score) / 3, 1.0)
        elif abs(z_score) > 1:
            mr_signal = -0.5 if z_score > 0 else 0.5
            signal_strength = abs(z_score) / 2

        # Bollinger Band mean reversion
        bb_mr_signal = 0
        if "bb_position" in df.columns:
            bb_pos = latest["bb_position"][0]
            if bb_pos > 0.95:
                bb_mr_signal = -1
            elif bb_pos < 0.05:
                bb_mr_signal = 1

        # Combine signals
        combined_signal = (mr_signal + bb_mr_signal) / 2

        mr_analysis.update({
            "z_score": z_score,
            "z_scores_multi": z_scores,
            "signal": combined_signal,
            "signal_strength": signal_strength,
            "bb_mr_signal": bb_mr_signal
        })

        return mr_analysis

    def analyze_technical_signals(self, df: pl.DataFrame) -> Dict:
        """Enhanced technical signals with advanced patterns."""
        if len(df) < 20:
            return {"signals": []}
        
        latest = df.tail(1)
        signals = []

        # Enhanced RSI signals
        if "rsi" in df.columns:
            rsi = latest["rsi"][0]
            rsi_signals = self._get_rsi_signals_advanced(rsi, df)
            signals.extend(rsi_signals)

        # Enhanced Bollinger Band signals
        if "bb_position" in df.columns:
            bb_signals = self._get_bb_signals_advanced(latest, df)
            signals.extend(bb_signals)

        # Enhanced MACD signals
        if all(col in df.columns for col in ["macd", "macd_signal"]):
            macd_signals = self._get_macd_signals_advanced(latest, df)
            signals.extend(macd_signals)

        # Moving average signals with multiple timeframes
        ma_signals = self._get_ma_signals_advanced(latest, df)
        signals.extend(ma_signals)

        # Volume-price signals
        if "volume" in df.columns:
            volume_signals = self._get_volume_signals_advanced(latest, df)
            signals.extend(volume_signals)

        # Advanced pattern signals
        pattern_signals = self._get_pattern_signals_advanced(df)
        signals.extend(pattern_signals)

        return {"signals": signals}

    def analyze_volume(self, df: pl.DataFrame) -> Dict:
        """Enhanced volume analysis."""
        if len(df) < 20 or "volume" not in df.columns:
            return {"error": "Insufficient volume data"}
        
        latest = df.tail(1)
        volume_analysis = {}
        
        # OBV analysis
        if "obv" in df.columns and "obv_momentum" in df.columns:
            obv_slope = df["obv"].tail(20).to_numpy()
            obv_trend = np.polyfit(range(len(obv_slope)), obv_slope, 1)[0]
            
            volume_analysis.update({
                "obv_momentum": latest["obv_momentum"][0],
                "obv_trend": 1 if obv_trend > 0 else -1,
                "obv_strength": abs(obv_trend)
            })

        # Volume trend analysis
        recent_vol = df["volume"].tail(20).mean()
        historical_vol = df["volume"].tail(100).mean() if len(df) >= 100 else recent_vol
        
        volume_ratio = recent_vol / (historical_vol + 1e-8)
        
        # Volume price trend
        volume_price_correlation = self._calculate_volume_price_correlation(df)
        
        # Volume divergence
        volume_divergence = self._check_divergence_advanced(
            df["close"].tail(20).to_numpy(),
            df["volume"].tail(20).to_numpy()
        )

        volume_analysis.update({
            "volume_ratio": volume_ratio,
            "volume_trend": "increasing" if volume_ratio > 1.2 else "decreasing" if volume_ratio < 0.8 else "stable",
            "volume_price_correlation": volume_price_correlation,
            "volume_price_divergence": volume_divergence
        })

        return volume_analysis

    def analyze_patterns(self, df: pl.DataFrame) -> Dict:
        """Enhanced pattern recognition."""
        if len(df) < 50:
            return {"pattern_signals": []}
        
        pattern_signals = []
        
        # Enhanced trend patterns
        if all(col in df.columns for col in ["higher_high", "lower_low"]):
            hh_count = df["higher_high"].tail(20).sum()
            ll_count = df["lower_low"].tail(20).sum()
            
            if hh_count > ll_count + 2:
                strength = min((hh_count - ll_count) / 20, 1.0)
                pattern_signals.append(("Strong Uptrend Pattern", 1, strength))
            elif ll_count > hh_count + 2:
                strength = min((ll_count - hh_count) / 20, 1.0)
                pattern_signals.append(("Strong Downtrend Pattern", -1, strength))

        # Support/Resistance patterns
        sr_patterns = self._identify_support_resistance_advanced(df)
        pattern_signals.extend(sr_patterns)

        # Price acceleration patterns
        if "price_acceleration" in df.columns:
            accel_patterns = self._analyze_acceleration_patterns(df)
            pattern_signals.extend(accel_patterns)

        # Reversal patterns
        reversal_patterns = self._detect_reversal_patterns(df)
        pattern_signals.extend(reversal_patterns)

        return {"pattern_signals": pattern_signals}

    def analyze_multi_timeframe(self, df: pl.DataFrame) -> Dict:
        """Enhanced multi-timeframe analysis."""
        if len(df) < 50:
            return {"error": "Insufficient data for multi-timeframe analysis"}
        
        timeframes = {}
        
        # Short-term (5 days)
        short_term = df.tail(5)
        timeframes["short"] = self._analyze_timeframe(short_term, "short")
        
        # Medium-term (20 days)
        medium_term = df.tail(20)
        timeframes["medium"] = self._analyze_timeframe(medium_term, "medium")
        
        # Long-term (50 days)
        if len(df) >= 50:
            long_term = df.tail(50)
            timeframes["long"] = self._analyze_timeframe(long_term, "long")

        # Calculate alignment and confluence
        alignment_score = self._calculate_timeframe_alignment(timeframes)
        confluence_strength = self._calculate_confluence_strength(timeframes)

        return {
            "timeframes": timeframes,
            "alignment": alignment_score,
            "confluence_strength": confluence_strength,
            "signal": 1 if alignment_score > 0.5 else -1 if alignment_score < -0.5 else 0
        }

    # Helper methods for enhanced analysis
    def _interpret_rsi_dynamic(self, rsi: float, df: pl.DataFrame) -> int:
        """Dynamic RSI interpretation based on volatility."""
        vol_regime = self.regime_detector.detect_volatility_regime(df)
        
        if vol_regime["regime"] == "high":
            overbought, oversold = 75, 25
        else:
            overbought, oversold = 70, 30
        
        if rsi > overbought:
            return -1
        elif rsi < oversold:
            return 1
        else:
            return 0

    def _check_divergence_advanced(self, price_data: np.ndarray, indicator_data: np.ndarray) -> float:
        """Advanced divergence detection."""
        if len(price_data) < 10 or len(indicator_data) < 10:
            return 0
        
        # Calculate trends using linear regression
        price_trend = np.polyfit(range(len(price_data)), price_data, 1)[0]
        indicator_trend = np.polyfit(range(len(indicator_data)), indicator_data, 1)[0]
        
        # Normalize trends
        price_trend_norm = price_trend / (np.mean(price_data) + 1e-8)
        indicator_trend_norm = indicator_trend / (np.mean(indicator_data) + 1e-8)
        
        # Calculate divergence strength
        divergence = price_trend_norm - indicator_trend_norm
        
        # Return divergence signal (-1 to 1)
        return np.tanh(divergence * 10)

    def _analyze_stochastic_advanced(self, stoch_k: float, stoch_d: float) -> Dict:
        """Advanced stochastic analysis."""
        signal = 0
        strength = 0
        
        if stoch_k > 80 and stoch_d > 80:
            signal = -1
            strength = min((stoch_k + stoch_d) / 200, 1.0)
        elif stoch_k < 20 and stoch_d < 20:
            signal = 1
            strength = min((40 - stoch_k - stoch_d) / 40, 1.0)
        elif stoch_k > stoch_d:
            signal = 0.5
            strength = abs(stoch_k - stoch_d) / 100
        else:
            signal = -0.5
            strength = abs(stoch_k - stoch_d) / 100
        
        return {
            "signal": signal,
            "strength": strength,
            "k": stoch_k,
            "d": stoch_d,
            "crossover": stoch_k - stoch_d
        }

    def _analyze_williams_advanced(self, williams_r: float) -> Dict:
        """Advanced Williams %R analysis."""
        signal = 0
        strength = 0
        
        if williams_r > -20:
            signal = -1
            strength = (williams_r + 20) / 20
        elif williams_r < -80:
            signal = 1
            strength = (-80 - williams_r) / 20
        else:
            signal = (williams_r + 50) / 50
            strength = abs(signal)
        
        return {
            "signal": signal,
            "strength": strength,
            "value": williams_r
        }

    def _analyze_cci_advanced(self, cci: float) -> Dict:
        """Advanced CCI analysis."""
        signal = 0
        strength = 0
        
        if cci > 100:
            signal = 1
            strength = min(cci / 200, 1.0)
        elif cci < -100:
            signal = -1
            strength = min(-cci / 200, 1.0)
        else:
            signal = cci / 200
            strength = abs(cci) / 100
        
        return {
            "signal": signal,
            "strength": strength,
            "value": cci
        }

    def _detect_volatility_clustering(self, vol_data: np.ndarray) -> Dict:
        """Detect volatility clustering patterns."""
        if len(vol_data) < 20:
            return {"clustering": False}
        
        # Calculate autocorrelation of squared returns
        vol_squared = vol_data ** 2
        autocorr = np.corrcoef(vol_squared[:-1], vol_squared[1:])[0, 1]
        
        clustering = autocorr > 0.3
        
        return {
            "clustering": clustering,
            "autocorrelation": autocorr,
            "persistence": clustering
        }

    def _get_rsi_signals_advanced(self, rsi: float, df: pl.DataFrame) -> List:
        """Get advanced RSI signals."""
        signals = []
        
        # Dynamic thresholds based on volatility
        vol_regime = self.regime_detector.detect_volatility_regime(df)
        
        if vol_regime["regime"] == "high":
            if rsi > 75:
                signals.append(("RSI Overbought (High Vol)", -1, 0.8))
            elif rsi < 25:
                signals.append(("RSI Oversold (High Vol)", 1, 0.8))
        else:
            if rsi > 70:
                signals.append(("RSI Overbought", -1, 0.7))
            elif rsi < 30:
                signals.append(("RSI Oversold", 1, 0.7))
        
        # RSI momentum
        if len(df) >= 5:
            rsi_momentum = df["rsi"].tail(1).item() - df["rsi"].tail(5).mean()
            if abs(rsi_momentum) > 5:
                direction = 1 if rsi_momentum > 0 else -1
                strength = min(abs(rsi_momentum) / 20, 1.0)
                signals.append(("RSI Momentum", direction, strength))
        
        return signals

    def _get_bb_signals_advanced(self, latest: pl.DataFrame, df: pl.DataFrame) -> List:
        """Get advanced Bollinger Band signals."""
        signals = []
        
        if "bb_position" not in df.columns:
            return signals
        
        bb_pos = latest["bb_position"][0]
        
        # BB squeeze consideration
        if all(col in df.columns for col in ["bb_upper", "bb_lower"]):
            bb_width = (latest["bb_upper"][0] - latest["bb_lower"][0]) / latest["close"][0]
            historical_bb_width = df.select(
                (pl.col("bb_upper") - pl.col("bb_lower")) / pl.col("close")
            ).tail(20).mean().item()
            
            is_squeeze = bb_width < historical_bb_width * 0.8
            
            if is_squeeze:
                signals.append(("BB Squeeze - Breakout Pending", 0, 0.6))
            else:
                if bb_pos > 0.95:
                    signals.append(("BB Upper Touch", -1, 0.7))
                elif bb_pos < 0.05:
                    signals.append(("BB Lower Touch", 1, 0.7))
                elif bb_pos > 0.8:
                    signals.append(("BB Upper Band Approach", -0.5, 0.5))
                elif bb_pos < 0.2:
                    signals.append(("BB Lower Band Approach", 0.5, 0.5))
        
        return signals

    def _get_macd_signals_advanced(self, latest: pl.DataFrame, df: pl.DataFrame) -> List:
        """Get advanced MACD signals."""
        signals = []
        
        macd = latest["macd"][0]
        macd_signal = latest["macd_signal"][0]
        macd_histogram = latest["macd_histogram"][0]
        
        # MACD crossover with strength
        if macd > macd_signal:
            strength = min(abs(macd_histogram) * 1000, 1.0)
            signals.append(("MACD Bullish Cross", 1, 0.8 * strength))
        else:
            strength = min(abs(macd_histogram) * 1000, 1.0)
            signals.append(("MACD Bearish Cross", -1, 0.8 * strength))
        
        # MACD momentum
        if len(df) >= 5:
            macd_momentum = df["macd"].tail(1).item() - df["macd"].tail(5).mean()
            if abs(macd_momentum) > 0.001:
                direction = 1 if macd_momentum > 0 else -1
                strength = min(abs(macd_momentum) * 500, 1.0)
                signals.append(("MACD Momentum", direction, strength))
        
        return signals

    def _get_ma_signals_advanced(self, latest: pl.DataFrame, df: pl.DataFrame) -> List:
        """Get advanced moving average signals."""
        signals = []
        
        if not all(col in df.columns for col in ["close", "sma_20", "sma_50"]):
            return signals
        
        close_price = latest["close"][0]
        sma_20 = latest["sma_20"][0]
        sma_50 = latest["sma_50"][0]
        
        # MA alignment with strength
        if close_price > sma_20 > sma_50:
            alignment_strength = min((close_price - sma_50) / sma_50 * 100, 1.0)
            signals.append(("MA Alignment Bullish", 1, 0.9 * alignment_strength))
        elif close_price < sma_20 < sma_50:
            alignment_strength = min((sma_50 - close_price) / sma_50 * 100, 1.0)
            signals.append(("MA Alignment Bearish", -1, 0.9 * alignment_strength))
        
        # MA slope analysis
        if len(df) >= 5:
            sma20_slope = df["sma_20"].tail(1).item() - df["sma_20"].tail(5).mean()
            sma50_slope = df["sma_50"].tail(1).item() - df["sma_50"].tail(5).mean()
            
            if sma20_slope > 0 and sma50_slope > 0:
                slope_strength = min((sma20_slope + sma50_slope) / close_price * 1000, 1.0)
                signals.append(("MA Slopes Rising", 1, slope_strength))
            elif sma20_slope < 0 and sma50_slope < 0:
                slope_strength = min(abs(sma20_slope + sma50_slope) / close_price * 1000, 1.0)
                signals.append(("MA Slopes Falling", -1, slope_strength))
        
        return signals

    def _get_volume_signals_advanced(self, latest: pl.DataFrame, df: pl.DataFrame) -> List:
        """Get advanced volume signals."""
        signals = []
        
        # Volume confirmation
        recent_vol = df["volume"].tail(5).mean()
        historical_vol = df["volume"].tail(20).mean()
        vol_ratio = recent_vol / (historical_vol + 1e-8)
        
        if vol_ratio > 1.5:
            signals.append(("High Volume Confirmation", 0.5, 0.6))
        elif vol_ratio < 0.5:
            signals.append(("Low Volume Warning", -0.5, 0.4))
        
        return signals

    def _get_pattern_signals_advanced(self, df: pl.DataFrame) -> List:
        """Get advanced pattern signals."""
        signals = []
        
        # Implement advanced pattern detection
        # This is a placeholder for more sophisticated pattern recognition
        
        return signals

    def _calculate_volume_price_correlation(self, df: pl.DataFrame) -> float:
        """Calculate volume-price correlation."""
        if len(df) < 20:
            return 0
        
        price_changes = df["close"].pct_change().tail(20).to_numpy()
        volume_changes = df["volume"].pct_change().tail(20).to_numpy()
        
        # Remove NaN values
        mask = ~(np.isnan(price_changes) | np.isnan(volume_changes))
        if np.sum(mask) < 10:
            return 0
        
        correlation = np.corrcoef(price_changes[mask], volume_changes[mask])[0, 1]
        return correlation if not np.isnan(correlation) else 0

    def _identify_support_resistance_advanced(self, df: pl.DataFrame) -> List:
        """Advanced support/resistance identification."""
        signals = []
        
        if len(df) < 50:
            return signals
        
        prices = df["close"].tail(50).to_numpy()
        current_price = prices[-1]
        
        # Find local extrema
        highs = []
        lows = []
        
        for i in range(2, len(prices) - 2):
            if (prices[i] > prices[i-1] and prices[i] > prices[i-2] and 
                prices[i] > prices[i+1] and prices[i] > prices[i+2]):
                highs.append(prices[i])
            elif (prices[i] < prices[i-1] and prices[i] < prices[i-2] and 
                  prices[i] < prices[i+1] and prices[i] < prices[i+2]):
                lows.append(prices[i])
        
        # Check proximity to support/resistance levels
        tolerance = current_price * 0.01  # 1% tolerance
        
        for high in highs:
            if abs(current_price - high) < tolerance:
                signals.append(("Near Resistance Level", -0.6, 0.7))
                break
        
        for low in lows:
            if abs(current_price - low) < tolerance:
                signals.append(("Near Support Level", 0.6, 0.7))
                break
        
        return signals

    def _analyze_acceleration_patterns(self, df: pl.DataFrame) -> List:
        """Analyze price acceleration patterns."""
        signals = []
        
        if "price_acceleration" not in df.columns or len(df) < 10:
            return signals
        
        accel_data = df["price_acceleration"].tail(10).to_numpy()
        recent_accel = np.mean(accel_data[-5:])
        
        if abs(recent_accel) > 0.001:
            direction = 1 if recent_accel > 0 else -1
            strength = min(abs(recent_accel) * 1000, 1.0)
            
            if recent_accel > 0:
                signals.append(("Price Accelerating Up", direction, strength))
            else:
                signals.append(("Price Accelerating Down", direction, strength))
        
        return signals

    def _detect_reversal_patterns(self, df: pl.DataFrame) -> List:
        """Detect potential reversal patterns."""
        signals = []
        
        # Implement reversal pattern detection
        # This could include double tops/bottoms, head and shoulders, etc.
        
        return signals

    def _analyze_timeframe(self, df_tf: pl.DataFrame, timeframe: str) -> Dict:
        """Analyze individual timeframe."""
        if len(df_tf) < 2:
            return {"trend": 0, "momentum": 0, "strength": 0}
        
        # Price trend
        start_price = df_tf["close"].head(1).item()
        end_price = df_tf["close"].tail(1).item()
        trend = 1 if end_price > start_price else -1
        
        # Momentum
        if "return_1d" in df_tf.columns:
            momentum = df_tf["return_1d"].mean()
        else:
            momentum = (end_price - start_price) / start_price
        
        # Strength
        price_range = df_tf["close"].max() - df_tf["close"].min()
        strength = abs(end_price - start_price) / (price_range + 1e-8)
        
        return {
            "trend": trend,
            "momentum": momentum,
            "strength": strength,
            "timeframe": timeframe
        }

    def _calculate_timeframe_alignment(self, timeframes: Dict) -> float:
        """Calculate alignment across timeframes."""
        trends = [tf_data["trend"] for tf_data in timeframes.values() 
                 if isinstance(tf_data, dict) and "trend" in tf_data]
        
        if not trends:
            return 0
        
        return np.mean(trends)

    def _calculate_confluence_strength(self, timeframes: Dict) -> float:
        """Calculate confluence strength across timeframes."""
        strengths = [tf_data["strength"] for tf_data in timeframes.values() 
                    if isinstance(tf_data, dict) and "strength" in tf_data]
        
        if not strengths:
            return 0
        
        return np.mean(strengths)

    def train_ml_models(self, df: pl.DataFrame) -> Dict:
        """Train the enhanced ML models."""
        return self.ml_analyzer.train(df)