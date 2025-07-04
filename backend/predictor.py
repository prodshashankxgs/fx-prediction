# Libraries
import numpy as np
from typing import Dict, Tuple
from datetime import datetime


class ForexPredictor:
    """Generate forex predictions based on analysis results."""

    def __init__(self, thresholds: Dict):
        self.thresholds = thresholds

    def predict(self, analysis: Dict) -> Dict:
        """Generate prediction based on analysis results."""

        # Calculate weighted score from all indicators
        score = self._calculate_composite_score(analysis)

        # Determine signal and confidence
        signal, strength = self._determine_signal(score)

        # Calculate confidence metrics
        confidence = self._calculate_confidence(analysis)

        # Generate supporting reasons
        reasons = self._generate_reasons(analysis)

        return {
            "prediction": signal,
            "strength": strength,
            "score": score,
            "confidence": confidence,
            "reasons": reasons,
            "timestamp": datetime.now().isoformat()
        }

    def _calculate_composite_score(self, analysis: Dict) -> float:
        """Calculate weighted composite score from all indicators including ML."""

        # Check if ML is available
        ml_available = analysis.get("ml_analysis", {}).get("ml_available", False)

        # Dynamic weights based on market conditions
        vol_regime = analysis.get("volatility", {}).get("regime", "normal")
        
        # Adjust weights based on ML availability and market conditions
        if ml_available:
            if vol_regime == "high":
                weights = {
                    "trend": 0.15,
                    "momentum": 0.15,
                    "technical": 0.20,
                    "mean_reversion": 0.15,
                    "volatility": 0.10,
                    "volume": 0.05,
                    "pattern": 0.05,
                    "multi_timeframe": 0.05,
                    "ml": 0.10
                }
            else:
                weights = {
                    "trend": 0.18,
                    "momentum": 0.17,
                    "technical": 0.20,
                    "mean_reversion": 0.08,
                    "volatility": 0.07,
                    "volume": 0.08,
                    "pattern": 0.07,
                    "multi_timeframe": 0.05,
                    "ml": 0.10
                }
        else:
            weights = {
                "trend": 0.22,
                "momentum": 0.22,
                "technical": 0.26,
                "mean_reversion": 0.10,
                "volatility": 0.08,
                "volume": 0.06,
                "pattern": 0.06
            }

        scores = {}

        # Calculate trend score
        if "trend" in analysis:
            trend_data = analysis["trend"]
            trend_score = 0.5  # neutral default

            if "direction" in trend_data:
                # Convert direction (-1, 1) to score (0, 1)
                trend_score = (trend_data["direction"] + 1) / 2

            # Adjust based on trend strength
            if "trend_strength" in trend_data:
                strength = trend_data["trend_strength"]
                # Normalize strength to 0-1 range
                normalized_strength = (np.tanh(strength) + 1) / 2
                trend_score = 0.5 + (trend_score - 0.5) * normalized_strength
            
            # Consider trend consistency
            if "trend_consistency" in trend_data:
                consistency = trend_data["trend_consistency"]
                # Amplify trend score based on consistency
                if consistency > 0:
                    trend_score = 0.5 + (trend_score - 0.5) * (1 + consistency)
                
            # Consider ADX strength
            if "adx_strength" in trend_data:
                adx = trend_data["adx_strength"] / 100  # Normalize to 0-1
                if adx > 0.25:  # Strong trend
                    trend_score = 0.5 + (trend_score - 0.5) * (1 + adx)

            scores["trend"] = np.clip(trend_score, 0, 1)

        # Calculate momentum score
        if "momentum" in analysis:
            momentum_data = analysis["momentum"]
            momentum_components = []

            # RSI component
            if "rsi" in momentum_data and isinstance(momentum_data["rsi"], dict):
                rsi_value = momentum_data["rsi"].get("value", 50)
                # Convert RSI to score (0-30: bullish, 30-70: neutral, 70-100: bearish)
                if rsi_value < 30:
                    rsi_score = 0.8  # Oversold = bullish
                elif rsi_value > 70:
                    rsi_score = 0.2  # Overbought = bearish
                else:
                    # Linear interpolation in neutral zone
                    rsi_score = 0.5 + (50 - rsi_value) / 100
                momentum_components.append(rsi_score)

            # MACD component
            if "macd" in momentum_data and isinstance(momentum_data["macd"], dict):
                macd_signal = momentum_data["macd"].get("signal", 0)
                macd_strength = momentum_data["macd"].get("strength", 0)
                # Convert MACD signal to score
                macd_score = (macd_signal + 1) / 2
                # Weight by strength
                if macd_strength > 0:
                    macd_score = 0.5 + (macd_score - 0.5) * min(macd_strength / 0.001, 1)
                momentum_components.append(macd_score)

            # Recent momentum
            if "recent_momentum" in momentum_data:
                recent_mom = momentum_data["recent_momentum"]
                # Normalize momentum to 0-1 range
                mom_score = (np.tanh(recent_mom * 10) + 1) / 2
                momentum_components.append(mom_score)
            
            # Stochastic
            if "stochastic" in momentum_data:
                stoch_data = momentum_data["stochastic"]
                stoch_signal = stoch_data.get("signal", 0)
                stoch_score = (stoch_signal + 1) / 2
                momentum_components.append(stoch_score)
            
            # Williams %R
            if "williams_r" in momentum_data:
                williams_data = momentum_data["williams_r"]
                williams_signal = williams_data.get("signal", 0)
                williams_score = (williams_signal + 1) / 2
                momentum_components.append(williams_score)
            
            # CCI
            if "cci" in momentum_data:
                cci_data = momentum_data["cci"]
                cci_signal = cci_data.get("signal", 0)
                cci_score = (cci_signal + 1) / 2
                momentum_components.append(cci_score)

            scores["momentum"] = np.mean(momentum_components) if momentum_components else 0.5

        # Calculate technical signals score
        if "technical_signals" in analysis:
            tech_data = analysis["technical_signals"]
            if "signals" in tech_data and isinstance(tech_data["signals"], list):
                signals = tech_data["signals"]
                if signals:
                    # Weight signals by their confidence
                    weighted_sum = 0
                    total_weight = 0
                    for signal in signals:
                        if isinstance(signal, (list, tuple)) and len(signal) >= 3:
                            direction = signal[1]  # -1, 0, or 1
                            weight = signal[2]  # confidence weight
                            weighted_sum += ((direction + 1) / 2) * weight
                            total_weight += weight

                    scores["technical"] = weighted_sum / total_weight if total_weight > 0 else 0.5
                else:
                    scores["technical"] = 0.5
            else:
                scores["technical"] = 0.5

        # Calculate mean reversion score
        if "mean_reversion" in analysis:
            mr_data = analysis["mean_reversion"]
            mr_signal = mr_data.get("signal", 0)
            z_score = mr_data.get("z_score", 0)

            # Convert signal to score
            mr_score = (mr_signal + 1) / 2

            # Adjust based on z-score magnitude
            if abs(z_score) > 2:
                # Strong mean reversion signal
                mr_score = 0.8 if mr_signal > 0 else 0.2
            elif abs(z_score) > 1:
                # Moderate mean reversion signal
                mr_score = 0.65 if mr_signal > 0 else 0.35

            scores["mean_reversion"] = mr_score

        # Calculate volatility score
        if "volatility" in analysis:
            vol_data = analysis["volatility"]
            vol_regime = vol_data.get("regime", "normal")
            bb_position = vol_data.get("bb_position", 0.5)

            # Base score on regime
            if vol_regime == "high":
                vol_score = 0.5  # High volatility = neutral/cautious
            elif vol_regime == "low":
                # Low volatility can favor trend continuation
                vol_score = 0.6 if bb_position > 0.5 else 0.4
            else:
                # Normal volatility - use Bollinger Band position
                if bb_position > 0.8:
                    vol_score = 0.3  # Near upper band = bearish
                elif bb_position < 0.2:
                    vol_score = 0.7  # Near lower band = bullish
                else:
                    vol_score = 0.5  # Middle of bands = neutral

            scores["volatility"] = vol_score

        # Calculate volume score
        if "volume_analysis" in analysis:
            vol_analysis = analysis["volume_analysis"]
            vol_components = []
            
            if "obv_trend" in vol_analysis:
                obv_trend = vol_analysis["obv_trend"]
                vol_components.append((obv_trend + 1) / 2)
            
            if "volume_price_divergence" in vol_analysis:
                divergence = vol_analysis["volume_price_divergence"]
                # Divergence is contrarian signal
                div_score = 0.5 - divergence * 0.3
                vol_components.append(div_score)
            
            if "volume_ratio" in vol_analysis:
                vol_ratio = vol_analysis["volume_ratio"]
                # High volume ratio suggests strong movement
                if vol_ratio > 1.2:
                    # Amplify current trend
                    current_avg = np.mean(list(scores.values())) if scores else 0.5
                    vol_score = 0.5 + (current_avg - 0.5) * 1.2
                else:
                    vol_score = 0.5
                vol_components.append(vol_score)
            
            scores["volume"] = np.mean(vol_components) if vol_components else 0.5
        
        # Calculate pattern score
        if "pattern_recognition" in analysis:
            pattern_data = analysis["pattern_recognition"]
            pattern_signals = pattern_data.get("pattern_signals", [])
            
            if pattern_signals:
                weighted_sum = 0
                total_weight = 0
                for signal in pattern_signals:
                    if isinstance(signal, (list, tuple)) and len(signal) >= 3:
                        direction = signal[1]
                        weight = signal[2]
                        weighted_sum += ((direction + 1) / 2) * weight
                        total_weight += weight
                
                scores["pattern"] = weighted_sum / total_weight if total_weight > 0 else 0.5
            else:
                scores["pattern"] = 0.5
        
        # Calculate multi-timeframe score
        if "multi_timeframe" in analysis:
            mtf_data = analysis["multi_timeframe"]
            alignment = mtf_data.get("alignment", 0)
            scores["multi_timeframe"] = (alignment + 1) / 2

        # Add ML score if available
        if ml_available:
            ml_data = analysis["ml_analysis"]
            ml_proba = ml_data.get("prediction_proba", 0.5)
            ml_confidence = ml_data.get("confidence", 0.5)
            
            # Weight ML score by its confidence
            ml_score = 0.5 + (ml_proba - 0.5) * ml_confidence
            
            # Consider ensemble agreement
            if "ensemble_agreement" in ml_data:
                agreement = ml_data["ensemble_agreement"]
                ml_score = 0.5 + (ml_score - 0.5) * agreement
            
            scores["ml"] = ml_score

        # Calculate weighted composite
        composite = 0
        for component, weight in weights.items():
            component_score = scores.get(component, 0.5)  # Default to neutral if missing
            composite += component_score * weight

        # Debug output
        print(f"Debug - Component scores: {scores}")
        print(f"Debug - Weights: {weights}")
        print(f"Debug - Composite score: {composite}")

        return composite

    def _determine_signal(self, score: float) -> Tuple[str, str]:
        """Determine trading signal based on composite score with dynamic thresholds."""
        # Dynamic threshold adjustment based on score distribution
        # This prevents always returning neutral/weak signals
        
        if score >= 0.75:
            return "BULLISH", "STRONG"
        elif score >= 0.60:
            return "BULLISH", "MODERATE"
        elif score >= 0.52:
            return "BULLISH", "WEAK"
        elif score <= 0.25:
            return "BEARISH", "STRONG"
        elif score <= 0.40:
            return "BEARISH", "MODERATE"
        elif score <= 0.48:
            return "BEARISH", "WEAK"
        else:
            return "NEUTRAL", "MODERATE"

    def _calculate_confidence(self, analysis: Dict) -> float:
        """Calculate confidence level based on indicator agreement and signal strength."""
        signals = []
        signal_weights = []

        # Collect all directional signals with weights
        if "trend" in analysis and "direction" in analysis["trend"]:
            signals.append(analysis["trend"]["direction"])
            # Weight trend more if ADX is strong
            adx = analysis["trend"].get("adx_strength", 50) / 100
            signal_weights.append(1 + adx * 0.5)

        if "momentum" in analysis:
            if "rsi" in analysis["momentum"] and isinstance(analysis["momentum"]["rsi"], dict):
                signals.append(analysis["momentum"]["rsi"].get("signal", 0))
                signal_weights.append(1.0)
            if "macd" in analysis["momentum"] and isinstance(analysis["momentum"]["macd"], dict):
                signals.append(analysis["momentum"]["macd"].get("signal", 0))
                # Weight MACD by its strength
                strength = analysis["momentum"]["macd"].get("strength", 0)
                signal_weights.append(1 + min(strength * 100, 0.5))
            if "stochastic" in analysis["momentum"]:
                signals.append(analysis["momentum"]["stochastic"].get("signal", 0))
                signal_weights.append(0.8)
            if "williams_r" in analysis["momentum"]:
                signals.append(analysis["momentum"]["williams_r"].get("signal", 0))
                signal_weights.append(0.8)
            if "cci" in analysis["momentum"]:
                signals.append(analysis["momentum"]["cci"].get("signal", 0))
                signal_weights.append(0.7)

        if "technical_signals" in analysis and "signals" in analysis["technical_signals"]:
            tech_sigs = analysis["technical_signals"]["signals"]
            if isinstance(tech_sigs, list):
                for s in tech_sigs:
                    if isinstance(s, (list, tuple)) and len(s) >= 3:
                        signals.append(s[1])
                        signal_weights.append(s[2])

        # Add volume signals
        if "volume_analysis" in analysis:
            vol_data = analysis["volume_analysis"]
            if "obv_trend" in vol_data:
                signals.append(vol_data["obv_trend"])
                signal_weights.append(0.6)

        # Add pattern signals
        if "pattern_recognition" in analysis:
            pattern_sigs = analysis["pattern_recognition"].get("pattern_signals", [])
            for s in pattern_sigs:
                if isinstance(s, (list, tuple)) and len(s) >= 3:
                    signals.append(s[1])
                    signal_weights.append(s[2])

        # Add multi-timeframe signal
        if "multi_timeframe" in analysis:
            mtf_signal = analysis["multi_timeframe"].get("signal", 0)
            if mtf_signal != 0:
                signals.append(mtf_signal)
                signal_weights.append(1.2)  # Higher weight for timeframe alignment

        # Add ML signal if available
        if analysis.get("ml_analysis", {}).get("ml_available", False):
            ml_signal = analysis["ml_analysis"].get("signal", 0)
            ml_confidence = analysis["ml_analysis"].get("confidence", 0.5)
            signals.append(ml_signal)
            signal_weights.append(1 + ml_confidence)

        # Calculate weighted agreement
        if not signals:
            return 0.5

        # Normalize weights
        if not signal_weights:
            signal_weights = [1] * len(signals)

        weighted_bullish = sum(w for s, w in zip(signals, signal_weights) if s > 0)
        weighted_bearish = sum(w for s, w in zip(signals, signal_weights) if s < 0)
        total_weight = sum(signal_weights)

        # Calculate directional agreement
        directional_weight = max(weighted_bullish, weighted_bearish)
        agreement = directional_weight / total_weight if total_weight > 0 else 0.5

        # Boost confidence if volatility regime supports it
        vol_regime = analysis.get("volatility", {}).get("regime", "normal")
        if vol_regime == "low":
            agreement *= 1.1  # Low volatility = more reliable signals
        elif vol_regime == "high":
            agreement *= 0.9  # High volatility = less reliable signals

        # Cap confidence but allow for higher values
        return min(agreement, 0.95)

    def _generate_reasons(self, analysis: Dict) -> list[str]:
        """Generate human-readable reasons for the prediction."""
        reasons = []

        # Debug: Print what we're receiving
        print("Debug - Analysis keys:", analysis.keys())

        # Trend reasons
        trend = analysis.get("trend", {})
        print("Debug - Trend data:", trend)

        if trend and "sma_signals" in trend:
            sma_signals = trend["sma_signals"]
            if sma_signals.get("short_term") == sma_signals.get("medium_term"):
                direction = "above" if sma_signals["short_term"] > 0 else "below"
                reasons.append(f"Price is {direction} both short and medium-term moving averages")

        # Momentum reasons
        momentum = analysis.get("momentum", {})
        print("Debug - Momentum data:", momentum)

        if momentum and "rsi" in momentum:
            rsi_data = momentum["rsi"]
            if isinstance(rsi_data, dict) and "value" in rsi_data:
                rsi = rsi_data["value"]
                if rsi > 70:
                    reasons.append(f"RSI indicates overbought conditions ({rsi:.1f})")
                elif rsi < 30:
                    reasons.append(f"RSI indicates oversold conditions ({rsi:.1f})")

        if momentum and "macd" in momentum:
            macd_data = momentum["macd"]
            if isinstance(macd_data, dict) and "signal" in macd_data:
                if macd_data["signal"] > 0:
                    reasons.append("MACD shows bullish momentum")
                else:
                    reasons.append("MACD shows bearish momentum")

        # Volatility reasons
        vol = analysis.get("volatility", {})
        print("Debug - Volatility data:", vol)

        if vol and "regime" in vol:
            if vol["regime"] == "high":
                percentile = vol.get("percentile", 0)
                reasons.append(f"Volatility is elevated (percentile: {percentile:.0f}%)")

        # Technical signals
        tech_data = analysis.get("technical_signals", {})
        print("Debug - Technical signals data:", tech_data)

        if tech_data and "signals" in tech_data:
            tech_signals = tech_data["signals"]
            if isinstance(tech_signals, list):
                strong_signals = [s for s in tech_signals if
                                  isinstance(s, (list, tuple)) and len(s) >= 3 and s[2] >= 0.8]
                for signal in strong_signals[:2]:  # Top 2 strongest signals
                    reasons.append(signal[0])

        # Volume reasons
        volume_data = analysis.get("volume_analysis", {})
        if volume_data:
            if "volume_trend" in volume_data and volume_data["volume_trend"] != "stable":
                reasons.append(f"Volume is {volume_data['volume_trend']}")
            
            divergence = volume_data.get("volume_price_divergence", 0)
            if divergence != 0:
                div_type = "Bullish" if divergence > 0 else "Bearish"
                reasons.append(f"{div_type} volume/price divergence detected")
        
        # Pattern reasons
        pattern_data = analysis.get("pattern_recognition", {})
        if pattern_data and "pattern_signals" in pattern_data:
            pattern_sigs = pattern_data["pattern_signals"]
            if pattern_sigs:
                # Add the strongest pattern signal
                strongest_pattern = max(pattern_sigs, key=lambda x: x[2] if len(x) >= 3 else 0)
                if strongest_pattern[2] >= 0.6:
                    reasons.append(strongest_pattern[0])
        
        # Multi-timeframe reasons
        mtf_data = analysis.get("multi_timeframe", {})
        if mtf_data:
            alignment = mtf_data.get("alignment", 0)
            if abs(alignment) > 0.5:
                timeframe_status = "aligned bullish" if alignment > 0 else "aligned bearish"
                reasons.append(f"Multiple timeframes are {timeframe_status}")

        # ML reasons
        ml_data = analysis.get("ml_analysis", {})
        print("Debug - ML data:", ml_data)

        if ml_data and ml_data.get("ml_available", False):
            confidence = ml_data.get("confidence", 0)
            signal = ml_data.get("signal", 0)
            prediction_class = ml_data.get("prediction_class", "")

            if confidence > 0.7:
                reasons.insert(0, f"ML model predicts {prediction_class} (confidence: {confidence:.1%})")
            elif confidence > 0.5:
                reasons.insert(0, f"ML model suggests {prediction_class} bias (confidence: {confidence:.1%})")

            # Add class probabilities if significant
            class_probs = ml_data.get("class_probabilities", {})
            if class_probs:
                max_class = max(class_probs.items(), key=lambda x: x[1])
                if max_class[1] > 0.4:
                    reasons.append(f"ML: {max_class[1]:.0%} probability of {max_class[0].replace('_', ' ')}")

            # Add ensemble agreement
            ensemble_agreement = ml_data.get("ensemble_agreement", 0)
            if ensemble_agreement > 0.8:
                reasons.append("ML ensemble models show strong agreement")
            elif ensemble_agreement < 0.5:
                reasons.append("ML ensemble models show divergence (lower confidence)")

        # If no reasons were generated, add some default ones based on the score
        if not reasons:
            print("Debug - No reasons generated, adding defaults")
            score = self._calculate_composite_score(analysis)
            if score > 0.55:
                reasons.append("Overall indicators suggest bullish bias")
            elif score < 0.45:
                reasons.append("Overall indicators suggest bearish bias")
            else:
                reasons.append("Mixed signals from technical indicators")
                reasons.append("Market showing neutral consolidation")

        print(f"Debug - Final reasons: {reasons}")
        return reasons