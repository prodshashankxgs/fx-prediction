# Libraries
import numpy as np
from typing import Dict, Tuple, List
from datetime import datetime
from scipy import stats

from backend.regime_detector import RegimeDetector


class EnhancedForexPredictor:
    """Enhanced forex predictor with dynamic thresholds and regime-aware predictions."""

    def __init__(self, base_thresholds: Dict):
        self.base_thresholds = base_thresholds
        self.regime_detector = RegimeDetector()
        self.prediction_history = []
        
    def predict_with_regime_adaptation(self, analysis: Dict) -> Dict:
        """Generate regime-aware prediction with dynamic thresholds."""
        
        # Get current market regime
        regime_info = analysis.get("regime_analysis", {})
        
        # Calculate base composite score
        base_score = self._calculate_enhanced_composite_score(analysis, regime_info)
        
        # Apply regime-specific adjustments
        adjusted_score = self._apply_regime_adjustments(base_score, regime_info)
        
        # Get dynamic thresholds based on regime
        dynamic_thresholds = self._get_dynamic_thresholds(regime_info)
        
        # Determine signal and strength
        signal, strength = self._determine_signal_with_dynamic_thresholds(
            adjusted_score, dynamic_thresholds
        )
        
        # Calculate enhanced confidence
        confidence = self._calculate_enhanced_confidence(analysis, regime_info)
        
        # Apply final filters
        filtered_result = self._apply_prediction_filters(
            signal, strength, confidence, analysis, regime_info
        )
        
        # Generate comprehensive reasons
        reasons = self._generate_enhanced_reasons(analysis, regime_info, filtered_result)
        
        # Create prediction result
        prediction = {
            "prediction": filtered_result["signal"],
            "strength": filtered_result["strength"],
            "score": adjusted_score,
            "base_score": base_score,
            "confidence": filtered_result["confidence"],
            "regime_adjusted": True,
            "dynamic_thresholds": dynamic_thresholds,
            "regime_info": regime_info,
            "reasons": reasons,
            "timestamp": datetime.now().isoformat(),
            "prediction_filters": filtered_result.get("filters_applied", [])
        }
        
        # Store in history for analysis
        self.prediction_history.append(prediction)
        
        return prediction

    def _calculate_enhanced_composite_score(self, analysis: Dict, regime_info: Dict) -> float:
        """Calculate composite score with regime awareness."""
        
        # Get market regime characteristics
        vol_regime = regime_info.get("volatility_regime", {}).get("regime", "normal")
        trend_regime = regime_info.get("trend_regime", {}).get("regime", "sideways")
        momentum_regime = regime_info.get("momentum_regime", {}).get("regime", "neutral")
        stress_level = regime_info.get("stress_conditions", {}).get("stress_level", "normal")
        
        # Regime-adaptive weights
        weights = self._get_regime_adaptive_weights(vol_regime, trend_regime, momentum_regime, stress_level)
        
        scores = {}
        
        # Enhanced trend scoring
        if "trend" in analysis:
            trend_data = analysis["trend"]
            trend_score = self._calculate_trend_score(trend_data, regime_info)
            scores["trend"] = trend_score
        
        # Enhanced momentum scoring  
        if "momentum" in analysis:
            momentum_data = analysis["momentum"]
            momentum_score = self._calculate_momentum_score(momentum_data, regime_info)
            scores["momentum"] = momentum_score
        
        # Technical signals with regime context
        if "technical_signals" in analysis:
            tech_score = self._calculate_technical_score(analysis["technical_signals"], regime_info)
            scores["technical"] = tech_score
        
        # Enhanced volatility scoring
        if "volatility" in analysis:
            vol_score = self._calculate_volatility_score(analysis["volatility"], regime_info)
            scores["volatility"] = vol_score
        
        # Mean reversion with regime context
        if "mean_reversion" in analysis:
            mr_score = self._calculate_mean_reversion_score(analysis["mean_reversion"], regime_info)
            scores["mean_reversion"] = mr_score
        
        # Volume analysis
        if "volume_analysis" in analysis:
            volume_score = self._calculate_volume_score(analysis["volume_analysis"], regime_info)
            scores["volume"] = volume_score
        
        # Pattern recognition
        if "pattern_recognition" in analysis:
            pattern_score = self._calculate_pattern_score(analysis["pattern_recognition"], regime_info)
            scores["pattern"] = pattern_score
        
        # Multi-timeframe analysis
        if "multi_timeframe" in analysis:
            mtf_score = self._calculate_mtf_score(analysis["multi_timeframe"], regime_info)
            scores["multi_timeframe"] = mtf_score
        
        # Enhanced ML scoring
        if "ml_analysis" in analysis and analysis["ml_analysis"].get("ml_available", False):
            ml_score = self._calculate_ml_score(analysis["ml_analysis"], regime_info)
            scores["ml"] = ml_score
        
        # Calculate weighted composite
        composite = 0
        total_weight = 0
        
        for component, weight in weights.items():
            if component in scores:
                composite += scores[component] * weight
                total_weight += weight
        
        # Normalize by actual weights used
        if total_weight > 0:
            composite = composite / total_weight
        else:
            composite = 0.5  # Neutral default
        
        return np.clip(composite, 0, 1)

    def _get_regime_adaptive_weights(self, vol_regime: str, trend_regime: str, 
                                   momentum_regime: str, stress_level: str) -> Dict:
        """Get adaptive weights based on market regime."""
        
        # Base weights
        weights = {
            "trend": 0.20,
            "momentum": 0.18,
            "technical": 0.22,
            "volatility": 0.08,
            "mean_reversion": 0.08,
            "volume": 0.06,
            "pattern": 0.06,
            "multi_timeframe": 0.06,
            "ml": 0.06
        }
        
        # Volatility regime adjustments
        if vol_regime == "high":
            weights["volatility"] += 0.05
            weights["mean_reversion"] += 0.03
            weights["trend"] -= 0.04
            weights["momentum"] -= 0.04
        elif vol_regime == "low":
            weights["trend"] += 0.04
            weights["momentum"] += 0.04
            weights["volatility"] -= 0.04
            weights["mean_reversion"] -= 0.04
        
        # Trend regime adjustments
        if trend_regime in ["uptrend", "downtrend"]:
            weights["trend"] += 0.08
            weights["multi_timeframe"] += 0.04
            weights["technical"] -= 0.06
            weights["mean_reversion"] -= 0.06
        elif trend_regime == "sideways":
            weights["mean_reversion"] += 0.08
            weights["technical"] += 0.04
            weights["trend"] -= 0.06
            weights["multi_timeframe"] -= 0.06
        
        # Momentum regime adjustments
        if momentum_regime in ["bullish", "bearish"]:
            weights["momentum"] += 0.06
            weights["ml"] += 0.02
            weights["volatility"] -= 0.04
            weights["mean_reversion"] -= 0.04
        
        # Stress level adjustments
        if stress_level == "high":
            weights["volatility"] += 0.08
            weights["volume"] += 0.04
            weights["trend"] -= 0.06
            weights["momentum"] -= 0.06
        
        # Ensure weights are positive and sum appropriately
        weights = {k: max(0.01, v) for k, v in weights.items()}
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}
        
        return weights

    def _calculate_trend_score(self, trend_data: Dict, regime_info: Dict) -> float:
        """Enhanced trend scoring with regime context."""
        base_direction = trend_data.get("direction", 0)
        trend_strength = trend_data.get("trend_strength", 0)
        trend_consistency = trend_data.get("trend_consistency", 0)
        
        # Base score
        score = (base_direction + 1) / 2
        
        # Adjust based on trend regime
        trend_regime = regime_info.get("trend_regime", {})
        if trend_regime.get("regime") in ["uptrend", "downtrend"]:
            regime_direction = 1 if trend_regime.get("regime") == "uptrend" else -1
            if np.sign(base_direction) == regime_direction:
                score = 0.5 + (score - 0.5) * (1 + trend_regime.get("strength", 0))
        
        # Strength adjustments
        if abs(trend_strength) > 0.01:
            score = 0.5 + (score - 0.5) * (1 + min(abs(trend_strength) * 50, 1))
        
        # Consistency adjustments
        if abs(trend_consistency) > 0.3:
            score = 0.5 + (score - 0.5) * (1 + abs(trend_consistency))
        
        return np.clip(score, 0, 1)

    def _calculate_momentum_score(self, momentum_data: Dict, regime_info: Dict) -> float:
        """Enhanced momentum scoring."""
        momentum_components = []
        
        # RSI component with regime context
        if "rsi" in momentum_data:
            rsi_data = momentum_data["rsi"]
            if isinstance(rsi_data, dict):
                rsi_value = rsi_data.get("value", 50)
                # Dynamic RSI thresholds based on volatility
                vol_regime = regime_info.get("volatility_regime", {}).get("regime", "normal")
                if vol_regime == "high":
                    overbought, oversold = 75, 25  # Wider bands in high vol
                else:
                    overbought, oversold = 70, 30
                
                if rsi_value > overbought:
                    rsi_score = 0.1 + (100 - rsi_value) / 100 * 0.3
                elif rsi_value < oversold:
                    rsi_score = 0.9 - rsi_value / 100 * 0.3
                else:
                    rsi_score = 0.5 + (50 - rsi_value) / 100
                
                momentum_components.append(rsi_score)
        
        # MACD with strength weighting
        if "macd" in momentum_data:
            macd_data = momentum_data["macd"]
            if isinstance(macd_data, dict):
                macd_signal = macd_data.get("signal", 0)
                macd_strength = macd_data.get("strength", 0)
                macd_score = (macd_signal + 1) / 2
                
                # Weight by strength
                if macd_strength > 0:
                    strength_multiplier = 1 + min(macd_strength * 500, 1)
                    macd_score = 0.5 + (macd_score - 0.5) * strength_multiplier
                
                momentum_components.append(macd_score)
        
        # Other momentum indicators
        for indicator in ["stochastic", "williams_r", "cci"]:
            if indicator in momentum_data:
                ind_data = momentum_data[indicator]
                if isinstance(ind_data, dict) and "signal" in ind_data:
                    signal = ind_data["signal"]
                    score = (signal + 1) / 2
                    momentum_components.append(score)
        
        # Recent momentum
        if "recent_momentum" in momentum_data:
            recent_mom = momentum_data["recent_momentum"]
            mom_score = (np.tanh(recent_mom * 10) + 1) / 2
            momentum_components.append(mom_score)
        
        return np.mean(momentum_components) if momentum_components else 0.5

    def _calculate_technical_score(self, tech_data: Dict, regime_info: Dict) -> float:
        """Calculate technical signals score with regime weighting."""
        signals = tech_data.get("signals", [])
        if not signals:
            return 0.5
        
        weighted_sum = 0
        total_weight = 0
        
        # Regime-based signal weighting
        stress_level = regime_info.get("stress_conditions", {}).get("stress_level", "normal")
        stress_multiplier = 0.7 if stress_level == "high" else 1.0
        
        for signal in signals:
            if isinstance(signal, (list, tuple)) and len(signal) >= 3:
                direction = signal[1]
                weight = signal[2] * stress_multiplier
                weighted_sum += ((direction + 1) / 2) * weight
                total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.5

    def _calculate_volatility_score(self, vol_data: Dict, regime_info: Dict) -> float:
        """Calculate volatility-based score."""
        vol_regime_data = regime_info.get("volatility_regime", {})
        vol_regime = vol_regime_data.get("regime", "normal")
        bb_position = vol_data.get("bb_position", 0.5)
        
        if vol_regime == "high":
            # In high volatility, be more cautious
            return 0.5
        elif vol_regime == "low":
            # In low volatility, favor trend continuation
            return 0.6 if bb_position > 0.5 else 0.4
        else:
            # Normal volatility - use Bollinger Band position
            if bb_position > 0.9:
                return 0.2  # Near upper band = bearish
            elif bb_position < 0.1:
                return 0.8  # Near lower band = bullish
            else:
                return 0.5

    def _calculate_mean_reversion_score(self, mr_data: Dict, regime_info: Dict) -> float:
        """Calculate mean reversion score with regime context."""
        mr_signal = mr_data.get("signal", 0)
        z_score = mr_data.get("z_score", 0)
        
        # Base score
        mr_score = (mr_signal + 1) / 2
        
        # Trend regime affects mean reversion
        trend_regime = regime_info.get("trend_regime", {}).get("regime", "sideways")
        
        if trend_regime == "sideways":
            # Stronger mean reversion in sideways markets
            if abs(z_score) > 2:
                mr_score = 0.8 if mr_signal > 0 else 0.2
            elif abs(z_score) > 1:
                mr_score = 0.65 if mr_signal > 0 else 0.35
        else:
            # Weaker mean reversion in trending markets
            mr_score = 0.5 + (mr_score - 0.5) * 0.7
        
        return mr_score

    def _calculate_volume_score(self, volume_data: Dict, regime_info: Dict) -> float:
        """Calculate volume-based score."""
        volume_components = []
        
        if "obv_trend" in volume_data:
            obv_trend = volume_data["obv_trend"]
            volume_components.append((obv_trend + 1) / 2)
        
        if "volume_price_divergence" in volume_data:
            divergence = volume_data["volume_price_divergence"]
            div_score = 0.5 - divergence * 0.3
            volume_components.append(div_score)
        
        if "volume_ratio" in volume_data:
            vol_ratio = volume_data["volume_ratio"]
            if vol_ratio > 1.5:  # High volume supports current move
                volume_components.append(0.7)
            elif vol_ratio < 0.5:  # Low volume weakens move
                volume_components.append(0.3)
            else:
                volume_components.append(0.5)
        
        return np.mean(volume_components) if volume_components else 0.5

    def _calculate_pattern_score(self, pattern_data: Dict, regime_info: Dict) -> float:
        """Calculate pattern recognition score."""
        pattern_signals = pattern_data.get("pattern_signals", [])
        
        if not pattern_signals:
            return 0.5
        
        weighted_sum = 0
        total_weight = 0
        
        for signal in pattern_signals:
            if isinstance(signal, (list, tuple)) and len(signal) >= 3:
                direction = signal[1]
                weight = signal[2]
                weighted_sum += ((direction + 1) / 2) * weight
                total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.5

    def _calculate_mtf_score(self, mtf_data: Dict, regime_info: Dict) -> float:
        """Calculate multi-timeframe score."""
        alignment = mtf_data.get("alignment", 0)
        score = (alignment + 1) / 2
        
        # Boost score if trend regime confirms alignment
        trend_regime = regime_info.get("trend_regime", {})
        if trend_regime.get("regime") in ["uptrend", "downtrend"]:
            regime_direction = 1 if trend_regime.get("regime") == "uptrend" else -1
            if np.sign(alignment) == regime_direction:
                score = 0.5 + (score - 0.5) * 1.3
        
        return np.clip(score, 0, 1)

    def _calculate_ml_score(self, ml_data: Dict, regime_info: Dict) -> float:
        """Calculate ML-based score with regime context."""
        ml_proba = ml_data.get("prediction_proba", 0.5)
        ml_confidence = ml_data.get("confidence", 0.5)
        
        # Base ML score
        ml_score = 0.5 + (ml_proba - 0.5) * ml_confidence
        
        # Ensemble agreement boosts confidence
        ensemble_agreement = ml_data.get("ensemble_agreement", 0.5)
        if ensemble_agreement > 0.8:
            ml_score = 0.5 + (ml_score - 0.5) * 1.2
        elif ensemble_agreement < 0.5:
            ml_score = 0.5 + (ml_score - 0.5) * 0.8
        
        # Stress conditions reduce ML confidence
        stress_level = regime_info.get("stress_conditions", {}).get("stress_level", "normal")
        if stress_level == "high":
            ml_score = 0.5 + (ml_score - 0.5) * 0.7
        
        return np.clip(ml_score, 0, 1)

    def _apply_regime_adjustments(self, base_score: float, regime_info: Dict) -> float:
        """Apply regime-specific adjustments to the base score."""
        adjusted_score = base_score
        
        # Volatility regime adjustments
        vol_regime = regime_info.get("volatility_regime", {})
        if vol_regime.get("regime") == "high":
            # Dampen extreme scores in high volatility
            adjusted_score = 0.5 + (adjusted_score - 0.5) * 0.8
        elif vol_regime.get("regime") == "low":
            # Amplify scores in low volatility
            adjusted_score = 0.5 + (adjusted_score - 0.5) * 1.2
        
        # Stress adjustments
        stress_conditions = regime_info.get("stress_conditions", {})
        if stress_conditions.get("stress_level") == "high":
            # Pull towards neutral in stress
            adjusted_score = 0.5 + (adjusted_score - 0.5) * 0.6
        
        # Trend momentum alignment
        trend_regime = regime_info.get("trend_regime", {})
        momentum_regime = regime_info.get("momentum_regime", {})
        
        if (trend_regime.get("regime") == "uptrend" and momentum_regime.get("regime") == "bullish"):
            adjusted_score = min(1.0, adjusted_score * 1.1)
        elif (trend_regime.get("regime") == "downtrend" and momentum_regime.get("regime") == "bearish"):
            adjusted_score = max(0.0, adjusted_score * 0.9)
        
        return np.clip(adjusted_score, 0, 1)

    def _get_dynamic_thresholds(self, regime_info: Dict) -> Dict:
        """Get dynamic thresholds based on market regime."""
        base_thresholds = self.base_thresholds.copy()
        
        # Volatility adjustments
        vol_regime = regime_info.get("volatility_regime", {}).get("regime", "normal")
        if vol_regime == "high":
            # Higher thresholds in volatile markets
            adjustment = 0.05
        elif vol_regime == "low":
            # Lower thresholds in calm markets  
            adjustment = -0.03
        else:
            adjustment = 0
        
        # Stress adjustments
        stress_level = regime_info.get("stress_conditions", {}).get("stress_level", "normal")
        if stress_level == "high":
            adjustment += 0.08
        elif stress_level == "moderate":
            adjustment += 0.03
        
        # Apply adjustments
        dynamic_thresholds = {}
        for key, value in base_thresholds.items():
            if "bullish" in key:
                dynamic_thresholds[key] = min(0.9, value + adjustment)
            elif "bearish" in key:
                dynamic_thresholds[key] = max(0.1, value - adjustment)
            else:
                dynamic_thresholds[key] = value
        
        return dynamic_thresholds

    def _determine_signal_with_dynamic_thresholds(self, score: float, 
                                                thresholds: Dict) -> Tuple[str, str]:
        """Determine signal using dynamic thresholds."""
        if score >= thresholds.get("strong_bullish", 0.75):
            return "BULLISH", "STRONG"
        elif score >= thresholds.get("bullish", 0.60):
            return "BULLISH", "MODERATE"
        elif score >= thresholds.get("weak_bullish", 0.52):
            return "BULLISH", "WEAK"
        elif score <= thresholds.get("strong_bearish", 0.25):
            return "BEARISH", "STRONG"
        elif score <= thresholds.get("bearish", 0.40):
            return "BEARISH", "MODERATE"
        elif score <= thresholds.get("weak_bearish", 0.48):
            return "BEARISH", "WEAK"
        else:
            return "NEUTRAL", "MODERATE"

    def _calculate_enhanced_confidence(self, analysis: Dict, regime_info: Dict) -> float:
        """Calculate enhanced confidence with regime awareness."""
        signals = []
        signal_weights = []
        
        # Collect signals from analysis
        # ... (similar to original but with regime adjustments)
        
        # Base confidence calculation
        base_confidence = 0.5  # Placeholder - implement full logic
        
        # Regime stability adjustment
        stability_info = regime_info.get("stability_analysis", {})
        if stability_info.get("stability") == "high":
            confidence_boost = 0.1
        elif stability_info.get("stability") == "low":
            confidence_boost = -0.1
        else:
            confidence_boost = 0
        
        final_confidence = base_confidence + confidence_boost
        return np.clip(final_confidence, 0.1, 0.95)

    def _apply_prediction_filters(self, signal: str, strength: str, confidence: float,
                                analysis: Dict, regime_info: Dict) -> Dict:
        """Apply final filters to prediction."""
        filters_applied = []
        
        # Minimum confidence filter
        min_confidence = regime_info.get("parameters", {}).get("confidence_threshold", 0.6)
        if confidence < min_confidence:
            signal = "NEUTRAL"
            strength = "WEAK"
            filters_applied.append("low_confidence")
        
        # Stress filter
        stress_level = regime_info.get("stress_conditions", {}).get("stress_level", "normal")
        if stress_level == "high" and confidence < 0.8:
            signal = "NEUTRAL"
            strength = "WEAK"
            filters_applied.append("high_stress")
        
        # Conflicting signals filter
        ml_available = analysis.get("ml_analysis", {}).get("ml_available", False)
        if ml_available:
            ml_signal = analysis["ml_analysis"].get("signal", 0)
            trend_signal = analysis.get("trend", {}).get("direction", 0)
            
            # If ML and trend strongly disagree, reduce to neutral
            if (ml_signal > 0 and trend_signal < -0.5) or (ml_signal < 0 and trend_signal > 0.5):
                if confidence < 0.8:
                    signal = "NEUTRAL"
                    strength = "WEAK"
                    filters_applied.append("conflicting_signals")
        
        return {
            "signal": signal,
            "strength": strength,
            "confidence": confidence,
            "filters_applied": filters_applied
        }

    def _generate_enhanced_reasons(self, analysis: Dict, regime_info: Dict, 
                                 filtered_result: Dict) -> List[str]:
        """Generate comprehensive reasons for the prediction."""
        reasons = []
        
        # Regime-based reasons
        vol_regime = regime_info.get("volatility_regime", {})
        if vol_regime.get("regime") == "high":
            reasons.append(f"High volatility regime detected (percentile: {vol_regime.get('percentile', 0):.0f}%)")
        
        trend_regime = regime_info.get("trend_regime", {})
        if trend_regime.get("regime") != "sideways":
            direction = trend_regime.get("regime", "")
            strength = trend_regime.get("strength", 0)
            reasons.append(f"Market in {direction} (strength: {strength:.1%})")
        
        # ML reasons with ensemble info
        ml_data = analysis.get("ml_analysis", {})
        if ml_data.get("ml_available", False):
            confidence = ml_data.get("confidence", 0)
            prediction_class = ml_data.get("prediction_class", "")
            ensemble_agreement = ml_data.get("ensemble_agreement", 0)
            
            if confidence > 0.7:
                reasons.insert(0, f"ML ensemble predicts {prediction_class} (confidence: {confidence:.1%}, agreement: {ensemble_agreement:.1%})")
        
        # Add filter explanations
        filters = filtered_result.get("filters_applied", [])
        if "high_stress" in filters:
            reasons.append("Prediction confidence reduced due to high market stress")
        if "conflicting_signals" in filters:
            reasons.append("Conflicting signals detected - reduced to neutral")
        
        # Technical reasons with regime context
        # ... (implement detailed technical reasons)
        
        # Default fallback
        if not reasons:
            score = filtered_result.get("confidence", 0.5)
            if score > 0.6:
                reasons.append("Overall indicators suggest bullish bias with regime confirmation")
            elif score < 0.4:
                reasons.append("Overall indicators suggest bearish bias with regime confirmation")
            else:
                reasons.append("Mixed signals from technical indicators and regime analysis")
        
        return reasons

    def predict(self, analysis: Dict) -> Dict:
        """Main prediction method - enhanced version."""
        return self.predict_with_regime_adaptation(analysis)

    def get_prediction_statistics(self) -> Dict:
        """Get statistics about recent predictions."""
        if not self.prediction_history:
            return {}
        
        recent_predictions = self.prediction_history[-50:]  # Last 50 predictions
        
        signal_counts = {}
        confidence_scores = []
        regime_adjustments = 0
        
        for pred in recent_predictions:
            signal = pred.get("prediction", "NEUTRAL")
            signal_counts[signal] = signal_counts.get(signal, 0) + 1
            confidence_scores.append(pred.get("confidence", 0.5))
            
            if pred.get("regime_adjusted", False):
                regime_adjustments += 1
        
        return {
            "total_predictions": len(recent_predictions),
            "signal_distribution": signal_counts,
            "average_confidence": np.mean(confidence_scores),
            "regime_adjustments_pct": regime_adjustments / len(recent_predictions) * 100
        }