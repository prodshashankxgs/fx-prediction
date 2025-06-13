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

        # Adjust weights based on ML availability
        if ml_available:
            weights = {
                "trend": 0.20,
                "momentum": 0.20,
                "technical": 0.25,
                "mean_reversion": 0.10,
                "volatility": 0.10,
                "ml": 0.15  # ML gets 15% weight when available
            }
        else:
            weights = {
                "trend": 0.25,
                "momentum": 0.25,
                "technical": 0.30,
                "mean_reversion": 0.10,
                "volatility": 0.10
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

            scores["trend"] = trend_score

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

        # Add ML score if available
        if ml_available:
            ml_data = analysis["ml_analysis"]
            ml_proba = ml_data.get("prediction_proba", 0.5)
            scores["ml"] = ml_proba

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
        """Determine trading signal based on composite score."""
        if score >= self.thresholds["strong_bullish"]:
            return "BULLISH", "STRONG"
        elif score >= self.thresholds["bullish"]:
            return "BULLISH", "MODERATE"
        elif score <= self.thresholds["strong_bearish"]:
            return "BEARISH", "STRONG"
        elif score <= self.thresholds["bearish"]:
            return "BEARISH", "MODERATE"
        else:
            return "NEUTRAL", "WEAK"

    def _calculate_confidence(self, analysis: Dict) -> float:
        """Calculate confidence level based on indicator agreement."""
        signals = []

        # Collect all directional signals
        if "trend" in analysis and "direction" in analysis["trend"]:
            signals.append(analysis["trend"]["direction"])

        if "momentum" in analysis:
            if "rsi" in analysis["momentum"] and isinstance(analysis["momentum"]["rsi"], dict):
                signals.append(analysis["momentum"]["rsi"].get("signal", 0))
            if "macd" in analysis["momentum"] and isinstance(analysis["momentum"]["macd"], dict):
                signals.append(analysis["momentum"]["macd"].get("signal", 0))

        if "technical_signals" in analysis and "signals" in analysis["technical_signals"]:
            tech_sigs = analysis["technical_signals"]["signals"]
            if isinstance(tech_sigs, list):
                signals.extend([s[1] for s in tech_sigs if isinstance(s, (list, tuple)) and len(s) >= 2])

        # Add ML signal if available
        if analysis.get("ml_analysis", {}).get("ml_available", False):
            ml_signal = analysis["ml_analysis"].get("signal", 0)
            signals.append(ml_signal)

        # Calculate agreement
        if not signals:
            return 0.5

        bullish = sum(1 for s in signals if s > 0)
        bearish = sum(1 for s in signals if s < 0)
        total = len(signals)

        agreement = max(bullish, bearish) / total
        return agreement

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

        # ML reasons
        ml_data = analysis.get("ml_analysis", {})
        print("Debug - ML data:", ml_data)

        if ml_data and ml_data.get("ml_available", False):
            confidence = ml_data.get("confidence", 0)
            signal = ml_data.get("signal", 0)

            if confidence > 0.7:
                if signal > 0:
                    reasons.insert(0, f"ML model shows strong bullish signal (confidence: {confidence:.1%})")
                elif signal < 0:
                    reasons.insert(0, f"ML model shows strong bearish signal (confidence: {confidence:.1%})")
            elif confidence > 0.5:
                if signal > 0:
                    reasons.insert(0, f"ML model indicates bullish bias (confidence: {confidence:.1%})")
                elif signal < 0:
                    reasons.insert(0, f"ML model indicates bearish bias (confidence: {confidence:.1%})")

            # Add top ML feature if available
            feature_importance = ml_data.get("feature_importance", {})
            if feature_importance:
                top_feature = max(feature_importance.items(), key=lambda x: x[1])
                reasons.append(f"ML: {top_feature[0]} is the most influential indicator")

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