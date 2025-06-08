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

        # ... existing score calculations ...

        # Add ML score if available
        if ml_available:
            ml_data = analysis["ml_analysis"]
            ml_score = (ml_data["signal"] + 1) / 2  # Convert -1,1 to 0,1
            scores["ml"] = ml_score

        # Calculate weighted composite
        composite = sum(scores.get(k, 0.5) * weights.get(k, 0) for k in weights)

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
        signals.append(analysis["trend"]["direction"])
        signals.append(analysis["momentum"]["rsi"]["signal"])
        signals.append(analysis["momentum"]["macd"]["signal"])

        tech_sigs = analysis["technical_signals"]["signals"]
        signals.extend([s[1] for s in tech_sigs])

        # Calculate agreement
        if not signals:
            return 0.5

        bullish = sum(1 for s in signals if s > 0)
        bearish = sum(1 for s in signals if s < 0)
        total = len(signals)

        agreement = max(bullish, bearish) / total
        return agreement

    # Updated _generate_reasons method for backend/predictor.py

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