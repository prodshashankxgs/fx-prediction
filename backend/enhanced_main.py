# Libraries
import argparse
import logging
import sys
import polars as pl
from datetime import datetime, timedelta

# Import enhanced modules
from backend.config import *
from backend.data_collector import PolygonClient, save_data_to_parquet, load_cached_data
from backend.features import FeatureEngineer
from backend.enhanced_analysis import EnhancedMarketAnalyzer
from backend.enhanced_predictor import EnhancedForexPredictor
from backend.utils import setup_logger, format_prediction_output

logger = setup_logger("enhanced_main")


class EnhancedForexPredictionApp:
    """Enhanced main application class with advanced ML and regime detection."""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.client = PolygonClient(POLYGON_API_KEY)
        self.feature_engineer = FeatureEngineer(TECHNICAL_INDICATORS)
        self.analyzer = EnhancedMarketAnalyzer(self.config)
        self.predictor = EnhancedForexPredictor(PREDICTION_THRESHOLDS)
        
        # Enhanced configuration
        self.ml_config = ML_CONFIG if 'ML_CONFIG' in globals() else {}
        
        # Performance tracking
        self.prediction_history = []
        self.model_performance = {}

    def run(self, pair: str, use_cache: bool = True, enhanced_mode: bool = True) -> Dict:
        """Run the enhanced prediction pipeline."""
        logger.info(f"Starting enhanced prediction for {pair}")

        # Validate currency pair
        if not self.client.validate_currency_pair(pair):
            raise ValueError(f"Unsupported currency pair: {pair}")

        # Get data
        logger.info("Fetching market data...")
        df = self._get_data(pair, use_cache)

        if len(df) < 100:
            logger.warning(f"Limited data available: {len(df)} rows. Some advanced features may not be available.")

        # Calculate enhanced features
        logger.info("Calculating enhanced technical indicators...")
        df = self.feature_engineer.calculate_all_features(df)
        
        logger.info(f"Feature engineering complete. DataFrame shape: {df.shape}")
        
        # Train enhanced ML models if not already trained
        if enhanced_mode:
            logger.info("Training enhanced ML models...")
            try:
                train_result = self.analyzer.train_ml_models(df)
                if train_result.get("status") == "success":
                    best_model = train_result.get("best_model", "unknown")
                    best_accuracy = train_result.get("best_accuracy", 0)
                    models_trained = train_result.get("models_trained", [])
                    
                    logger.info(f"Enhanced ML models trained successfully")
                    logger.info(f"Best model: {best_model} (Accuracy: {best_accuracy:.2%})")
                    logger.info(f"Models trained: {', '.join(models_trained)}")
                    
                    # Store model performance
                    self.model_performance[pair] = train_result
                else:
                    logger.warning(f"Enhanced ML training failed: {train_result.get('error', 'Unknown error')}")
            except Exception as e:
                logger.error(f"Enhanced ML training error: {str(e)}")

        # Run enhanced analysis
        logger.info("Running enhanced market analysis...")
        analysis_results = self.analyzer.analyze(df)
        
        # Log regime information
        regime_info = analysis_results.get("regime_analysis", {})
        if regime_info:
            vol_regime = regime_info.get("volatility_regime", {}).get("regime", "unknown")
            trend_regime = regime_info.get("trend_regime", {}).get("regime", "unknown") 
            stress_level = regime_info.get("stress_conditions", {}).get("stress_level", "unknown")
            
            logger.info(f"Market regime - Volatility: {vol_regime}, Trend: {trend_regime}, Stress: {stress_level}")

        # Generate enhanced prediction
        logger.info("Generating enhanced prediction...")
        prediction = self.predictor.predict_with_regime_adaptation(analysis_results)
        
        # Add metadata
        prediction.update({
            "pair": pair,
            "data_points": len(df),
            "features_calculated": len(df.columns),
            "enhanced_mode": enhanced_mode,
            "model_performance": self.model_performance.get(pair, {})
        })
        
        # Store prediction for analysis
        self.prediction_history.append(prediction)
        
        # Display results
        self._display_enhanced_results(prediction, pair)

        return prediction

    def run_batch_analysis(self, pairs: List[str], use_cache: bool = True) -> Dict[str, Dict]:
        """Run analysis on multiple currency pairs."""
        logger.info(f"Starting batch analysis for {len(pairs)} pairs")
        
        results = {}
        
        for pair in pairs:
            try:
                logger.info(f"Processing {pair}...")
                result = self.run(pair, use_cache, enhanced_mode=True)
                results[pair] = result
                
                # Brief summary
                signal = result.get("prediction", "NEUTRAL")
                confidence = result.get("confidence", 0)
                logger.info(f"{pair}: {signal} (confidence: {confidence:.1%})")
                
            except Exception as e:
                logger.error(f"Error processing {pair}: {str(e)}")
                results[pair] = {"error": str(e)}
        
        return results

    def train_all_models(self, pair: str, retrain: bool = False) -> Dict:
        """Train all available models for a currency pair."""
        logger.info(f"Training all models for {pair}")
        
        # Get data
        df = self._get_data(pair, use_cache=not retrain)
        df = self.feature_engineer.calculate_all_features(df)
        
        # Train models
        training_results = self.analyzer.train_ml_models(df)
        
        # Store results
        self.model_performance[pair] = training_results
        
        return training_results

    def get_model_explanations(self, pair: str) -> Dict:
        """Get feature importance and model explanations."""
        if pair not in self.model_performance:
            return {"error": "No trained models found for this pair"}
        
        # Get feature importance from ML analyzer
        feature_importance = self.analyzer.ml_analyzer.get_feature_importance()
        
        # Get recent prediction reasons
        recent_predictions = [p for p in self.prediction_history if p.get("pair") == pair]
        latest_reasons = recent_predictions[-1].get("reasons", []) if recent_predictions else []
        
        return {
            "feature_importance": feature_importance,
            "latest_prediction_reasons": latest_reasons,
            "model_performance": self.model_performance.get(pair, {}),
            "prediction_statistics": self.predictor.get_prediction_statistics()
        }

    def analyze_regime_stability(self, pair: str, days: int = 30) -> Dict:
        """Analyze regime stability over time."""
        # Get extended data
        df = self.client.get_forex_data(
            pair=pair,
            timeframe="day", 
            start_date=(datetime.now() - timedelta(days=days+100)).strftime("%Y-%m-%d")
        )
        
        if len(df) < days:
            return {"error": "Insufficient data for regime analysis"}
        
        df = self.feature_engineer.calculate_all_features(df)
        
        # Analyze regime stability
        stability_analysis = self.analyzer.regime_detector.analyze_regime_stability(df, lookback=days)
        
        # Get current regime
        current_regime = self.analyzer.regime_detector.get_regime_adaptive_parameters(df)
        
        return {
            "stability_analysis": stability_analysis,
            "current_regime": current_regime,
            "analysis_period": f"{days} days",
            "data_points": len(df)
        }

    def backtest_strategy(self, pair: str, days: int = 90) -> Dict:
        """Simple backtest of the prediction strategy."""
        logger.info(f"Running backtest for {pair} over {days} days")
        
        # Get historical data
        start_date = (datetime.now() - timedelta(days=days+50)).strftime("%Y-%m-%d")
        df = self.client.get_forex_data(pair=pair, timeframe="day", start_date=start_date)
        
        if len(df) < days + 50:
            return {"error": "Insufficient data for backtesting"}
        
        df = self.feature_engineer.calculate_all_features(df)
        
        # Simulate predictions over time
        predictions = []
        actual_returns = []
        
        # Use walk-forward approach
        for i in range(50, len(df) - 1):  # Start after 50 days for indicators
            # Use data up to day i for prediction
            train_data = df.slice(0, i)
            
            # Analyze with limited data
            analysis = self.analyzer.analyze(train_data)
            prediction = self.predictor.predict_with_regime_adaptation(analysis)
            
            # Get actual return for next day
            current_price = df.slice(i, 1).select("close").item()
            next_price = df.slice(i + 1, 1).select("close").item()
            actual_return = (next_price - current_price) / current_price
            
            predictions.append(prediction)
            actual_returns.append(actual_return)
        
        # Calculate performance metrics
        performance = self._calculate_backtest_performance(predictions, actual_returns)
        
        return {
            "performance": performance,
            "total_predictions": len(predictions),
            "backtest_period": f"{days} days",
            "pair": pair
        }

    def _get_data(self, pair: str, use_cache: bool) -> pl.DataFrame:
        """Get data from cache or API with enhanced error handling."""
        cache_file = f"{DATA_PATH}/{pair.replace(':', '_')}_daily.parquet"

        if use_cache:
            df = load_cached_data(cache_file)
            if df is not None:
                logger.info(f"Using cached data: {len(df)} rows")
                return df

        # Fetch from API with extended history for better analysis
        try:
            df = self.client.get_forex_data(
                pair=pair,
                timeframe="day",
                start_date=(datetime.now() - timedelta(days=500)).strftime("%Y-%m-%d")  # Extended history
            )
            
            logger.info(f"Fetched {len(df)} rows from API")
            
            # Save to cache
            save_data_to_parquet(df, cache_file)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            # Try with shorter history as fallback
            try:
                df = self.client.get_forex_data(
                    pair=pair,
                    timeframe="day",
                    start_date=(datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
                )
                logger.info(f"Fetched {len(df)} rows from API (fallback)")
                save_data_to_parquet(df, cache_file)
                return df
            except Exception as e2:
                logger.error(f"Fallback data fetch also failed: {str(e2)}")
                raise e2

    def _display_enhanced_results(self, prediction: Dict, pair: str):
        """Display enhanced prediction results."""
        print("\n" + "="*80)
        print(f"ENHANCED FOREX PREDICTION: {pair}")
        print("="*80)
        
        # Main prediction
        signal = prediction.get("prediction", "NEUTRAL")
        strength = prediction.get("strength", "MODERATE")
        confidence = prediction.get("confidence", 0.5)
        score = prediction.get("score", 0.5)
        
        print(f"📈 PREDICTION: {signal} ({strength})")
        print(f"🎯 CONFIDENCE: {confidence:.1%}")
        print(f"📊 SCORE: {score:.3f}")
        
        # Regime information
        regime_info = prediction.get("regime_info", {})
        if regime_info:
            print(f"\n🌡️  MARKET REGIME:")
            vol_regime = regime_info.get("volatility_regime", {})
            print(f"   Volatility: {vol_regime.get('regime', 'unknown').upper()} "
                  f"(percentile: {vol_regime.get('percentile', 0):.0f}%)")
            
            trend_regime = regime_info.get("trend_regime", {})
            print(f"   Trend: {trend_regime.get('regime', 'unknown').upper()} "
                  f"(strength: {trend_regime.get('strength', 0):.1%})")
            
            stress_conditions = regime_info.get("stress_conditions", {})
            print(f"   Market Stress: {stress_conditions.get('stress_level', 'unknown').upper()}")
        
        # ML information
        ml_info = prediction.get("model_performance", {})
        if ml_info and ml_info.get("status") == "success":
            print(f"\n🤖 ML MODELS:")
            print(f"   Best Model: {ml_info.get('best_model', 'unknown')}")
            print(f"   Accuracy: {ml_info.get('best_accuracy', 0):.1%}")
            models_trained = ml_info.get("models_trained", [])
            if models_trained:
                print(f"   Models: {', '.join(models_trained)}")
        
        # Dynamic thresholds
        dynamic_thresholds = prediction.get("dynamic_thresholds", {})
        if dynamic_thresholds:
            print(f"\n⚙️  DYNAMIC THRESHOLDS:")
            for key, value in dynamic_thresholds.items():
                if "bullish" in key or "bearish" in key:
                    print(f"   {key.replace('_', ' ').title()}: {value:.3f}")
        
        # Prediction filters
        filters_applied = prediction.get("prediction_filters", [])
        if filters_applied:
            print(f"\n🔍 FILTERS APPLIED: {', '.join(filters_applied)}")
        
        # Key reasons
        reasons = prediction.get("reasons", [])
        if reasons:
            print(f"\n💡 KEY REASONS:")
            for i, reason in enumerate(reasons[:5], 1):  # Show top 5 reasons
                print(f"   {i}. {reason}")
        
        # Technical summary
        data_points = prediction.get("data_points", 0)
        features = prediction.get("features_calculated", 0)
        enhanced_mode = prediction.get("enhanced_mode", False)
        
        print(f"\n📋 TECHNICAL INFO:")
        print(f"   Data Points: {data_points:,}")
        print(f"   Features: {features}")
        print(f"   Enhanced Mode: {'✅' if enhanced_mode else '❌'}")
        print(f"   Timestamp: {prediction.get('timestamp', 'unknown')}")
        
        print("="*80)

    def _calculate_backtest_performance(self, predictions: List[Dict], actual_returns: List[float]) -> Dict:
        """Calculate backtest performance metrics."""
        if not predictions or not actual_returns:
            return {"error": "No predictions or returns data"}
        
        # Extract prediction signals
        signals = []
        confidences = []
        
        for pred in predictions:
            signal = pred.get("prediction", "NEUTRAL")
            confidence = pred.get("confidence", 0.5)
            
            if signal == "BULLISH":
                signals.append(1)
            elif signal == "BEARISH":
                signals.append(-1)
            else:
                signals.append(0)
            
            confidences.append(confidence)
        
        signals = np.array(signals)
        actual_returns = np.array(actual_returns)
        confidences = np.array(confidences)
        
        # Calculate strategy returns (assuming we trade based on signals)
        strategy_returns = signals * actual_returns
        
        # Performance metrics
        total_return = np.sum(strategy_returns)
        hit_rate = np.mean((signals * actual_returns) > 0) if len(signals) > 0 else 0
        
        # Risk metrics
        volatility = np.std(strategy_returns) * np.sqrt(252)  # Annualized
        sharpe_ratio = (np.mean(strategy_returns) * 252) / (volatility + 1e-8)
        
        # Directional accuracy
        correct_direction = np.sum((signals > 0) & (actual_returns > 0)) + np.sum((signals < 0) & (actual_returns < 0))
        total_directional = np.sum(signals != 0)
        directional_accuracy = correct_direction / total_directional if total_directional > 0 else 0
        
        return {
            "total_return": total_return,
            "hit_rate": hit_rate,
            "directional_accuracy": directional_accuracy,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "total_trades": total_directional,
            "avg_confidence": np.mean(confidences)
        }


def main():
    """Enhanced main entry point."""
    parser = argparse.ArgumentParser(description="Enhanced Forex Rate Prediction Tool")
    parser.add_argument("pair", type=str, help="Currency pair (e.g., USD:GBP, EUR:USD)")
    parser.add_argument("--no-cache", action="store_true", help="Force fresh data fetch")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--batch", nargs="+", help="Run batch analysis on multiple pairs")
    parser.add_argument("--backtest", type=int, default=0, help="Run backtest for N days")
    parser.add_argument("--regime-analysis", action="store_true", help="Run regime stability analysis")
    parser.add_argument("--model-explanation", action="store_true", help="Show model explanations")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        app = EnhancedForexPredictionApp()
        
        if args.batch:
            # Batch analysis
            results = app.run_batch_analysis(args.batch, use_cache=not args.no_cache)
            print("\n📊 BATCH ANALYSIS SUMMARY:")
            for pair, result in results.items():
                if "error" not in result:
                    signal = result.get("prediction", "NEUTRAL")
                    confidence = result.get("confidence", 0)
                    print(f"   {pair}: {signal} ({confidence:.1%})")
                else:
                    print(f"   {pair}: ERROR - {result['error']}")
        
        elif args.backtest > 0:
            # Backtest mode
            backtest_results = app.backtest_strategy(args.pair.upper(), args.backtest)
            if "error" not in backtest_results:
                perf = backtest_results["performance"]
                print(f"\n📈 BACKTEST RESULTS ({args.backtest} days):")
                print(f"   Total Return: {perf['total_return']:.2%}")
                print(f"   Hit Rate: {perf['hit_rate']:.1%}")
                print(f"   Directional Accuracy: {perf['directional_accuracy']:.1%}")
                print(f"   Sharpe Ratio: {perf['sharpe_ratio']:.2f}")
                print(f"   Total Trades: {perf['total_trades']}")
            else:
                print(f"Backtest error: {backtest_results['error']}")
        
        elif args.regime_analysis:
            # Regime analysis mode
            regime_results = app.analyze_regime_stability(args.pair.upper())
            if "error" not in regime_results:
                stability = regime_results["stability_analysis"]
                current = regime_results["current_regime"]
                print(f"\n🌡️  REGIME STABILITY ANALYSIS:")
                print(f"   Stability: {stability.get('stability', 'unknown').upper()}")
                print(f"   Stability Score: {stability.get('stability_score', 0):.1%}")
                print(f"   Regime Changes: {stability.get('regime_changes', 0)}")
            else:
                print(f"Regime analysis error: {regime_results['error']}")
        
        elif args.model_explanation:
            # Model explanation mode
            explanations = app.get_model_explanations(args.pair.upper())
            if "error" not in explanations:
                print(f"\n🤖 MODEL EXPLANATIONS:")
                feature_importance = explanations.get("feature_importance", {})
                if feature_importance:
                    print("   Top Features:")
                    for i, (feature, importance) in enumerate(list(feature_importance.items())[:10], 1):
                        print(f"   {i:2d}. {feature}: {importance:.3f}")
            else:
                print(f"Model explanation error: {explanations['error']}")
        
        else:
            # Standard prediction mode
            app.run(args.pair.upper(), use_cache=not args.no_cache)
            
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()