# Libraries
import argparse
import logging
import sys
import polars as pl

# Import modules
from backend.config import *
from backend.data_collector import PolygonClient, save_data_to_parquet, load_cached_data
from backend.features import FeatureEngineer
from backend.analysis import MarketAnalyzer
from backend.predictor import ForexPredictor
from backend.utils import setup_logger, format_prediction_output

logger = setup_logger("main")


class ForexPredictionApp:
    """Main application class."""

    def __init__(self):
        self.client = PolygonClient(POLYGON_API_KEY)
        self.feature_engineer = FeatureEngineer(TECHNICAL_INDICATORS)
        self.analyzer = MarketAnalyzer()
        self.predictor = ForexPredictor(PREDICTION_THRESHOLDS)

    def run(self, pair: str, use_cache: bool = True):
        """Run the prediction pipeline."""
        logger.info(f"Starting prediction for {pair}")

        # Validate currency pair
        if not self.client.validate_currency_pair(pair):
            raise ValueError(f"Unsupported currency pair: {pair}")

        # Get data
        logger.info("Fetching market data...")
        df = self._get_data(pair, use_cache)

        # Calculate features
        logger.info("Calculating technical indicators...")
        df = self.feature_engineer.calculate_all_features(df)

        # Run analysis
        logger.info("Analyzing market conditions...")
        analysis_results = self.analyzer.analyze(df)

        # Generate prediction
        logger.info("Generating prediction...")
        prediction = self.predictor.predict(analysis_results)

        # Display results
        print(format_prediction_output(prediction, pair))

        return prediction

    def _get_data(self, pair: str, use_cache: bool) -> pl.DataFrame:
        """Get data from cache or API."""
        cache_file = f"{DATA_PATH}/{pair.replace(':', '_')}_daily.parquet"

        if use_cache:
            df = load_cached_data(cache_file)
            if df is not None:
                logger.info("Using cached data")
                return df

        # Fetch from API
        df = self.client.get_forex_data(
            pair=pair,
            timeframe="day",
            start_date=(datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        )

        # Save to cache
        save_data_to_parquet(df, cache_file)

        return df


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Forex Rate Prediction Tool")
    parser.add_argument(
        "pair",
        type=str,
        help="Currency pair (e.g., USD:GBP, EUR:USD)"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force fresh data fetch"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        app = ForexPredictionApp()
        app.run(args.pair.upper(), use_cache=not args.no_cache)
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)


def train_ml_model(self, pair: str) -> dict:
    """Train ML model for enhanced predictions."""
    logger.info(f"Training ML model for {pair}")

    # Get data
    df = self._get_data(pair, use_cache=True)

    # Calculate features
    df = self.feature_engineer.calculate_all_features(df)

    # Train ML model
    ml_results = self.analyzer.ml_analyzer.train(df)

    return ml_results


if __name__ == "__main__":
    main()