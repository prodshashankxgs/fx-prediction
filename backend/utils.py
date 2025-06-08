# Libraries
import polars as pl
import numpy as np
from typing import Dict, List
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def setup_logger(name: str) -> logging.Logger:
    """Set up logger for a module."""
    return logging.getLogger(name)


def validate_dataframe(df: pl.DataFrame, required_columns: List[str]) -> bool:
    """Validate that DataFrame contains required columns."""
    return all(col in df.columns for col in required_columns)


def calculate_performance_metrics(predictions: List[Dict], actual_results: List[float]) -> Dict:
    """Calculate performance metrics for predictions."""
    if len(predictions) != len(actual_results):
        raise ValueError("Predictions and results must have same length")

    correct = 0
    total = len(predictions)

    for pred, actual in zip(predictions, actual_results):
        pred_direction = 1 if pred["prediction"] == "BULLISH" else -1
        actual_direction = 1 if actual > 0 else -1

        if pred_direction == actual_direction:
            correct += 1

    accuracy = correct / total if total > 0 else 0

    return {
        "accuracy": accuracy,
        "total_predictions": total,
        "correct_predictions": correct
    }


def format_prediction_output(prediction: Dict, pair: str) -> str:
    """Format prediction results for display."""
    output = f"\n{'=' * 50}\n"
    output += f"FOREX PREDICTION REPORT\n"
    output += f"Currency Pair: {pair}\n"
    output += f"Timestamp: {prediction['timestamp']}\n"
    output += f"{'=' * 50}\n\n"

    output += f"PREDICTION: {prediction['prediction']}\n"
    output += f"Strength: {prediction['strength']}\n"
    output += f"Confidence: {prediction['confidence']:.1%}\n"
    output += f"Composite Score: {prediction['score']:.3f}\n\n"

    output += "Key Reasons:\n"
    for i, reason in enumerate(prediction['reasons'], 1):
        output += f"  {i}. {reason}\n"

    output += f"\n{'=' * 50}\n"

    return output