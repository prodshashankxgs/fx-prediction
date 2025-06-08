# Libraries
import numpy as np
import polars as pl
from typing import Dict, List, Tuple, Optional
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import warnings

warnings.filterwarnings('ignore')


class MLAnalyzer:
    """ML analyzer that provides signals similar to technical indicators."""

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_columns = []

    def prepare_features(self, df: pl.DataFrame) -> Optional[np.ndarray]:
        """Extract features that already exist from technical analysis."""

        # Use only features that are already calculated by FeatureEngineer
        available_features = []

        # Core technical features
        technical_features = [
            'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'bb_position', 'atr',
            'volatility_20d', 'volatility_50d',
            'momentum_20d', 'momentum_50d',
            'price_to_sma20', 'price_to_sma50',
            'return_1d', 'return_5d', 'return_20d',
            'skew_20d', 'skew_50d'
        ]

        # Only use features that exist in the dataframe
        for feat in technical_features:
            if feat in df.columns:
                available_features.append(feat)

        if len(available_features) < 5:  # Need minimum features
            return None

        self.feature_columns = available_features

        # Get clean data
        df_clean = df.select(available_features).drop_nulls()

        if len(df_clean) < 50:  # Need minimum samples
            return None

        return df_clean.to_numpy()

    def train(self, df: pl.DataFrame) -> Dict:
        """Train a lightweight ML model on historical data."""

        # Prepare features
        X = self.prepare_features(df)
        if X is None:
            return {"error": "Insufficient data for ML training"}

        # Create target: 1 if next day return is positive
        returns = df.select('return_1d').drop_nulls().to_numpy().flatten()
        if len(returns) < len(X) + 1:
            return {"error": "Insufficient return data"}

        # Align features with next-day returns
        y = (returns[1:len(X) + 1] > 0).astype(int)
        X = X[:-1]  # Remove last row to align with targets

        # Use RandomForest - no OpenMP required
        self.model = RandomForestClassifier(
            n_estimators=50,
            max_depth=4,
            min_samples_split=10,
            random_state=42,
            n_jobs=1  # Single thread to avoid any issues
        )

        # Train with time series split validation
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)

            # Train and evaluate
            self.model.fit(X_train_scaled, y_train)
            score = self.model.score(X_val_scaled, y_val)
            scores.append(score)

        # Final training on all data
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True

        return {
            "status": "success",
            "validation_accuracy": np.mean(scores),
            "feature_count": len(self.feature_columns),
            "model_type": "RandomForest"
        }

    def analyze(self, df: pl.DataFrame) -> Dict:
        """Analyze current market conditions using ML, returning results similar to other analyzers."""

        if not self.is_trained:
            return {
                "ml_available": False,
                "signal": 0,
                "confidence": 0.5,
                "prediction_proba": 0.5
            }

        # Get latest features
        X = self.prepare_features(df)
        if X is None or len(X) == 0:
            return {
                "ml_available": False,
                "signal": 0,
                "confidence": 0.5,
                "prediction_proba": 0.5
            }

        # Use only the latest data point
        X_latest = X[-1:]
        X_scaled = self.scaler.transform(X_latest)

        # Get prediction probability
        proba = self.model.predict_proba(X_scaled)[0]
        bullish_proba = proba[1]  # Probability of positive return

        # Convert to signal (-1 to 1) like other indicators
        if bullish_proba > 0.6:
            signal = 1
        elif bullish_proba < 0.4:
            signal = -1
        else:
            signal = 0

        # Calculate confidence (how far from 0.5)
        confidence = abs(bullish_proba - 0.5) * 2

        # Get feature importance for the most important features
        feature_importance = {}
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            for feat, imp in zip(self.feature_columns, importances):
                if imp > 0.05:  # Only significant features
                    feature_importance[feat] = float(imp)

        return {
            "ml_available": True,
            "signal": signal,
            "confidence": confidence,
            "prediction_proba": bullish_proba,
            "feature_importance": feature_importance,
            "features_used": len(self.feature_columns)
        }

    def save(self, filepath: str):
        """Save the trained model."""
        if self.is_trained:
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'is_trained': self.is_trained
            }, filepath)

    def load(self, filepath: str):
        """Load a trained model."""
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_columns = data['feature_columns']
        self.is_trained = data['is_trained']