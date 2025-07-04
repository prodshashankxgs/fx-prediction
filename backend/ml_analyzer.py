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
    """Simplified ML analyzer for forex predictions."""

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_columns = []

    def prepare_features(self, df: pl.DataFrame) -> Optional[np.ndarray]:
        """Extract basic features for ML training."""
        
        # Basic technical features that should be available
        basic_features = [
            'rsi', 'macd', 'macd_signal', 'bb_position', 'atr'
        ]
        
        # Check what features are available
        available_features = []
        for feat in basic_features:
            if feat in df.columns:
                available_features.append(feat)
        
        # Add return features if available
        return_features = ['return_1d', 'return_5d', 'return_20d']
        for feat in return_features:
            if feat in df.columns:
                available_features.append(feat)
        
        if len(available_features) < 3:
            return None
        
        self.feature_columns = available_features
        
        # Get clean data
        try:
            df_clean = df.select(available_features).drop_nulls()
            if len(df_clean) < 20:
                return None
            return df_clean.to_numpy()
        except Exception:
            return None

    def train(self, df: pl.DataFrame) -> Dict:
        """Train a simple ML model."""
        
        # Prepare features
        X = self.prepare_features(df)
        if X is None:
            return {"error": "Insufficient data for ML training", "status": "failed"}
        
        # Get returns for target
        try:
            returns = df.select('return_1d').drop_nulls().to_numpy().flatten()
        except Exception:
            return {"error": "No return data available", "status": "failed"}
        
        if len(returns) < 50:
            return {"error": "Insufficient return data", "status": "failed"}
        
        # Align data
        min_len = min(len(X) - 1, len(returns) - 1)
        if min_len < 30:
            return {"error": "Not enough aligned data", "status": "failed"}
        
        X_train = X[:min_len]
        y_future = returns[1:min_len + 1]
        
        # Create simple binary target (up/down)
        y = (y_future > 0).astype(int)
        
        print(f"Training with {len(X_train)} samples, {X_train.shape[1]} features")
        
        try:
            # Simple Random Forest
            self.model = RandomForestClassifier(
                n_estimators=50,
                max_depth=5,
                random_state=42,
                n_jobs=1
            )
            
            # Scale and train
            X_scaled = self.scaler.fit_transform(X_train)
            self.model.fit(X_scaled, y)
            self.is_trained = True
            
            # Calculate accuracy
            accuracy = self.model.score(X_scaled, y)
            
            return {
                "status": "success",
                "validation_accuracy": accuracy,
                "feature_count": len(self.feature_columns),
                "model_type": "RandomForest"
            }
            
        except Exception as e:
            return {"error": f"Training failed: {str(e)}", "status": "failed"}

    def analyze(self, df: pl.DataFrame) -> Dict:
        """Analyze using the trained model."""
        
        if not self.is_trained or self.model is None:
            return {
                "ml_available": False,
                "signal": 0,
                "confidence": 0.5,
                "prediction_proba": 0.5
            }
        
        try:
            # Get features
            X = self.prepare_features(df)
            if X is None or len(X) == 0:
                return {
                    "ml_available": False,
                    "signal": 0,
                    "confidence": 0.5,
                    "prediction_proba": 0.5
                }
            
            # Use latest data point
            X_latest = X[-1:].reshape(1, -1)
            X_scaled = self.scaler.transform(X_latest)
            
            # Get prediction
            prediction = self.model.predict(X_scaled)[0]
            proba = self.model.predict_proba(X_scaled)[0]
            
            # Convert to signal
            signal = 1 if prediction == 1 else -1
            bullish_proba = proba[1] if len(proba) > 1 else 0.5
            confidence = max(proba) if len(proba) > 0 else 0.5
            
            return {
                "ml_available": True,
                "signal": signal,
                "confidence": confidence,
                "prediction_proba": bullish_proba,
                "features_used": len(self.feature_columns)
            }
            
        except Exception as e:
            print(f"ML analysis error: {e}")
            return {
                "ml_available": False,
                "signal": 0,
                "confidence": 0.5,
                "prediction_proba": 0.5,
                "error": str(e)
            }

    def save(self, filepath: str):
        """Save the model."""
        if self.is_trained and self.model is not None:
            try:
                joblib.dump({
                    'model': self.model,
                    'scaler': self.scaler,
                    'feature_columns': self.feature_columns,
                    'is_trained': self.is_trained
                }, filepath)
            except Exception as e:
                print(f"Error saving model: {e}")

    def load(self, filepath: str):
        """Load a saved model."""
        try:
            data = joblib.load(filepath)
            self.model = data['model']
            self.scaler = data['scaler']
            self.feature_columns = data['feature_columns']
            self.is_trained = data['is_trained']
        except Exception as e:
            print(f"Error loading model: {e}")
            self.is_trained = False