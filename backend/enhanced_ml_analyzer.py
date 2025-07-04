# Libraries
import numpy as np
import polars as pl
from typing import Dict, List, Tuple, Optional, Any
import joblib
import warnings
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline

# Try to import XGBoost and LightGBM, fallback if not available
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("LightGBM not available. Install with: pip install lightgbm")

warnings.filterwarnings('ignore')


class EnhancedMLAnalyzer:
    """Enhanced ML analyzer with ensemble models and proper time series validation."""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.models = {}
        self.ensemble_model = None
        self.scaler = RobustScaler()  # More robust to outliers
        self.feature_selector = None
        self.is_trained = False
        self.feature_columns = []
        self.feature_importance = {}
        self.validation_scores = {}
        self.regime_models = {}  # Separate models for different market regimes
        
    def prepare_enhanced_features(self, df: pl.DataFrame) -> Optional[np.ndarray]:
        """Extract comprehensive features for ML training."""
        
        # Core technical features
        core_features = [
            'rsi', 'macd', 'macd_signal', 'bb_position', 'atr',
            'stoch_k', 'stoch_d', 'williams_r', 'cci'
        ]
        
        # Advanced indicators
        advanced_features = [
            'rvi', 'cmo', 'efficiency_ratio', 'price_spread',
            'order_flow_ratio', 'liquidity_proxy'
        ]
        
        # Statistical features
        statistical_features = [
            'volatility_20d', 'volatility_50d', 'skew_20d', 'skew_50d',
            'momentum_20d', 'momentum_50d'
        ]
        
        # Time-based features
        time_features = [
            'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos',
            'short_window_return', 'medium_window_return', 'long_window_return'
        ]
        
        # Lag features
        lag_features = []
        for lag in [1, 2, 3, 5]:
            lag_features.extend([
                f'return_1d_lag_{lag}', f'rsi_lag_{lag}', f'volatility_lag_{lag}'
            ])
        
        # Interaction features
        interaction_features = [
            'rsi_vol_interaction', 'macd_bb_interaction', 'price_volume_interaction',
            'sma_rsi_interaction', 'vol_momentum_interaction', 'short_long_momentum'
        ]
        
        # Regime features
        regime_features = [
            'vol_zscore', 'trend_score', 'momentum_zscore', 'stress_indicator',
            'vol_regime', 'trend_regime'
        ]
        
        # Combine all features
        all_features = (core_features + advanced_features + statistical_features + 
                       time_features + lag_features + interaction_features + regime_features)
        
        # Check available features
        available_features = [feat for feat in all_features if feat in df.columns]
        
        if len(available_features) < 10:
            print(f"Warning: Only {len(available_features)} features available")
            return None
        
        self.feature_columns = available_features
        print(f"Using {len(available_features)} features for ML training")
        
        # Get clean data
        try:
            df_clean = df.select(available_features).drop_nulls()
            if len(df_clean) < 50:
                print("Warning: Insufficient clean data after null removal")
                return None
            return df_clean.to_numpy()
        except Exception as e:
            print(f"Error preparing features: {e}")
            return None

    def create_enhanced_targets(self, df: pl.DataFrame) -> Optional[np.ndarray]:
        """Create multi-class targets based on return magnitude and regime."""
        try:
            returns = df.select('return_1d').drop_nulls().to_numpy().flatten()
            
            if len(returns) < 50:
                return None
            
            # Dynamic thresholds based on volatility
            vol_data = df.select('volatility_20d').drop_nulls().to_numpy().flatten()
            if len(vol_data) > 0:
                vol_percentile = np.percentile(vol_data, 75)
                dynamic_threshold = max(0.002, vol_percentile * 0.5)
            else:
                dynamic_threshold = 0.002
            
            # Multi-class classification
            thresholds = {
                'strong_bull': dynamic_threshold * 2,
                'bull': dynamic_threshold,
                'neutral': dynamic_threshold * 0.5,
                'bear': -dynamic_threshold,
                'strong_bear': -dynamic_threshold * 2
            }
            
            # Create targets: 0=strong_bear, 1=bear, 2=neutral, 3=bull, 4=strong_bull
            targets = np.where(returns > thresholds['strong_bull'], 4,
                      np.where(returns > thresholds['bull'], 3,
                      np.where(returns > thresholds['neutral'], 2,
                      np.where(returns < thresholds['strong_bear'], 0,
                      np.where(returns < thresholds['bear'], 1, 2)))))
            
            print(f"Target distribution: {np.bincount(targets)}")
            return targets
            
        except Exception as e:
            print(f"Error creating targets: {e}")
            return None

    def create_ensemble_models(self) -> Dict:
        """Create ensemble of different ML models."""
        models = {}
        
        # Random Forest (always available)
        models['rf'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        
        # XGBoost (if available)
        if HAS_XGB:
            models['xgb'] = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='mlogloss'
            )
        
        # LightGBM (if available)
        if HAS_LGB:
            models['lgb'] = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1
            )
        
        return models

    def time_series_cross_validate(self, X: np.ndarray, y: np.ndarray, 
                                  model: Any, n_splits: int = 5) -> Dict:
        """Perform time series cross-validation."""
        tscv = TimeSeriesSplit(n_splits=n_splits, gap=5)
        
        scores = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Scale features
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Train model
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_val_scaled)
            
            # Calculate metrics
            scores['accuracy'].append(accuracy_score(y_val, y_pred))
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_val, y_pred, average='weighted', zero_division=0
            )
            scores['precision'].append(precision)
            scores['recall'].append(recall)
            scores['f1'].append(f1)
        
        # Return mean scores
        return {k: np.mean(v) for k, v in scores.items()}

    def train(self, df: pl.DataFrame) -> Dict:
        """Train ensemble ML models with proper validation."""
        
        # Prepare features and targets
        X = self.prepare_enhanced_features(df)
        if X is None:
            return {"error": "Failed to prepare features", "status": "failed"}
        
        y = self.create_enhanced_targets(df)
        if y is None:
            return {"error": "Failed to create targets", "status": "failed"}
        
        # Align data
        min_len = min(len(X) - 1, len(y) - 1)
        if min_len < 100:
            return {"error": "Insufficient data for training", "status": "failed"}
        
        X_train = X[:min_len]
        y_future = y[1:min_len + 1]
        
        print(f"Training with {len(X_train)} samples, {X_train.shape[1]} features")
        
        try:
            # Feature selection
            selector = SelectKBest(score_func=mutual_info_classif, k=min(30, X_train.shape[1]))
            X_selected = selector.fit_transform(X_train, y_future)
            selected_features = [self.feature_columns[i] for i in selector.get_support(indices=True)]
            self.feature_selector = selector
            
            print(f"Selected {len(selected_features)} most informative features")
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X_selected)
            
            # Create and train individual models
            models = self.create_ensemble_models()
            model_scores = {}
            
            for name, model in models.items():
                print(f"Training {name} model...")
                
                # Time series cross-validation
                cv_scores = self.time_series_cross_validate(X_scaled, y_future, model)
                model_scores[name] = cv_scores
                
                # Train on full data
                model.fit(X_scaled, y_future)
                self.models[name] = model
                
                # Feature importance
                if hasattr(model, 'feature_importances_'):
                    importance = model.feature_importances_
                    self.feature_importance[name] = dict(zip(selected_features, importance))
            
            # Create voting ensemble if multiple models available
            if len(models) > 1:
                voting_models = [(name, model) for name, model in models.items()]
                self.ensemble_model = VotingClassifier(
                    estimators=voting_models,
                    voting='soft'
                )
                self.ensemble_model.fit(X_scaled, y_future)
                
                # Ensemble cross-validation
                ensemble_scores = self.time_series_cross_validate(X_scaled, y_future, self.ensemble_model)
                model_scores['ensemble'] = ensemble_scores
            
            self.validation_scores = model_scores
            self.is_trained = True
            
            # Return training results
            best_model = max(model_scores.items(), key=lambda x: x[1]['accuracy'])
            
            return {
                "status": "success",
                "best_model": best_model[0],
                "best_accuracy": best_model[1]['accuracy'],
                "model_scores": model_scores,
                "feature_count": len(selected_features),
                "models_trained": list(models.keys())
            }
            
        except Exception as e:
            print(f"Training error: {e}")
            return {"error": f"Training failed: {str(e)}", "status": "failed"}

    def analyze_with_regime(self, df: pl.DataFrame) -> Dict:
        """Analyze using ensemble models with regime awareness."""
        
        if not self.is_trained:
            return {
                "ml_available": False,
                "signal": 0,
                "confidence": 0.5,
                "prediction_proba": 0.5
            }
        
        try:
            # Prepare features
            X = self.prepare_enhanced_features(df)
            if X is None or len(X) == 0:
                return {
                    "ml_available": False,
                    "signal": 0,
                    "confidence": 0.5,
                    "prediction_proba": 0.5
                }
            
            # Use latest data point
            X_latest = X[-1:].reshape(1, -1)
            
            # Apply feature selection
            if self.feature_selector:
                X_latest = self.feature_selector.transform(X_latest)
            
            # Scale features
            X_scaled = self.scaler.transform(X_latest)
            
            # Get predictions from all models
            predictions = {}
            probabilities = {}
            
            for name, model in self.models.items():
                pred = model.predict(X_scaled)[0]
                proba = model.predict_proba(X_scaled)[0]
                predictions[name] = pred
                probabilities[name] = proba
            
            # Ensemble prediction if available
            if self.ensemble_model:
                ensemble_pred = self.ensemble_model.predict(X_scaled)[0]
                ensemble_proba = self.ensemble_model.predict_proba(X_scaled)[0]
                predictions['ensemble'] = ensemble_pred
                probabilities['ensemble'] = ensemble_proba
            
            # Use best performing model or ensemble
            best_model = max(self.validation_scores.items(), key=lambda x: x[1]['accuracy'])[0]
            if best_model in predictions:
                final_prediction = predictions[best_model]
                final_proba = probabilities[best_model]
            else:
                final_prediction = predictions[list(predictions.keys())[0]]
                final_proba = probabilities[list(probabilities.keys())[0]]
            
            # Convert to signal (-1, 0, 1)
            if final_prediction >= 3:  # Bull or strong bull
                signal = 1
            elif final_prediction <= 1:  # Bear or strong bear
                signal = -1
            else:  # Neutral
                signal = 0
            
            # Calculate confidence
            confidence = np.max(final_proba)
            
            # Bullish probability (classes 3 and 4)
            bullish_proba = np.sum(final_proba[3:]) if len(final_proba) > 3 else final_proba[-1]
            
            # Model agreement
            agreement = self._calculate_model_agreement(predictions)
            
            return {
                "ml_available": True,
                "signal": signal,
                "confidence": confidence,
                "prediction_proba": bullish_proba,
                "prediction_class": self._get_class_name(final_prediction),
                "class_probabilities": self._format_class_probabilities(final_proba),
                "model_predictions": predictions,
                "ensemble_agreement": agreement,
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

    def _calculate_model_agreement(self, predictions: Dict) -> float:
        """Calculate agreement between different models."""
        if len(predictions) < 2:
            return 1.0
        
        pred_values = list(predictions.values())
        # Calculate standard deviation of predictions (lower = more agreement)
        agreement = 1.0 - (np.std(pred_values) / 2.0)  # Normalize by max possible std
        return max(0.0, min(1.0, agreement))

    def _get_class_name(self, prediction: int) -> str:
        """Convert prediction class to readable name."""
        class_names = {
            0: "strong_bearish",
            1: "bearish", 
            2: "neutral",
            3: "bullish",
            4: "strong_bullish"
        }
        return class_names.get(prediction, "neutral")

    def _format_class_probabilities(self, probabilities: np.ndarray) -> Dict:
        """Format class probabilities for display."""
        class_names = ["strong_bearish", "bearish", "neutral", "bullish", "strong_bullish"]
        return {class_names[i]: prob for i, prob in enumerate(probabilities) if i < len(class_names)}

    def get_feature_importance(self, model_name: str = None) -> Dict:
        """Get feature importance from trained models."""
        if not self.is_trained or not self.feature_importance:
            return {}
        
        if model_name and model_name in self.feature_importance:
            return self.feature_importance[model_name]
        
        # Return average importance across all models
        all_features = set()
        for importance_dict in self.feature_importance.values():
            all_features.update(importance_dict.keys())
        
        avg_importance = {}
        for feature in all_features:
            importances = [
                importance_dict.get(feature, 0) 
                for importance_dict in self.feature_importance.values()
            ]
            avg_importance[feature] = np.mean(importances)
        
        # Sort by importance
        return dict(sorted(avg_importance.items(), key=lambda x: x[1], reverse=True))

    def save(self, filepath: str):
        """Save the trained models and components."""
        if self.is_trained:
            try:
                save_data = {
                    'models': self.models,
                    'ensemble_model': self.ensemble_model,
                    'scaler': self.scaler,
                    'feature_selector': self.feature_selector,
                    'feature_columns': self.feature_columns,
                    'feature_importance': self.feature_importance,
                    'validation_scores': self.validation_scores,
                    'is_trained': self.is_trained
                }
                joblib.dump(save_data, filepath)
                print(f"Enhanced ML models saved to {filepath}")
            except Exception as e:
                print(f"Error saving models: {e}")

    def load(self, filepath: str):
        """Load saved models and components."""
        try:
            save_data = joblib.load(filepath)
            self.models = save_data['models']
            self.ensemble_model = save_data['ensemble_model']
            self.scaler = save_data['scaler']
            self.feature_selector = save_data['feature_selector']
            self.feature_columns = save_data['feature_columns']
            self.feature_importance = save_data['feature_importance']
            self.validation_scores = save_data['validation_scores']
            self.is_trained = save_data['is_trained']
            print(f"Enhanced ML models loaded from {filepath}")
        except Exception as e:
            print(f"Error loading models: {e}")
            self.is_trained = False

    def analyze(self, df: pl.DataFrame) -> Dict:
        """Main analysis method for compatibility."""
        return self.analyze_with_regime(df)