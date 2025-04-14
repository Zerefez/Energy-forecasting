import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import os
import logging

logger = logging.getLogger('energy_forecasting')

class ModelTrainer:
    """Train and evaluate machine learning models"""
    
    def __init__(self, model_type: str = 'random_forest'):
        """Initialize with specified model type"""
        self.model_type = model_type
        self.model = self._get_model()
        self.feature_importance = None
    
    def _get_model(self):
        """Get model based on type"""
        if self.model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                random_state=42
            )
        else:
            raise ValueError(f"Model type {self.model_type} not supported")
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model on the provided data"""
        logger.info(f"Training {self.model_type} model on {X.shape[0]} samples")
        
        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Evaluate on validation set
        val_predictions = self.model.predict(X_val)
        mae = mean_absolute_error(y_val, val_predictions)
        rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
        r2 = r2_score(y_val, val_predictions)
        
        logger.info(f"Model validation results: MAE = {mae:.2f}, RMSE = {rmse:.2f}, R² = {r2:.4f}")
        
        # Store feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
    
    def save_model(self, path: str = "models/") -> None:
        """Save the trained model to disk"""
        os.makedirs(path, exist_ok=True)
        model_path = os.path.join(path, f"{self.model_type}_model.joblib")
        joblib.dump(self.model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save feature importance if available
        if self.feature_importance is not None:
            importance_path = os.path.join(path, "feature_importance.csv")
            self.feature_importance.to_csv(importance_path, index=False)
            logger.info(f"Feature importance saved to {importance_path}")
    
    def load_model(self, path: str) -> None:
        """Load a saved model"""
        self.model = joblib.load(path)
        logger.info(f"Model loaded from {path}") 