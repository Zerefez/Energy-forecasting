import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging
from typing import Tuple

logger = logging.getLogger('energy_forecasting')

class DataProcessor:
    """Process and prepare data for machine learning models"""
    
    def __init__(self):
        """Initialize the data processor"""
        self.scaler = StandardScaler()
    
    def merge_datasets(self, price_data: pd.DataFrame, weather_data: pd.DataFrame) -> pd.DataFrame:
        """Merge price and weather datasets on timestamp and market"""
        logger.info("Merging price and weather datasets")
        
        # Make copies to avoid modifying the original dataframes
        price_df = price_data.copy()
        weather_df = weather_data.copy()
        
        # Ensure timestamp columns are datetime objects
        if not pd.api.types.is_datetime64_any_dtype(price_df['timestamp']):
            price_df['timestamp'] = pd.to_datetime(price_df['timestamp'])
        if not pd.api.types.is_datetime64_any_dtype(weather_df['timestamp']):
            weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'])
        
        # Normalize timestamps to exact hours (removing microseconds, etc.)
        price_df['timestamp_str'] = price_df['timestamp'].dt.strftime('%Y-%m-%d %H:00:00')
        weather_df['timestamp_str'] = weather_df['timestamp'].dt.strftime('%Y-%m-%d %H:00:00')
        
        price_df['timestamp'] = pd.to_datetime(price_df['timestamp_str'])
        weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp_str'])
        
        # Drop the temporary string columns
        price_df = price_df.drop('timestamp_str', axis=1)
        weather_df = weather_df.drop('timestamp_str', axis=1)
            
        # Check for market column consistency
        if 'market' not in price_df.columns or 'market' not in weather_df.columns:
            logger.error("Market column missing from one or both datasets")
            # Create a default if missing
            if 'market' not in price_df.columns:
                price_df['market'] = 'unknown'
            if 'market' not in weather_df.columns:
                weather_df['market'] = 'unknown'
        
        # Perform the merge
        merged_data = pd.merge(price_df, weather_df, on=['timestamp', 'market'], how='inner')
        
        # Log merge results
        logger.info(f"Merged dataset contains {len(merged_data)} records")
        
        # If merged data is empty, provide debugging information and try alternative approach
        if merged_data.empty and not price_df.empty and not weather_df.empty:
            logger.warning("Merge resulted in empty dataset despite having input data")
            logger.info(f"Price data has {len(price_df)} rows with timestamp range: {price_df['timestamp'].min()} to {price_df['timestamp'].max()}")
            logger.info(f"Weather data has {len(weather_df)} rows with timestamp range: {weather_df['timestamp'].min()} to {weather_df['timestamp'].max()}")
            
            # Try a different approach using string representation of timestamps
            logger.info("Attempting merge using string representation of timestamps...")
            
            # Convert timestamps to string representation (year-month-day hour)
            price_df['ts_key'] = price_df['timestamp'].dt.strftime('%Y-%m-%d %H')
            weather_df['ts_key'] = weather_df['timestamp'].dt.strftime('%Y-%m-%d %H')
            
            # Merge on string timestamp and market
            merged_data = pd.merge(price_df, weather_df, on=['ts_key', 'market'], how='inner')
            
            if not merged_data.empty:
                logger.info(f"Alternative merge successful, got {len(merged_data)} records")
                
                # Keep only one timestamp column
                merged_data = merged_data.drop('timestamp_y', axis=1)
                merged_data = merged_data.rename(columns={'timestamp_x': 'timestamp'})
                merged_data = merged_data.drop('ts_key', axis=1)
            else:
                logger.error("All merge attempts failed. Check data alignment carefully.")
                
                # Check for any matching timestamps and markets
                price_keys = set(zip(price_df['ts_key'], price_df['market']))
                weather_keys = set(zip(weather_df['ts_key'], weather_df['market']))
                common_keys = price_keys.intersection(weather_keys)
                logger.info(f"Found {len(common_keys)} common timestamp-market combinations")
                
                if len(common_keys) > 0:
                    logger.info("Some common keys exist but merge failed. Sample common keys:")
                    for i, (ts, mkt) in enumerate(list(common_keys)[:5]):
                        logger.info(f"{i+1}: {ts}, {mkt}")
        
        return merged_data
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional features for the model"""
        logger.info("Creating additional features")
        
        # Extract time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        
        # Create lagged features (previous price points)
        for lag in [1, 3, 12, 24]:
            df[f'price_lag_{lag}'] = df.groupby('market')['price'].shift(lag)
        
        # Create rolling window features
        for window in [6, 12, 24]:
            df[f'price_rolling_mean_{window}'] = df.groupby('market')['price'].rolling(window=window).mean().reset_index(0, drop=True)
            df[f'price_rolling_std_{window}'] = df.groupby('market')['price'].rolling(window=window).std().reset_index(0, drop=True)
        
        # Demand/supply indicators
        df['temp_squared'] = df['temperature'] ** 2  # Non-linear relationship with energy use
        
        # Interaction features
        df['wind_solar_interaction'] = df['wind_speed'] * df['solar_irradiance']
        
        # Drop rows with NaN values resulting from lag features
        df = df.dropna()
        
        logger.info(f"Created features, dataset now has {df.shape[1]} columns and {len(df)} rows")
        return df
    
    def prepare_model_data(self, df: pd.DataFrame, target_col: str = 'price', forecast_horizon: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for model training with a specified forecast horizon"""
        logger.info(f"Preparing model data with {forecast_horizon} hour forecast horizon")
        
        # Check if dataframe is empty
        if df.empty:
            logger.error("Cannot prepare model data: Input dataframe is empty")
            return np.array([]), np.array([])
        
        # Create target: future price
        df['target'] = df.groupby('market')[target_col].shift(-forecast_horizon)
        
        # Drop rows with NaN targets
        df = df.dropna(subset=['target'])
        
        # Check if dataframe is still valid after removing NaNs
        if df.empty:
            logger.error("No valid rows remain after removing NaN values")
            return np.array([]), np.array([])
        
        # Select features and target
        feature_cols = [col for col in df.columns if col not in ['timestamp', 'market', 'price', 'target']]
        X = df[feature_cols].copy()
        y = df['target'].values
        
        # Scale numerical features
        numerical_cols = [col for col in X.columns if X[col].dtype in [np.float64, np.int64]]
        
        # Only apply scaling if we have data
        if not X.empty and len(numerical_cols) > 0:
            X[numerical_cols] = self.scaler.fit_transform(X[numerical_cols])
        
        logger.info(f"Model data prepared with {len(feature_cols)} features and {len(X)} samples")
        return X, y 