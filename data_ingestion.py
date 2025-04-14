import pandas as pd
import numpy as np
import os
import logging
import yaml
from datetime import datetime, timedelta
from typing import Dict

logger = logging.getLogger('energy_forecasting')

class DataIngestion:
    """Handle data acquisition from energy market APIs"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize with configuration"""
        self.config = self._load_config(config_path)
        self.api_key = os.environ.get("ENERGY_API_KEY", "demo_key")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using default configuration.")
            return {
                "data_sources": {
                    "entsoe": "https://transparency.entsoe.eu/api",
                    "nordpool": "https://www.nordpoolgroup.com/api/marketdata"
                },
                "markets": ["DE", "FR", "NL", "DK"],
                "lookback_days": 30
            }
    
    def fetch_price_data(self, market: str = "DE") -> pd.DataFrame:
        """
        Fetch market price data for a specific European power market
        Note: In a real project, you'd use actual API endpoints that require authentication
        For this demo, we'll simulate data
        """
        logger.info(f"Fetching price data for market: {market}")
        
        # In a real implementation, you would use:
        # response = requests.get(
        #     f"{self.config['data_sources']['entsoe']}/actual_url_endpoint",
        #     params={"market": market, "api_key": self.api_key}
        # )
        # data = response.json()
        
        # For demonstration, generate simulated price data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.config["lookback_days"])
        
        # Generate dates at hourly intervals
        date_range = pd.date_range(start=start_date, end=end_date, freq='h')
        
        # Create synthetic price data with realistic patterns
        np.random.seed(42)  # For reproducibility
        
        # Base price depending on market
        base_prices = {"DE": 45, "FR": 50, "NL": 48, "DK": 42}
        base_price = base_prices.get(market, 45)
        
        # Generate synthetic data with daily and weekly patterns plus some noise
        hours = np.array([d.hour for d in date_range])
        weekdays = np.array([d.weekday() for d in date_range])
        
        # Daily pattern: higher during day, lower at night
        daily_pattern = 15 * np.sin(np.pi * hours / 12 - 6)
        
        # Weekly pattern: lower on weekends
        weekly_pattern = -5 * np.where(weekdays >= 5, 1, 0)
        
        # Random noise
        noise = np.random.normal(0, 5, len(date_range))
        
        # Combined price
        prices = base_price + daily_pattern + weekly_pattern + noise
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': date_range,
            'price': prices,
            'market': market
        })
        
        # Add volume data (simulated)
        df['volume'] = base_price * 10 + 500 * np.sin(np.pi * hours / 12) + np.random.normal(0, 200, len(date_range))
        
        # Ensure timestamp is truncated to hour precision
        df['timestamp'] = df['timestamp'].dt.floor('h')
        
        logger.info(f"Successfully retrieved {len(df)} records for market {market}")
        return df
    
    def fetch_weather_data(self, market: str = "DE") -> pd.DataFrame:
        """Fetch weather data that might impact energy production/consumption"""
        logger.info(f"Fetching weather data for market: {market}")
        
        # Similar to price data, we'll simulate this
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.config["lookback_days"])
        date_range = pd.date_range(start=start_date, end=end_date, freq='h')
        
        # Temperature data with realistic patterns
        hours = np.array([d.hour for d in date_range])
        days = np.array([(d - start_date).days for d in date_range])
        
        # Base temperature with seasonal component
        base_temp = {
            "DE": 15 - 5 * np.sin(np.pi * days / 180),  # Germany
            "FR": 18 - 6 * np.sin(np.pi * days / 180),  # France
            "NL": 14 - 5 * np.sin(np.pi * days / 180),  # Netherlands
            "DK": 12 - 6 * np.sin(np.pi * days / 180)   # Denmark
        }.get(market, 15)
        
        # Daily temperature pattern
        daily_pattern = 5 * np.sin(np.pi * hours / 12 - 3)
        
        # Random weather fluctuations
        noise = np.random.normal(0, 2, len(date_range))
        
        temperature = base_temp + daily_pattern + noise
        
        # Wind data (higher values = more wind)
        wind_speed = 5 + 3 * np.sin(np.pi * days / 15) + np.random.normal(0, 2, len(date_range))
        
        # Solar irradiance (daytime only)
        solar = np.maximum(0, 700 * np.sin(np.pi * hours / 24) ** 2 + np.random.normal(0, 50, len(date_range)))
        solar = np.where((hours < 6) | (hours > 21), 0, solar)  # No solar at night
        
        df = pd.DataFrame({
            'timestamp': date_range,
            'temperature': temperature,
            'wind_speed': wind_speed,
            'solar_irradiance': solar,
            'market': market
        })
        
        # Ensure timestamp is truncated to hour precision
        df['timestamp'] = df['timestamp'].dt.floor('h')
        
        logger.info(f"Successfully retrieved {len(df)} weather records for market {market}")
        return df 