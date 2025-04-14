#!/usr/bin/env python
# Energy Price Forecasting System
# MFT Energy Application Project

import os
import yaml
import pandas as pd
from typing import Dict, List
from logger import setup_logger
from data_ingestion import DataIngestion
from data_processor import DataProcessor
from model_trainer import ModelTrainer
from visualization import Visualization
import matplotlib.pyplot as plt

# Set up logging
logger = setup_logger()

def load_config(config_path: str = "config.yaml") -> Dict:
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
            "markets": ["DE", "FR", "NL"],
            "lookback_days": 30,
            "model": {
                "type": "random_forest",
                "forecast_horizon": 3
            },
            "paths": {
                "models_dir": "models/",
                "data_dir": "data/"
            }
        }

def run_pipeline(config_path: str = "config.yaml") -> None:
    """Run the complete data analysis and modeling pipeline"""
    logger.info("Starting energy price forecasting pipeline")
    
    # Load configuration
    config = load_config(config_path)
    markets = config.get("markets", ["DE", "FR", "NL"])
    
    # Initialize components
    data_ingestion = DataIngestion(config_path)
    data_processor = DataProcessor()
    model_trainer = ModelTrainer(config.get("model", {}).get("type", "random_forest"))
    
    all_price_data = []
    all_weather_data = []
    
    # Fetch data for each market
    for market in markets:
        price_data = data_ingestion.fetch_price_data(market)
        weather_data = data_ingestion.fetch_weather_data(market)
        
        all_price_data.append(price_data)
        all_weather_data.append(weather_data)
    
    # Combine data from all markets
    price_df = pd.concat(all_price_data, ignore_index=True)
    weather_df = pd.concat(all_weather_data, ignore_index=True)
    
    # Merge and process data
    merged_df = data_processor.merge_datasets(price_df, weather_df)
    
    # Check if merged dataset is empty
    if merged_df.empty:
        logger.error("Merged dataset is empty. Checking data sources...")
        
        # Debug information
        logger.info(f"Price data shape: {price_df.shape}, Weather data shape: {weather_df.shape}")
        
        # Print sample of timestamps to check for matching issues
        if not price_df.empty and not weather_df.empty:
            logger.info(f"Price data timestamp sample: {price_df['timestamp'].head()}")
            logger.info(f"Weather data timestamp sample: {weather_df['timestamp'].head()}")
            
            # Check timestamp formats
            if price_df['timestamp'].dtype != weather_df['timestamp'].dtype:
                logger.info("Converting timestamp formats to ensure matching...")
                price_df['timestamp'] = pd.to_datetime(price_df['timestamp'])
                weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'])
                
                # Try merging again
                merged_df = data_processor.merge_datasets(price_df, weather_df)
        
        # If still empty, generate error message with more details
        if merged_df.empty:
            logger.error("Failed to create a valid merged dataset. Exiting pipeline.")
            return
    
    processed_df = data_processor.create_features(merged_df)
    
    # Save processed data
    data_dir = config.get("paths", {}).get("data_dir", "data/")
    os.makedirs(data_dir, exist_ok=True)
    processed_df.to_csv(os.path.join(data_dir, "processed_data.csv"), index=False)
    logger.info(f"Processed data saved to {os.path.join(data_dir, 'processed_data.csv')}")
    
    # Prepare data for modeling
    forecast_horizon = config.get("model", {}).get("forecast_horizon", 3)
    X, y = data_processor.prepare_model_data(processed_df, forecast_horizon=forecast_horizon)
    
    # Train model
    model_trainer.train(X, y)
    
    # If we couldn't get valid data, stop the pipeline
    if X.size == 0 or y.size == 0:
        logger.error("No valid data available for training. Stopping pipeline.")
        return
    
    # Save model
    models_dir = config.get("paths", {}).get("models_dir", "models/")
    model_trainer.save_model(models_dir)
    
    # Create visualizations
    viz = Visualization()
    
    # Only create visualizations if we have data
    if not merged_df.empty:
        # Plot price series for all markets
        for market in markets:
            fig = viz.plot_price_series(merged_df, market=market)
            fig.savefig(os.path.join(data_dir, f"{market.lower()}_price_series.png"))
            plt.close(fig)  # Close the figure to free memory
        
        # Plot feature importance
        if model_trainer.feature_importance is not None:
            fig2 = viz.plot_feature_importance(model_trainer.feature_importance)
            fig2.savefig(os.path.join(data_dir, "feature_importance.png"))
            plt.close(fig2)  # Close the figure to free memory
        
        # Plot daily patterns for all markets
        for market in markets:
            fig3 = viz.plot_daily_patterns(processed_df, market=market)
            fig3.savefig(os.path.join(data_dir, f"{market.lower()}_daily_patterns.png"))
            plt.close(fig3)  # Close the figure to free memory
        
        logger.info("Pipeline completed successfully")
    else:
        logger.error("Cannot create visualizations: No valid data available")

if __name__ == "__main__":
    run_pipeline()