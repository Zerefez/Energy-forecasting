import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Visualization:
    """Create visualizations for data analysis and model results"""
    
    @staticmethod
    def plot_price_series(df: pd.DataFrame, market: str = "DE") -> plt.Figure:
        """Plot historical price series for a specific market"""
        market_data = df[df['market'] == market].copy()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(market_data['timestamp'], market_data['price'], label=f'{market} Price')
        
        # Format plot
        ax.set_title(f'Energy Price Series - {market} Market', fontsize=15)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price (€/MWh)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 15) -> plt.Figure:
        """Plot feature importance from model"""
        top_features = importance_df.head(top_n)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(x='importance', y='feature', data=top_features, ax=ax)
        
        ax.set_title('Top Feature Importance', fontsize=15)
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_prediction_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, dates: pd.Series = None) -> plt.Figure:
        """Plot predictions vs actual values"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if dates is not None:
            ax.plot(dates, y_true, label='Actual', alpha=0.7)
            ax.plot(dates, y_pred, label='Predicted', alpha=0.7)
            ax.set_xlabel('Date', fontsize=12)
        else:
            ax.plot(y_true, label='Actual', alpha=0.7)
            ax.plot(y_pred, label='Predicted', alpha=0.7)
            ax.set_xlabel('Sample Index', fontsize=12)
        
        # Format plot
        ax.set_title('Predicted vs Actual Energy Prices', fontsize=15)
        ax.set_ylabel('Price (€/MWh)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_daily_patterns(df: pd.DataFrame, market: str = "DE") -> plt.Figure:
        """Plot average daily price patterns"""
        market_data = df[df['market'] == market].copy()
        
        # Create hour of day and day of week features if they don't exist
        if 'hour' not in market_data.columns:
            market_data['hour'] = market_data['timestamp'].dt.hour
        if 'day_of_week' not in market_data.columns:
            market_data['day_of_week'] = market_data['timestamp'].dt.dayofweek
        
        # Group by hour and day of week
        hourly_avg = market_data.groupby('hour')['price'].mean().reset_index()
        daily_avg = market_data.groupby('day_of_week')['price'].mean().reset_index()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot hourly pattern
        sns.lineplot(x='hour', y='price', data=hourly_avg, ax=ax1, marker='o')
        ax1.set_title(f'Average Hourly Price Pattern - {market}', fontsize=14)
        ax1.set_xlabel('Hour of Day', fontsize=12)
        ax1.set_ylabel('Average Price (€/MWh)', fontsize=12)
        ax1.set_xticks(range(0, 24, 2))
        ax1.grid(True, alpha=0.3)
        
        # Plot daily pattern
        sns.barplot(x='day_of_week', y='price', data=daily_avg, ax=ax2)
        ax2.set_title(f'Average Daily Price Pattern - {market}', fontsize=14)
        ax2.set_xlabel('Day of Week (0=Monday)', fontsize=12)
        ax2.set_ylabel('Average Price (€/MWh)', fontsize=12)
        ax2.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig 