# Energy Price Forecasting System

A machine learning system for forecasting energy prices across European power markets.

## Project Structure

```
Energy-forecasting/
├── data/                     # Directory for data and visualization outputs
├── models/                   # Directory for saved ML models
├── config.yaml               # Configuration file
├── data_ingestion.py         # Data acquisition module
├── data_processor.py         # Data processing and feature engineering
├── logger.py                 # Logging setup
├── main.py                   # Main execution script
├── model_trainer.py          # Model training and evaluation
├── requirements.txt          # Python dependencies
├── visualization.py          # Data and results visualization
└── __init__.py               # Package initialization
```

## Features

- Simulated data generation for energy prices and relevant weather data
- Feature engineering tailored for energy market forecasting
- Time series prediction with configurable forecast horizon
- Visualizations for data analysis and model evaluation
- Modular architecture for easy extension

## Setup

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Optional: Set environment variable for API key (if using real data sources):
   ```
   export ENERGY_API_KEY=your_api_key
   ```

## Usage

Run the complete pipeline:

```
python main.py
```

## Configuration

The system can be configured via the `config.yaml` file:

- **data_sources**: APIs for data acquisition
- **markets**: List of power markets to analyze (e.g., DK, DE, FR, NL)
- **lookback_days**: Number of days of historical data to use
- **model**: Configuration for forecasting model
- **paths**: Directory paths for data and model storage

## Extending the System

- Add new data sources in the `DataIngestion` class
- Implement new feature engineering techniques in the `DataProcessor` class
- Add new models by extending the `ModelTrainer` class
- Create new visualizations in the `Visualization` class
