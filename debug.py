import pandas as pd
from data_ingestion import DataIngestion
from data_processor import DataProcessor

def debug_merge_process():
    # Initialize components
    data_ingestion = DataIngestion()
    data_processor = DataProcessor()
    
    # Get data for a single market for simplicity
    market = "DE"
    price_data = data_ingestion.fetch_price_data(market)
    weather_data = data_ingestion.fetch_weather_data(market)
    
    print(f"Price data shape: {price_data.shape}")
    print(f"Weather data shape: {weather_data.shape}")
    
    # Show first few rows of each dataset
    print("\nPrice data first 3 rows:")
    print(price_data.head(3))
    print("\nWeather data first 3 rows:")
    print(weather_data.head(3))
    
    # Check timestamp format
    print("\nPrice data timestamp type:", type(price_data['timestamp'].iloc[0]))
    print("Weather data timestamp type:", type(weather_data['timestamp'].iloc[0]))
    
    # Check timestamp values explicitly
    print("\nSample price timestamps:")
    for i in range(5):
        ts = price_data['timestamp'].iloc[i]
        print(f"{i}: {ts} ({ts.timestamp()})")
    
    print("\nSample weather timestamps:")
    for i in range(5):
        ts = weather_data['timestamp'].iloc[i]
        print(f"{i}: {ts} ({ts.timestamp()})")
    
    # Try with more explicit timestamp normalization
    price_data['timestamp_str'] = price_data['timestamp'].dt.strftime('%Y-%m-%d %H:00:00')
    weather_data['timestamp_str'] = weather_data['timestamp'].dt.strftime('%Y-%m-%d %H:00:00')
    
    price_data['timestamp'] = pd.to_datetime(price_data['timestamp_str'])
    weather_data['timestamp'] = pd.to_datetime(weather_data['timestamp_str'])
    
    # Drop the string columns
    price_data = price_data.drop('timestamp_str', axis=1)
    weather_data = weather_data.drop('timestamp_str', axis=1)
    
    print("\nAfter normalization:")
    print("Sample price timestamps:")
    for i in range(5):
        ts = price_data['timestamp'].iloc[i]
        print(f"{i}: {ts} ({ts.timestamp()})")
    
    print("\nSample weather timestamps:")
    for i in range(5):
        ts = weather_data['timestamp'].iloc[i]
        print(f"{i}: {ts} ({ts.timestamp()})")
    
    # Try merging again
    merged_data = data_processor.merge_datasets(price_data, weather_data)
    print(f"\nMerged data shape: {merged_data.shape}")
    
    if not merged_data.empty:
        print("\nMerged data first 3 rows:")
        print(merged_data.head(3))
    else:
        print("\nMerge still results in empty dataset!")
        
        # Try even more direct approach
        print("\nTrying alternative merge approach...")
        
        # Convert to strings and merge on string representation
        price_data['ts_key'] = price_data['timestamp'].dt.strftime('%Y-%m-%d %H')
        weather_data['ts_key'] = weather_data['timestamp'].dt.strftime('%Y-%m-%d %H')
        
        # Manual merge
        manual_merge = pd.merge(price_data, weather_data, on=['ts_key', 'market'], how='inner')
        print(f"Manual merge shape: {manual_merge.shape}")
        
        if not manual_merge.empty:
            # Keep only one timestamp column
            manual_merge = manual_merge.drop('timestamp_y', axis=1)
            manual_merge = manual_merge.rename(columns={'timestamp_x': 'timestamp'})
            manual_merge = manual_merge.drop('ts_key', axis=1)
            
            print("\nManual merge first 3 rows:")
            print(manual_merge.head(3))
            
            # Save this for further processing
            manual_merge.to_csv('data/debug_merged.csv', index=False)
            print("Saved manual merge to data/debug_merged.csv")

if __name__ == "__main__":
    debug_merge_process() 