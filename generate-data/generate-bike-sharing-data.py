import os
import sys
from datetime import datetime
import pandas as pd

"""
The data was taken from: http://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
The processing is inspired by the way that Christoph Molnar describes in 4.1 Bike Rentals (Regression) from his book: Interpretable Machine Learning
(https://christophm.github.io/interpretable-ml-book/)
"""

def get_project_root():
    """Returns the root directory of the project."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def load_csv(file_path):
    """Loads a CSV file into a pandas DataFrame."""
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        sys.exit(f"File not found: {file_path}")
    except pd.errors.EmptyDataError:
        sys.exit(f"No data: {file_path} is empty.")
    except Exception as e:
        sys.exit(f"Error loading {file_path}: {e}")

def map_season(season_num):
    """Maps season numbers to season names."""
    return {1: 'spring', 2: 'summer', 3: 'fall', 4: 'winter'}.get(season_num, 'unknown')

def map_weathersit(weathersit_num):
    """Maps weathersit numbers to descriptive weather situations."""
    return {
        1: 'clear, few clouds, partly cloudy',
        2: 'mist + cloudy, mist + broken clouds, mist + few clouds, mist',
        3: 'light snow, light rain + thunderstorm + scattered clouds, light rain + scattered clouds',
        4: 'heavy rain + ice pallets + thunderstorm + mist, snow + fog'
    }.get(weathersit_num, 'unknown')

def process_day_data(day_df):
    """Processes the daily bike sharing data."""
    # Select and rename required features
    required_features = ['dteday', 'season', 'yr', 'holiday', 'workingday', 'weathersit', 'temp', 'hum', 'windspeed', 'cnt']
    day_df = day_df[required_features]
    
    # Map 'season' and 'weathersit' to descriptive names
    day_df['season'] = day_df['season'].apply(map_season)
    day_df['weathersit'] = day_df['weathersit'].apply(map_weathersit)
    
    # Convert 'yr' to actual year
    day_df['yr'] = day_df['yr'].apply(lambda x: 2011 if x == 0 else 2012)
    
    # Convert 'dteday' to datetime and calculate 'days_since_2011_01_01'
    day_df['dteday'] = pd.to_datetime(day_df['dteday'], format='%Y-%m-%d')
    start_date = datetime(2011, 1, 1)
    day_df['days_since_2011_01_01'] = (day_df['dteday'] - start_date).dt.days
    
    # Convert normalised 'temp', 'hum', and 'windspeed' to Celsius, percentage, and km/h
    day_df['temp_celsius'] = day_df['temp'] * 41
    day_df['humidity_percent'] = day_df['hum'] * 100
    day_df['windspeed_kmh'] = day_df['windspeed'] * 67

    # Rename columns for clarity
    day_df = day_df.rename(columns={
        'cnt': 'total_count',
        'holiday': 'is_holiday',
        'workingday': 'is_workingday',
        'season': 'season',
        'weathersit': 'weather_situation'
    })

    # Select and reorder final features
    final_features = [
        'total_count',
        'season',
        'is_holiday',
        'yr',
        'days_since_2011_01_01',
        'is_workingday',
        'weather_situation',
        'temp_celsius',
        'humidity_percent',
        'windspeed_kmh'
    ]
    processed_df = day_df[final_features]

    return processed_df

def main():
    """Main function to process the Bike Sharing Dataset."""
    project_root = get_project_root()
    input_dir = os.path.join(project_root, 'dataset-generation', 'bike-sharing')
    output_dir = os.path.join(project_root, 'data')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'bike_sharing_processed.csv')

    day_csv_path = os.path.join(input_dir, 'day.csv')

    # Load data and process daily data
    day_df = load_csv(day_csv_path)
    processed_day_df = process_day_data(day_df)

    # Save processed data
    processed_day_df.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")

if __name__ == "__main__":
    main()