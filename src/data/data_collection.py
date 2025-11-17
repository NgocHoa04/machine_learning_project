"""
Weather Data Collection Module

This module provides functionality to collect weather data from Visual Crossing API
for Hanoi, Vietnam. It supports both daily and hourly data collection with proper
error handling, logging, and data validation.

Author: Weather Analytics Team
Date: 2025-11-17
"""

import requests
import pandas as pd
import os
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple
from pathlib import Path
from dotenv import load_dotenv


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WeatherDataCollector:
    """
    A class to handle weather data collection from Visual Crossing API.
    
    Attributes:
        api_key (str): API key for Visual Crossing
        base_url (str): Base URL for the API
        location (str): Location for weather data
    """
    
    def __init__(self, api_key: Optional[str] = None, location: str = "Hanoi,Vietnam"):
        """
        Initialize the WeatherDataCollector.
        
        Args:
            api_key: Visual Crossing API key. If None, loads from environment.
            location: Location string for weather data collection.
            
        Raises:
            ValueError: If API key is not found.
        """
        # Load environment variables
        self._load_environment()
        
        # Set API key
        self.api_key = api_key or os.getenv("VISUAL_CROSSING_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key not found! Please set VISUAL_CROSSING_API_KEY in .env file "
                "or pass it as a parameter."
            )
        
        self.base_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
        self.location = location
        logger.info(f"WeatherDataCollector initialized for location: {location}")
    
    @staticmethod
    def _load_environment() -> None:
        """Load environment variables from .env file."""
        # Try multiple possible paths for .env file
        possible_paths = [
            Path('.env'),
            Path('../.env'),
            Path('../../.env'),
            Path(__file__).parent.parent.parent / '.env'
        ]
        
        for env_path in possible_paths:
            if env_path.exists():
                load_dotenv(dotenv_path=env_path)
                logger.info(f"Loaded environment from: {env_path}")
                return
        
        logger.warning("No .env file found in common locations")
    
    def fetch_weather_data(
        self,
        start_date: str,
        end_date: str,
        include_hourly: bool = True
    ) -> Optional[Dict]:
        """
        Fetch weather data from Visual Crossing API.
        
        Args:
            start_date: Start date in format YYYY-MM-DD
            end_date: End date in format YYYY-MM-DD
            include_hourly: Whether to include hourly data
            
        Returns:
            Dictionary containing weather data, or None if request fails
            
        Raises:
            requests.exceptions.RequestException: If API request fails
        """
        # Build API URL
        url = f"{self.base_url}/{self.location}/{start_date}/{end_date}"
        
        # API parameters
        params = {
            'unitGroup': 'metric',  # Use metric units (Celsius, km/h, etc.)
            'key': self.api_key,
            'include': 'days,hours' if include_hourly else 'days',
            'contentType': 'json'
        }
        
        try:
            logger.info(
                f"Fetching weather data for {self.location} "
                f"from {start_date} to {end_date}..."
            )
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            logger.info(
                f"Data fetched successfully! "
                f"Retrieved {len(data.get('days', []))} days of data."
            )
            
            return data
            
        except requests.exceptions.Timeout:
            logger.error("Request timed out after 30 seconds")
            raise
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error occurred: {e}")
            if response.status_code == 401:
                logger.error("Authentication failed. Please check your API key.")
            elif response.status_code == 429:
                logger.error("Rate limit exceeded. Please wait before making more requests.")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching data: {e}")
            raise
    
    def process_daily_data(self, weather_data: Dict) -> Optional[pd.DataFrame]:
        """
        Convert daily weather data to DataFrame.
        
        Args:
            weather_data: Raw weather data from API
            
        Returns:
            DataFrame containing daily weather data, or None if processing fails
        """
        if not weather_data or 'days' not in weather_data:
            logger.error("Invalid weather data: missing 'days' key")
            return None
        
        try:
            daily_data = []
            for day in weather_data['days']:
                # Remove nested hourly data if present
                day_copy = {k: v for k, v in day.items() if k != 'hours'}
                daily_data.append(day_copy)
            
            df_daily = pd.DataFrame(daily_data)
            
            # Convert datetime column to proper format
            if 'datetime' in df_daily.columns:
                df_daily['datetime'] = pd.to_datetime(df_daily['datetime'])
            
            logger.info(
                f"Daily data processed successfully. "
                f"Shape: {df_daily.shape}, Columns: {len(df_daily.columns)}"
            )
            
            return df_daily
            
        except Exception as e:
            logger.error(f"Error processing daily data: {e}")
            return None
    
    def process_hourly_data(self, weather_data: Dict) -> Optional[pd.DataFrame]:
        """
        Convert hourly weather data to DataFrame.
        
        Args:
            weather_data: Raw weather data from API
            
        Returns:
            DataFrame containing hourly weather data, or None if processing fails
        """
        if not weather_data or 'days' not in weather_data:
            logger.error("Invalid weather data: missing 'days' key")
            return None
        
        try:
            hourly_data = []
            for day in weather_data['days']:
                if 'hours' not in day:
                    continue
                    
                date = day['datetime']
                for hour in day['hours']:
                    hour_data = hour.copy()
                    hour_data['date'] = date
                    hourly_data.append(hour_data)
            
            if not hourly_data:
                logger.warning("No hourly data found in weather data")
                return None
            
            df_hourly = pd.DataFrame(hourly_data)
            
            # Create full datetime column
            if 'date' in df_hourly.columns and 'datetime' in df_hourly.columns:
                df_hourly['full_datetime'] = pd.to_datetime(
                    df_hourly['date'] + ' ' + df_hourly['datetime']
                )
            
            logger.info(
                f"Hourly data processed successfully. "
                f"Shape: {df_hourly.shape}, Columns: {len(df_hourly.columns)}"
            )
            
            return df_hourly
            
        except Exception as e:
            logger.error(f"Error processing hourly data: {e}")
            return None
    
    def collect_and_save(
        self,
        start_date: str,
        end_date: str,
        output_dir: str = "dataset/raw",
        save_daily: bool = True,
        save_hourly: bool = True
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Collect weather data and save to CSV files.
        
        Args:
            start_date: Start date in format YYYY-MM-DD
            end_date: End date in format YYYY-MM-DD
            output_dir: Directory to save the output files
            save_daily: Whether to save daily data
            save_hourly: Whether to save hourly data
            
        Returns:
            Tuple of (daily_df, hourly_df)
        """
        # Fetch data
        weather_data = self.fetch_weather_data(start_date, end_date)
        
        if not weather_data:
            logger.error("Failed to fetch weather data")
            return None, None
        
        # Process data
        df_daily = None
        df_hourly = None
        
        if save_daily:
            df_daily = self.process_daily_data(weather_data)
        
        if save_hourly:
            df_hourly = self.process_hourly_data(weather_data)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save daily data
        if df_daily is not None and save_daily:
            daily_filename = "Hanoi_Daily.csv"
            daily_filepath = output_path / daily_filename
            df_daily.to_csv(daily_filepath, index=False, encoding='utf-8')
            logger.info(f"✓ Daily data saved to: {daily_filepath} (Shape: {df_daily.shape})")
        
        # Save hourly data
        if df_hourly is not None and save_hourly:
            hourly_filename = "Hanoi_Hourly.csv"
            hourly_filepath = output_path / hourly_filename
            df_hourly.to_csv(hourly_filepath, index=False, encoding='utf-8')
            logger.info(f"✓ Hourly data saved to: {hourly_filepath} (Shape: {df_hourly.shape})")
        
        return df_daily, df_hourly
    
    def collect_recent_data(
        self,
        days: int = 15,
        output_dir: str = "dataset/raw"
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Collect weather data for the recent N days.
        
        Args:
            days: Number of days to collect (from today backwards)
            output_dir: Directory to save the output files
            
        Returns:
            Tuple of (daily_df, hourly_df)
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")
        
        logger.info(f"Collecting recent {days} days of data ({start_date_str} to {end_date_str})")
        
        return self.collect_and_save(start_date_str, end_date_str, output_dir)


def main():
    """
    Main function to demonstrate usage of WeatherDataCollector.
    """
    try:
        # Initialize collector
        collector = WeatherDataCollector(location="Hanoi,Vietnam")
        
        # Collect recent 15 days of data
        df_daily, df_hourly = collector.collect_recent_data(days=15)
        
        # Display summary
        if df_daily is not None:
            print("\n=== Daily Data Summary ===")
            print(f"Shape: {df_daily.shape}")
            print(f"Date range: {df_daily['datetime'].min()} to {df_daily['datetime'].max()}")
            print("\nFirst few rows:")
            print(df_daily.head())
        
        if df_hourly is not None:
            print("\n=== Hourly Data Summary ===")
            print(f"Shape: {df_hourly.shape}")
            if 'full_datetime' in df_hourly.columns:
                print(f"Date range: {df_hourly['full_datetime'].min()} to {df_hourly['full_datetime'].max()}")
            print("\nFirst few rows:")
            print(df_hourly.head())
        
        logger.info("Data collection completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()
