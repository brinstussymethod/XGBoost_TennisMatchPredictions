"""
Data Loader Module for ATP Tennis Prediction
Handles loading, basic cleaning, and splitting of tennis match data
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path


class TennisDataLoader:
    """Load and preprocess ATP tennis match data"""
    
    def __init__(self, data_path='data/raw/atp_tennis.csv'):
        """
        Initialize the data loader
        
        Args:
            data_path (str): Path to the CSV file containing match data
        """
        self.data_path = data_path
        self.df = None
        
    def load_data(self):
        """
        Load the CSV file into a pandas DataFrame
        
        Returns:
            pd.DataFrame: Loaded data
        """
        print(f"Loading data from {self.data_path}...")
        self.df = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.df):,} matches")
        return self.df
    
    def clean_data(self):
        """
        Clean the data:
        - Convert dates to datetime
        - Handle missing values
        - Standardize formats
        
        Returns:
            pd.DataFrame: Cleaned data
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")

        print("\nCleaning data...")
        
        # Convert Date to datetime
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        
        # Extract year for easier filtering
        self.df['Year'] = self.df['Date'].dt.year
        
        # Handle missing rankings (replace -1 with NaN)
        self.df['Rank_1'] = self.df['Rank_1'].replace(-1, np.nan)
        self.df['Rank_2'] = self.df['Rank_2'].replace(-1, np.nan)
        
        # Handle missing points
        self.df['Pts_1'] = self.df['Pts_1'].replace(-1, np.nan)
        self.df['Pts_2'] = self.df['Pts_2'].replace(-1, np.nan)
        
        # Handle missing odds
        self.df['Odd_1'] = self.df['Odd_1'].replace(-1.0, np.nan)
        self.df['Odd_2'] = self.df['Odd_2'].replace(-1.0, np.nan)
        
        # Remove any completely duplicate rows
        initial_len = len(self.df)
        self.df = self.df.drop_duplicates()
        removed = initial_len - len(self.df)
        if removed > 0:
            print(f"   Removed {removed} duplicate matches")
        
        # Sort by date (CRITICAL for ELO calculation)
        self.df = self.df.sort_values('Date').reset_index(drop=True)

        print(f"Data cleaned. {len(self.df):,} matches remaining")

        return self.df
    
    def split_train_test(self, test_year=2025):
        """
        Split data into training and test sets by year
        
        Args:
            test_year (int): Year to use as test set (default: 2025)
            
        Returns:
            tuple: (train_df, test_df)
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")

        print(f"\nSplitting data: Train (before {test_year}) vs Test ({test_year})")
        
        train_df = self.df[self.df['Year'] < test_year].copy()
        test_df = self.df[self.df['Year'] >= test_year].copy()
        
        print(f"   Training set: {len(train_df):,} matches ({train_df['Year'].min()}-{train_df['Year'].max()})")
        print(f"   Test set: {len(test_df):,} matches ({test_df['Year'].min()}-{test_df['Year'].max()})")
        
        return train_df, test_df
    
    def get_tournament_data(self, tournament_name, year=None):
        """
        Get data for a specific tournament
        
        Args:
            tournament_name (str): Name of tournament (e.g., "Wimbledon")
            year (int, optional): Specific year
            
        Returns:
            pd.DataFrame: Tournament data
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        df_filtered = self.df[self.df['Tournament'] == tournament_name].copy()
        
        if year is not None:
            df_filtered = df_filtered[df_filtered['Year'] == year]
        
        return df_filtered
    
    def get_data_summary(self):
        """
        Print a summary of the loaded data
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")

        print("\n" + "="*60)
        print("DATA SUMMARY")
        print("="*60)

        print(f"\nDate Range: {self.df['Date'].min().date()} to {self.df['Date'].max().date()}")
        print(f"Total Matches: {len(self.df):,}")
        print(f"Tournaments: {self.df['Tournament'].nunique()}")
        print(f"Players: {len(set(self.df['Player_1']) | set(self.df['Player_2']))}")

        print(f"\nMatches by Year:")
        year_counts = self.df['Year'].value_counts().sort_index()
        for year, count in year_counts.items():
            print(f"   {year}: {count:,} matches")

        print(f"\nSurface Distribution:")
        surface_counts = self.df['Surface'].value_counts()
        for surface, count in surface_counts.items():
            percentage = (count / len(self.df)) * 100
            print(f"   {surface}: {count:,} matches ({percentage:.1f}%)")

        print(f"\nGrand Slams in Dataset:")
        grand_slams = ['Australian Open', 'French Open', 'Wimbledon', 'US Open']
        for slam in grand_slams:
            count = len(self.df[self.df['Tournament'] == slam])
            if count > 0:
                print(f"   {slam}: {count:,} matches")
        
        print("\n" + "="*60)


def load_and_prepare_data(data_path='data/raw/atp_tennis.csv', test_year=2025):
    """
    Convenience function to load and prepare data in one step
    
    Args:
        data_path (str): Path to CSV file
        test_year (int): Year to use as test set
        
    Returns:
        tuple: (train_df, test_df, loader)
    """
    loader = TennisDataLoader(data_path)
    loader.load_data()
    loader.clean_data()
    loader.get_data_summary()
    
    train_df, test_df = loader.split_train_test(test_year)
    
    return train_df, test_df, loader


if __name__ == "__main__":
    # Example usage
    train, test, loader = load_and_prepare_data()
    print(f"\nData ready! Training: {len(train):,} | Test: {len(test):,}")
