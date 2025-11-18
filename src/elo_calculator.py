"""
ELO Rating Calculator for ATP Tennis Prediction
Implements ELO rating system with dynamic K-factor based on player experience
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta


class EloCalculator:
    """
    Calculate and maintain ELO ratings for tennis players
    
    Features:
    - Dynamic K-factor based on player experience (like chess)
    - Surface-specific ELO ratings (grass, clay, hard)
    - Handles player breaks/inactivity
    """
    
    def __init__(self, initial_rating=1500, base_k=32, min_k=16, max_k=48):
        """
        Initialize ELO calculator
        
        Args:
            initial_rating (int): Starting ELO for new players
            base_k (int): Base K-factor
            min_k (int): Minimum K-factor for experienced players
            max_k (int): Maximum K-factor for new/returning players
        """
        self.initial_rating = initial_rating
        self.base_k = base_k
        self.min_k = min_k
        self.max_k = max_k
        
        # Store ratings for each player
        self.ratings = defaultdict(lambda: initial_rating)
        self.surface_ratings = {
            'Hard': defaultdict(lambda: initial_rating),
            'Clay': defaultdict(lambda: initial_rating),
            'Grass': defaultdict(lambda: initial_rating),
            'Carpet': defaultdict(lambda: initial_rating)
        }
        
        # Track player statistics for dynamic K-factor
        self.match_count = defaultdict(int)
        self.last_match_date = {}
        
    def get_dynamic_k_factor(self, player, current_date):
        """
        Calculate dynamic K-factor based on player experience and activity
        
        Formula inspired by chess ELO system:
        - New players (< 30 matches): Higher K-factor (more volatile ratings)
        - Experienced players (> 30 matches): Lower K-factor (stable ratings)
        - Returning players (long break): Increased K-factor
        
        Args:
            player (str): Player name
            current_date (datetime): Current match date
            
        Returns:
            float: Dynamic K-factor
        """
        matches = self.match_count[player]
        
        # Base K-factor from match experience
        if matches < 10:
            k = self.max_k  # Very new, highly volatile
        elif matches < 30:
            k = self.base_k  # Still learning
        else:
            k = self.min_k + (self.base_k - self.min_k) * np.exp(-matches / 50)
        
        # Increase K-factor if player took a long break
        if player in self.last_match_date:
            days_since_last = (current_date - self.last_match_date[player]).days
            if days_since_last > 180:  # 6 months break
                k *= 1.3  # Increase by 30%
            elif days_since_last > 90:  # 3 months break
                k *= 1.15  # Increase by 15%
        
        return min(k, self.max_k)  # Cap at maximum
    
    def expected_score(self, rating_1, rating_2):
        """
        Calculate expected score (win probability) using ELO formula
        
        Args:
            rating_1 (float): Player 1's rating
            rating_2 (float): Player 2's rating
            
        Returns:
            float: Expected score for player 1 (0 to 1)
        """
        return 1 / (1 + 10 ** ((rating_2 - rating_1) / 400))
    
    def update_ratings(self, player_1, player_2, winner, surface, match_date, 
                       best_of=3, is_grand_slam=False):
        """
        Update ELO ratings after a match
        
        Args:
            player_1 (str): First player
            player_2 (str): Second player
            winner (str): Match winner
            surface (str): Court surface
            match_date (datetime): Match date
            best_of (int): Best of 3 or 5 sets
            is_grand_slam (bool): Whether this is a Grand Slam
            
        Returns:
            tuple: (new_rating_1, new_rating_2, expected_1, expected_2)
        """
        # Get current ratings
        rating_1 = self.ratings[player_1]
        rating_2 = self.ratings[player_2]
        
        surface_rating_1 = self.surface_ratings[surface][player_1]
        surface_rating_2 = self.surface_ratings[surface][player_2]
        
        # Calculate expected scores
        expected_1 = self.expected_score(rating_1, rating_2)
        expected_2 = 1 - expected_1
        
        surface_expected_1 = self.expected_score(surface_rating_1, surface_rating_2)
        surface_expected_2 = 1 - surface_expected_1
        
        # Actual scores
        score_1 = 1.0 if winner == player_1 else 0.0
        score_2 = 1.0 if winner == player_2 else 0.0
        
        # Get dynamic K-factors
        k1 = self.get_dynamic_k_factor(player_1, match_date)
        k2 = self.get_dynamic_k_factor(player_2, match_date)
        
        # Adjust K-factor for important matches
        if is_grand_slam:
            k1 *= 1.2
            k2 *= 1.2
        elif best_of == 5:
            k1 *= 1.1
            k2 *= 1.1
        
        # Update overall ratings
        new_rating_1 = rating_1 + k1 * (score_1 - expected_1)
        new_rating_2 = rating_2 + k2 * (score_2 - expected_2)
        
        self.ratings[player_1] = new_rating_1
        self.ratings[player_2] = new_rating_2
        
        # Update surface-specific ratings
        new_surface_rating_1 = surface_rating_1 + k1 * (score_1 - surface_expected_1)
        new_surface_rating_2 = surface_rating_2 + k2 * (score_2 - surface_expected_2)
        
        self.surface_ratings[surface][player_1] = new_surface_rating_1
        self.surface_ratings[surface][player_2] = new_surface_rating_2
        
        # Update statistics
        self.match_count[player_1] += 1
        self.match_count[player_2] += 1
        self.last_match_date[player_1] = match_date
        self.last_match_date[player_2] = match_date
        
        return new_rating_1, new_rating_2, expected_1, expected_2
    
    def get_rating(self, player, surface=None):
        """
        Get current rating for a player
        
        Args:
            player (str): Player name
            surface (str, optional): Get surface-specific rating
            
        Returns:
            float: Current ELO rating
        """
        if surface:
            return self.surface_ratings[surface][player]
        return self.ratings[player]
    
    def get_top_players(self, n=20, surface=None):
        """
        Get top N players by ELO rating
        
        Args:
            n (int): Number of players to return
            surface (str, optional): Get surface-specific rankings
            
        Returns:
            list: List of (player, rating) tuples
        """
        if surface:
            ratings_dict = dict(self.surface_ratings[surface])
        else:
            ratings_dict = dict(self.ratings)
        
        sorted_players = sorted(ratings_dict.items(), key=lambda x: x[1], reverse=True)
        return sorted_players[:n]


def calculate_elo_for_dataframe(df, initial_rating=1500):
    """
    Calculate ELO ratings for all matches in a DataFrame
    
    Args:
        df (pd.DataFrame): DataFrame with match data (must be sorted by date!)
        initial_rating (int): Starting ELO
        
    Returns:
        pd.DataFrame: DataFrame with added ELO columns
    """
    print("🎮 Calculating ELO ratings...")
    print("   This may take a few minutes for large datasets...")
    
    # Initialize calculator
    elo = EloCalculator(initial_rating=initial_rating)
    
    # Lists to store pre-match ELO ratings
    elo_1_before = []
    elo_2_before = []
    surface_elo_1_before = []
    surface_elo_2_before = []
    expected_1_list = []
    expected_2_list = []
    
    # Process each match in chronological order
    for idx, row in df.iterrows():
        player_1 = row['Player_1']
        player_2 = row['Player_2']
        winner = row['Winner']
        surface = row['Surface']
        match_date = row['Date']
        best_of = row['Best of']
        
        # Check if Grand Slam
        is_grand_slam = row['Series'] == 'Grand Slam'
        
        # Get ratings BEFORE the match
        rating_1_before = elo.get_rating(player_1)
        rating_2_before = elo.get_rating(player_2)
        surface_rating_1_before = elo.get_rating(player_1, surface)
        surface_rating_2_before = elo.get_rating(player_2, surface)
        
        # Store pre-match ratings
        elo_1_before.append(rating_1_before)
        elo_2_before.append(rating_2_before)
        surface_elo_1_before.append(surface_rating_1_before)
        surface_elo_2_before.append(surface_rating_2_before)
        
        # Calculate expected scores
        expected_1 = elo.expected_score(rating_1_before, rating_2_before)
        expected_2 = 1 - expected_1
        expected_1_list.append(expected_1)
        expected_2_list.append(expected_2)
        
        # Update ratings AFTER the match
        elo.update_ratings(player_1, player_2, winner, surface, match_date, 
                          best_of, is_grand_slam)
        
        # Progress indicator
        if (idx + 1) % 5000 == 0:
            print(f"   Processed {idx + 1:,} matches...")
    
    # Add ELO columns to dataframe
    df_with_elo = df.copy()
    df_with_elo['elo_1'] = elo_1_before
    df_with_elo['elo_2'] = elo_2_before
    df_with_elo['surface_elo_1'] = surface_elo_1_before
    df_with_elo['surface_elo_2'] = surface_elo_2_before
    df_with_elo['elo_diff'] = df_with_elo['elo_1'] - df_with_elo['elo_2']
    df_with_elo['surface_elo_diff'] = df_with_elo['surface_elo_1'] - df_with_elo['surface_elo_2']
    df_with_elo['expected_1'] = expected_1_list
    df_with_elo['expected_2'] = expected_2_list
    
    print(f"✅ ELO calculation complete!")
    print(f"   Average ELO: {df_with_elo['elo_1'].mean():.1f}")
    print(f"   ELO range: {df_with_elo['elo_1'].min():.1f} - {df_with_elo['elo_1'].max():.1f}")
    
    return df_with_elo, elo


if __name__ == "__main__":
    print("ELO Calculator Module - Ready to use!")
