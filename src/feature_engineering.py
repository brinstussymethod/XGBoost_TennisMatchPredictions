"""
Feature Engineering Module for ATP Tennis Prediction
Creates features from raw match data for machine learning models
"""

import pandas as pd
import numpy as np
from collections import defaultdict


class FeatureEngineer:
    """Create features for tennis match prediction"""
    
    def __init__(self):
        """Initialize feature engineer"""
        self.player_stats = defaultdict(lambda: {
            'matches': 0,
            'wins': 0,
            'recent_matches': [],
            'surface_stats': defaultdict(lambda: {'matches': 0, 'wins': 0})
        })
        
    def create_basic_features(self, df):
        """
        Create basic features from match data
        
        Args:
            df (pd.DataFrame): Match data with ELO ratings
            
        Returns:
            pd.DataFrame: DataFrame with basic features
        """
        print("🔧 Creating basic features...")
        
        df_features = df.copy()
        
        # Ranking features (handle missing rankings)
        df_features['rank_diff'] = df_features['Rank_1'].fillna(1000) - df_features['Rank_2'].fillna(1000)
        df_features['rank_1_log'] = np.log1p(df_features['Rank_1'].fillna(1000))
        df_features['rank_2_log'] = np.log1p(df_features['Rank_2'].fillna(1000))
        
        # Points features
        df_features['points_diff'] = df_features['Pts_1'].fillna(0) - df_features['Pts_2'].fillna(0)
        
        # Surface encoding (one-hot)
        surface_dummies = pd.get_dummies(df_features['Surface'], prefix='surface')
        df_features = pd.concat([df_features, surface_dummies], axis=1)
        
        # Best of encoding
        df_features['is_best_of_5'] = (df_features['Best of'] == 5).astype(int)
        
        # Tournament importance
        df_features['is_grand_slam'] = (df_features['Series'] == 'Grand Slam').astype(int)
        df_features['is_masters'] = (df_features['Series'].str.contains('Masters', na=False)).astype(int)
        
        # Betting odds features (if available)
        df_features['odds_diff'] = df_features['Odd_1'].fillna(2.0) - df_features['Odd_2'].fillna(2.0)
        df_features['odds_implied_prob_1'] = 1 / df_features['Odd_1'].fillna(2.0)
        df_features['odds_implied_prob_2'] = 1 / df_features['Odd_2'].fillna(2.0)
        
        print(f"✅ Basic features created")
        
        return df_features
    
    def calculate_recent_form(self, df, window=10):
        """
        Calculate recent form statistics for each player
        
        Args:
            df (pd.DataFrame): Match data (must be sorted by date!)
            window (int): Number of recent matches to consider
            
        Returns:
            pd.DataFrame: DataFrame with recent form features
        """
        print(f"📊 Calculating recent form (last {window} matches)...")
        
        # Initialize lists to store features
        recent_win_pct_1 = []
        recent_win_pct_2 = []
        form_diff = []
        
        # Track recent matches for each player
        player_recent = defaultdict(list)
        
        for idx, row in df.iterrows():
            player_1 = row['Player_1']
            player_2 = row['Player_2']
            winner = row['Winner']
            
            # Get recent form for both players
            recent_1 = player_recent[player_1][-window:] if player_1 in player_recent else []
            recent_2 = player_recent[player_2][-window:] if player_2 in player_recent else []
            
            # Calculate win percentages
            win_pct_1 = sum(recent_1) / len(recent_1) if len(recent_1) > 0 else 0.5
            win_pct_2 = sum(recent_2) / len(recent_2) if len(recent_2) > 0 else 0.5
            
            recent_win_pct_1.append(win_pct_1)
            recent_win_pct_2.append(win_pct_2)
            form_diff.append(win_pct_1 - win_pct_2)
            
            # Update recent matches (1 for win, 0 for loss)
            player_recent[player_1].append(1 if winner == player_1 else 0)
            player_recent[player_2].append(1 if winner == player_2 else 0)
            
            # Progress
            if (idx + 1) % 5000 == 0:
                print(f"   Processed {idx + 1:,} matches...")
        
        df_form = df.copy()
        df_form['recent_win_pct_1'] = recent_win_pct_1
        df_form['recent_win_pct_2'] = recent_win_pct_2
        df_form['form_diff'] = form_diff
        
        print(f"✅ Recent form calculated")
        
        return df_form
    
    def calculate_head_to_head(self, df):
        """
        Calculate head-to-head record between players
        
        Args:
            df (pd.DataFrame): Match data (must be sorted by date!)
            
        Returns:
            pd.DataFrame: DataFrame with H2H features
        """
        print("🤝 Calculating head-to-head records...")
        
        # Track H2H records
        h2h_records = defaultdict(lambda: defaultdict(int))
        
        h2h_wins_1 = []
        h2h_wins_2 = []
        h2h_total = []
        h2h_win_pct_1 = []
        
        for idx, row in df.iterrows():
            player_1 = row['Player_1']
            player_2 = row['Player_2']
            winner = row['Winner']
            
            # Create consistent key (alphabetical order)
            players = tuple(sorted([player_1, player_2]))
            
            # Get current H2H record
            wins_1 = h2h_records[players][player_1]
            wins_2 = h2h_records[players][player_2]
            total = wins_1 + wins_2
            
            h2h_wins_1.append(wins_1)
            h2h_wins_2.append(wins_2)
            h2h_total.append(total)
            
            # Calculate win percentage (0.5 if no history)
            win_pct = wins_1 / total if total > 0 else 0.5
            h2h_win_pct_1.append(win_pct)
            
            # Update H2H record
            h2h_records[players][winner] += 1
            
            # Progress
            if (idx + 1) % 10000 == 0:
                print(f"   Processed {idx + 1:,} matches...")
        
        df_h2h = df.copy()
        df_h2h['h2h_wins_1'] = h2h_wins_1
        df_h2h['h2h_wins_2'] = h2h_wins_2
        df_h2h['h2h_total'] = h2h_total
        df_h2h['h2h_win_pct_1'] = h2h_win_pct_1
        
        print(f"✅ Head-to-head records calculated")
        
        return df_h2h
    
    def calculate_surface_stats(self, df):
        """
        Calculate surface-specific statistics for each player
        
        Args:
            df (pd.DataFrame): Match data (must be sorted by date!)
            
        Returns:
            pd.DataFrame: DataFrame with surface-specific features
        """
        print("🏟️ Calculating surface-specific stats...")
        
        # Track surface stats
        surface_stats = defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'matches': 0}))
        
        surface_win_pct_1 = []
        surface_win_pct_2 = []
        surface_matches_1 = []
        surface_matches_2 = []
        
        for idx, row in df.iterrows():
            player_1 = row['Player_1']
            player_2 = row['Player_2']
            winner = row['Winner']
            surface = row['Surface']
            
            # Get current surface stats
            stats_1 = surface_stats[player_1][surface]
            stats_2 = surface_stats[player_2][surface]
            
            # Calculate win percentages (0.5 if no history)
            win_pct_1 = stats_1['wins'] / stats_1['matches'] if stats_1['matches'] > 0 else 0.5
            win_pct_2 = stats_2['wins'] / stats_2['matches'] if stats_2['matches'] > 0 else 0.5
            
            surface_win_pct_1.append(win_pct_1)
            surface_win_pct_2.append(win_pct_2)
            surface_matches_1.append(stats_1['matches'])
            surface_matches_2.append(stats_2['matches'])
            
            # Update surface stats
            surface_stats[player_1][surface]['matches'] += 1
            surface_stats[player_2][surface]['matches'] += 1
            
            if winner == player_1:
                surface_stats[player_1][surface]['wins'] += 1
            else:
                surface_stats[player_2][surface]['wins'] += 1
            
            # Progress
            if (idx + 1) % 10000 == 0:
                print(f"   Processed {idx + 1:,} matches...")
        
        df_surface = df.copy()
        df_surface['surface_win_pct_1'] = surface_win_pct_1
        df_surface['surface_win_pct_2'] = surface_win_pct_2
        df_surface['surface_matches_1'] = surface_matches_1
        df_surface['surface_matches_2'] = surface_matches_2
        df_surface['surface_experience_diff'] = df_surface['surface_matches_1'] - df_surface['surface_matches_2']
        
        print(f"✅ Surface-specific stats calculated")
        
        return df_surface
    
    def create_target(self, df):
        """
        Create target variable (1 if Player_1 won, 0 if Player_2 won)
        
        Args:
            df (pd.DataFrame): Match data
            
        Returns:
            pd.DataFrame: DataFrame with target variable
        """
        df_target = df.copy()
        df_target['target'] = (df_target['Winner'] == df_target['Player_1']).astype(int)
        return df_target


def engineer_all_features(df):
    """
    Create all features for the dataset
    
    Args:
        df (pd.DataFrame): Match data with ELO ratings (must be sorted by date!)
        
    Returns:
        pd.DataFrame: DataFrame with all engineered features
    """
    print("\n" + "="*60)
    print("🔧 FEATURE ENGINEERING")
    print("="*60 + "\n")
    
    engineer = FeatureEngineer()
    
    # Create features step by step
    df = engineer.create_basic_features(df)
    df = engineer.calculate_recent_form(df, window=10)
    df = engineer.calculate_head_to_head(df)
    df = engineer.calculate_surface_stats(df)
    df = engineer.create_target(df)
    
    print("\n" + "="*60)
    print(f"✅ Feature engineering complete!")
    print(f"   Total features: {len(df.columns)}")
    print("="*60 + "\n")
    
    return df


if __name__ == "__main__":
    print("Feature Engineering Module - Ready to use!")
