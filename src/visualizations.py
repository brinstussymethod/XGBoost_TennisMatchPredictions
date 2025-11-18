"""
Visualization Module for ATP Tennis Prediction
Creates charts and graphs for analysis and presentation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def plot_elo_evolution(df, players, surface=None, save_path=None):
    """
    Plot ELO rating evolution over time for specific players
    
    Args:
        df (pd.DataFrame): DataFrame with ELO ratings
        players (list): List of player names to plot
        surface (str, optional): Plot surface-specific ELO
        save_path (str, optional): Path to save figure
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for player in players:
        # Get matches for this player
        player_matches = df[(df['Player_1'] == player) | (df['Player_2'] == player)].copy()
        
        if len(player_matches) == 0:
            print(f"Warning: No matches found for {player}")
            continue
        
        # Get ELO ratings
        player_matches['elo'] = player_matches.apply(
            lambda row: row['elo_1'] if row['Player_1'] == player else row['elo_2'],
            axis=1
        )
        
        if surface:
            elo_col = f'surface_elo_1' if 'surface_elo_1' in player_matches.columns else 'elo_1'
            player_matches['elo'] = player_matches.apply(
                lambda row: row[elo_col] if row['Player_1'] == player else row[elo_col.replace('_1', '_2')],
                axis=1
            )
        
        # Plot
        ax.plot(player_matches['Date'], player_matches['elo'], 
                label=player, linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('ELO Rating', fontsize=12)
    title = f'ELO Rating Evolution Over Time'
    if surface:
        title += f' ({surface} Surface)'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to {save_path}")
    
    plt.show()


def plot_feature_importance(feature_importance_df, top_n=20, save_path=None):
    """
    Plot feature importance from model
    
    Args:
        feature_importance_df (pd.DataFrame): Feature importance DataFrame
        top_n (int): Number of top features to show
        save_path (str, optional): Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    top_features = feature_importance_df.head(top_n)
    
    ax.barh(range(len(top_features)), top_features['importance'], color='steelblue')
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to {save_path}")
    
    plt.show()


def plot_prediction_accuracy(results_df, group_by='Tournament', save_path=None):
    """
    Plot prediction accuracy by tournament/round/surface
    
    Args:
        results_df (pd.DataFrame): DataFrame with predictions and actual results
        group_by (str): Column to group by
        save_path (str, optional): Path to save figure
    """
    # Calculate accuracy by group
    accuracy_by_group = results_df.groupby(group_by).apply(
        lambda x: (x['prediction'] == x['target']).mean()
    ).sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bars = ax.barh(range(len(accuracy_by_group)), accuracy_by_group.values, color='green', alpha=0.7)
    ax.set_yticks(range(len(accuracy_by_group)))
    ax.set_yticklabels(accuracy_by_group.index)
    ax.set_xlabel('Accuracy', fontsize=12)
    ax.set_title(f'Prediction Accuracy by {group_by}', fontsize=14, fontweight='bold')
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Random Guess (50%)')
    ax.axvline(x=accuracy_by_group.mean(), color='blue', linestyle='--', alpha=0.5, 
               label=f'Average ({accuracy_by_group.mean():.1%})')
    ax.invert_yaxis()
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add percentage labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{width:.1%}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to {save_path}")
    
    plt.show()


def plot_confusion_matrix(confusion_matrix, save_path=None):
    """
    Plot confusion matrix
    
    Args:
        confusion_matrix (np.array): Confusion matrix
        save_path (str, optional): Path to save figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Player 2 Wins', 'Player 1 Wins'],
                yticklabels=['Player 2 Wins', 'Player 1 Wins'],
                ax=ax)
    
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to {save_path}")
    
    plt.show()


def plot_accuracy_by_round(results_df, tournament=None, save_path=None):
    """
    Plot accuracy by tournament round
    
    Args:
        results_df (pd.DataFrame): DataFrame with predictions
        tournament (str, optional): Filter to specific tournament
        save_path (str, optional): Path to save figure
    """
    if tournament:
        results_df = results_df[results_df['Tournament'] == tournament]
    
    # Define round order
    round_order = ['1st Round', '2nd Round', '3rd Round', '4th Round',  
                   'Quarterfinals', 'Semifinals', 'The Final']
    
    # Calculate accuracy by round
    accuracy_by_round = results_df.groupby('Round').apply(
        lambda x: (x['prediction'] == x['target']).mean()
    )
    
    # Reorder
    accuracy_by_round = accuracy_by_round.reindex([r for r in round_order if r in accuracy_by_round.index])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars = ax.bar(range(len(accuracy_by_round)), accuracy_by_round.values, 
                  color='purple', alpha=0.7)
    ax.set_xticks(range(len(accuracy_by_round)))
    ax.set_xticklabels(accuracy_by_round.index, rotation=45, ha='right')
    ax.set_ylabel('Accuracy', fontsize=12)
    title = 'Prediction Accuracy by Round'
    if tournament:
        title += f' ({tournament})'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random (50%)')
    ax.axhline(y=accuracy_by_round.mean(), color='blue', linestyle='--', alpha=0.5,
               label=f'Average ({accuracy_by_round.mean():.1%})')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1])
    
    # Add percentage labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                f'{height:.1%}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to {save_path}")
    
    plt.show()


def plot_calibration_curve(y_true, y_pred_proba, n_bins=10, save_path=None):
    """
    Plot calibration curve (reliability diagram)
    
    Args:
        y_true (array): True labels
        y_pred_proba (array): Predicted probabilities
        n_bins (int): Number of bins
        save_path (str, optional): Path to save figure
    """
    from sklearn.calibration import calibration_curve
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=n_bins)
    
    ax.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Model')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    
    ax.set_xlabel('Predicted Probability', fontsize=12)
    ax.set_ylabel('True Probability', fontsize=12)
    ax.set_title('Calibration Curve', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to {save_path}")
    
    plt.show()


def create_tournament_bracket_viz(tournament_df, predictions_df, tournament_name='Wimbledon 2025'):
    """
    Create a simple tournament bracket visualization
    
    Args:
        tournament_df (pd.DataFrame): Tournament data
        predictions_df (pd.DataFrame): Predictions with accuracy
        tournament_name (str): Name for the plot
    """
    print(f"\n{'='*60}")
    print(f"🏆 {tournament_name} - Prediction Summary")
    print(f"{'='*60}\n")
    
    # Calculate accuracy by round
    round_accuracy = predictions_df.groupby('Round').apply(
        lambda x: (x['prediction'] == x['target']).mean()
    )
    
    print("📊 Accuracy by Round:")
    for round_name, accuracy in round_accuracy.items():
        matches = len(predictions_df[predictions_df['Round'] == round_name])
        correct = (predictions_df[predictions_df['Round'] == round_name]['prediction'] == 
                  predictions_df[predictions_df['Round'] == round_name]['target']).sum()
        print(f"   {round_name}: {accuracy:.1%} ({correct}/{matches} correct)")
    
    overall_accuracy = (predictions_df['prediction'] == predictions_df['target']).mean()
    total_matches = len(predictions_df)
    total_correct = (predictions_df['prediction'] == predictions_df['target']).sum()
    
    print(f"\n🎯 Overall Accuracy: {overall_accuracy:.1%} ({total_correct}/{total_matches} matches)")
    
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    print("Visualization Module - Ready to use!")
