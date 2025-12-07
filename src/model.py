"""
XGBoost Model Module for ATP Tennis Prediction
Handles model training, prediction, and evaluation
FIXED VERSION - Proper DataFrame handling
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle


class TennisPredictionModel:
    """XGBoost model for predicting tennis match outcomes"""

    def __init__(self, **xgb_params):
        """
        Initialize the model

        Args:
            **xgb_params: XGBoost parameters
        """
        # Default parameters (can be overridden)
        default_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }

        # Update with user parameters
        default_params.update(xgb_params)

        self.model = xgb.XGBClassifier(**default_params)
        self.feature_names = None
        self.feature_importance = None

    def prepare_features(self, df, feature_cols=None):
        """
        Prepare features and target for model

        Args:
            df (pd.DataFrame): DataFrame with features
            feature_cols (list, optional): List of feature columns to use

        Returns:
            tuple: (X, y)
        """
        if feature_cols is None:
            # Default feature set
            feature_cols = [
                # ELO features
                'elo_1', 'elo_2', 'elo_diff',
                'surface_elo_1', 'surface_elo_2', 'surface_elo_diff',

                # Ranking features
                'rank_diff', 'rank_1_log', 'rank_2_log',

                # Points features
                'points_diff',

                # Recent form
                'recent_win_pct_1', 'recent_win_pct_2', 'form_diff',

                # Head-to-head
                'h2h_wins_1', 'h2h_wins_2', 'h2h_total', 'h2h_win_pct_1',

                # Surface-specific
                'surface_win_pct_1', 'surface_win_pct_2',
                'surface_matches_1', 'surface_matches_2', 'surface_experience_diff',

                # Match importance
                'is_best_of_5', 'is_grand_slam', 'is_masters',

                # Betting odds (if available)
                'odds_implied_prob_1', 'odds_implied_prob_2', 'odds_diff'
            ]

            # Add surface one-hot encoding if available
            surface_cols = [col for col in df.columns if col.startswith('surface_') and col not in feature_cols]
            feature_cols.extend(surface_cols)

        # Filter to available columns
        available_cols = [col for col in feature_cols if col in df.columns]

        # CRITICAL FIX: Ensure proper DataFrame handling
        X = df[available_cols].copy()

        # Convert all columns to numeric
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = pd.to_numeric(X[col], errors='coerce')

        # Handle bad values
        X = X.fillna(0)
        X = X.replace([np.inf, -np.inf], 0)

        # Ensure we have a clean DataFrame (not nested)
        X = pd.DataFrame(X, columns=available_cols)

        y = df['target'].copy() if 'target' in df.columns else None

        self.feature_names = available_cols

        return X, y

    def train(self, df_train, df_val=None, verbose=True):
        """
        Train the model

        Args:
            df_train (pd.DataFrame): Training data
            df_val (pd.DataFrame, optional): Validation data
            verbose (bool): Print training progress

        Returns:
            dict: Training metrics
        """
        print("\n" + "=" * 60)
        print("TRAINING XGBOOST MODEL")
        print("=" * 60 + "\n")

        # Prepare features
        X_train, y_train = self.prepare_features(df_train)

        print(f"Training set: {len(X_train):,} matches")
        print(f"Features: {len(self.feature_names)}")

        if df_val is not None:
            X_val, y_val = self.prepare_features(df_val, self.feature_names)
            print(f"Validation set: {len(X_val):,} matches")

            eval_set = [(X_train, y_train), (X_val, y_val)]

            # Train with early stopping
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=verbose
            )

            # Evaluate on validation set
            val_pred = self.model.predict(X_val)
            val_accuracy = accuracy_score(y_val, val_pred)

            print(f"\nValidation Accuracy: {val_accuracy:.4f} ({val_accuracy * 100:.2f}%)")

            metrics = {
                'train_accuracy': accuracy_score(y_train, self.model.predict(X_train)),
                'val_accuracy': val_accuracy
            }
        else:
            # Train without validation
            self.model.fit(X_train, y_train, verbose=verbose)

            metrics = {
                'train_accuracy': accuracy_score(y_train, self.model.predict(X_train))
            }

            print(f"\nTraining Accuracy: {metrics['train_accuracy']:.4f}")

        # Get feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\nTop 10 Most Important Features:")
        for idx, row in self.feature_importance.head(10).iterrows():
            print(f"   {row['feature']}: {row['importance']:.4f}")

        print("\n" + "=" * 60)

        return metrics

    def predict(self, df, return_proba=False):
        """
        Make predictions

        Args:
            df (pd.DataFrame): Data to predict on
            return_proba (bool): Return probabilities instead of binary predictions

        Returns:
            np.array: Predictions
        """
        X, _ = self.prepare_features(df, self.feature_names)

        if return_proba:
            return self.model.predict_proba(X)[:, 1]  # Probability of Player_1 winning
        else:
            return self.model.predict(X)

    def evaluate(self, df, print_report=True):
        """
        Evaluate model performance

        Args:
            df (pd.DataFrame): Data to evaluate on
            print_report (bool): Print detailed classification report

        Returns:
            dict: Evaluation metrics
        """
        X, y = self.prepare_features(df, self.feature_names)

        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]

        accuracy = accuracy_score(y, predictions)

        if print_report:
            print("\n" + "=" * 60)
            print("MODEL EVALUATION")
            print("=" * 60 + "\n")

            print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
            print(f"Total Predictions: {len(predictions):,}")
            print(f"Correct Predictions: {(predictions == y).sum():,}")
            print(f"Wrong Predictions: {(predictions != y).sum():,}")

            print("\nClassification Report:")
            print(classification_report(y, predictions,
                                        target_names=['Player_2 Wins', 'Player_1 Wins']))

            print("\nConfusion Matrix:")
            cm = confusion_matrix(y, predictions)
            print(f"   True Negatives (P2 predicted & won): {cm[0, 0]}")
            print(f"   False Positives (P1 predicted, P2 won): {cm[0, 1]}")
            print(f"   False Negatives (P2 predicted, P1 won): {cm[1, 0]}")
            print(f"   True Positives (P1 predicted & won): {cm[1, 1]}")

            print("\n" + "=" * 60)

        metrics = {
            'accuracy': accuracy,
            'predictions': predictions,
            'probabilities': probabilities,
            'confusion_matrix': confusion_matrix(y, predictions)
        }

        return metrics

    def save_model(self, filepath):
        """
        Save model to file

        Args:
            filepath (str): Path to save model
        """
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_names': self.feature_names,
                'feature_importance': self.feature_importance
            }, f)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """
        Load model from file

        Args:
            filepath (str): Path to load model from
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.feature_names = data['feature_names']
            self.feature_importance = data['feature_importance']
        print(f"Model loaded from {filepath}")


if __name__ == "__main__":
    print("Tennis Prediction Model Module - Ready to use!")