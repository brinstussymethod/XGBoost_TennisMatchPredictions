"""
Ensemble Methods for Tennis Prediction
Combining multiple models for better predictions
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier,
    AdaBoostClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


class TennisEnsemble:
    """Ensemble of multiple models for tennis prediction"""

    def __init__(self):
        self.models = {}
        self.ensemble = None

    def train_individual_models(self, X_train, y_train, X_val, y_val):
        """
        Train multiple different algorithms
        Each captures different patterns
        """
        print("Training individual models...\n")

        # 1. XGBoost (your current model)
        print("1. XGBoost...")
        xgb_model = xgb.XGBClassifier(
            max_depth=8,
            learning_rate=0.05,
            n_estimators=300,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        xgb_model.fit(X_train, y_train)
        xgb_acc = accuracy_score(y_val, xgb_model.predict(X_val))
        print(f"   Validation accuracy: {xgb_acc:.4f}")
        self.models['xgboost'] = xgb_model

        # 2. LightGBM (faster, often better than XGBoost)
        print("\n2. LightGBM...")
        lgbm_model = LGBMClassifier(
            max_depth=8,
            learning_rate=0.05,
            n_estimators=300,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
        lgbm_model.fit(X_train, y_train)
        lgbm_acc = accuracy_score(y_val, lgbm_model.predict(X_val))
        print(f"   Validation accuracy: {lgbm_acc:.4f}")
        self.models['lightgbm'] = lgbm_model

        # 3. CatBoost (handles categorical features well)
        print("\n3. CatBoost...")
        catboost_model = CatBoostClassifier(
            depth=8,
            learning_rate=0.05,
            iterations=300,
            random_state=42,
            verbose=False
        )
        catboost_model.fit(X_train, y_train)
        catboost_acc = accuracy_score(y_val, catboost_model.predict(X_val))
        print(f"   Validation accuracy: {catboost_acc:.4f}")
        self.models['catboost'] = catboost_model

        # 4. Random Forest
        print("\n4. Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        rf_acc = accuracy_score(y_val, rf_model.predict(X_val))
        print(f"   Validation accuracy: {rf_acc:.4f}")
        self.models['random_forest'] = rf_model

        # 5. Gradient Boosting
        print("\n5. Gradient Boosting...")
        gb_model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        gb_model.fit(X_train, y_train)
        gb_acc = accuracy_score(y_val, gb_model.predict(X_val))
        print(f"   Validation accuracy: {gb_acc:.4f}")
        self.models['gradient_boosting'] = gb_model

        # 6. Neural Network
        print("\n6. Neural Network...")
        nn_model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            learning_rate_init=0.001,
            max_iter=500,
            random_state=42
        )
        nn_model.fit(X_train, y_train)
        nn_acc = accuracy_score(y_val, nn_model.predict(X_val))
        print(f"   Validation accuracy: {nn_acc:.4f}")
        self.models['neural_network'] = nn_model

        print("\n" + "="*60)
        print("Individual Model Summary:")
        results = {
            'XGBoost': xgb_acc,
            'LightGBM': lgbm_acc,
            'CatBoost': catboost_acc,
            'Random Forest': rf_acc,
            'Gradient Boosting': gb_acc,
            'Neural Network': nn_acc
        }
        for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
            print(f"   {name}: {acc:.4f}")

        return self.models

    def voting_ensemble(self, X_train, y_train, X_val, y_val, voting='soft'):
        """
        Voting Ensemble - Each model votes, majority wins
        voting='soft' uses probability averaging (better)
        voting='hard' uses majority vote
        """
        print(f"\nTraining Voting Ensemble ({voting})...")

        # Select best performing models
        estimators = [
            ('xgb', self.models['xgboost']),
            ('lgbm', self.models['lightgbm']),
            ('catboost', self.models['catboost']),
            ('rf', self.models['random_forest'])
        ]

        voting_clf = VotingClassifier(
            estimators=estimators,
            voting=voting,
            n_jobs=-1
        )

        voting_clf.fit(X_train, y_train)

        val_pred = voting_clf.predict(X_val)
        val_acc = accuracy_score(y_val, val_pred)

        print(f"Voting Ensemble accuracy: {val_acc:.4f}")

        return voting_clf

    def stacking_ensemble(self, X_train, y_train, X_val, y_val):
        """
        Stacking Ensemble - Use models as features for meta-model
        Often gives best results
        RECOMMENDED
        """
        print("\nTraining Stacking Ensemble...")

        # Base models (level 0)
        base_models = [
            ('xgb', self.models['xgboost']),
            ('lgbm', self.models['lightgbm']),
            ('catboost', self.models['catboost']),
            ('rf', self.models['random_forest']),
            ('gb', self.models['gradient_boosting'])
        ]

        # Meta-model (level 1) - learns how to combine base models
        meta_model = LogisticRegression(max_iter=1000)

        stacking_clf = StackingClassifier(
            estimators=base_models,
            final_estimator=meta_model,
            cv=5,
            n_jobs=-1
        )

        stacking_clf.fit(X_train, y_train)

        val_pred = stacking_clf.predict(X_val)
        val_acc = accuracy_score(y_val, val_pred)

        print(f"Stacking Ensemble accuracy: {val_acc:.4f}")

        return stacking_clf

    def weighted_average_ensemble(self, X_val, y_val):
        """
        Weighted Average - Give more weight to better models
        Simple but effective
        """
        print("\nCreating Weighted Average Ensemble...")

        # Get predictions from each model
        predictions = {}
        for name, model in self.models.items():
            pred_proba = model.predict_proba(X_val)[:, 1]
            predictions[name] = pred_proba

        # Calculate weights based on validation accuracy
        weights = {}
        for name, model in self.models.items():
            val_pred = model.predict(X_val)
            acc = accuracy_score(y_val, val_pred)
            weights[name] = acc

        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}

        print("Model weights:")
        for name, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
            print(f"   {name}: {weight:.4f}")

        # Weighted average of predictions
        weighted_pred_proba = np.zeros(len(X_val))
        for name, pred_proba in predictions.items():
            weighted_pred_proba += weights[name] * pred_proba

        weighted_pred = (weighted_pred_proba >= 0.5).astype(int)
        weighted_acc = accuracy_score(y_val, weighted_pred)

        print(f"\nWeighted Average accuracy: {weighted_acc:.4f}")

        return weighted_pred_proba, weights

    def predict_ensemble(self, X, weights=None):
        """
        Make predictions using weighted ensemble
        """
        if weights is None:
            # Equal weights
            weights = {name: 1.0/len(self.models) for name in self.models.keys()}

        # Weighted average
        pred_proba = np.zeros(len(X))
        for name, model in self.models.items():
            pred_proba += weights[name] * model.predict_proba(X)[:, 1]

        return pred_proba


def train_full_ensemble(X_train, y_train, X_val, y_val):
    """
    Complete ensemble training pipeline
    MAIN FUNCTION TO USE
    """
    print("="*60)
    print("TRAINING FULL ENSEMBLE")
    print("="*60)

    ensemble = TennisEnsemble()

    # Step 1: Train individual models
    ensemble.train_individual_models(X_train, y_train, X_val, y_val)

    # Step 2: Try different ensemble methods
    print("\n" + "="*60)
    print("ENSEMBLE COMBINATIONS")
    print("="*60)

    # Voting
    voting_clf = ensemble.voting_ensemble(X_train, y_train, X_val, y_val)

    # Stacking (usually best)
    stacking_clf = ensemble.stacking_ensemble(X_train, y_train, X_val, y_val)

    # Weighted average
    weighted_pred, weights = ensemble.weighted_average_ensemble(X_val, y_val)

    print("\n" + "="*60)
    print("RECOMMENDATION:")
    print("Use Stacking Ensemble for best accuracy")
    print("Expected improvement: +2-4% over single model")
    print("="*60)

    return ensemble, stacking_clf


if __name__ == "__main__":
    print("Ensemble Methods Module - Ready to use!")
    print("\nKey insight: Ensemble models typically outperform single models")
    print("Expected accuracy gain: 2-5%")
