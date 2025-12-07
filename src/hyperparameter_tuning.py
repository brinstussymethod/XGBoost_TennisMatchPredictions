"""
Hyperparameter Tuning for Tennis Prediction
Methods to optimize model performance
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, log_loss
import optuna


def grid_search_xgboost(X_train, y_train, X_val, y_val):
    """
    Grid Search for XGBoost hyperparameters
    Exhaustive search over specified parameter values
    """
    print("Running Grid Search...")

    param_grid = {
        'max_depth': [4, 6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'n_estimators': [100, 200, 300, 500],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2, 0.3],
        'reg_alpha': [0, 0.1, 0.5, 1.0],  # L1 regularization
        'reg_lambda': [0, 0.1, 0.5, 1.0]  # L2 regularization
    }

    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42
    )

    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=3,
        scoring='accuracy',
        verbose=2,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")

    # Validate on validation set
    best_model = grid_search.best_estimator_
    val_pred = best_model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_pred)
    print(f"Validation accuracy: {val_accuracy:.4f}")

    return grid_search.best_estimator_, grid_search.best_params_


def random_search_xgboost(X_train, y_train, X_val, y_val, n_iter=100):
    """
    Random Search for XGBoost hyperparameters
    Faster than grid search, samples random combinations
    """
    print("Running Random Search...")

    param_distributions = {
        'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
        'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2, 0.3],
        'n_estimators': [50, 100, 200, 300, 500, 700, 1000],
        'min_child_weight': [1, 2, 3, 4, 5],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'gamma': [0, 0.05, 0.1, 0.2, 0.3, 0.5],
        'reg_alpha': [0, 0.01, 0.1, 0.5, 1.0, 2.0],
        'reg_lambda': [0, 0.01, 0.1, 0.5, 1.0, 2.0]
    }

    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42
    )

    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=3,
        scoring='accuracy',
        verbose=2,
        n_jobs=-1,
        random_state=42
    )

    random_search.fit(X_train, y_train)

    print(f"\nBest parameters: {random_search.best_params_}")
    print(f"Best CV score: {random_search.best_score_:.4f}")

    # Validate
    best_model = random_search.best_estimator_
    val_pred = best_model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_pred)
    print(f"Validation accuracy: {val_accuracy:.4f}")

    return random_search.best_estimator_, random_search.best_params_


def bayesian_optimization_optuna(X_train, y_train, X_val, y_val, n_trials=100):
    """
    Bayesian Optimization using Optuna
    Most efficient - learns from previous trials
    RECOMMENDED METHOD
    """
    print("Running Bayesian Optimization with Optuna...")

    def objective(trial):
        # Define hyperparameter search space
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 0.5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 2.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 2.0),
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'random_state': 42
        }

        # Train model
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, verbose=False)

        # Evaluate on validation set
        val_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, val_pred)

        return accuracy

    # Create study and optimize
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nBest parameters: {study.best_params}")
    print(f"Best accuracy: {study.best_value:.4f}")

    # Train final model with best parameters
    best_params = study.best_params
    best_params['objective'] = 'binary:logistic'
    best_params['eval_metric'] = 'logloss'
    best_params['random_state'] = 42

    best_model = xgb.XGBClassifier(**best_params)
    best_model.fit(X_train, y_train)

    return best_model, study.best_params


def quick_parameter_boost(X_train, y_train, X_val, y_val):
    """
    Quick parameter improvements based on research
    These parameters typically improve sports prediction
    """
    print("Training with optimized parameters...")

    # Based on research in sports prediction
    optimized_params = {
        'max_depth': 8,              # Deeper trees for complex interactions
        'learning_rate': 0.05,       # Slower learning, better generalization
        'n_estimators': 500,         # More trees (YES, increase this!)
        'min_child_weight': 3,       # Reduce overfitting
        'subsample': 0.8,            # Row sampling
        'colsample_bytree': 0.8,     # Column sampling
        'colsample_bylevel': 0.8,    # Column sampling by tree level
        'gamma': 0.1,                # Minimum loss reduction
        'reg_alpha': 0.1,            # L1 regularization
        'reg_lambda': 1.0,           # L2 regularization
        'scale_pos_weight': 1,       # Balance positive/negative classes
        'objective': 'binary:logistic',
        'eval_metric': ['logloss', 'auc'],
        'random_state': 42
    }

    model = xgb.XGBClassifier(**optimized_params)

    # Train with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=100
    )

    # Evaluate
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)

    print(f"\nTrain accuracy: {train_acc:.4f}")
    print(f"Validation accuracy: {val_acc:.4f}")

    return model, optimized_params


if __name__ == "__main__":
    print("Hyperparameter Tuning Module - Ready to use!")
    print("\nRecommended approach:")
    print("1. Start with quick_parameter_boost() - Fast, good results")
    print("2. If time permits, use bayesian_optimization_optuna() - Best results")
    print("3. Avoid grid_search - Too slow for large parameter spaces")
