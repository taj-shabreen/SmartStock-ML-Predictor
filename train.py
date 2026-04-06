# ============================================================
# train.py
# Model definitions, training, and saving pipeline
# ============================================================

import numpy as np
import pandas as pd
import time
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor


# ============================================================
# MODEL DEFINITIONS
# ============================================================

def get_classification_models() -> dict:
    """
    Returns a dictionary of all classification models with default hyperparameters.
    Tuned for speed + performance balance.
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=500, random_state=42, C=1.0, solver='lbfgs', n_jobs=-1
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=80, max_depth=7, random_state=42, n_jobs=-1
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=7, metric='euclidean', n_jobs=-1, algorithm='ball_tree'
        ),
        "Support Vector Machine": SVC(
            kernel='rbf', C=1.0, probability=True, random_state=42,
            cache_size=500, max_iter=500
        ),
        "Naive Bayes": GaussianNB(),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=60, learning_rate=0.1, random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=80, learning_rate=0.1, max_depth=4,
            random_state=42, subsample=0.8
        ),
        "XGBoost": XGBClassifier(
            n_estimators=80, learning_rate=0.1, max_depth=4,
            use_label_encoder=False, eval_metric='logloss',
            random_state=42, verbosity=0, n_jobs=-1,
            tree_method='hist'
        ),
    }
    return models


def get_regression_models() -> dict:
    """
    Returns a dictionary of all regression models with default hyperparameters.
    Tuned for speed + performance balance.
    """
    models = {
        "Linear Regression": LinearRegression(n_jobs=-1),
        "Decision Tree Regressor": DecisionTreeRegressor(
            max_depth=5, random_state=42
        ),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=80, max_depth=7, random_state=42, n_jobs=-1
        ),
        "KNN Regressor": KNeighborsRegressor(
            n_neighbors=7, metric='euclidean', n_jobs=-1, algorithm='ball_tree'
        ),
        "Support Vector Regressor": SVR(
            kernel='rbf', C=1.0, epsilon=0.1, cache_size=500, max_iter=500
        ),
        "AdaBoost Regressor": AdaBoostRegressor(
            n_estimators=60, learning_rate=0.1, random_state=42
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=80, learning_rate=0.1, max_depth=4,
            random_state=42, subsample=0.8
        ),
        "XGBoost Regressor": XGBRegressor(
            n_estimators=80, learning_rate=0.1, max_depth=4,
            random_state=42, verbosity=0, n_jobs=-1,
            tree_method='hist'
        ),
    }
    return models


# ============================================================
# TRAINING PIPELINE
# ============================================================

def prepare_train_test_split(X: pd.DataFrame, y: pd.Series,
                              test_size: float = 0.2,
                              scale: bool = True) -> tuple:
    """
    Split data into training and test sets with optional scaling.
    Uses time-series aware splitting (no shuffle).

    Args:
        X: Feature DataFrame
        y: Target Series
        test_size: Fraction of data for testing
        scale: Whether to apply StandardScaler

    Returns:
        X_train, X_test, y_train, y_test, scaler (or None)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler


def train_all_classification_models(X_train, X_test, y_train, y_test,
                                     feature_names: list) -> dict:
    """
    Train ALL classification models and return results.

    Returns:
        dict with model name → {model, predictions, probabilities, training_time}
    """
    models = get_classification_models()
    results = {}

    print("\n" + "="*60)
    print("TRAINING CLASSIFICATION MODELS")
    print("="*60)

    for name, model in models.items():
        print(f"  Training: {name}...", end=" ")
        start_time = time.time()

        try:
            model.fit(X_train, y_train)
            training_time = time.time() - start_time

            y_pred = model.predict(X_test)

            # Get prediction probabilities if available
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test)[:, 1]
            else:
                y_prob = None

            results[name] = {
                'model': model,
                'y_pred': y_pred,
                'y_prob': y_prob,
                'training_time': training_time,
                'feature_names': feature_names,
            }

            print(f"Done in {training_time:.2f}s")

        except Exception as e:
            print(f"FAILED: {str(e)}")
            continue

    return results


def train_all_regression_models(X_train, X_test, y_train, y_test,
                                  feature_names: list) -> dict:
    """
    Train ALL regression models and return results.

    Returns:
        dict with model name → {model, predictions, training_time}
    """
    models = get_regression_models()
    results = {}

    print("\n" + "="*60)
    print("TRAINING REGRESSION MODELS")
    print("="*60)

    for name, model in models.items():
        print(f"  Training: {name}...", end=" ")
        start_time = time.time()

        try:
            model.fit(X_train, y_train)
            training_time = time.time() - start_time

            y_pred = model.predict(X_test)

            results[name] = {
                'model': model,
                'y_pred': y_pred,
                'training_time': training_time,
                'feature_names': feature_names,
            }

            print(f"Done in {training_time:.2f}s")

        except Exception as e:
            print(f"FAILED: {str(e)}")
            continue

    return results


# ============================================================
# MODEL SAVING / LOADING
# ============================================================

def save_models(clf_results: dict, reg_results: dict,
                clf_scaler, reg_scaler,
                ticker: str, save_dir: str = "models") -> dict:
    """
    Save trained models, scalers and metadata to disk.

    Args:
        clf_results: Classification model results dict
        reg_results: Regression model results dict
        clf_scaler: Scaler used for classification
        reg_scaler: Scaler used for regression
        ticker: Stock ticker symbol
        save_dir: Directory to save models

    Returns:
        dict with saved file paths
    """
    os.makedirs(save_dir, exist_ok=True)
    saved_paths = {}
    ticker_clean = ticker.replace('.', '_').upper()

    # Save classification models
    for name, result in clf_results.items():
        filename = f"{ticker_clean}_clf_{name.replace(' ', '_').lower()}.pkl"
        path = os.path.join(save_dir, filename)
        joblib.dump(result['model'], path)
        saved_paths[f'clf_{name}'] = path

    # Save regression models
    for name, result in reg_results.items():
        filename = f"{ticker_clean}_reg_{name.replace(' ', '_').lower()}.pkl"
        path = os.path.join(save_dir, filename)
        joblib.dump(result['model'], path)
        saved_paths[f'reg_{name}'] = path

    # Save scalers
    if clf_scaler:
        path = os.path.join(save_dir, f"{ticker_clean}_clf_scaler.pkl")
        joblib.dump(clf_scaler, path)
        saved_paths['clf_scaler'] = path

    if reg_scaler:
        path = os.path.join(save_dir, f"{ticker_clean}_reg_scaler.pkl")
        joblib.dump(reg_scaler, path)
        saved_paths['reg_scaler'] = path

    print(f"\n[INFO] Saved {len(saved_paths)} model files to '{save_dir}/'")
    return saved_paths


def load_model(model_path: str):
    """Load a saved model from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


def get_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """
    Extract feature importance from tree-based models.

    Args:
        model: Trained sklearn model
        feature_names: List of feature column names

    Returns:
        DataFrame with feature importance scores sorted descending
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        return df
    elif hasattr(model, 'coef_'):
        # Linear models
        coefs = np.abs(model.coef_).flatten()
        if len(coefs) == len(feature_names):
            df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': coefs
            }).sort_values('Importance', ascending=False)
            return df
    return None