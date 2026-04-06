# ============================================================
# utils/helpers.py
# Utility helper functions used across the project
# ============================================================

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime


def ensure_directories():
    """Create all required project directories if they don't exist."""
    dirs = ['data', 'models', 'plots', 'results']
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("[INFO] Project directories verified.")


def save_results_csv(df: pd.DataFrame, filename: str,
                     results_dir: str = "results") -> str:
    """Save a DataFrame as CSV to results directory."""
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, filename)
    df.to_csv(path, index=True)
    print(f"[INFO] Saved results: {path}")
    return path


def format_number(n, decimals: int = 2) -> str:
    """Format a number with commas and specified decimal places."""
    if n is None:
        return "N/A"
    if abs(n) >= 1e9:
        return f"{n/1e9:.{decimals}f}B"
    if abs(n) >= 1e6:
        return f"{n/1e6:.{decimals}f}M"
    if abs(n) >= 1e3:
        return f"{n/1e3:.{decimals}f}K"
    return f"{n:.{decimals}f}"


def timestamp_str() -> str:
    """Return current timestamp string for file naming."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def validate_ticker(ticker: str) -> bool:
    """Basic validation of stock ticker format."""
    if not ticker or len(ticker) > 20:
        return False
    # Allow letters, digits, dots, hyphens (for international tickers)
    import re
    return bool(re.match(r'^[A-Za-z0-9.\-\^]+$', ticker))


def get_market_phase(df: pd.DataFrame) -> str:
    """
    Determine overall market trend (Bullish / Bearish / Sideways)
    based on SMA crossover.
    """
    if 'SMA_20' not in df.columns or 'SMA_50' not in df.columns:
        return "Unknown"

    latest = df.iloc[-1]
    if latest['SMA_20'] > latest['SMA_50'] and latest['Close'] > latest['SMA_20']:
        return "🟢 Bullish"
    elif latest['SMA_20'] < latest['SMA_50'] and latest['Close'] < latest['SMA_20']:
        return "🔴 Bearish"
    else:
        return "🟡 Sideways"


def compute_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.04) -> float:
    """
    Compute annualized Sharpe Ratio for a return series.
    Default risk-free rate = 4% annual
    """
    daily_rf = risk_free_rate / 252
    excess = returns - daily_rf
    if excess.std() == 0:
        return 0.0
    sharpe = (excess.mean() / excess.std()) * np.sqrt(252)
    return round(sharpe, 3)


def compute_max_drawdown(prices: pd.Series) -> float:
    """Compute maximum drawdown percentage."""
    peak = prices.cummax()
    drawdown = (prices - peak) / peak
    return round(drawdown.min() * 100, 2)


def describe_model_type(model_name: str) -> str:
    """Return a short description for each model type."""
    descriptions = {
        "Logistic Regression": "Linear classifier using sigmoid function. Fast, interpretable baseline.",
        "Decision Tree": "Tree-based rules. Fully interpretable but can overfit.",
        "Random Forest": "Ensemble of decision trees with bagging. Robust and handles noise well.",
        "K-Nearest Neighbors": "Instance-based learning. Predicts based on k-nearest neighbors.",
        "Support Vector Machine": "Finds optimal hyperplane with maximum margin. Effective in high dimensions.",
        "Naive Bayes": "Probabilistic classifier based on Bayes theorem. Very fast training.",
        "AdaBoost": "Adaptive boosting. Sequentially corrects errors of weak learners.",
        "Gradient Boosting": "Builds trees sequentially, each correcting the previous. High accuracy.",
        "XGBoost": "Optimized gradient boosting with regularization. Often best performer.",
        "Linear Regression": "Models linear relationship between features and target price.",
        "Decision Tree Regressor": "Tree-based rules for continuous price prediction.",
        "Random Forest Regressor": "Ensemble regressor averaging many decision trees.",
        "KNN Regressor": "Predicts price as average of k-nearest neighbor prices.",
        "Support Vector Regressor": "SVR with epsilon-insensitive tube for price regression.",
        "AdaBoost Regressor": "Boosted ensemble for price prediction.",
        "Gradient Boosting Regressor": "Sequential gradient-boosted trees for price.",
        "XGBoost Regressor": "High-performance XGBoost for continuous price prediction.",
    }
    return descriptions.get(model_name, "ML model for stock prediction.")