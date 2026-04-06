# ============================================================
# evaluation.py
# Model evaluation, metrics computation, and best model selection
# ============================================================

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)


# ============================================================
# CLASSIFICATION EVALUATION
# ============================================================

def evaluate_classification_models(results: dict, y_test) -> pd.DataFrame:
    """
    Compute all classification metrics for each model.

    Metrics: Accuracy, Precision, Recall, F1-Score, Training Time

    Args:
        results: dict from train_all_classification_models()
        y_test: True labels

    Returns:
        DataFrame with metrics for each model, sorted by F1-Score
    """
    metrics_list = []

    for name, result in results.items():
        y_pred = result['y_pred']

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        t_time = result['training_time']

        metrics_list.append({
            'Model': name,
            'Accuracy': round(acc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1-Score': round(f1, 4),
            'Training Time (s)': round(t_time, 3),
        })

    df = pd.DataFrame(metrics_list)
    df = df.sort_values('F1-Score', ascending=False).reset_index(drop=True)
    df.index = df.index + 1  # Rank starts at 1

    return df


def get_confusion_matrix(results: dict, y_test, model_name: str) -> np.ndarray:
    """Get confusion matrix for a specific model."""
    if model_name not in results:
        raise ValueError(f"Model '{model_name}' not found in results.")
    y_pred = results[model_name]['y_pred']
    return confusion_matrix(y_test, y_pred)


def get_classification_report(results: dict, y_test, model_name: str) -> str:
    """Get sklearn classification report string for a model."""
    y_pred = results[model_name]['y_pred']
    return classification_report(y_test, y_pred, target_names=['DOWN (0)', 'UP (1)'])


# ============================================================
# REGRESSION EVALUATION
# ============================================================

def evaluate_regression_models(results: dict, y_test) -> pd.DataFrame:
    """
    Compute all regression metrics for each model.

    Metrics: MAE, MSE, RMSE, R² Score, Training Time

    Args:
        results: dict from train_all_regression_models()
        y_test: True price values

    Returns:
        DataFrame with metrics for each model, sorted by R²
    """
    metrics_list = []

    for name, result in results.items():
        y_pred = result['y_pred']

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        t_time = result['training_time']

        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_test - y_pred) / (np.abs(y_test) + 1e-9))) * 100

        metrics_list.append({
            'Model': name,
            'MAE': round(mae, 4),
            'MSE': round(mse, 4),
            'RMSE': round(rmse, 4),
            'MAPE (%)': round(mape, 4),
            'R² Score': round(r2, 4),
            'Training Time (s)': round(t_time, 3),
        })

    df = pd.DataFrame(metrics_list)
    df = df.sort_values('R² Score', ascending=False).reset_index(drop=True)
    df.index = df.index + 1  # Rank starts at 1

    return df


# ============================================================
# BEST MODEL SELECTION
# ============================================================

def select_best_classification_model(metrics_df: pd.DataFrame,
                                      results: dict) -> dict:
    """
    Automatically select the best classification model.

    Selection Criteria:
    - Primary: Highest F1-Score (balances precision & recall)
    - Secondary: Highest Accuracy (tiebreaker)

    Args:
        metrics_df: Output of evaluate_classification_models()
        results: Raw model results dict

    Returns:
        dict with best model info and reasoning
    """
    # Best = first row (already sorted by F1-Score desc)
    best_row = metrics_df.iloc[0]
    best_name = best_row['Model']

    # Generate reasoning
    f1 = best_row['F1-Score']
    acc = best_row['Accuracy']
    prec = best_row['Precision']
    rec = best_row['Recall']

    reasoning = (
        f"**{best_name}** was selected as the best classification model based on:\n\n"
        f"- 🏆 **Highest F1-Score**: {f1:.4f} — F1 balances precision and recall, making it "
        f"robust for imbalanced UP/DOWN distributions in stock movements.\n"
        f"- 📊 **Accuracy**: {acc:.4f} — Overall correct predictions on test set.\n"
        f"- 🎯 **Precision**: {prec:.4f} — Of predicted UP signals, {prec*100:.1f}% were correct.\n"
        f"- 🔍 **Recall**: {rec:.4f} — Detected {rec*100:.1f}% of actual UP movements."
    )

    return {
        'name': best_name,
        'model': results[best_name]['model'],
        'metrics': best_row.to_dict(),
        'reasoning': reasoning,
        'rank': 1,
    }


def select_best_regression_model(metrics_df: pd.DataFrame,
                                   results: dict) -> dict:
    """
    Automatically select the best regression model.

    Selection Criteria:
    - Primary: Highest R² Score (variance explained)
    - Secondary: Lowest RMSE (tiebreaker)

    Args:
        metrics_df: Output of evaluate_regression_models()
        results: Raw model results dict

    Returns:
        dict with best model info and reasoning
    """
    best_row = metrics_df.iloc[0]
    best_name = best_row['Model']

    r2 = best_row['R² Score']
    rmse = best_row['RMSE']
    mae = best_row['MAE']
    mape = best_row['MAPE (%)']

    reasoning = (
        f"**{best_name}** was selected as the best regression model based on:\n\n"
        f"- 🏆 **Highest R² Score**: {r2:.4f} — Explains {r2*100:.1f}% of variance in stock price.\n"
        f"- 📉 **RMSE**: {rmse:.4f} — Average prediction error in price units.\n"
        f"- 📊 **MAE**: {mae:.4f} — Mean absolute error in price units.\n"
        f"- 📈 **MAPE**: {mape:.2f}% — Mean absolute percentage error from true price."
    )

    return {
        'name': best_name,
        'model': results[best_name]['model'],
        'metrics': best_row.to_dict(),
        'reasoning': reasoning,
        'predictions': results[best_name]['y_pred'],
        'rank': 1,
    }


# ============================================================
# PREDICTION ON LATEST DATA
# ============================================================

def predict_next_day(best_clf_model, best_reg_model,
                     clf_scaler, reg_scaler,
                     latest_features: np.ndarray,
                     best_clf_name: str, best_reg_name: str) -> dict:
    """
    Make next-day prediction using the best models.

    Args:
        best_clf_model: Best classification model
        best_reg_model: Best regression model
        clf_scaler: Scaler for classification features
        reg_scaler: Scaler for regression features
        latest_features: Most recent feature row (1D array)
        best_clf_name: Name of best clf model
        best_reg_name: Name of best reg model

    Returns:
        dict with price prediction, direction, and confidence
    """
    # Reshape to 2D
    X = latest_features.reshape(1, -1)

    # Scale
    if clf_scaler:
        X_clf = clf_scaler.transform(X)
    else:
        X_clf = X

    if reg_scaler:
        X_reg = reg_scaler.transform(X)
    else:
        X_reg = X

    # Classification prediction
    direction_pred = best_clf_model.predict(X_clf)[0]
    direction_label = "📈 UP" if direction_pred == 1 else "📉 DOWN"

    # Confidence from probability
    if hasattr(best_clf_model, 'predict_proba'):
        proba = best_clf_model.predict_proba(X_clf)[0]
        confidence = max(proba) * 100
    else:
        confidence = None

    # Regression prediction (price)
    price_pred = best_reg_model.predict(X_reg)[0]

    return {
        'predicted_price': round(float(price_pred), 2),
        'predicted_direction': direction_label,
        'direction_value': int(direction_pred),
        'confidence': round(confidence, 2) if confidence else None,
        'clf_model': best_clf_name,
        'reg_model': best_reg_name,
    }


def summarize_results(clf_metrics: pd.DataFrame,
                      reg_metrics: pd.DataFrame,
                      best_clf: dict, best_reg: dict) -> str:
    """Generate a text summary of all results."""
    summary = []
    summary.append("=" * 60)
    summary.append("CLASSIFICATION RESULTS SUMMARY")
    summary.append("=" * 60)
    summary.append(clf_metrics.to_string())
    summary.append(f"\nBEST MODEL: {best_clf['name']}")
    summary.append("")
    summary.append("=" * 60)
    summary.append("REGRESSION RESULTS SUMMARY")
    summary.append("=" * 60)
    summary.append(reg_metrics.to_string())
    summary.append(f"\nBEST MODEL: {best_reg['name']}")
    return "\n".join(summary)