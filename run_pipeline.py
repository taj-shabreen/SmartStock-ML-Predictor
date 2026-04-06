#!/usr/bin/env python3
# ============================================================
# run_pipeline.py
# Command-line pipeline runner (no UI)
# Usage: python run_pipeline.py --ticker AAPL --start 2020-01-01 --end 2024-01-01
# ============================================================

import argparse
import sys
import os
import warnings
warnings.filterwarnings('ignore')

from utils.data_fetcher import fetch_stock_data, preprocess_data, get_stock_info
from train import (
    prepare_train_test_split,
    train_all_classification_models,
    train_all_regression_models,
    save_models,
    get_feature_importance,
)
from evaluation import (
    evaluate_classification_models,
    evaluate_regression_models,
    select_best_classification_model,
    select_best_regression_model,
    predict_next_day,
    summarize_results,
)
from utils.visualizer import (
    plot_stock_price_trend,
    plot_classification_comparison,
    plot_confusion_matrix,
    plot_regression_predictions,
    plot_regression_comparison,
    plot_feature_importance,
    plot_technical_indicators,
    plot_training_time_comparison,
    save_plot,
)
from utils.helpers import ensure_directories
from evaluation import get_confusion_matrix


def run_full_pipeline(ticker: str, start: str, end: str,
                      test_size: float = 0.2,
                      save: bool = True) -> dict:
    """
    Full end-to-end ML pipeline.

    Args:
        ticker: Stock symbol
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        test_size: Fraction of data for testing
        save: Whether to save models and plots

    Returns:
        dict with all results
    """
    ensure_directories()

    print("\n" + "="*70)
    print(f"  STOCK MARKET ML PIPELINE — {ticker}")
    print("="*70)

    # ---- STEP 1: Fetch & Preprocess ----
    print(f"\n[STEP 1] Fetching data for {ticker}...")
    raw_df = fetch_stock_data(ticker, start, end)
    info = get_stock_info(ticker)
    print(f"         Company: {info.get('name', ticker)}")
    print(f"         Sector:  {info.get('sector', 'N/A')}")

    processed_df, feature_cols, X, y_clf, y_reg = preprocess_data(raw_df)

    # ---- STEP 2: Split Data ----
    print(f"\n[STEP 2] Splitting data (test={test_size*100:.0f}%)...")
    X_tr_c, X_te_c, y_tr_c, y_te_c, clf_scaler = prepare_train_test_split(
        X, y_clf, test_size=test_size, scale=True
    )
    X_tr_r, X_te_r, y_tr_r, y_te_r, reg_scaler = prepare_train_test_split(
        X, y_reg, test_size=test_size, scale=True
    )
    print(f"         Train samples: {len(X_tr_c)} | Test samples: {len(X_te_c)}")

    # ---- STEP 3: Train ----
    print("\n[STEP 3] Training Classification Models...")
    clf_results = train_all_classification_models(
        X_tr_c, X_te_c, y_tr_c, y_te_c, feature_cols
    )

    print("\n[STEP 4] Training Regression Models...")
    reg_results = train_all_regression_models(
        X_tr_r, X_te_r, y_tr_r, y_te_r, feature_cols
    )

    # ---- STEP 4: Evaluate ----
    print("\n[STEP 5] Evaluating Models...")
    clf_metrics = evaluate_classification_models(clf_results, y_te_c)
    reg_metrics = evaluate_regression_models(reg_results, y_te_r)

    best_clf = select_best_classification_model(clf_metrics, clf_results)
    best_reg = select_best_regression_model(reg_metrics, reg_results)

    # ---- Print Results ----
    print("\n" + "="*70)
    print("CLASSIFICATION RESULTS")
    print("="*70)
    print(clf_metrics.to_string())
    print(f"\n🏆 Best Classifier: {best_clf['name']}")
    print(best_clf['reasoning'])

    print("\n" + "="*70)
    print("REGRESSION RESULTS")
    print("="*70)
    print(reg_metrics.to_string())
    print(f"\n🏆 Best Regressor: {best_reg['name']}")
    print(best_reg['reasoning'])

    # ---- STEP 5: Predict ----
    print("\n[STEP 6] Generating Next-Day Prediction...")
    X_latest = X.iloc[-1].values
    prediction = predict_next_day(
        best_clf['model'], best_reg['model'],
        clf_scaler, reg_scaler,
        X_latest,
        best_clf['name'], best_reg['name'],
    )
    current_price = raw_df['Close'].iloc[-1]

    print(f"\n{'='*70}")
    print("📊 NEXT TRADING DAY PREDICTION")
    print(f"{'='*70}")
    print(f"  Current Price:     ${current_price:.2f}")
    print(f"  Predicted Price:   ${prediction['predicted_price']:.2f}")
    change = prediction['predicted_price'] - current_price
    print(f"  Expected Change:   {'+' if change >= 0 else ''}{change:.2f} ({change/current_price*100:+.2f}%)")
    print(f"  Direction:         {prediction['predicted_direction']}")
    if prediction['confidence']:
        print(f"  Confidence:        {prediction['confidence']:.1f}%")
    print(f"  Classification By: {prediction['clf_model']}")
    print(f"  Regression By:     {prediction['reg_model']}")

    # ---- STEP 6: Save & Plot ----
    if save:
        print("\n[STEP 7] Generating & Saving Visualizations...")

        fig1 = plot_stock_price_trend(processed_df, ticker)
        save_plot(fig1, f"{ticker}_price_trend.png")

        fig2 = plot_technical_indicators(processed_df, ticker)
        save_plot(fig2, f"{ticker}_technical_indicators.png")

        fig3 = plot_classification_comparison(clf_metrics, best_clf['name'])
        save_plot(fig3, f"{ticker}_classification_comparison.png")

        cm = get_confusion_matrix(clf_results, y_te_c, best_clf['name'])
        fig4 = plot_confusion_matrix(cm, best_clf['name'])
        save_plot(fig4, f"{ticker}_confusion_matrix.png")

        pred_dict = {n: r['y_pred'] for n, r in reg_results.items()}
        fig5 = plot_regression_predictions(y_te_r, pred_dict, best_reg['name'], ticker)
        save_plot(fig5, f"{ticker}_regression_predictions.png")

        fig6 = plot_regression_comparison(reg_metrics, best_reg['name'])
        save_plot(fig6, f"{ticker}_regression_comparison.png")

        fi_df = get_feature_importance(best_clf['model'], feature_cols)
        if fi_df is not None:
            fig7 = plot_feature_importance(fi_df, best_clf['name'])
            save_plot(fig7, f"{ticker}_feature_importance.png")

        fig8 = plot_training_time_comparison(clf_metrics, reg_metrics)
        save_plot(fig8, f"{ticker}_training_time.png")

        print("  Saved all plots to plots/")

        print("\n[STEP 8] Saving Models...")
        saved_paths = save_models(
            clf_results, reg_results,
            clf_scaler, reg_scaler,
            ticker
        )

        print("\n[STEP 9] Saving Results CSVs...")
        from utils.helpers import save_results_csv
        save_results_csv(clf_metrics, f"{ticker}_clf_metrics.csv")
        save_results_csv(reg_metrics, f"{ticker}_reg_metrics.csv")

    print(f"\n{'='*70}")
    print("✅ PIPELINE COMPLETE")
    print(f"{'='*70}\n")

    return {
        'raw_df': raw_df,
        'clf_metrics': clf_metrics,
        'reg_metrics': reg_metrics,
        'best_clf': best_clf,
        'best_reg': best_reg,
        'prediction': prediction,
    }


# ============================================================
# CLI Entry Point
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stock Market ML Pipeline (CLI mode)"
    )
    parser.add_argument("--ticker", type=str, default="AAPL",
                        help="Stock ticker symbol (e.g., AAPL, TSLA, INFY.NS)")
    parser.add_argument("--start", type=str, default="2020-01-01",
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2024-01-01",
                        help="End date (YYYY-MM-DD)")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Test set fraction (0.0 - 0.5, default 0.2)")
    parser.add_argument("--no-save", action="store_true",
                        help="Skip saving models/plots")

    args = parser.parse_args()

    results = run_full_pipeline(
        ticker=args.ticker.upper(),
        start=args.start,
        end=args.end,
        test_size=args.test_size,
        save=not args.no_save,
    )