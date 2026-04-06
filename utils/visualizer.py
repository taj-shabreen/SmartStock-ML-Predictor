# ============================================================
# utils/visualizer.py
# All visualization functions for model results and stock data
# ============================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Streamlit
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# ---- Global Style Settings ----
PALETTE = {
    'bg': '#0d1117',
    'card': '#161b22',
    'accent': '#58a6ff',
    'green': '#3fb950',
    'red': '#f85149',
    'yellow': '#e3b341',
    'text': '#c9d1d9',
    'grid': '#21262d',
}

def set_dark_style():
    """Apply consistent dark theme to all matplotlib plots."""
    plt.rcParams.update({
        'figure.facecolor': PALETTE['bg'],
        'axes.facecolor': PALETTE['card'],
        'axes.edgecolor': PALETTE['grid'],
        'axes.labelcolor': PALETTE['text'],
        'text.color': PALETTE['text'],
        'xtick.color': PALETTE['text'],
        'ytick.color': PALETTE['text'],
        'grid.color': PALETTE['grid'],
        'grid.linewidth': 0.5,
        'font.family': 'DejaVu Sans',
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.titleweight': 'bold',
        'figure.dpi': 120,
    })


def save_plot(fig, filename: str, plots_dir: str = "plots") -> str:
    """Save a matplotlib figure and return path."""
    os.makedirs(plots_dir, exist_ok=True)
    path = os.path.join(plots_dir, filename)
    fig.savefig(path, bbox_inches='tight', dpi=120, facecolor=PALETTE['bg'])
    plt.close(fig)
    return path


# ============================================================
# 1. STOCK PRICE TREND CHART
# ============================================================

def plot_stock_price_trend(df: pd.DataFrame, ticker: str) -> plt.Figure:
    """
    Plot stock price history with volume and key moving averages.
    """
    set_dark_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                    gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle(f'{ticker} — Stock Price History & Technical Overview',
                 fontsize=15, fontweight='bold', color=PALETTE['text'], y=1.01)

    # Price + Moving Averages
    ax1.plot(df.index, df['Close'], color=PALETTE['accent'],
             linewidth=1.5, label='Close Price', zorder=3)

    for col, color in [('SMA_20', '#f0a500'), ('SMA_50', PALETTE['green'])]:
        if col in df.columns:
            ax1.plot(df.index, df[col], linewidth=1, linestyle='--',
                     color=color, alpha=0.85,
                     label=col.replace('_', ' '))

    # Bollinger Bands fill
    if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns:
        ax1.fill_between(df.index, df['BB_Lower'], df['BB_Upper'],
                         alpha=0.08, color=PALETTE['accent'], label='Bollinger Bands')
        ax1.plot(df.index, df['BB_Upper'], linewidth=0.6,
                 linestyle=':', color=PALETTE['accent'], alpha=0.5)
        ax1.plot(df.index, df['BB_Lower'], linewidth=0.6,
                 linestyle=':', color=PALETTE['accent'], alpha=0.5)

    ax1.set_ylabel('Price', color=PALETTE['text'])
    ax1.legend(loc='upper left', fontsize=9, framealpha=0.3)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelbottom=False)

    # Volume bars
    colors_vol = [PALETTE['green'] if df['Close'].iloc[i] >= df['Open'].iloc[i]
                  else PALETTE['red'] for i in range(len(df))]
    ax2.bar(df.index, df['Volume'], color=colors_vol, alpha=0.7, width=0.8)
    ax2.set_ylabel('Volume', color=PALETTE['text'])
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ============================================================
# 2. CLASSIFICATION ACCURACY COMPARISON
# ============================================================

def plot_classification_comparison(metrics_df: pd.DataFrame,
                                    best_model_name: str) -> plt.Figure:
    """
    Grouped bar chart comparing Accuracy, Precision, Recall, F1 across all models.
    """
    set_dark_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Classification Models — Performance Comparison',
                 fontsize=15, fontweight='bold', color=PALETTE['text'])

    models = metrics_df['Model'].tolist()
    x = np.arange(len(models))
    width = 0.2

    metric_configs = [
        ('Accuracy', PALETTE['accent']),
        ('Precision', PALETTE['yellow']),
        ('Recall', PALETTE['green']),
        ('F1-Score', '#da8fff'),
    ]

    # Left: Grouped bar chart
    ax = axes[0]
    for i, (metric, color) in enumerate(metric_configs):
        vals = metrics_df[metric].values
        bars = ax.bar(x + i * width, vals, width,
                      label=metric, color=color, alpha=0.85)

        # Highlight best model
        best_idx = metrics_df[metrics_df['Model'] == best_model_name].index[0] - 1
        bars[best_idx].set_edgecolor('#ffd700')
        bars[best_idx].set_linewidth(2)

    ax.set_xlabel('Model', color=PALETTE['text'])
    ax.set_ylabel('Score', color=PALETTE['text'])
    ax.set_title('All Metrics Comparison', color=PALETTE['text'])
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=9, framealpha=0.3)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0.5, color=PALETTE['red'], linestyle='--',
               linewidth=0.8, alpha=0.6, label='50% baseline')

    # Annotate best
    ax.annotate(f'★ Best', xy=(best_idx + width * 1.5,
                               metrics_df.iloc[best_idx]['F1-Score'] + 0.03),
                ha='center', color='#ffd700', fontsize=8)

    # Right: F1-Score horizontal bar chart (ranked)
    ax2 = axes[1]
    colors = [PALETTE['yellow'] if m == best_model_name
              else PALETTE['accent'] for m in models]
    bars = ax2.barh(models, metrics_df['F1-Score'].values,
                    color=colors, alpha=0.85, height=0.6)

    # Value labels
    for bar, val in zip(bars, metrics_df['F1-Score'].values):
        ax2.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                 f'{val:.3f}', va='center', fontsize=9, color=PALETTE['text'])

    ax2.set_xlabel('F1-Score', color=PALETTE['text'])
    ax2.set_title('F1-Score Ranking', color=PALETTE['text'])
    ax2.set_xlim(0, 1.15)
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.invert_yaxis()

    plt.tight_layout()
    return fig


# ============================================================
# 3. CONFUSION MATRIX HEATMAP
# ============================================================

def plot_confusion_matrix(cm: np.ndarray, model_name: str) -> plt.Figure:
    """Plot a styled confusion matrix heatmap."""
    set_dark_style()
    fig, ax = plt.subplots(figsize=(6, 5))

    # Custom colormap
    cmap = sns.color_palette([PALETTE['card'], PALETTE['accent']], as_cmap=True)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['DOWN (0)', 'UP (1)'],
                yticklabels=['DOWN (0)', 'UP (1)'],
                ax=ax, linewidths=2,
                annot_kws={'size': 16, 'weight': 'bold'})

    ax.set_xlabel('Predicted Label', fontsize=12, color=PALETTE['text'])
    ax.set_ylabel('True Label', fontsize=12, color=PALETTE['text'])
    ax.set_title(f'Confusion Matrix — {model_name}', fontsize=13,
                 fontweight='bold', color=PALETTE['text'])

    # Stats overlay
    tn, fp, fn, tp = cm.ravel()
    total = tn + fp + fn + tp
    acc = (tn + tp) / total

    ax.text(2.3, 1.5, f'Accuracy\n{acc:.2%}',
            fontsize=11, color=PALETTE['green'],
            ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor=PALETTE['card'],
                      edgecolor=PALETTE['green'], alpha=0.8))

    plt.tight_layout()
    return fig


# ============================================================
# 4. REGRESSION: ACTUAL vs PREDICTED
# ============================================================

def plot_regression_predictions(y_test, y_pred_dict: dict,
                                  best_model_name: str,
                                  ticker: str) -> plt.Figure:
    """
    Plot actual vs predicted prices for the best regression model,
    and include scatter plots comparing all models.
    """
    set_dark_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f'{ticker} — Regression: Actual vs Predicted Price',
                 fontsize=15, fontweight='bold', color=PALETTE['text'])

    y_test_arr = np.array(y_test)
    best_pred = y_pred_dict[best_model_name]

    # Left: Time-series line plot
    ax1 = axes[0]
    ax1.plot(range(len(y_test_arr)), y_test_arr,
             color=PALETTE['accent'], linewidth=1.5,
             label='Actual Price', zorder=3)
    ax1.plot(range(len(best_pred)), best_pred,
             color=PALETTE['green'], linewidth=1.5,
             linestyle='--', label=f'Predicted ({best_model_name})',
             zorder=2, alpha=0.9)

    ax1.fill_between(range(len(y_test_arr)), y_test_arr, best_pred,
                     alpha=0.1, color=PALETTE['yellow'])

    ax1.set_xlabel('Time Steps (Test Period)', color=PALETTE['text'])
    ax1.set_ylabel('Price', color=PALETTE['text'])
    ax1.set_title(f'Best Model: {best_model_name}', color=PALETTE['text'])
    ax1.legend(fontsize=9, framealpha=0.3)
    ax1.grid(True, alpha=0.3)

    # Right: Scatter plot actual vs predicted
    ax2 = axes[1]
    ax2.scatter(y_test_arr, best_pred, alpha=0.5,
                color=PALETTE['accent'], s=20, zorder=2)

    # Perfect prediction line
    min_val = min(y_test_arr.min(), best_pred.min())
    max_val = max(y_test_arr.max(), best_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val],
             color=PALETTE['yellow'], linewidth=1.5,
             linestyle='--', label='Perfect Prediction')

    ax2.set_xlabel('Actual Price', color=PALETTE['text'])
    ax2.set_ylabel('Predicted Price', color=PALETTE['text'])
    ax2.set_title('Actual vs Predicted Scatter', color=PALETTE['text'])
    ax2.legend(fontsize=9, framealpha=0.3)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ============================================================
# 5. REGRESSION METRICS COMPARISON
# ============================================================

def plot_regression_comparison(metrics_df: pd.DataFrame,
                                best_model_name: str) -> plt.Figure:
    """Bar charts comparing R², RMSE, MAE across regression models."""
    set_dark_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.suptitle('Regression Models — Performance Comparison',
                 fontsize=15, fontweight='bold', color=PALETTE['text'])

    models = metrics_df['Model'].tolist()
    best_idx = metrics_df[metrics_df['Model'] == best_model_name].index[0] - 1

    configs = [
        ('R² Score', PALETTE['green'], True, 'Higher is better'),
        ('RMSE', PALETTE['red'], False, 'Lower is better'),
        ('MAE', PALETTE['yellow'], False, 'Lower is better'),
    ]

    for ax, (metric, color, higher_better, note) in zip(axes, configs):
        vals = metrics_df[metric].values
        colors = [PALETTE['yellow'] if i == best_idx else color
                  for i in range(len(models))]

        bars = ax.barh(models, vals, color=colors, alpha=0.85, height=0.6)

        # Value labels
        for bar, val in zip(bars, vals):
            ax.text(bar.get_width() + max(vals) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f'{val:.3f}', va='center', fontsize=8.5,
                    color=PALETTE['text'])

        ax.set_xlabel(metric, color=PALETTE['text'])
        ax.set_title(f'{metric}\n({note})', color=PALETTE['text'])
        ax.grid(True, alpha=0.3, axis='x')
        ax.invert_yaxis()

        # Star best model
        ax.annotate('★', xy=(vals[best_idx], best_idx),
                    xytext=(-15, 0), textcoords='offset points',
                    color='#ffd700', fontsize=12, va='center')

    plt.tight_layout()
    return fig


# ============================================================
# 6. FEATURE IMPORTANCE
# ============================================================

def plot_feature_importance(importance_df: pd.DataFrame,
                             model_name: str,
                             top_n: int = 20) -> plt.Figure:
    """
    Horizontal bar chart for feature importance (top N features).
    """
    set_dark_style()
    df = importance_df.head(top_n)

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))

    # Color gradient by importance
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(df)))[::-1]
    bars = ax.barh(df['Feature'], df['Importance'],
                   color=colors, edgecolor='none', height=0.7)

    # Value labels
    for bar, val in zip(bars, df['Importance']):
        ax.text(bar.get_width() + max(df['Importance']) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=9, color=PALETTE['text'])

    ax.set_xlabel('Importance Score', color=PALETTE['text'])
    ax.set_title(f'Top {top_n} Feature Importances — {model_name}',
                 fontsize=13, fontweight='bold', color=PALETTE['text'])
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()

    plt.tight_layout()
    return fig


# ============================================================
# 7. RSI + MACD TECHNICAL DASHBOARD
# ============================================================

def plot_technical_indicators(df: pd.DataFrame, ticker: str) -> plt.Figure:
    """
    Multi-panel technical indicator dashboard.
    Panels: Price, RSI, MACD, Volatility
    """
    set_dark_style()
    fig, axes = plt.subplots(4, 1, figsize=(14, 14),
                              gridspec_kw={'height_ratios': [3, 1.5, 1.5, 1.5]})
    fig.suptitle(f'{ticker} — Technical Indicator Dashboard',
                 fontsize=15, fontweight='bold', color=PALETTE['text'])

    # Panel 1: Price + SMA
    ax1 = axes[0]
    ax1.plot(df.index, df['Close'], color=PALETTE['accent'],
             linewidth=1.5, label='Close')
    if 'SMA_20' in df.columns:
        ax1.plot(df.index, df['SMA_20'], color=PALETTE['yellow'],
                 linewidth=1, linestyle='--', label='SMA 20')
    if 'EMA_12' in df.columns:
        ax1.plot(df.index, df['EMA_12'], color=PALETTE['green'],
                 linewidth=1, linestyle='--', label='EMA 12')
    ax1.set_ylabel('Price', color=PALETTE['text'])
    ax1.legend(fontsize=9, framealpha=0.3)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelbottom=False)

    # Panel 2: RSI
    ax2 = axes[1]
    if 'RSI_14' in df.columns:
        ax2.plot(df.index, df['RSI_14'], color='#da8fff',
                 linewidth=1.2, label='RSI (14)')
        ax2.axhline(70, color=PALETTE['red'], linewidth=0.8,
                    linestyle='--', alpha=0.7, label='Overbought (70)')
        ax2.axhline(30, color=PALETTE['green'], linewidth=0.8,
                    linestyle='--', alpha=0.7, label='Oversold (30)')
        ax2.fill_between(df.index, df['RSI_14'], 70,
                         where=(df['RSI_14'] >= 70),
                         interpolate=True, color=PALETTE['red'], alpha=0.15)
        ax2.fill_between(df.index, df['RSI_14'], 30,
                         where=(df['RSI_14'] <= 30),
                         interpolate=True, color=PALETTE['green'], alpha=0.15)
        ax2.set_ylim(0, 100)
    ax2.set_ylabel('RSI', color=PALETTE['text'])
    ax2.legend(fontsize=8, framealpha=0.3)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelbottom=False)

    # Panel 3: MACD
    ax3 = axes[2]
    if 'MACD' in df.columns:
        ax3.plot(df.index, df['MACD'], color=PALETTE['accent'],
                 linewidth=1.2, label='MACD')
        ax3.plot(df.index, df['MACD_Signal'], color=PALETTE['yellow'],
                 linewidth=1.0, linestyle='--', label='Signal')
        if 'MACD_Hist' in df.columns:
            colors_macd = [PALETTE['green'] if v >= 0 else PALETTE['red']
                           for v in df['MACD_Hist']]
            ax3.bar(df.index, df['MACD_Hist'],
                    color=colors_macd, alpha=0.5, width=0.8, label='Histogram')
        ax3.axhline(0, color=PALETTE['text'], linewidth=0.5, alpha=0.5)
    ax3.set_ylabel('MACD', color=PALETTE['text'])
    ax3.legend(fontsize=8, framealpha=0.3)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(labelbottom=False)

    # Panel 4: Volatility
    ax4 = axes[3]
    if 'Volatility_20' in df.columns:
        ax4.plot(df.index, df['Volatility_20'] * 100,
                 color=PALETTE['yellow'], linewidth=1.2,
                 label='20-Day Volatility (%)')
        ax4.fill_between(df.index, df['Volatility_20'] * 100,
                         alpha=0.15, color=PALETTE['yellow'])
    ax4.set_ylabel('Volatility (%)', color=PALETTE['text'])
    ax4.set_xlabel('Date', color=PALETTE['text'])
    ax4.legend(fontsize=8, framealpha=0.3)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ============================================================
# 8. TRAINING TIME COMPARISON
# ============================================================

def plot_training_time_comparison(clf_metrics: pd.DataFrame,
                                   reg_metrics: pd.DataFrame) -> plt.Figure:
    """Side-by-side bar charts for training time comparison."""
    set_dark_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Model Training Time Comparison',
                 fontsize=15, fontweight='bold', color=PALETTE['text'])

    for ax, metrics_df, title in [
        (axes[0], clf_metrics, 'Classification Models'),
        (axes[1], reg_metrics, 'Regression Models'),
    ]:
        models = metrics_df['Model'].tolist()
        times = metrics_df['Training Time (s)'].values

        colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(models)))
        bars = ax.barh(models, times, color=colors, alpha=0.85, height=0.6)

        for bar, t in zip(bars, times):
            ax.text(bar.get_width() + max(times) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f'{t:.3f}s', va='center', fontsize=9,
                    color=PALETTE['text'])

        ax.set_xlabel('Training Time (seconds)', color=PALETTE['text'])
        ax.set_title(title, color=PALETTE['text'])
        ax.grid(True, alpha=0.3, axis='x')
        ax.invert_yaxis()

    plt.tight_layout()
    return fig