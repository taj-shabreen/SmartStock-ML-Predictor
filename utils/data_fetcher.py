# ============================================================
# utils/data_fetcher.py
# Handles fetching stock data from yfinance and preprocessing
# ============================================================

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import warnings

warnings.filterwarnings('ignore')


def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch historical stock data using yfinance with multiple fallback methods.

    Tries several approaches to handle yfinance API quirks:
      1. yf.Ticker().history()  — primary method
      2. yf.download()          — fallback, handles more tickers
      3. Auto-extended period   — if date range yields nothing
      4. With retries and delays — handle rate limiting

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'TSLA', 'INFY.NS', '^IXIC' for Nasdaq)
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format

    Returns:
        DataFrame with OHLCV columns
    """
    ticker = ticker.strip().upper()

    # Known valid tickers that should work (whitelist for debugging)
    VALID_US_STOCKS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'V', 'JNJ']

    # Only check for exchange names - NOT valid stock tickers
    EXCHANGE_NAMES = ['NASDAQ', 'NYSE', 'NSE', 'BSE', 'LSE', 'TSX', 'EURONEXT', 'DAX', 'FTSE']

    if ticker in EXCHANGE_NAMES:
        raise ValueError(
            f"'{ticker}' is an exchange name, not a valid stock ticker.\n\n"
            f"Please use specific stock tickers like:\n"
            f"  • 'AAPL' (Apple Inc. - trades on NASDAQ)\n"
            f"  • 'MSFT' (Microsoft - trades on NASDAQ)\n"
            f"  • '^IXIC' (Nasdaq Composite Index)\n"
            f"  • '^GSPC' (S&P 500 Index)\n"
            f"  • 'INFY.NS' (Infosys - NSE India)\n\n"
            f"Tip: Visit https://finance.yahoo.com to find valid ticker symbols."
        )

    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize columns, strip timezone, drop NaNs."""
        if df is None or df.empty:
            return pd.DataFrame()

        # yf.download returns MultiIndex columns when group_by not set
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Normalise column names (handle both Title Case and lower case)
        df.columns = [c.strip().title() for c in df.columns]

        needed = ['Open', 'High', 'Low', 'Close', 'Volume']
        available = [c for c in needed if c in df.columns]
        if not available:
            return pd.DataFrame()
        df = df[available].copy()

        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.index = pd.to_datetime(df.index)
        df.dropna(subset=['Close'], inplace=True)
        return df

    # Helper to try fetching with retry
    def fetch_with_retry(fetch_func, max_retries=1, delay=0.3):
        for attempt in range(max_retries):
            try:
                result = fetch_func()
                if result is not None and not result.empty:
                    return result
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"[WARN] Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    print(f"[WARN] All retries failed: {e}")
        return None

    # ── Method 1: Ticker.history() with retry ──────────────────────────
    def method1():
        tk = yf.Ticker(ticker)
        df = tk.history(start=start_date, end=end_date, auto_adjust=True)
        if not df.empty:
            return _clean(df)
        return pd.DataFrame()

    df = fetch_with_retry(method1)
    if df is not None and len(df) >= 20:
        print(f"[INFO] Method 1 OK — {len(df)} rows for {ticker}")
        return df

    # ── Method 2: yf.download() with retry ─────────────────────────────
    def method2():
        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if not df.empty and isinstance(df, pd.DataFrame):
            return _clean(df)
        return pd.DataFrame()

    df = fetch_with_retry(method2)
    if df is not None and len(df) >= 20:
        print(f"[INFO] Method 2 OK — {len(df)} rows for {ticker}")
        return df

    # ── Method 3: yf.download() with period string ─────────────────────
    def method3():
        days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
        period = "max" if days > 1800 else ("5y" if days > 900 else "2y")

        df = yf.download(
            ticker,
            period=period,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if not df.empty and isinstance(df, pd.DataFrame):
            df = _clean(df)
            if not df.empty:
                return df.loc[start_date:end_date]
        return pd.DataFrame()

    df = fetch_with_retry(method3)
    if df is not None and len(df) >= 15:
        print(f"[INFO] Method 3 OK — {len(df)} rows for {ticker}")
        return df

    # ── Method 4: Try with longer period and no date filter ────────────
    def method4():
        tk = yf.Ticker(ticker)
        df = tk.history(period="max", auto_adjust=True)
        if not df.empty:
            df = _clean(df)
            if not df.empty:
                return df.loc[start_date:end_date] if start_date in df.index else df
        return pd.DataFrame()

    df = fetch_with_retry(method4)
    if df is not None and len(df) >= 10:
        print(f"[INFO] Method 4 OK — {len(df)} rows for {ticker}")
        return df

    # ── All methods exhausted ─────────────────────────────────────────
    raise RuntimeError(
        f"❌ Could not fetch data for ticker '{ticker}'.\n\n"
        f"Possible reasons:\n"
        f"  • The ticker symbol is incorrect or delisted\n"
        f"  • No trading data exists between {start_date} and {end_date}\n"
        f"  • The stock wasn't publicly traded during this period\n\n"
        f"Valid examples:\n"
        f"  • US Stocks: AAPL, MSFT, TSLA, GOOGL, AMZN\n"
        f"  • Indian Stocks: INFY.NS, RELIANCE.NS, TCS.NS\n"
        f"  • Indices: ^GSPC (S&P 500), ^IXIC (Nasdaq), ^NSEI (Nifty 50)\n\n"
        f"Tip: Verify '{ticker}' at https://finance.yahoo.com"
    )


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a rich set of technical indicators for feature engineering.

    Indicators computed:
    - Simple Moving Averages (SMA): 5, 10, 20, 50 days
    - Exponential Moving Averages (EMA): 12, 26 days
    - Moving Average Convergence Divergence (MACD)
    - Relative Strength Index (RSI): 14 days
    - Bollinger Bands (Upper, Lower, Width)
    - Average True Range (ATR) - volatility
    - On-Balance Volume (OBV)
    - Rate of Change (ROC)
    - Williams %R
    - Daily Returns
    - Log Returns
    - Rolling Volatility

    Args:
        df: DataFrame with OHLCV columns

    Returns:
        DataFrame with all original + engineered features
    """
    data = df.copy()

    # Check if we have enough data
    if len(data) < 50:
        print(f"[WARN] Only {len(data)} rows available. Some indicators may have NaN values.")

    # ---- Price-based Features ----
    data['Daily_Return'] = data['Close'].pct_change()
    data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1) + 1e-9)

    # ---- Simple Moving Averages ----
    for window in [5, 10, 20, 50]:
        if len(data) > window:
            data[f'SMA_{window}'] = data['Close'].rolling(window=window, min_periods=max(1, window // 2)).mean()

    # ---- Exponential Moving Averages ----
    if len(data) > 12:
        data['EMA_12'] = data['Close'].ewm(span=12, adjust=False, min_periods=6).mean()
    if len(data) > 26:
        data['EMA_26'] = data['Close'].ewm(span=26, adjust=False, min_periods=13).mean()

    # ---- MACD ----
    if 'EMA_12' in data.columns and 'EMA_26' in data.columns:
        data['MACD'] = data['EMA_12'] - data['EMA_26']
        data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False, min_periods=5).mean()
        data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']

    # ---- RSI (Relative Strength Index) ----
    if len(data) > 14:
        data['RSI_14'] = compute_rsi(data['Close'], period=14)

    # ---- Bollinger Bands ----
    bb_window = 20
    if len(data) > bb_window:
        rolling_mean = data['Close'].rolling(window=bb_window, min_periods=10).mean()
        rolling_std = data['Close'].rolling(window=bb_window, min_periods=10).std()
        data['BB_Upper'] = rolling_mean + (rolling_std * 2)
        data['BB_Lower'] = rolling_mean - (rolling_std * 2)
        data['BB_Width'] = data['BB_Upper'] - data['BB_Lower']
        data['BB_Position'] = (data['Close'] - data['BB_Lower']) / (data['BB_Width'] + 1e-9)

    # ---- Average True Range (ATR) ----
    if len(data) > 14:
        data['ATR_14'] = compute_atr(data, period=14)

    # ---- On-Balance Volume (OBV) ----
    data['OBV'] = compute_obv(data)

    # ---- Rate of Change (ROC) ----
    if len(data) > 10:
        data['ROC_10'] = ((data['Close'] - data['Close'].shift(10)) / (data['Close'].shift(10) + 1e-9)) * 100

    # ---- Williams %R ----
    if len(data) > 14:
        data['Williams_R'] = compute_williams_r(data, period=14)

    # ---- Rolling Volatility ----
    if len(data) > 10:
        data['Volatility_10'] = data['Daily_Return'].rolling(window=10, min_periods=5).std() * np.sqrt(252)
    if len(data) > 20:
        data['Volatility_20'] = data['Daily_Return'].rolling(window=20, min_periods=10).std() * np.sqrt(252)

    # ---- Price Ratios ----
    data['High_Low_Ratio'] = data['High'] / (data['Low'] + 1e-9)
    data['Close_Open_Ratio'] = data['Close'] / (data['Open'] + 1e-9)

    # ---- Volume Features ----
    if len(data) > 10:
        data['Volume_SMA_10'] = data['Volume'].rolling(window=10, min_periods=5).mean()
        data['Volume_Ratio'] = data['Volume'] / (data['Volume_SMA_10'] + 1e-9)

    # ---- Lagged Close Prices ----
    for lag in [1, 2, 3, 5]:
        data[f'Close_Lag_{lag}'] = data['Close'].shift(lag)

    return data


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI for a price series."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period, min_periods=period // 2).mean()
    avg_loss = loss.rolling(window=period, min_periods=period // 2).mean()

    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    high_low = df['High'] - df['Low']
    high_close = abs(df['High'] - df['Close'].shift(1))
    low_close = abs(df['Low'] - df['Close'].shift(1))

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period, min_periods=period // 2).mean()
    return atr


def compute_obv(df: pd.DataFrame) -> pd.Series:
    """Calculate On-Balance Volume — vectorized (no Python loop)."""
    close = df['Close'].values
    volume = df['Volume'].values
    direction = np.sign(np.diff(close, prepend=close[0]))
    direction[0] = 0  # first row has no prior close
    obv = np.cumsum(direction * volume)
    return pd.Series(obv, index=df.index)


def compute_williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Williams %R."""
    highest_high = df['High'].rolling(window=period, min_periods=period // 2).max()
    lowest_low = df['Low'].rolling(window=period, min_periods=period // 2).min()
    williams_r = -100 * ((highest_high - df['Close']) / (highest_high - lowest_low + 1e-9))
    return williams_r


def create_target_variables(df: pd.DataFrame, future_days: int = 1) -> pd.DataFrame:
    """
    Create both classification and regression target variables.

    Classification Target: 1 (UP) if next day close > today's close, else 0 (DOWN)
    Regression Target: Next day's closing price

    Args:
        df: DataFrame with computed features
        future_days: Number of days ahead to predict (default = 1)

    Returns:
        DataFrame with added target columns
    """
    data = df.copy()

    # Regression target: future closing price
    data['Target_Price'] = data['Close'].shift(-future_days)

    # Classification target: UP(1) or DOWN(0) movement
    data['Target_Direction'] = (data['Close'].shift(-future_days) > data['Close']).astype(int)

    return data


def preprocess_data(df: pd.DataFrame) -> tuple:
    """
    Full preprocessing pipeline: features → cleaning → splitting.

    Returns:
        Tuple of (processed_df, feature_columns, X, y_clf, y_reg)
    """
    if df is None or df.empty:
        raise ValueError("Cannot preprocess empty DataFrame")

    if len(df) < 30:
        raise ValueError(
            f"Only {len(df)} rows of data available. Need at least 30 trading days for meaningful features.")

    # Compute all indicators
    data = compute_technical_indicators(df)

    # Create targets
    data = create_target_variables(data)

    # Drop rows with NaN (from rolling windows)
    data.dropna(inplace=True)

    if len(data) < 20:
        raise ValueError(f"After preprocessing, only {len(data)} rows remain. Need more historical data.")

    # Define feature columns (exclude targets and raw OHLCV)
    exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume',
                    'Target_Price', 'Target_Direction']
    feature_cols = [c for c in data.columns if c not in exclude_cols]

    # Remove any columns that are all NaN
    feature_cols = [c for c in feature_cols if data[c].notna().any()]

    X = data[feature_cols]
    y_clf = data['Target_Direction']
    y_reg = data['Target_Price']

    print(f"[INFO] Features: {len(feature_cols)} | Samples: {len(data)}")
    print(f"[INFO] Class distribution (UP/DOWN): {y_clf.value_counts().to_dict()}")

    return data, feature_cols, X, y_clf, y_reg


def get_stock_info(ticker: str) -> dict:
    """
    Fetch basic stock info for display.
    Uses fast_info only (avoids the slow stock.info call which can block 10-30s).
    Correctly detects currency (USD for US stocks, INR for .NS/.BO, GBP for .L, etc.)
    Never raises — always returns a safe default dict.
    """
    # Infer currency from ticker suffix as a reliable fallback
    def _infer_currency(t: str) -> str:
        t = t.upper()
        if t.endswith('.NS') or t.endswith('.BO'):
            return 'INR'
        if t.endswith('.L'):
            return 'GBP'
        if t.endswith('.HK'):
            return 'HKD'
        if t.endswith('.T'):
            return 'JPY'
        if t.endswith('.SS') or t.endswith('.SZ'):
            return 'CNY'
        if t.endswith('.AX'):
            return 'AUD'
        if t.endswith('.TO'):
            return 'CAD'
        if t.endswith('.DE') or t.endswith('.PA') or t.endswith('.MC'):
            return 'EUR'
        # Default to USD for plain tickers (US stocks, indices)
        return 'USD'

    default = {
        'name': ticker, 'sector': 'N/A', 'industry': 'N/A',
        'currency': _infer_currency(ticker), 'exchange': 'N/A', 'market_cap': None,
    }
    try:
        stock = yf.Ticker(ticker)
        # fast_info is a lightweight call — no heavy JSON fetch
        fi = stock.fast_info
        detected = getattr(fi, 'currency', None)
        # Use API-detected currency if available, else fallback to inferred
        default['currency'] = (detected or _infer_currency(ticker)).upper()
        default['exchange'] = getattr(fi, 'exchange', 'N/A') or 'N/A'
        mc = getattr(fi, 'market_cap', None)
        default['market_cap'] = float(mc) if mc else None
        return default
    except Exception:
        return default


def get_currency_symbol(currency: str) -> str:
    """Return the currency symbol for a given currency code."""
    symbols = {
        'USD': '$',
        'INR': '₹',
        'GBP': '£',
        'EUR': '€',
        'JPY': '¥',
        'CNY': '¥',
        'HKD': 'HK$',
        'AUD': 'A$',
        'CAD': 'C$',
        'CHF': 'Fr',
        'KRW': '₩',
        'SGD': 'S$',
    }
    return symbols.get((currency or 'USD').upper(), '$')