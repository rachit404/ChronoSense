import pandas as pd
import numpy as np
import os
from typing import List
# from src.pre_data_analysis import (
#     precompute_basic_features,
#     save_pre_analysis
# )

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def precompute_basic_features(df: pd.DataFrame, price_col: str,
                              windows: List[int] = [7, 30],
                              rsi_window: int = 14) -> pd.DataFrame:
    """
    Compute basic statistical/time-series features described by the user:
      - rolling mean, median, std for windows
      - daily returns and cumulative returns
      - SMA (rolling mean) and EMA
      - rolling correlations with Volume if present
      - volatility (std of returns)
      - RSI and MACD
    
    Returns a new DataFrame with added columns.
    """
    df = df.copy()
    price = df[price_col].astype(float)
    
    # Daily returns
    df['daily_return'] = price.pct_change()
    # Cumulative return from start
    df['cumulative_return'] = (1 + df['daily_return']).cumprod() - 1

    # Rolling stats and MAs/EMAs
    for w in windows:
        df[f'rolling_mean_{w}'] = price.rolling(window=w, min_periods=1).mean()
        df[f'rolling_median_{w}'] = price.rolling(window=w, min_periods=1).median()
        df[f'rolling_std_{w}'] = price.rolling(window=w, min_periods=1).std(ddof=0)
        # EMA and SMA (SMA = rolling mean)
        df[f'ema_{w}'] = _ema(price, span=w)
        df[f'sma_{w}'] = df[f'rolling_mean_{w}']

    # Volatility: std of returns over windows
    for w in windows:
        df[f'volatility_{w}'] = df['daily_return'].rolling(window=w, min_periods=1).std(ddof=0)

    # Rolling correlation with volume if volume exists
    if 'Volume' in df.columns:
        for w in windows:
            df[f'rolling_corr_price_volume_{w}'] = price.rolling(window=w, min_periods=1).corr(df['Volume'])

    # RSI implementation (Wilder's smoothing)
    delta = price.diff()
    up = delta.clip(lower=0.0)
    down = -1 * delta.clip(upper=0.0)
    # Use exponential moving average of gains/losses (Wilder's)
    roll_up = up.ewm(alpha=1/rsi_window, adjust=False, min_periods=1).mean()
    roll_down = down.ewm(alpha=1/rsi_window, adjust=False, min_periods=1).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi'] = df['rsi'].fillna(0)

    # MACD: EMA12 - EMA26 and signal line 9-day EMA of MACD
    ema_short = _ema(price, span=12)
    ema_long = _ema(price, span=26)
    df['macd'] = ema_short - ema_long
    df['macd_signal'] = _ema(df['macd'].fillna(0), span=9)
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # Momentum: difference between current price and price n periods ago (we'll include for windows)
    for w in windows:
        df[f'momentum_{w}'] = price - price.shift(w)
    
    # Clean up infinite values if any and keep numeric columns consistent
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    return df

def save_pre_analysis(df: pd.DataFrame, original_path: str) -> str:
    """
    Save dataframe to a new CSV with suffix "_pre_analysis" before the file extension.
    Returns the path to the saved CSV.
    """
    base, ext = os.path.splitext(original_path)
    new_path = f"{base}_pre_analysis{ext}"
    df.to_csv(new_path, index=True)
    return new_path