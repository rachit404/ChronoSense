import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

# from src.time_series_analysis import (
#     compute_trend_seasonality,
#     detect_anomalies,
#     forecast_arima,
#     forecast_sarima,
#     forecast_lstm
# )

# -------------------------
# Trend & Seasonality
# -------------------------
def compute_trend_seasonality(df: pd.DataFrame, price_col: str, model: str = 'additive', period: int = 30) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(subset=[price_col])
    result = seasonal_decompose(df[price_col], model=model, period=period, extrapolate_trend='freq')
    df['trend'] = result.trend
    df['seasonal'] = result.seasonal
    df['residual'] = result.resid
    return df

# -------------------------
# Anomaly Detection
# -------------------------
def detect_anomalies(df: pd.DataFrame, price_col: str, z_thresh: float = 3.0) -> pd.DataFrame:
    df = df.copy()
    # Z-score method
    df['zscore'] = (df[price_col] - df[price_col].mean()) / df[price_col].std(ddof=0)
    df['zscore_anomaly'] = df['zscore'].abs() > z_thresh

    # Isolation Forest
    scaler = StandardScaler()
    scaled_vals = scaler.fit_transform(df[[price_col]].fillna(0))
    iso = IsolationForest(contamination=0.01, random_state=42)
    df['isolation_anomaly'] = iso.fit_predict(scaled_vals) == -1
    return df


# ----------------------------Models------------------------
# ==========================================================
# 1️⃣ ARIMA (with in-sample predictions)
# ==========================================================
def forecast_arima(df: pd.DataFrame, price_col: str, steps: int = 7):
    """
    ARIMA(5,1,0) model producing both in-sample predictions and out-of-sample forecasts.
    """
    df = df.dropna(subset=[price_col]).copy()

    model = ARIMA(df[price_col], order=(5, 1, 0))
    model_fit = model.fit()

    # In-sample predictions
    pred_in_sample = model_fit.predict(start=1, end=len(df))

    # Out-of-sample forecast
    forecast_out = model_fit.forecast(steps=steps)

    # Combine into one DataFrame
    df_pred = pd.DataFrame({
        "forecast_date": df.index,
        "forecast_model": "ARIMA",
        "forecast_type": "in-sample",
        "forecast_value": pred_in_sample.values
    })

    df_forecast = pd.DataFrame({
        "forecast_date": pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=steps),
        "forecast_model": "ARIMA",
        "forecast_type": "out-of-sample",
        "forecast_value": forecast_out.values
    })
    print("[INFO] ARIMA forecast generated.")

    return pd.concat([df_pred, df_forecast], ignore_index=True)


# ==========================================================
# 2️⃣ SARIMA (with in-sample predictions)
# ==========================================================
def forecast_sarima(
    df: pd.DataFrame,
    price_col: str,
    steps: int = 7,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12)
):
    """
    SARIMA model producing both in-sample predictions and out-of-sample forecasts.
    """
    df = df.dropna(subset=[price_col]).copy()

    model = SARIMAX(df[price_col], order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)

    # In-sample predictions
    pred_in_sample = model_fit.predict(start=1, end=len(df))

    # Out-of-sample forecast
    forecast_out = model_fit.forecast(steps=steps)

    df_pred = pd.DataFrame({
        "forecast_date": df.index,
        "forecast_model": "SARIMA",
        "forecast_type": "in-sample",
        "forecast_value": pred_in_sample.values
    })

    df_forecast = pd.DataFrame({
        "forecast_date": pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=steps),
        "forecast_model": "SARIMA",
        "forecast_type": "out-of-sample",
        "forecast_value": forecast_out.values
    })
    print("[INFO] SARIMA forecast generated.")

    return pd.concat([df_pred, df_forecast], ignore_index=True)


# ==========================================================
# 3️⃣ LSTM (with in-sample predictions)
# ==========================================================
def forecast_lstm(df: pd.DataFrame, price_col: str, steps: int = 7, epochs: int = 5):
    """
    LSTM model producing both in-sample predictions (reconstructed) and out-of-sample forecasts.
    """
    from numpy import array

    df = df.dropna(subset=[price_col]).copy()

    prices = df[price_col].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices)

    # Prepare supervised data
    X, y = [], []
    for i in range(len(scaled) - 1):
        X.append(scaled[i])
        y.append(scaled[i + 1])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], 1, X.shape[1]))

    # Model
    model = Sequential([
        LSTM(50, activation="relu", input_shape=(1, 1)),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=epochs, verbose=0)

    # In-sample predictions
    preds_in = model.predict(X, verbose=0)
    preds_in = scaler.inverse_transform(preds_in).flatten()

    df_pred = pd.DataFrame({
        "forecast_date": df.index[1:],  # skip first (no prediction)
        "forecast_model": "LSTM",
        "forecast_type": "in-sample",
        "forecast_value": preds_in
    })

    # Out-of-sample forecasts
    last_val = scaled[-1].reshape((1, 1, 1))
    preds_out = []
    for _ in range(steps):
        next_pred = model.predict(last_val, verbose=0)
        preds_out.append(next_pred[0, 0])
        last_val = next_pred.reshape((1, 1, 1))

    preds_out = scaler.inverse_transform(np.array(preds_out).reshape(-1, 1)).flatten()

    df_forecast = pd.DataFrame({
        "forecast_date": pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=steps),
        "forecast_model": "LSTM",
        "forecast_type": "out-of-sample",
        "forecast_value": preds_out
    })
    print("[INFO] LSTM forecast generated.")

    return pd.concat([df_pred, df_forecast], ignore_index=True)
