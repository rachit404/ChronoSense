import matplotlib.pyplot as plt
import os
from datetime import datetime
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def generate_timeseries_visuals(
    df,
    price_col="Close",
    arima_forecast=None,
    sarima_forecast=None,
    lstm_forecast=None,
    output_dir="images",
    run_id=None
):
    """
    Generate and save time-series visualizations with filenames synchronized
    to metadata keys used in LangChain summaries (e.g., image_forecast_arima).
    
    Each file is versioned with run_id for uniqueness.
    Returns:
        dict: { "image_forecast_arima": "images/forecast_arima_20250101_123000.png", ... }
    """

    os.makedirs(output_dir, exist_ok=True)

    # -----------------------
    # Unique run suffix
    # -----------------------
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{run_id}"

    # Helper to save and return full metadata key mapping
    def save_fig(fig, base_name, label=None):
        """
        Saves figure and returns (metadata_key, file_path)
        """
        file_name = f"{base_name}{suffix}.png"
        file_path = os.path.join(output_dir, file_name)
        plt.tight_layout()
        fig.savefig(file_path, dpi=200)
        plt.close(fig)

        metadata_key = f"image_{base_name}" if label is None else f"image_{label}"
        return metadata_key, file_path

    saved_files = {}

    # -----------------------
    # 1️ Trend & Seasonality
    # -----------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df[price_col], label="Price", color="black", linewidth=1)
    if "trend" in df.columns:
        ax.plot(df.index, df["trend"], label="Trend", color="blue", linewidth=2)
    if "seasonal" in df.columns:
        ax.plot(df.index, df["seasonal"], label="Seasonality", color="forestgreen", alpha=0.6)
    if "residual" in df.columns:
        ax.plot(df.index, df["residual"], label="Residual", color="orange", linestyle="--", alpha=0.7)
    ax.set_title("Trend, Seasonality and Residual Decomposition")
    ax.legend()
    k, v = save_fig(fig, "trend_seasonality")
    saved_files[k] = v

    # -----------------------
    # 2️ Volatility over Time
    # -----------------------
    vol_cols = [c for c in df.columns if c.startswith("volatility_")]
    if vol_cols:
        fig, ax = plt.subplots(figsize=(10, 4))
        for col in vol_cols:
            ax.plot(df.index, df[col], label=col)
        ax.set_title("Volatility (Rolling Std of Returns)")
        ax.legend()
        k, v = save_fig(fig, "volatility")
        saved_files[k] = v

    # -----------------------
    # 3 Forecast Plots
    # -----------------------
    if arima_forecast is not None and not arima_forecast.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df.index, df[price_col], label="Historical", color="red")
        ax.plot(
            arima_forecast["forecast_date"],
            arima_forecast["forecast_value"],
            label="ARIMA Forecast",
            color="darkgreen"
        )
        ax.set_title("ARIMA Forecast")
        ax.legend()
        k, v = save_fig(fig, "forecast_arima")
        saved_files[k] = v

    if sarima_forecast is not None and not sarima_forecast.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df.index, df[price_col], label="Historical", color="navy")
        ax.plot(
            sarima_forecast["forecast_date"],
            sarima_forecast["forecast_value"],
            label="SARIMA Forecast",
            color="crimson"
        )
        ax.set_title("SARIMA Forecast (Seasonal Model)")
        ax.legend()
        k, v = save_fig(fig, "forecast_sarima")
        saved_files[k] = v

    if lstm_forecast is not None and not lstm_forecast.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df.index, df[price_col], label="Historical", color="black")
        ax.plot(
            lstm_forecast["forecast_date"],
            lstm_forecast["forecast_value"],
            label="LSTM Forecast",
            color="purple"
        )
        ax.set_title("LSTM Forecast")
        ax.legend()
        k, v = save_fig(fig, "forecast_lstm")
        saved_files[k] = v

    # -----------------------
    # 4️ ACF & PACF
    # -----------------------
    if len(df) > 20:
        fig, ax = plt.subplots(figsize=(8, 4))
        plot_acf(df[price_col].dropna(), ax=ax, lags=30)
        ax.set_title("Autocorrelation Function (ACF)")
        k, v = save_fig(fig, "acf_plot")
        saved_files[k] = v

        fig, ax = plt.subplots(figsize=(8, 4))
        plot_pacf(df[price_col].dropna(), ax=ax, lags=30)
        ax.set_title("Partial Autocorrelation Function (PACF)")
        k, v = save_fig(fig, "pacf_plot")
        saved_files[k] = v

    # -----------------------
    # 5️ RSI & MACD
    # -----------------------
    if "rsi" in df.columns or "macd" in df.columns:
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        if "rsi" in df.columns:
            axes[0].plot(df.index, df["rsi"], color="teal")
            axes[0].set_title("Relative Strength Index (RSI)")
            axes[0].axhline(70, color="red", linestyle="--", alpha=0.7)
            axes[0].axhline(30, color="green", linestyle="--", alpha=0.7)
        if "macd" in df.columns and "macd_signal" in df.columns:
            axes[1].plot(df.index, df["macd"], label="MACD", color="purple")
            axes[1].plot(df.index, df["macd_signal"], label="Signal", color="orange")
            axes[1].set_title("MACD Indicator")
            axes[1].legend()
        k, v = save_fig(fig, "rsi_macd")
        saved_files[k] = v

    print(f"✅ Saved all analysis visualizations (run_id={run_id}) to: {output_dir}")
    return saved_files
