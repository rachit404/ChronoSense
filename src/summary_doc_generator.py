# =====================================================
# Summary for Embedding (with Forecast Integration)
# =====================================================
from langchain_core.documents import Document
import pandas as pd

def create_langchain_summaries(
    df: pd.DataFrame,
    price_col: str,
    freq: str = 'M',
    arima_forecast: pd.DataFrame = None,
    sarima_forecast: pd.DataFrame = None,
    lstm_forecast: pd.DataFrame = None
) -> list:
    """
    Create multiple LangChain Documents summarizing trends/anomalies per time chunk (e.g., month),
    and integrate model forecasts (ARIMA, SARIMA, LSTM) corresponding to the same or next period.

    Enhancements:
    - Adds explicit time tags (Year, Month, Period Range)
    - Includes volatility interpretation (low/moderate/high)
    - Merges forecasts if their dates fall within or just after the period
    - Automatically attaches visualization file names (if exist)
    """

    docs = []

    # Ensure DataFrame has DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        date_cols = [c for c in df.columns if 'date' in c.lower()]
        if date_cols:
            df[date_cols[0]] = pd.to_datetime(df[date_cols[0]], errors='coerce')
            df = df.set_index(date_cols[0])
        else:
            raise ValueError("No DatetimeIndex or 'Date' column found for resampling.")

    # -------------------------------
    # Group data by time frequency
    # -------------------------------
    grouped = df.resample(freq)

    # Helper: safely fetch forecast values for a given date range
    def get_forecast_for_period(forecast_df, start_date, end_date):
        if forecast_df is None or "forecast_date" not in forecast_df.columns:
            return None
        mask = (forecast_df["forecast_date"] >= pd.Timestamp(start_date)) & (
            forecast_df["forecast_date"] <= pd.Timestamp(end_date)
        )
        if mask.any():
            values = forecast_df.loc[mask, "forecast_value"]
            if not values.empty:
                return round(values.mean(), 4)
        return None

    # -------------------------------
    # Iterate through each monthly chunk
    # -------------------------------
    for period, group in grouped:
        if len(group) < 5:
            continue

        # --- Basic stats ---
        mean_price = group[price_col].mean()
        vol = group["daily_return"].std() if "daily_return" in group else group[price_col].pct_change().std()
        anomalies = group["zscore_anomaly"].sum() if "zscore_anomaly" in group else 0
        trend_desc = "increasing" if group["trend"].iloc[-1] > group["trend"].iloc[0] else "decreasing"

        # --- Time context ---
        start_date = group.index.min().date()
        end_date = group.index.max().date()
        year = period.year
        month = period.month
        month_name = period.strftime("%B")

        # --- Interpret volatility level ---
        if vol < 0.01:
            vol_level = "low"
        elif vol < 0.03:
            vol_level = "moderate"
        else:
            vol_level = "high"

        # --- Attach model forecasts for this period or the next ---
        arima_val = get_forecast_for_period(arima_forecast, start_date, end_date)
        sarima_val = get_forecast_for_period(sarima_forecast, start_date, end_date)
        lstm_val = get_forecast_for_period(lstm_forecast, start_date, end_date)

        forecast_text_parts = []
        if any([arima_val, sarima_val, lstm_val]):
            forecast_text_parts.append("\nForecasts:")
            if arima_val:
                forecast_text_parts.append(f"- ARIMA predicted average {price_col} ≈ {arima_val:.2f}")
            if sarima_val:
                forecast_text_parts.append(f"- SARIMA predicted average {price_col} ≈ {sarima_val:.2f}")
            if lstm_val:
                forecast_text_parts.append(f"- LSTM predicted average {price_col} ≈ {lstm_val:.2f}")
        forecast_text = "\n".join(forecast_text_parts)

        # --- Enhanced textual summary ---
        text = (
            f"Period Summary: {month_name} {year}\n"
            f"Date Range: {start_date} to {end_date}\n"
            f"Year: {year}, Month: {month:02d}\n"
            f"Trend: The {price_col} trend during this period was {trend_desc}.\n"
            f"Average {price_col}: {mean_price:.2f}\n"
            f"Volatility: {vol:.4f} ({vol_level})\n"
            f"Detected anomalies: {anomalies}\n"
            f"{forecast_text}\n\n"
            f"Summary: In {month_name} {year}, the {price_col} showed a {trend_desc} trend with "
            f"{vol_level} volatility (std {vol:.4f}). The mean price was {mean_price:.2f}, and "
            f"{anomalies} anomalies were detected between {start_date} and {end_date}."
        )

        # --- Metadata for retrieval & filtering ---
        meta = {
            "year": int(year),
            "month": int(month),
            "month_name": str(month_name),
            "period_start": str(start_date),
            "period_end": str(end_date),
            "mean_price": float(round(mean_price, 4)),
            "volatility": float(round(vol, 4)),
            "volatility_level": str(vol_level),
            "num_anomalies": int(anomalies),
            "trend": str(trend_desc),
            "forecast_arima": float(arima_val) if arima_val is not None else 0.0,
            "forecast_sarima": float(sarima_val) if sarima_val is not None else 0.0,
            "forecast_lstm": float(lstm_val) if lstm_val is not None else 0.0,
        }


        # --- Optional image metadata if files exist ---
        # img_dir = "images"
        # for img_name in [
        #     "trend_seasonality.png", "volatility.png", "forecast_arima.png",
        #     "forecast_sarima.png", "forecast_lstm.png", "acf_plot.png",
        #     "pacf_plot.png", "rsi_macd.png"
        # ]:
        #     img_path = os.path.join(img_dir, img_name)
        #     if os.path.exists(img_path):
        #         meta[f"image_{img_name.split('.')[0]}"] = img_path

        docs.append(Document(page_content=text, metadata=meta))

    print(f"✅ Created {len(docs)} LangChain summaries (with forecasts & visuals).")
    return docs