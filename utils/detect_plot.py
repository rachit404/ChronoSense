def detect_plot_types(user_query: str, run_id: str = "007") -> list:
    """
    Determine relevant plot filenames based on the user query.
    Returns a list of actual file names matching the user's intent.
    """
    q = user_query.lower()
    matches = []

    # Forecast-related
    if "arima" in q:
        matches.append(f"forecast_arima_{run_id}.png")
    if "sarima" in q:
        matches.append(f"forecast_sarima_{run_id}.png")
    if "lstm" in q:
        matches.append(f"forecast_lstm_{run_id}.png")

    # Technical indicators
    if "rsi" in q or "macd" in q:
        matches.append(f"rsi_macd_{run_id}.png")

    # Statistical diagnostics
    if "acf" in q:
        matches.append(f"acf_plot_{run_id}.png")
    if "pacf" in q:
        matches.append(f"pacf_plot_{run_id}.png")

    # Trend / Seasonality
    if "trend" in q or "season" in q:
        matches.append(f"trend_seasonality_{run_id}.png")

    # Volatility
    if "volatility" in q or "variance" in q:
        matches.append(f"volatility_{run_id}.png")

    # General "forecast" — include all major forecast types
    if "forecast" in q and not matches:
        matches.extend([
            f"forecast_arima_{run_id}.png",
            f"forecast_sarima_{run_id}.png",
            f"forecast_lstm_{run_id}.png"
        ])

    # Remove duplicates, preserve order
    return list(dict.fromkeys(matches))


# def detect_plot_type(user_query: str) -> str:
#     """
#     Determine which visualization to display based on the user query.
#     """
#     q = user_query.lower()
#     if "arima" in q:
#         return "image_forecast_arima"
#     elif "sarima" in q:
#         return "image_forecast_sarima"
#     elif "lstm" in q:
#         return "image_forecast_lstm"
#     elif "rsi" in q or "macd" in q:
#         return "image_rsi_macd"
#     elif "acf" in q:
#         return "image_acf_plot"
#     elif "pacf" in q:
#         return "image_pacf_plot"
#     elif "trend" in q or "season" in q:
#         return "image_trend_seasonality"
#     elif "volatility" in q or "variance" in q:
#         return "image_volatility"
#     else:
#         return None