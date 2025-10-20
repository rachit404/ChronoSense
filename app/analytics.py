import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import io
import warnings
from app.rag_pipeline import answer_query  # Make sure this path matches your project

# Suppress Prophet warnings
warnings.filterwarnings("ignore")

def plot_time_series(df, datetime_col, target_col):
    """
    Generate a simple time-series plot and return as bytes for Streamlit.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df[datetime_col], df[target_col], marker='o')
    ax.set_xlabel("Date")
    ax.set_ylabel(target_col)
    ax.set_title(f"{target_col} over Time")
    fig.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    return buf

def forecast_time_series(df, datetime_col, target_col, periods=30):
    """
    Perform a simple forecast using Prophet.
    """
    df_prophet = df[[datetime_col, target_col]].rename(columns={datetime_col: 'ds', target_col: 'y'})
    model = Prophet()
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

def compute_basic_stats(df, target_col):
    """
    Return descriptive statistics for a column.
    """
    return df[target_col].describe().to_dict()

def handle_analytical_query(df, datetime_col, query, qa_chain=None):
    """
    Process analytical queries with optional LLM assistance.
    Example query types:
      - "trend of sales"
      - "forecast temperature next 10 days"
      - "summary website_visits"
    """
    query_lower = query.lower()

    numeric_cols = [col.lower() for col in df.select_dtypes(include=['number']).columns]
    col_map = {col.lower(): col for col in df.select_dtypes(include=['number']).columns}

    # Trend / plotting
    for col_lower in numeric_cols:
        if f"trend {col_lower}" in query_lower or f"plot {col_lower}" in query_lower:
            plot_buf = plot_time_series(df, datetime_col, col_map[col_lower])
            return {"type": "plot", "buffer": plot_buf, "description": f"Trend of {col_map[col_lower]}"}

    # Forecast
    for col_lower in numeric_cols:
        if f"forecast {col_lower}" in query_lower:
            forecast_df = forecast_time_series(df, datetime_col, col_map[col_lower])
            return {"type": "forecast", "data": forecast_df, "description": f"Forecast of {col_map[col_lower]}"}

    # Descriptive stats
    for col_lower in numeric_cols:
        if f"summary {col_lower}" in query_lower or f"stats {col_lower}" in query_lower:
            stats = compute_basic_stats(df, col_map[col_lower])
            return {"type": "stats", "data": stats, "description": f"Descriptive stats for {col_map[col_lower]}"}

    # Fallback: use RAG LLM chain if available
    if qa_chain:
        answer, sources = answer_query(query, qa_chain)
        return {"type": "llm", "answer": answer, "sources": sources}

    return {"type": "error", "message": "Query not understood or no LLM chain provided."}
