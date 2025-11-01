# =====================================================
# MAIN ANALYSIS PIPELINE (ENHANCED)
# =====================================================

# Imports
import pandas as pd
from pathlib import Path
from utils.slice_df import slice_df
from src.data_loader import load_data
from src.pre_data_analysis import (
    precompute_basic_features,
    save_pre_analysis
)
from src.time_series_analysis import (
    compute_trend_seasonality,
    detect_anomalies,
    forecast_arima,
    forecast_sarima,
    forecast_lstm
)
from src.visualization_generator import generate_timeseries_visuals
from src.summary_doc_generator import create_langchain_summaries

from src.vector_pipeline import EmbeddingEngine, VectorStore
from src.chat_groq_client import ChatGroqClient

def chrono_sense_pipeline(file_name: str, col: str):
    data_path = "data/" + file_name
    df = load_data(data_path, col=col)

    print("[INFO] Data loaded. Index preview:", df.index[:3])

    df = precompute_basic_features(df, col)
    df = compute_trend_seasonality(df, col)
    df = detect_anomalies(df, col)

    # -----------------------------------------------------
    # 2️⃣ Forecasts (ARIMA, SARIMA, LSTM)
    # -----------------------------------------------------
    sliced_df = slice_df(df, slice_range=(750, None))
    print("⏳ Running forecasts...")
    arima_forecast = forecast_arima(sliced_df, col, steps=500)
    sarima_forecast = forecast_sarima(sliced_df, col, steps=500, order=(2,1,10), seasonal_order=(1,1,1,12))
    lstm_forecast = forecast_lstm(sliced_df, col, epochs=15, steps=500)

    # Merge forecasts (optional consolidated view)
    df_forecasts = (
        arima_forecast[["forecast_date", "forecast_value"]]
        .rename(columns={"forecast_value": "forecast_arima"})
        .merge(
            sarima_forecast[["forecast_date", "forecast_value"]].rename(columns={"forecast_value": "forecast_sarima"}),
            on="forecast_date",
            how="outer"
        )
        .merge(
            lstm_forecast[["forecast_date", "forecast_value"]].rename(columns={"forecast_value": "forecast_lstm"}),
            on="forecast_date",
            how="outer"
        )
    )
    print("✅ Forecasts merged. Preview:")
    print(df_forecasts.head())
    save_pre_analysis(df, Path("../data/pre_analysis_enhanced.csv"))



    # -----------------------------------------------------
    # 3️⃣ Generate and Save Visuals
    # -----------------------------------------------------
    run_id = "007"
    output_dir = Path(f"visualizations/{run_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    visuals = generate_timeseries_visuals(
        df,
        price_col=col,
        arima_forecast=arima_forecast,
        sarima_forecast=sarima_forecast,
        lstm_forecast=lstm_forecast,
        output_dir=str(output_dir),
        run_id=run_id
    )

    # -----------------------------------------------------
    # 4️⃣ Create LangChain Summaries (auto-links visuals)
    # -----------------------------------------------------
    docs = create_langchain_summaries(df, price_col=col, freq="M")
    for d in docs:
        d.metadata.update(visuals)

    print("✅ Example Document Summary:")
    print("─────────────────────────────")
    print(docs[-1].page_content)
    print("─────────────────────────────")
    print("Metadata:", docs[-1].metadata)

    # -----------------------------------------------------
    # 5️⃣ Optional: Save intermediate data
    # -----------------------------------------------------
    merged_output = Path("../data/processed_forecasts.csv")
    df_forecasts.to_csv(merged_output, index=False)
    print(f"📁 Saved forecast results to: {merged_output.resolve()}")

    print("🎯 Pipeline complete. Ready for embedding or query stage.")


    # Step 1: Embed
    embedder = EmbeddingEngine(model_name="all-MiniLM-L6-v2")
    doc_embeddings = embedder.embed_documents(docs)

    # Step 2: Store in Chroma
    store = VectorStore(persist_dir="./chroma_store", collection_name="goog_timeseries")
    store.clear()
    store.add_documents(docs, doc_embeddings)

    user_query = "Explain the volatility trend for March 2024"
    query_vec = embedder.embed_query(user_query)
    results = store.query(query_vec, top_k=3)

    for r in results:
        print("\n🧩 Match:")
        print("Text:", r["text"])
        print("Score:", round(r["score"], 4))
        print("Metadata:", r["metadata"])
    
    return True
        
        

    # queries = [
    #     "Explain the volatility trend for March 2024",
    #     "What was the average closing price in 2021?",
    #     "Did the SARIMA model detect any seasonal shifts in 2024?",
    #     "Compare LSTM and ARIMA forecasts for early 2025"
    # ]
def chat_query(query: str, groq_llm: ChatGroqClient):
    # groq_llm = ChatGroqClient()
    # for query in queries:
        embedder = EmbeddingEngine(model_name="all-MiniLM-L6-v2")
        query_vec = embedder.embed_query(query)
        store = VectorStore(persist_dir="./chroma_store", collection_name="goog_timeseries")
        results = store.query(query_vec, top_k=50)
        # Combine retrieved texts into a single context string
        context = "\n---\n".join([r["text"] for r in results])
        answer = groq_llm.ask(context, query)
        print(f"\n💬 User Query: {query}")
        print(f"\n💬 Context: {context}")
        print("💬 LLM Answer:")
        print(answer)
        return answer
    
    
    
    
    
# # --- CUSTOM CHAT UI ---
# st.markdown("""
#     <style>
#     .chat-message {
#         display: flex;
#         margin-bottom: 10px;
#     }
#     .chat-message.user {
#         justify-content: flex-end;
#     }
#     .chat-message.bot {
#         justify-content: flex-start;
#     }
#     .message {
#         max-width: 70%;
#         padding: 10px 15px;
#         border-radius: 12px;
#         line-height: 1.4;
#         font-size: 16px;
#     }
#     .user .message {
#         background-color: #0066FF;
#         color: white;
#         border-bottom-right-radius: 0;
#     }
#     .bot .message {
#         background-color: #262730;
#         color: #FFFFFF;
#         border-bottom-left-radius: 0;
#     }
#     </style>
# """, unsafe_allow_html=True)

# # --- DISPLAY CHAT HISTORY ---
# for chat in st.session_state.chat_history:
#     if chat["role"] == "user":
#         st.markdown(
#             f"""
#             <div class="chat-message user">
#                 <div class="message">{chat['message']}</div>
#             </div>
#             """,
#             unsafe_allow_html=True,
#         )
#     else:
#         st.markdown(
#             f"""
#             <div class="chat-message bot">
#                 <div class="message"> {chat['message']}</div>
#             </div>
#             """,
#             unsafe_allow_html=True,
#         )