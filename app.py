# import numpy as np

# from utils.extract_numeric_values import extract_numeric_values

# from src.data_loader import load_time_series_documents
# from src.cnn_encoder import CNNEmbeddingPipeline


# if __name__ == "__main__":
#     docs = load_time_series_documents("data")
#     print(f"\nExample document:")
#     print(f"metadata:\n{docs[0].metadata}")
#     print(f"page_content:\n{docs[0].page_content}")
    
#     close_prices = extract_numeric_values(docs, col='Close')
#     print(f"[INFO] Close Prices: {close_prices[:5]}")
    
#     # 3. Create windows of size 256
#     window_size = 256
#     windows = []
#     for i in range(0, len(close_prices) - window_size + 1):
#         window = close_prices[i:i + window_size]
#         windows.append(np.array(window, dtype=np.float32))

#     print(f"[INFO] Created {len(windows)} windows of size {window_size}")
    
#     # 4. Initialize CNN embedding pipeline
#     pipeline = CNNEmbeddingPipeline(window_size=window_size, embedding_dim=128)

#     # 5. Generate embeddings
#     embeddings = pipeline.embed_windows(windows)
# app.py

# app.py

from src.embed_and_store import TimeSeriesEmbedder
from src.search import TimeSeriesRetriever
from src.query_embed_and_store import QueryEmbedAndStore

if __name__ == "__main__":
    # -------------------------------
    # Step 1: Embed and store time-series data
    # -------------------------------
    ts_embedder = TimeSeriesEmbedder(
        data_dir="data",
        window_size=256,
        embedding_dim=384,
        batch_size=64,
        persist_directory="chroma_db"
    )
    ts_embedder.run()

    # -------------------------------
    # Step 2: Embed and store text queries
    # -------------------------------
    query_embedder = QueryEmbedAndStore(
        client=ts_embedder.client,
        collection_name="query_embeddings",
        model_name="all-MiniLM-L6-v2"
    )

    # Example queries
    queries = [
        "What was the closing price trend in October 2020?",
        "Show recent stock spikes in S&P 500",
        "Analyze Bitcoin closing trends in November 2021"
    ]

    # Store query embeddings
    query_embedder.store_queries(queries, metadatas=[{"type": "finance"}]*len(queries))

    # -------------------------------
    # Step 3: Initialize retriever for numeric windows
    # -------------------------------
    ts_retriever = TimeSeriesRetriever(
        persist_directory="chroma_db",
        collection_name="financial_timeseries"
    )

    # -------------------------------
    # Step 4: Retrieve context for each query
    # -------------------------------
    for query in queries:
        context = ts_retriever.retrieve_context_str(query, top_k=5)
        print(f"\nQuery: {query}")
        print(f"Retrieved context:\n{context}")



