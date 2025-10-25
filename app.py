import numpy as np

from utils.extract_numeric_values import extract_numeric_values

from src.data_loader import load_time_series_documents
from src.cnn_encoder import CNNEmbeddingPipeline


if __name__ == "__main__":
    docs = load_time_series_documents("data")
    print(f"\nExample document:")
    print(f"metadata:\n{docs[0].metadata}")
    print(f"page_content:\n{docs[0].page_content}")
    
    close_prices = extract_numeric_values(docs, col='Close')
    print(f"[INFO] Close Prices: {close_prices[:5]}")
    
    # 3. Create windows of size 256
    window_size = 256
    windows = []
    for i in range(0, len(close_prices) - window_size + 1):
        window = close_prices[i:i + window_size]
        windows.append(np.array(window, dtype=np.float32))

    print(f"[INFO] Created {len(windows)} windows of size {window_size}")
    
    # 4. Initialize CNN embedding pipeline
    pipeline = CNNEmbeddingPipeline(window_size=window_size, embedding_dim=128)

    # 5. Generate embeddings
    embeddings = pipeline.embed_windows(windows)
    