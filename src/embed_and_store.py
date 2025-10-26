# src/embed_and_store.py

from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import chromadb
from chromadb.config import Settings

from utils.extract_numeric_values import extract_numeric_values
from src.data_loader import load_time_series_documents
from src.cnn_encoder import CNNEmbeddingPipeline


class TimeSeriesEmbedder:
    """
    Loads time-series data → extracts numeric values → creates windows → generates embeddings → stores in ChromaDB.
    """

    def __init__(
        self,
        data_dir: str = "data",
        window_size: int = 256,
        embedding_dim: int = 384,
        batch_size: int = 64,
        persist_directory: str = "chroma_db",
        collection_name: str = "financial_timeseries"
    ):
        self.data_dir = data_dir
        self.window_size = window_size
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        # Initialize ChromaDB with correct client API
        settings = Settings(persist_directory=persist_directory)
        self.client = chromadb.Client(settings=settings)
        print(f"[INFO] Initialized ChromaDB client with persist_directory: {persist_directory}")

        # Create or get collection
        if collection_name in [c.name for c in self.client.list_collections()]:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"[INFO] Loaded existing collection: {collection_name}")
        else:
            self.collection = self.client.create_collection(name=collection_name)
            print(f"[INFO] Created new collection: {collection_name}")

        # Initialize CNN embedder
        self.pipeline = CNNEmbeddingPipeline(
            window_size=self.window_size,
            embedding_dim=self.embedding_dim,
            batch_size=self.batch_size
        )
        print(f"[INFO] CNN Embedding pipeline initialized: window_size={self.window_size}, embedding_dim={self.embedding_dim}")

    def load_documents(self) -> List[Any]:
        docs = load_time_series_documents(self.data_dir)
        print(f"[INFO] Loaded {len(docs)} documents from {self.data_dir}")
        print(f"\nExample document:")
        print(f"metadata:\n{docs[0].metadata}")
        print(f"page_content:\n{docs[0].page_content}")
        return docs

    def create_windows(self, docs: List[Any], col: str = "Close"):
        prices = extract_numeric_values(docs, col)
        print(f"[INFO] Extracted {len(prices)} numeric values from column '{col}'")

        windows: List[np.ndarray] = []
        metadata_list: List[Dict[str, Any]] = []

        for i in range(0, len(prices) - self.window_size + 1):
            w = np.array(prices[i : i + self.window_size], dtype=np.float32)
            windows.append(w)

            metadata = {
                "source": docs[i].metadata.get("source", "unknown"),
                "start_index": i,
                "end_index": i + self.window_size,
                "start_date": docs[i].metadata.get("Date", None),
                "end_date": docs[i + self.window_size - 1].metadata.get("Date", None),
            }
            metadata_list.append(metadata)


        print(f"[INFO] Created {len(windows)} windows of size {self.window_size}")
        return windows, metadata_list

    def generate_embeddings(self, windows: List[np.ndarray]) -> np.ndarray:
        print("[INFO] Generating embeddings …")
        embeddings = self.pipeline.embed_windows(windows)
        print(f"[INFO] Embeddings shape: {embeddings.shape}")
        return embeddings

    def store_embeddings(self, embeddings: np.ndarray, metadata_list: List[Dict[str, Any]]):
        ids = [f"window_{i}" for i in range(len(embeddings))]
        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            metadatas=metadata_list
        )
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        print(f"[INFO] Stored {len(embeddings)} embeddings in ChromaDB persist directory.")

    def run(self):
        docs = self.load_documents()
        windows, metadata_list = self.create_windows(docs, col="Close")
        embeddings = self.generate_embeddings(windows)
        self.store_embeddings(embeddings, metadata_list)
        print("[✅ DONE] Time-series embedding & storage pipeline complete.")
