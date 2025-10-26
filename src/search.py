# src/search.py

from typing import List
import chromadb
from chromadb.config import Settings
from src.cnn_encoder import CNNEmbeddingPipeline
from src.query_embed_and_store import QueryEmbedAndStore
import numpy as np

class TimeSeriesRetriever:
    """
    Retriever for numeric time-series windows stored in ChromaDB using CNN embeddings.
    Text queries are embedded using SentenceTransformer (QueryEmbedAndStore).
    """

    def __init__(
        self,
        persist_directory: str = "chroma_db",
        collection_name: str = "financial_timeseries",
        window_size: int = 256,
        embedding_dim: int = 128,
        batch_size: int = 64,
        query_model_name: str = "all-MiniLM-L6-v2"
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        # Initialize ChromaDB client for numeric embeddings
        settings = Settings(persist_directory=persist_directory)
        self.client = chromadb.Client(settings=settings)

        # Load collection for numeric embeddings
        self.collection = self.client.get_collection(self.collection_name)

        # Numeric CNN embedder (for internal use or similarity calculations if needed)
        self.db_embedder = CNNEmbeddingPipeline(
            window_size=window_size,
            embedding_dim=embedding_dim,
            batch_size=batch_size
        )

        # Query embedder using SentenceTransformer (separate from numeric embeddings)
        self.query_embedder = QueryEmbedAndStore(
            client=self.client,
            collection_name="query_embeddings",
            model_name=query_model_name
        )

        print(f"[INFO] TimeSeriesRetriever initialized with numeric collection '{collection_name}'")

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a text query using QueryEmbedAndStore.
        """
        return self.query_embedder.embed_single_query(query)

    def retrieve_top_k(self, query: str, top_k: int = 5) -> List[dict]:
        """
        Retrieve top-k most relevant numeric time-series windows from ChromaDB.
        """
        # 1. Embed the query
        query_vec = self.embed_query(query)

        # --- NOTE ---
        # The query vector dimension must match the numeric embeddings (128) to avoid errors.
        # If using SentenceTransformer (384) embeddings, store/query in a separate collection.

        # 2. Query the numeric collection (requires compatible embeddings)
        results = self.collection.query(
            query_embeddings=[query_vec.tolist()],  # Ensure it's a list of 1D vectors
            n_results=top_k
        )

        # 3. Process results
        retrieved = []
        for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
            retrieved.append({
                "source": meta.get("source", "unknown"),
                "start_index": meta.get("start_index"),
                "end_index": meta.get("end_index"),
                "start_date": meta.get("start_date"),
                "end_date": meta.get("end_date"),
                "doc": doc
            })

        return retrieved

    def retrieve_context_str(self, query: str, top_k: int = 5) -> str:
        """
        Retrieve top-k numeric context windows as a formatted string for RAG.
        """
        retrieved = self.retrieve_top_k(query, top_k)
        context = "\n".join([
            f"{item['start_date']} to {item['end_date']}: {item['source']}"
            for item in retrieved
        ])
        return context
