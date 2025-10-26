# src/query_embed_and_store.py

from typing import List, Dict, Any
import numpy as np
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

class QueryEmbedAndStore:
    """
    Embeds natural language queries and stores them in a separate ChromaDB collection.
    """

    def __init__(
        self,
        client: chromadb.Client,
        collection_name: str = "query_embeddings",
        model_name: str = "all-MiniLM-L6-v2"
    ):
        self.client = client
        self.collection_name = collection_name

        # Load or create collection
        if collection_name in [c.name for c in self.client.list_collections()]:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"[INFO] Loaded existing query collection: '{collection_name}'")
        else:
            self.collection = self.client.create_collection(name=collection_name)
            print(f"[INFO] Created new query collection: '{collection_name}'")

        # Load SentenceTransformer model
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        print(f"[INFO] Loaded SentenceTransformer model: {model_name}")

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Convert a list of text queries into embeddings.
        """
        if not texts:
            raise ValueError("Input 'texts' list cannot be empty.")
        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        print(f"[INFO] Generated embeddings: {embeddings.shape}")
        return embeddings

    def embed_single_query(self, query: str) -> np.ndarray:
        """
        Embed a single text query.
        """
        embedding = self.embed_texts([query])[0]
        print(f"[INFO] Single query embedding shape: {embedding.shape}")
        return embedding

    def store_queries(self, queries: List[str], metadatas: List[Dict[str, Any]] = None):
        """
        Embed a list of queries and store them in ChromaDB.
        """
        embeddings = self.embed_texts(queries)
        ids = [f"query_{i}" for i in range(len(queries))]

        if metadatas is None:
            metadatas = [{} for _ in queries]

        self.collection.add(ids=ids, embeddings=embeddings.tolist(), metadatas=metadatas)
        print(f"[INFO] Stored {len(queries)} query embeddings in ChromaDB collection '{self.collection_name}'")

    def embed_and_store_single(self, query: str, metadata: Dict[str, Any] = None):
        """
        Embed a single query and store it in ChromaDB.
        """
        emb = self.embed_single_query(query)
        self.collection.add(
            ids=[f"query_0"],
            embeddings=[emb.tolist()],
            metadatas=[metadata or {}]
        )
        print(f"[INFO] Stored single query embedding in collection '{self.collection_name}'")
