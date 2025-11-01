# ==============================================
# vector_pipeline.py
# ==============================================

from typing import List
from langchain_core.documents import Document
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

# -------------------------------------------------
# 1️⃣ Embedding Engine
# -------------------------------------------------
class EmbeddingEngine:
    """
    A wrapper around SentenceTransformer for encoding LangChain Documents.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        print(f"✅ Loaded embedding model: {model_name} on {device}")

    def embed_documents(self, docs: List[Document]) -> np.ndarray:
        """
        Encodes a list of LangChain Document objects into dense vectors.
        """
        texts = [doc.page_content for doc in docs]
        embeddings = self.model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """
        Encodes a single query string into a dense vector.
        """
        return self.model.encode([query], convert_to_numpy=True)[0]


# -------------------------------------------------
# 2️⃣ Vector Store (ChromaDB)
# -------------------------------------------------
class VectorStore:
    """
    A persistent vector store using ChromaDB.
    """

    def __init__(self, persist_dir: str = "./chroma_store", collection_name: str = "timeseries_docs", model_name: str = "all-MiniLM-L6-v2"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.model_name = model_name
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn
        )
        print(f"✅ Connected to ChromaDB collection: {collection_name}")

    def add_documents(self, docs: List[Document], embeddings: np.ndarray):
        """
        Adds documents + embeddings to Chroma collection.
        """
        ids = [f"doc_{i}" for i in range(len(docs))]
        metadatas = [doc.metadata for doc in docs]
        texts = [doc.page_content for doc in docs]
        self.collection.add(documents=texts, embeddings=embeddings.tolist(), metadatas=metadatas, ids=ids)
        print(f"📚 Added {len(docs)} documents to ChromaDB")

    def query(self, query_embedding: np.ndarray, top_k: int = 3):
        """
        Retrieves top-k similar documents given a query embedding.
        """
        results = self.collection.query(query_embeddings=[query_embedding.tolist()], n_results=top_k)
        hits = [
            {"text": results["documents"][0][i], "score": results["distances"][0][i], "metadata": results["metadatas"][0][i]}
            for i in range(len(results["documents"][0]))
        ]
        return hits

    def clear(self):
        """Clears all documents in the ChromaDB collection safely."""
        all_ids = self.collection.get()["ids"]
        if all_ids:
            self.collection.delete(ids=all_ids)
            print(f"🧹 Cleared {len(all_ids)} documents from ChromaDB collection.")
        else:
            print("🧹 Collection already empty.")

