from langchain.embeddings import GroqEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd

embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Path to persist Chroma vector store
VECTOR_STORE_PATH = "./models/vector_store/chroma_db"

def dataset_to_documents(df, datetime_col):
    """
    Convert each row of the dataset into a text Document for embeddings.
    """
    documents = []
    for _, row in df.iterrows():
        # Convert row to text
        text = " | ".join([f"{col}: {row[col]}" for col in df.columns])
        metadata = { "datetime": str(row[datetime_col]) }
        documents.append(Document(page_content=text, metadata=metadata))
    return documents

def create_chroma_vectorstore(documents):
    """
    Generate embeddings and store in Chroma vector store.
    """
    vectordb = Chroma.from_documents(
        documents,
        embeddings_model,
        persist_directory=VECTOR_STORE_PATH
    )
    vectordb.persist()
    return vectordb
