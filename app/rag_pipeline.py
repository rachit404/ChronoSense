from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from utils.env import GROQ_API_KEY
from utils.logger import logger  # ✅ import your custom logger


# Initialize Groq chat model
try:
    chat_model = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.3-70b-versatile", temperature=0.0)
    logger.info("Groq Chat model initialized successfully: llama-3.3-70b-versatile")
except Exception as e:
    logger.exception(f"Failed to initialize Groq Chat model: {e}")
    raise


# Initialize free embedding function
try:
    embedding_fn = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    logger.info("HuggingFace embeddings initialized (model: all-MiniLM-L6-v2)")
except Exception as e:
    logger.exception(f"Error initializing HuggingFace embeddings: {e}")
    raise


# Path to Chroma DB
VECTOR_STORE_PATH = "./models/vector_store/chroma_db"


def get_retriever():
    """
    Load Chroma vector store with embeddings and return a retriever object.
    """
    try:
        vectordb = Chroma(
            persist_directory=VECTOR_STORE_PATH,
            embedding_function=embedding_fn
        )
        logger.info(f"Chroma vector store loaded from: {VECTOR_STORE_PATH}")
        return vectordb.as_retriever(search_kwargs={"k": 5})
    except Exception as e:
        logger.exception(f"Failed to initialize Chroma retriever: {e}")
        raise


def create_rag_chain():
    """
    Create a retrieval chain combining Groq chat model and Chroma retriever
    using langchain-classic style.
    """
    try:
        retriever = get_retriever()

        # Custom prompt template
        prompt_template = """
        You are ChronoSense, a chatbot for time-series analytics. 
        Use the following context to answer the user's question accurately.

        Context:
        {context}

        Question:
        {input}

        Answer:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "input"])

        # Step 1: Create a combine documents chain
        combine_docs_chain = create_stuff_documents_chain(
            llm=chat_model,
            prompt=prompt
        )
        logger.debug("Combine documents chain created successfully.")

        # Step 2: Create the retrieval chain
        rag_chain = create_retrieval_chain(retriever, combine_docs_chain)
        logger.info("RAG chain successfully created.")

        return rag_chain

    except Exception as e:
        logger.exception(f"Error creating RAG chain: {e}")
        raise


def answer_query(query, rag_chain):
    """
    Get answer from RAG pipeline given a user query.
    """
    logger.info(f"Processing user query: {query}")
    try:
        result = rag_chain.invoke({"input": query})
        logger.debug(f"Raw RAG output: {result}")

        answer = result.get("answer") or result.get("output_text") or "No response generated."
        sources = result.get("context") or result.get("source_documents", [])
        logger.info("RAG response successfully generated.")
        return answer, sources

    except Exception as e:
        logger.exception(f"Error during RAG query processing: {e}")
        raise
