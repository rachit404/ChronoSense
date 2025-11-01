"""
chat_groq_client.py

A lightweight, task-specific class to interact with ChatGroq Llama-3.3-70B-Versatile.
It takes a context (retrieved from Chroma) and a user query, builds a clear prompt,
and returns the model's answer.

Usage:
    from chat_groq_client import ChatGroqClient

    llm = ChatGroqClient(api_key="YOUR_GROQ_API_KEY")
    answer = llm.ask(context_str, user_query)
    print(answer)
"""

import os
from typing import Optional
from langchain_groq.chat_models import ChatGroq
from dotenv import load_dotenv
load_dotenv()


class ChatGroqClient:
    """
    Minimal wrapper for Groq's Llama-3.3-70B-Versatile model via LangChain.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.2,
        max_tokens: int = 512,
    ):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("❌ GROQ_API_KEY is not set. Provide it via constructor or environment variable.")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        # Initialize Groq model through LangChain wrapper
        self.llm = ChatGroq(
            api_key=self.api_key,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    def build_prompt(self, context: str, query: str) -> str:
        """
        Builds a focused prompt using retrieved context and user query.
        """
        instruction = (
            "You are a financial analysis assistant. "
            "Use the provided context (stock trends, volatility, and anomalies) "
            "to answer the user query accurately and concisely."
        )
        return f"{instruction}\n\nCONTEXT:\n{context}\n\nQUESTION:\n{query}\n\nAnswer clearly."

    def ask(self, context: str, query: str) -> str:
        """
        Sends the prompt to Groq and returns the model's reply.
        """
        prompt = self.build_prompt(context, query)
        response = self.llm.invoke(prompt)
        # Depending on LangChain version, ChatGroq may return a dict or AIMessage
        if hasattr(response, "content"):
            return response.content
        if isinstance(response, dict):
            return response.get("content") or str(response)
        return str(response)
