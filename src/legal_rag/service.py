from __future__ import annotations

try:
    from dotenv import load_dotenv

    load_dotenv()  # GEMINI_API_KEY / Langfuse keys from .env
except ImportError:
    pass

import bentoml

from legal_rag.config import TOP_K
from legal_rag.rag import ask as rag_ask


@bentoml.service(
    resources={"cpu": "2"},
    traffic={"timeout": 300},  # Ollama generation + Gemini review can be slow
)
class LegalRAG:
    """Egyptian Civil Code RAG as an HTTP API."""

    @bentoml.api
    def ask(self, question: str, top_k: int = TOP_K) -> dict:
        result = rag_ask(question=question, top_k=top_k)

        return {
            "question": result["question"],
            "answer": result["answer"],
            "sources": result["sources"],
        }