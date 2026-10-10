
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from legal_rag.config import TOP_K
from legal_rag.rag import ask as rag_ask

app = FastAPI(
    title="Arabic Legal RAG API",
    description="Egyptian Civil Code question-answering API",
    version="1.0.0",
)


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(..., description="Legal question")

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Question must not be empty.")
        return value


class AskResponse(BaseModel):
    answer: str = Field(..., min_length=1)
    sources: list[str]


class HealthResponse(BaseModel):
    status: Literal["healthy"]
    documents_indexed: int = Field(..., ge=0)


def get_documents_indexed() -> int:
    """Count indexed records in the processed corpus."""
    from legal_rag.config import PROJECT_ROOT

    path = PROJECT_ROOT / "data" / "processed" / "civil_code.json"

    if not path.exists():
        return 0

    with path.open("r", encoding="utf-8") as file:
        documents = json.load(file)

    if isinstance(documents, list):
        return len(documents)

    return 0


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    try:
        result = rag_ask(
            question=request.question,
            top_k=TOP_K,
        )

        # Reject empty answers, including whitespace-only answers.
        answer = str(result.get("answer") or "").strip()
        if not answer:
            raise HTTPException(
                status_code=502,
                detail="The RAG pipeline returned an empty answer.",
            )

        sources = result.get("sources") or []

        return AskResponse(
            answer=answer,
            sources=[str(source) for source in sources],
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to process the legal question.",
        ) from exc


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        documents_indexed=get_documents_indexed(),
    )
