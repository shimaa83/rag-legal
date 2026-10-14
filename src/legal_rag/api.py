from __future__ import annotations

import json
import logging
import threading
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.concurrency import run_in_threadpool

from legal_rag.cache import cache_stats, ping
from legal_rag.config import PROJECT_ROOT, WARMUP_ON_STARTUP
from legal_rag.rag import CANDIDATE_K, TOP_K, LLMError
from legal_rag.rag import ask as rag_ask
from legal_rag.rag import flush_traces, is_warm, warm_up


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the search index, the reranker and the Ollama models ONCE, before
    # the first request. uvicorn starts serving only after this finishes.
    if WARMUP_ON_STARTUP:
        try:
            report = await run_in_threadpool(warm_up)
            logger.info("Startup warm-up finished: %s", report)
        except Exception:  # noqa: BLE001 - the API must still start
            logger.exception("Startup warm-up failed; models load lazily")

    yield

    # Make sure the last Langfuse observations are sent.
    await run_in_threadpool(flush_traces)


app = FastAPI(
    title="Arabic Legal RAG API",
    description="Egyptian Civil Code question-answering API",
    version="1.4.0",
    lifespan=lifespan,
)

# Prometheus metrics for Grafana
Instrumentator().instrument(app).expose(
    app,
    endpoint="/metrics",
    include_in_schema=False,
)


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(..., max_length=2000)
    top_k: int = Field(default=TOP_K, ge=1, le=100)
    candidate_k: int = Field(default=CANDIDATE_K, ge=1, le=200)
    verify: bool | None = None
    use_cache: bool = True

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Question must not be empty.")
        return value


class Source(BaseModel):
    article_number: int | str | None = None
    citation: str
    source_page: int | str | None = None


class AskResponse(BaseModel):
    answer: str = Field(..., min_length=1)
    sources: list[Source]
    verified: bool | None = None
    context: str = ""
    timings: dict[str, float] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: Literal["healthy"]
    documents_indexed: int = Field(..., ge=0)
    models_loaded: bool = False
    redis: Literal["up", "down", "disabled"] = "disabled"
    cache: dict[str, int | bool] = Field(default_factory=dict)


_docs_lock = threading.Lock()
_docs_count: int | None = None


def get_documents_indexed() -> int:
    """Count indexed documents. A missing file is NOT cached (retry later)."""
    global _docs_count

    if _docs_count is not None:
        return _docs_count

    path = PROJECT_ROOT / "data" / "processed" / "civil_code.json"

    if not path.exists():
        return 0

    with _docs_lock:
        if _docs_count is None:
            with path.open("r", encoding="utf-8") as file:
                documents = json.load(file)

            _docs_count = len(documents) if isinstance(documents, list) else 0

    return _docs_count


# Plain `def` on purpose: FastAPI runs it in a worker thread, so the long
# blocking Ollama calls do not freeze the event loop.
@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    try:
        result = rag_ask(
            question=request.question,
            top_k=request.top_k,
            candidate_k=max(request.candidate_k, request.top_k),
            verify=request.verify,
            use_cache=request.use_cache,
        )

    except LLMError as exc:
        logger.error("LLM backend error: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="The language model backend is unavailable.",
        ) from exc

    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    except Exception as exc:
        logger.exception("Unhandled error while answering a legal question")
        raise HTTPException(
            status_code=500,
            detail="Failed to process the legal question.",
        ) from exc

    answer = str(result.get("answer") or "").strip()

    if not answer:
        raise HTTPException(
            status_code=502,
            detail="The RAG pipeline returned an empty answer.",
        )

    return AskResponse(
        answer=answer,
        sources=[Source(**source) for source in result.get("sources", [])],
        verified=result.get("verified"),
        context=result.get("context") or "",
        timings=result.get("timings") or {},
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    redis_state = ping()

    return HealthResponse(
        status="healthy",
        documents_indexed=get_documents_indexed(),
        models_loaded=is_warm(),
        redis={True: "up", False: "down", None: "disabled"}[redis_state],
        cache=cache_stats(),
    )