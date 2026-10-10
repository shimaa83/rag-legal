
from __future__ import annotations

import json
import os
import re
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from google import genai
from langfuse import get_client, observe

from legal_rag.config import (
    API_JUDGE_MODEL,
    GENERATOR_MODEL,
    INSUFFICIENT_CONTEXT,
    PROMPT_TEMPLATE,
    VERIFICATION_PROMPT_TEMPLATE,
)
from legal_rag.retrieval import hybrid_search
from legal_rag.reranker import ArabicReranker


# ============================================================
# Configuration
# ============================================================

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434/api/chat",
)

TOP_K = int(os.getenv("RAG_TOP_K", "5"))
CANDIDATE_K = int(os.getenv("RAG_CANDIDATE_K", "20"))
MAX_NEW_TOKENS = int(os.getenv("RAG_MAX_NEW_TOKENS", "800"))

GENERATOR_MODEL_NAME = os.getenv(
    "GENERATOR_MODEL",
    GENERATOR_MODEL,
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

RRF_K = int(os.getenv("RAG_RRF_K", "20"))
VECTOR_WEIGHT = float(os.getenv("RAG_VECTOR_WEIGHT", "1.0"))
KEYWORD_WEIGHT = float(os.getenv("RAG_KEYWORD_WEIGHT", "0.25"))

PLACEHOLDERS = re.compile(
    r"\{(question|context|answer|insufficient_context)\}"
)

langfuse = get_client()

# Load the reranker only once per Python process.
_reranker: ArabicReranker | None = None


def get_reranker() -> ArabicReranker:
    """Return the shared reranker instance."""
    global _reranker

    if _reranker is None:
        _reranker = ArabicReranker()

    return _reranker


# ============================================================
# Prompt utilities
# ============================================================

def render_template(template: str, **values: Any) -> str:
    """Replace supported placeholders in a single pass."""
    return PLACEHOLDERS.sub(
        lambda match: str(
            values.get(match.group(1), match.group(0))
        ),
        template,
    )


# ============================================================
# LLM calls
# ============================================================

def call_generator_ollama(
    prompt: str,
    model: str | None = None,
) -> str:
    """Generate an answer using a local Ollama model."""
    model = model or GENERATOR_MODEL_NAME

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": MAX_NEW_TOKENS,
        },
    }

    request = Request(
        OLLAMA_URL,
        data=json.dumps(
            payload,
            ensure_ascii=False,
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=180) as response:
            result = json.loads(
                response.read().decode("utf-8")
            )

    except HTTPError as exc:
        details = exc.read().decode(
            "utf-8",
            errors="replace",
        )
        raise RuntimeError(
            f"Ollama HTTP error {exc.code}: {details}"
        ) from exc

    except (URLError, TimeoutError) as exc:
        raise RuntimeError(
            f"Cannot connect to Ollama at {OLLAMA_URL}. "
            "Check that Ollama is running."
        ) from exc

    answer = (
        result.get("message", {}).get("content") or ""
    ).strip()

    if not answer:
        raise RuntimeError(
            f"Ollama returned an empty answer for {model}."
        )

    return answer


def call_judge(prompt: str) -> str:
    """Review an answer using Gemini."""
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. "
            "Set it in the environment."
        )

    client = genai.Client(api_key=GEMINI_API_KEY)

    response = client.models.generate_content(
        model=API_JUDGE_MODEL,
        contents=prompt,
    )

    answer = (response.text or "").strip()

    if not answer:
        raise RuntimeError(
            "Gemini judge returned an empty response."
        )

    return answer


# ============================================================
# Context and sources
# ============================================================

def get_article_text(document: dict[str, Any]) -> str:
    """Return the first non-empty Arabic text field."""
    for field in (
        "text_ar_normalized",
        "ar_text",
        "text_ar",
    ):
        value = document.get(field)

        if isinstance(value, str) and value.strip():
            return value.strip()

    return ""


def is_repealed(document: dict[str, Any]) -> bool:
    """Support boolean and string repeal flags."""
    value = document.get("is_repealed", False)

    if isinstance(value, str):
        return value.strip().lower() in {
            "true",
            "1",
            "yes",
            "نعم",
        }

    return bool(value)


def build_context(
    retrieved_results: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """
    Build bilingual legal context and deduplicated sources.

    Repealed articles are excluded from context and sources.
    """
    parts: list[str] = []
    sources: list[dict[str, Any]] = []
    seen: set[str] = set()

    for doc in retrieved_results:
        if not isinstance(doc, dict):
            continue

        if is_repealed(doc):
            continue

        text_ar = get_article_text(doc)

        text_en_value = doc.get("text_en")
        text_en = (
            text_en_value.strip()
            if isinstance(text_en_value, str)
            else ""
        )

        if not text_ar and not text_en:
            continue

        number = doc.get("article_number", "غير محدد")

        citation = doc.get("citation") or (
            f"القانون المدني المصري، المادة {number}"
        )

        number_key = str(number)

        # Avoid repeating the same article in the context.
        if number_key in seen:
            continue

        seen.add(number_key)

        parts.append(
            f"[{citation}]\n"
            f"رقم المادة: {number}\n"
            f"النص العربي:\n{text_ar or 'غير متاح'}\n"
            f"English translation:\n"
            f"{text_en or 'Not available'}"
        )

        sources.append(
            {
                "article_number": number,
                "citation": citation,
                "source_page": doc.get("source_page"),
            }
        )

    context = "\n\n---\n\n".join(parts)

    return context, sources


# ============================================================
# Generation and verification
# ============================================================

@observe(name="legal-rag.generate")
def generate_answer(
    question: str,
    context: str,
    model: str | None = None,
    prompt_template: str = PROMPT_TEMPLATE,
) -> str:
    """Generate the initial answer using Ollama."""
    prompt = render_template(
        prompt_template,
        question=question,
        context=context,
        insufficient_context=INSUFFICIENT_CONTEXT,
    )

    answer = call_generator_ollama(prompt, model)

    langfuse.update_current_span(
        input={
            "question": question,
            "context": context,
            "model": model or GENERATOR_MODEL_NAME,
        },
        output=answer,
    )

    return answer


@observe(name="legal-rag.verify")
def verify_answer(
    question: str,
    context: str,
    answer: str,
) -> str:
    """Ask Gemini to review the generated answer."""
    prompt = render_template(
        VERIFICATION_PROMPT_TEMPLATE,
        question=question,
        context=context,
        answer=answer,
        insufficient_context=INSUFFICIENT_CONTEXT,
    )

    try:
        final = call_judge(prompt)

    except Exception as exc:
        langfuse.update_current_span(
            output={
                "verification_status": "failed",
                "error": str(exc),
            }
        )

        # Preserve the existing fail-closed behavior.
        return INSUFFICIENT_CONTEXT

    return (
        INSUFFICIENT_CONTEXT
        if INSUFFICIENT_CONTEXT in final
        else final
    )


# ============================================================
# Main RAG pipeline
# ============================================================

@observe(name="legal-rag.ask")
def ask(
    question: str,
    top_k: int = TOP_K,
    candidate_k: int = CANDIDATE_K,
    model: str | None = None,
    prompt_template: str = PROMPT_TEMPLATE,
) -> dict[str, Any]:
    """
    RAG pipeline:
    1. Hybrid retrieval with vector + keyword search.
    2. Fuse candidates using Reciprocal Rank Fusion (RRF).
    3. Remove repealed articles.
    4. Rerank candidates using a multilingual Cross-Encoder.
    5. Build Arabic-English context and citations.
    6. Generate with Ollama.
    7. Review the answer using Gemini.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError(
            "Question must be a non-empty string."
        )

    if (
        not isinstance(top_k, int)
        or isinstance(top_k, bool)
        or not isinstance(candidate_k, int)
        or isinstance(candidate_k, bool)
        or top_k < 1
        or candidate_k < 1
    ):
        raise ValueError(
            "top_k and candidate_k must be positive integers."
        )

    if candidate_k < top_k:
        raise ValueError(
            "candidate_k must be greater than or equal to top_k."
        )

    question = question.strip()

    langfuse.update_current_span(
        input={
            "question": question,
            "top_k": top_k,
            "candidate_k": candidate_k,
            "generator_model": model or GENERATOR_MODEL_NAME,
            "judge_model": API_JUDGE_MODEL,
            "reranker": "Horizon-Labs/multilingual-reranker-small",
            "rrf_k": RRF_K,
            "vector_weight": VECTOR_WEIGHT,
            "keyword_weight": KEYWORD_WEIGHT,
        }
    )

    # Step 1: Retrieve and fuse a larger candidate set.
    candidates = hybrid_search(
        question=question,
        top_k=candidate_k,
        candidate_k=candidate_k,
        rrf_k=RRF_K,
        vector_weight=VECTOR_WEIGHT,
        keyword_weight=KEYWORD_WEIGHT,
    )

    # Step 2: Exclude repealed articles before reranking.
    active_candidates = [
        doc
        for doc in candidates
        if isinstance(doc, dict) and not is_repealed(doc)
    ]

    # Step 3: Rerank candidates and retain the best top_k.
    reranked_results = get_reranker().rerank(
        question=question,
        results=active_candidates,
        top_k=top_k,
    )

    # Step 4: Build context and citations from reranked results.
    context, sources = build_context(reranked_results)

    # Step 5: Generate and verify the answer.
    if context.strip():
        initial = generate_answer(
            question=question,
            context=context,
            model=model,
            prompt_template=prompt_template,
        )

        answer = verify_answer(
            question=question,
            context=context,
            answer=initial,
        )

    else:
        answer = INSUFFICIENT_CONTEXT
        sources = []

    result = {
        "question": question,
        "answer": answer,
        "sources": sources,
        "retrieved_results": reranked_results,
        "context": context,
    }

    langfuse.update_current_span(output=result)

    return result
