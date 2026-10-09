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

# ============================================================
# Configuration
# ============================================================

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
TOP_K = int(os.getenv("RAG_TOP_K", "5"))
CANDIDATE_K = int(os.getenv("RAG_CANDIDATE_K", "20"))
MAX_NEW_TOKENS = int(os.getenv("RAG_MAX_NEW_TOKENS", "800"))
GENERATOR_MODEL_NAME = os.getenv("GENERATOR_MODEL", GENERATOR_MODEL)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

PLACEHOLDERS = re.compile(r"\{(question|context|answer|insufficient_context)\}")

langfuse = get_client()


def render_template(template: str, **values: Any) -> str:
    """Fill only the supported placeholders (single pass, safe for any text)."""
    return PLACEHOLDERS.sub(lambda m: str(values.get(m.group(1), m.group(0))), template)


# ============================================================
# LLM calls
# ============================================================

def call_generator_ollama(prompt: str, model: str | None = None) -> str:
    """Generate an answer with a local Ollama model."""
    model = model or GENERATOR_MODEL_NAME
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": MAX_NEW_TOKENS},
    }
    request = Request(
        OLLAMA_URL,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=180) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP error {exc.code}: {details}") from exc
    except (URLError, TimeoutError) as exc:
        raise RuntimeError(
            f"Cannot connect to Ollama. Is it running at {OLLAMA_URL}?"
        ) from exc

    answer = (result.get("message", {}).get("content") or "").strip()
    if not answer:
        raise RuntimeError(f"Ollama returned an empty answer for {model}.")
    return answer


def call_judge(prompt: str) -> str:
    """Review an answer with Gemini."""
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is missing. Set it in the environment.")

    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(model=API_JUDGE_MODEL, contents=prompt)

    answer = (response.text or "").strip()
    if not answer:
        raise RuntimeError("Gemini judge returned an empty response.")
    return answer


# ============================================================
# Context and sources
# ============================================================

def get_article_text(document: dict[str, Any]) -> str:
    """Return the first non-empty Arabic text field of an article."""
    for field in ("text_ar_normalized", "ar_text", "text_ar"):
        value = document.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def is_repealed(document: dict[str, Any]) -> bool:
    """Support both boolean and string values."""
    value = document.get("is_repealed", False)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "نعم"}
    return bool(value)


def build_context(
    retrieved_results: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Build the legal context and deduplicated sources (repealed articles excluded)."""
    parts: list[str] = []
    sources: list[dict[str, Any]] = []
    seen: set[str] = set()

    for doc in retrieved_results:
        if not isinstance(doc, dict) or is_repealed(doc):
            continue

        text = get_article_text(doc)
        if not text:
            continue

        number = doc.get("article_number", "غير محدد")
        citation = doc.get("citation") or f"القانون المدني المصري، المادة {number}"

        parts.append(f"[{citation}]\nرقم المادة: {number}\nنص المادة:\n{text}")

        if str(number) not in seen:
            seen.add(str(number))
            sources.append(
                {
                    "article_number": number,
                    "citation": citation,
                    "source_page": doc.get("source_page"),
                }
            )

    return "\n\n---\n\n".join(parts), sources


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
    """Generate the initial answer with Ollama."""
    prompt = render_template(
        prompt_template,
        question=question,
        context=context,
        insufficient_context=INSUFFICIENT_CONTEXT,
    )
    answer = call_generator_ollama(prompt, model)

    langfuse.update_current_span(
        input={"question": question, "context": context, "model": model or GENERATOR_MODEL_NAME},
        output=answer,
    )
    return answer


@observe(name="legal-rag.verify")
def verify_answer(question: str, context: str, answer: str) -> str:
    """Let Gemini review the answer and return the final one."""
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
            output={"verification_status": "failed", "error": str(exc)}
        )
        return INSUFFICIENT_CONTEXT

    return INSUFFICIENT_CONTEXT if INSUFFICIENT_CONTEXT in final else final


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
    1. Hybrid retrieval  2. Build context  3. Ollama generation
    4. Gemini review     5. Return the final answer + sources
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string.")

    question = question.strip()

    langfuse.update_current_span(
        input={
            "question": question,
            "top_k": top_k,
            "candidate_k": candidate_k,
            "generator_model": model or GENERATOR_MODEL_NAME,
            "judge_model": API_JUDGE_MODEL,
        }
    )

    retrieved = hybrid_search(question=question, top_k=top_k, candidate_k=candidate_k)
    context, sources = build_context(retrieved)

    if context.strip():
        initial = generate_answer(question, context, model, prompt_template)
        answer = verify_answer(question, context, initial)
    else:
        answer, sources = INSUFFICIENT_CONTEXT, []

    result = {
        "question": question,
        "answer": answer,
        "sources": sources,
        "retrieved_results": retrieved,
        "context": context,
    }
    langfuse.update_current_span(output=result)
    return result