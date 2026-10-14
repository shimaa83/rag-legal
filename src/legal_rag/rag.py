from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from langfuse import get_client, observe

from legal_rag.config import (
    ARABIC_DIGITS,
    CACHE_TTL_ANSWER,
    CACHE_TTL_RETRIEVAL,
    CANDIDATE_K,
    ENABLE_VERIFICATION,
    ENGLISH_DIGITS,
    GENERATOR_MODEL,
    INSUFFICIENT_CONTEXT,
    PROCESSED_JSON_PATH,
    PROMPT_TEMPLATE,
    REPEALED_ARTICLES,
    TOP_K,
    VERIFICATION_PROMPT_TEMPLATE,
    VERIFIER_MODEL,
)
from legal_rag.cache import (
    cache_get,
    cache_set,
    file_fingerprint,
    hash_text,
    make_key,
    single_flight,
)
from legal_rag.reranker import ArabicReranker
from legal_rag.retrieval import hybrid_search

logger = logging.getLogger(__name__)

__all__ = [
    "CANDIDATE_K",
    "ENABLE_VERIFICATION",
    "GENERATOR_MODEL_NAME",
    "LLMError",
    "TOP_K",
    "VERIFIER_MODEL_NAME",
    "ask",
    "flush_traces",
    "get_article_text",
    "is_warm",
    "warm_up",
]


# ============================================================
# Configuration
# ============================================================

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "300"))

# Ollama's default context window is small (2k-4k tokens). Five bilingual
# articles + the prompt overflow it and the start of the prompt is silently
# cut, so we always send an explicit num_ctx.
# IMPORTANT: every call (warm-up, generator, verifier) must send the SAME
# num_ctx, otherwise Ollama reloads the model from disk.
OLLAMA_NUM_CTX = int(os.getenv("RAG_NUM_CTX", "8192"))


def _parse_keep_alive(value: str) -> int | str:
    """Ollama wants a number for -1 / seconds and a string for '30m'."""
    value = value.strip()
    try:
        return int(value)
    except ValueError:
        return value


# -1 = keep the model in memory forever (loaded once, never reloaded).
OLLAMA_KEEP_ALIVE = _parse_keep_alive(os.getenv("OLLAMA_KEEP_ALIVE", "-1"))

MAX_NEW_TOKENS = int(os.getenv("RAG_MAX_NEW_TOKENS", "800"))

GENERATOR_MODEL_NAME = GENERATOR_MODEL
VERIFIER_MODEL_NAME = VERIFIER_MODEL
RERANKER_NAME = "Horizon-Labs/multilingual-reranker-small"

RRF_K = int(os.getenv("RAG_RRF_K", "20"))
VECTOR_WEIGHT = float(os.getenv("RAG_VECTOR_WEIGHT", "1.0"))
KEYWORD_WEIGHT = float(os.getenv("RAG_KEYWORD_WEIGHT", "0.25"))

PLACEHOLDERS = re.compile(
    r"\{(question|context|answer|insufficient_context)\}"
)

# Chinese / Japanese / Korean characters (qwen sometimes leaks them).
CJK_RE = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")

# "الإجابة النهائية:" / "الإجابة:" prefixes small models like to add.
ANSWER_PREFIX_RE = re.compile(
    r"^\s*(?:\*\*)?(?:الإجابة النهائية|الاجابة النهائية|الإجابة|الاجابة)"
    r"(?:\*\*)?\s*[:：]\s*"
)

_DIGIT_MAP = str.maketrans(ARABIC_DIGITS, ENGLISH_DIGITS)

langfuse = get_client()

# The data file fingerprint is part of every cache key: when the processed
# data changes, old cached answers are never reused.
DATA_FINGERPRINT = file_fingerprint(PROCESSED_JSON_PATH)


class LLMError(RuntimeError):
    """Raised when the local Ollama backend fails."""


# ============================================================
# Shared reranker (loaded once per process, thread-safe)
# ============================================================

_reranker: ArabicReranker | None = None
_reranker_lock = threading.Lock()


def get_reranker() -> ArabicReranker:
    """Return the shared reranker instance."""
    global _reranker

    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                _reranker = ArabicReranker()

    return _reranker


def _trace(**kwargs: Any) -> None:
    """Update the current Langfuse span without ever breaking the pipeline."""
    try:
        langfuse.update_current_span(**kwargs)
    except Exception:  # noqa: BLE001 - tracing must never fail a request
        logger.debug("Langfuse update failed", exc_info=True)


def flush_traces() -> None:
    """Send pending Langfuse events (call on shutdown)."""
    try:
        langfuse.flush()
    except Exception:  # noqa: BLE001
        logger.debug("Langfuse flush failed", exc_info=True)


# ============================================================
# Prompt utilities
# ============================================================

def render_template(template: str, **values: Any) -> str:
    """Replace supported placeholders in a single pass."""
    return PLACEHOLDERS.sub(
        lambda match: str(values.get(match.group(1), match.group(0))),
        template,
    )


# ============================================================
# LLM calls (Ollama only)
# ============================================================

def call_ollama(
    prompt: str,
    model: str,
    temperature: float = 0.1,
    max_tokens: int | None = None,
) -> str:
    """Send one prompt to a local Ollama model and return the text."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens or MAX_NEW_TOKENS,
            "num_ctx": OLLAMA_NUM_CTX,
        },
    }

    request = Request(
        OLLAMA_URL,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=OLLAMA_TIMEOUT) as response:
            result = json.loads(response.read().decode("utf-8"))

    except HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise LLMError(f"Ollama HTTP error {exc.code}: {details}") from exc

    except (URLError, TimeoutError, OSError) as exc:
        raise LLMError(
            f"Cannot connect to Ollama at {OLLAMA_URL}. "
            "Check that Ollama is running."
        ) from exc

    except json.JSONDecodeError as exc:
        raise LLMError(f"Ollama returned invalid JSON: {exc}") from exc

    answer = (result.get("message", {}).get("content") or "").strip()

    if not answer:
        raise LLMError(f"Ollama returned an empty answer for {model}.")

    return answer


def call_generator_ollama(prompt: str, model: str | None = None) -> str:
    """Generate an answer using the generator model."""
    return call_ollama(prompt, model or GENERATOR_MODEL_NAME, temperature=0.1)


def call_verifier_ollama(prompt: str) -> str:
    """Review an answer using the local verifier model (deterministic)."""
    return call_ollama(prompt, VERIFIER_MODEL_NAME, temperature=0.0)


# ============================================================
# Warm-up: load every model once, before the first user request
# ============================================================

_warm_lock = threading.Lock()
_warm = False
WARMUP_QUESTION = "ما هو سن الرشد في القانون المدني المصري؟"


def is_warm() -> bool:
    return _warm


def warm_up(include_llm: bool = True) -> dict[str, Any]:
    """
    Load the search index, the reranker and the Ollama models.

    Safe to call many times: after a fully successful warm-up it returns
    immediately. Each step is isolated, so one failure (e.g. Ollama not
    started yet) never stops the API from starting.
    """
    global _warm

    with _warm_lock:
        if _warm:
            return {"status": "already_warm"}

        report: dict[str, Any] = {}
        failed = False

        # 1) search index + embedding model, 2) reranker weights
        started = time.perf_counter()
        try:
            candidates = hybrid_search(
                question=WARMUP_QUESTION,
                top_k=CANDIDATE_K,
                candidate_k=CANDIDATE_K,
                rrf_k=RRF_K,
                vector_weight=VECTOR_WEIGHT,
                keyword_weight=KEYWORD_WEIGHT,
            )
            report["retrieval"] = round(time.perf_counter() - started, 2)

            started = time.perf_counter()
            active = [d for d in candidates if isinstance(d, dict)]
            if active:
                get_reranker().rerank(
                    question=WARMUP_QUESTION, results=active, top_k=1
                )
            else:
                get_reranker()
            report["reranker"] = round(time.perf_counter() - started, 2)
        except Exception as exc:  # noqa: BLE001
            failed = True
            report["retrieval_or_reranker_error"] = str(exc)
            logger.exception("Warm-up of retrieval/reranker failed")

        # 3) Ollama models (same num_ctx + keep_alive as real requests)
        if include_llm:
            for model in dict.fromkeys(
                [GENERATOR_MODEL_NAME, VERIFIER_MODEL_NAME]
            ):
                started = time.perf_counter()
                try:
                    call_ollama(".", model, temperature=0.0, max_tokens=1)
                    report[f"ollama:{model}"] = round(
                        time.perf_counter() - started, 2
                    )
                except LLMError as exc:
                    failed = True
                    report[f"ollama:{model}_error"] = str(exc)
                    logger.warning("Ollama warm-up failed for %s: %s", model, exc)

        _warm = not failed
        logger.info("Warm-up report: %s", report)
        return report


# ============================================================
# Context and sources
# ============================================================

def get_article_text(document: dict[str, Any]) -> str:
    """Return the first non-empty Arabic text field."""
    for field in ("text_ar_normalized", "ar_text", "text_ar"):
        value = document.get(field)

        if isinstance(value, str) and value.strip():
            return value.strip()

    return ""


def _article_int(value: Any) -> int | None:
    """Parse an article number written with Arabic or English digits."""
    try:
        return int(str(value).strip().translate(_DIGIT_MAP))
    except (TypeError, ValueError):
        return None


def is_repealed(document: dict[str, Any]) -> bool:
    """True if the article is flagged as repealed or is in the repealed range."""
    value = document.get("is_repealed", False)

    if isinstance(value, str):
        flagged = value.strip().lower() in {"true", "1", "yes", "نعم"}
    else:
        flagged = bool(value)

    if flagged:
        return True

    return _article_int(document.get("article_number")) in REPEALED_ARTICLES


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
        if not isinstance(doc, dict) or is_repealed(doc):
            continue

        text_ar = get_article_text(doc)

        text_en_value = doc.get("text_en")
        text_en = (
            text_en_value.strip() if isinstance(text_en_value, str) else ""
        )

        if not text_ar and not text_en:
            continue

        raw_number = doc.get("article_number")
        number = raw_number if raw_number not in (None, "") else "غير محدد"

        citation = doc.get("citation") or (
            f"القانون المدني المصري، المادة {number}"
        )

        # Avoid repeating the same article (articles without a number
        # are never merged together). "٥٠" and "50" are the same article.
        if raw_number not in (None, ""):
            parsed = _article_int(raw_number)
            number_key = str(parsed) if parsed is not None else str(number)

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
    """Generate the initial answer using Ollama."""
    prompt = render_template(
        prompt_template,
        question=question,
        context=context,
        insufficient_context=INSUFFICIENT_CONTEXT,
    )

    answer = call_generator_ollama(prompt, model)

    _trace(
        input={
            "question": question,
            "context": context,
            "model": model or GENERATOR_MODEL_NAME,
        },
        output=answer,
    )

    return answer


def clean_verified_answer(text: str) -> str:
    """Strip code fences and 'final answer:' prefixes from the verifier."""
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    return ANSWER_PREFIX_RE.sub("", text.strip()).strip()


@observe(name="legal-rag.verify")
def verify_answer(
    question: str,
    context: str,
    answer: str,
) -> tuple[str, bool]:
    """
    Review the draft answer with the local verifier model (qwen2.5:3b).

    Returns (final_answer, verified). If the verifier fails or returns an
    unusable output (empty / contains Chinese), the draft is kept and
    verified=False so the UI can show that the review did not complete.
    """
    prompt = render_template(
        VERIFICATION_PROMPT_TEMPLATE,
        question=question,
        context=context,
        answer=answer,
        insufficient_context=INSUFFICIENT_CONTEXT,
    )

    try:
        final = clean_verified_answer(call_verifier_ollama(prompt))

    except LLMError as exc:
        logger.warning("Verifier failed, keeping the draft answer: %s", exc)
        _trace(output={"verification_status": "failed", "error": str(exc)})
        return answer, False

    if not final or CJK_RE.search(final):
        logger.warning("Verifier output unusable, keeping the draft answer.")
        _trace(output={"verification_status": "unusable", "output": final})
        return answer, False

    if INSUFFICIENT_CONTEXT in final:
        return INSUFFICIENT_CONTEXT, True

    return final, True


# ============================================================
# Main RAG pipeline
# ============================================================

def _lookup_answer(answer_key: str, question: str) -> dict[str, Any] | None:
    """Return a cached answer (with fresh timings) or None."""
    started = time.perf_counter()
    cached = cache_get(answer_key)

    if not (isinstance(cached, dict) and cached.get("answer")):
        return None

    cached["question"] = question
    cached["cached"] = True
    cached["timings"] = {"cache": round(time.perf_counter() - started, 3)}

    _trace(
        input={"question": question},
        output=cached,
        metadata={"cache": "answer_hit"},
    )
    return cached


def _run_pipeline(
    *,
    question: str,
    top_k: int,
    candidate_k: int,
    model: str | None,
    prompt_template: str,
    use_verification: bool,
    use_cache: bool,
    answer_key: str,
) -> dict[str, Any]:
    timings: dict[str, float] = {}
    pipeline_started = time.perf_counter()

    _trace(
        input={
            "question": question,
            "top_k": top_k,
            "candidate_k": candidate_k,
            "generator_model": model or GENERATOR_MODEL_NAME,
            "verifier_model": VERIFIER_MODEL_NAME,
            "verification": use_verification,
            "reranker": RERANKER_NAME,
            "rrf_k": RRF_K,
            "vector_weight": VECTOR_WEIGHT,
            "keyword_weight": KEYWORD_WEIGHT,
        },
        metadata={"cache": "answer_miss" if use_cache else "bypassed"},
    )

    # Step 1: Retrieve and fuse a larger candidate set.
    started = time.perf_counter()
    retrieval_key = make_key(
        "retrieval",
        question,
        candidate_k=candidate_k,
        rrf_k=RRF_K,
        vector_weight=VECTOR_WEIGHT,
        keyword_weight=KEYWORD_WEIGHT,
        data=DATA_FINGERPRINT,
    )
    candidates = cache_get(retrieval_key) if use_cache else None

    if not isinstance(candidates, list):
        candidates = hybrid_search(
            question=question,
            top_k=candidate_k,
            candidate_k=candidate_k,
            rrf_k=RRF_K,
            vector_weight=VECTOR_WEIGHT,
            keyword_weight=KEYWORD_WEIGHT,
        )
        if use_cache and candidates:
            cache_set(retrieval_key, candidates, CACHE_TTL_RETRIEVAL)

    timings["retrieval"] = round(time.perf_counter() - started, 2)

    # Step 2: Exclude repealed articles before reranking.
    active_candidates = [
        doc
        for doc in candidates
        if isinstance(doc, dict) and not is_repealed(doc)
    ]

    # Step 3: Rerank candidates and retain the best top_k.
    started = time.perf_counter()
    if active_candidates:
        reranked_results = get_reranker().rerank(
            question=question,
            results=active_candidates,
            top_k=top_k,
        )
    else:
        reranked_results = []
    timings["rerank"] = round(time.perf_counter() - started, 2)

    # Step 4: Build context and citations from reranked results.
    context, sources = build_context(reranked_results)

    # Step 5: Generate and (optionally) verify the answer.
    verified: bool | None = None

    if context.strip():
        started = time.perf_counter()
        answer = generate_answer(
            question=question,
            context=context,
            model=model,
            prompt_template=prompt_template,
        )
        timings["generate"] = round(time.perf_counter() - started, 2)

        if use_verification:
            started = time.perf_counter()
            answer, verified = verify_answer(
                question=question,
                context=context,
                answer=answer,
            )
            timings["verify"] = round(time.perf_counter() - started, 2)

    else:
        answer = INSUFFICIENT_CONTEXT
        sources = []

    timings["total"] = round(time.perf_counter() - pipeline_started, 2)

    result = {
        "question": question,
        "answer": answer,
        "sources": sources,
        "verified": verified,
        "timings": timings,
        "retrieved_results": reranked_results,
        "context": context,
    }

    # Cache only good answers: a failed verification is not stored,
    # so the next identical request gets another chance.
    if use_cache and context.strip() and not (
        use_verification and verified is False
    ):
        cache_set(answer_key, result, CACHE_TTL_ANSWER)

    _trace(output=result)

    return result


@observe(name="legal-rag.ask")
def ask(
    question: str,
    top_k: int = TOP_K,
    candidate_k: int = CANDIDATE_K,
    model: str | None = None,
    prompt_template: str = PROMPT_TEMPLATE,
    verify: bool | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    """
    RAG pipeline:
    1. Hybrid retrieval with vector + keyword search.
    2. Fuse candidates using Reciprocal Rank Fusion (RRF).
    3. Remove repealed articles.
    4. Rerank candidates using a multilingual Cross-Encoder.
    5. Build Arabic-English context and citations.
    6. Generate with Ollama.
    7. Review the answer with the local verifier (qwen2.5:3b via Ollama).

    `verify=None` uses ENABLE_VERIFICATION from the config.
    `use_cache=False` bypasses Redis completely (used by the evaluation, so
    metrics measure the real pipeline and not stored answers).
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string.")

    if (
        not isinstance(top_k, int)
        or isinstance(top_k, bool)
        or not isinstance(candidate_k, int)
        or isinstance(candidate_k, bool)
        or top_k < 1
        or candidate_k < 1
    ):
        raise ValueError("top_k and candidate_k must be positive integers.")

    if candidate_k < top_k:
        raise ValueError(
            "candidate_k must be greater than or equal to top_k."
        )

    question = question.strip()
    use_verification = ENABLE_VERIFICATION if verify is None else bool(verify)

    # Everything that can change the answer is part of the key.
    answer_key = make_key(
        "answer",
        question,
        top_k=top_k,
        candidate_k=candidate_k,
        generator=model or GENERATOR_MODEL_NAME,
        verifier=VERIFIER_MODEL_NAME,
        verify=use_verification,
        prompt=hash_text(prompt_template),
        verify_prompt=hash_text(VERIFICATION_PROMPT_TEMPLATE),
        reranker=RERANKER_NAME,
        rrf_k=RRF_K,
        vector_weight=VECTOR_WEIGHT,
        keyword_weight=KEYWORD_WEIGHT,
        num_ctx=OLLAMA_NUM_CTX,
        max_tokens=MAX_NEW_TOKENS,
        data=DATA_FINGERPRINT,
    )

    run_kwargs = {
        "question": question,
        "top_k": top_k,
        "candidate_k": candidate_k,
        "model": model,
        "prompt_template": prompt_template,
        "use_verification": use_verification,
        "use_cache": use_cache,
        "answer_key": answer_key,
    }

    if not use_cache:
        return _run_pipeline(**run_kwargs)

    hit = _lookup_answer(answer_key, question)
    if hit is not None:
        return hit

    # Identical concurrent questions: the first one computes, the others
    # wait and then read the stored answer (no duplicate LLM work).
    with single_flight(answer_key):
        hit = _lookup_answer(answer_key, question)
        if hit is not None:
            return hit

        return _run_pipeline(**run_kwargs)