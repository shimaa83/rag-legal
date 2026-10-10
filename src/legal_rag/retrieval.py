
from __future__ import annotations

import re
from typing import Any

from legal_rag.database import get_connection
from legal_rag.embeddings import ArabicEmbedder


ARABIC_STOPWORDS = {
    "ما", "ماذا", "هي", "هو", "هل", "من", "في",
    "على", "إلى", "عن", "و", "أو", "أن", "إن",
    "لا", "لم", "لن", "مع", "هذا", "هذه",
}

ENGLISH_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were",
    "what", "which", "who", "how", "does", "do",
    "of", "in", "on", "at", "to", "for", "from",
    "with", "by", "under", "and", "or", "as", "be",
    "can", "could", "should", "would", "about",
}


def vector_search(
    question: str,
    top_k: int = 5,
    embedder: ArabicEmbedder | None = None,
) -> list[dict[str, Any]]:
    """Retrieve similar legal chunks using multilingual embeddings."""

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    if embedder is None:
        embedder = ArabicEmbedder()

    query_embedding = embedder.embed_query(question)

    query = """
        SELECT
            chunk_id,
            article_number,
            text_ar_normalized,
            text_en,
            citation,
            source_page,
            is_repealed,
            1 - (embedding <=> %s::vector) AS similarity
        FROM legal_chunks
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """

    with get_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            query,
            (
                query_embedding,
                query_embedding,
                top_k,
            ),
        )
        rows = cursor.fetchall()

    return [
        {
            "chunk_id": row[0],
            "article_number": row[1],
            "text_ar_normalized": row[2],
            "text_en": row[3],
            "citation": row[4],
            "source_page": row[5],
            "is_repealed": row[6],
            "similarity": float(row[7]),
        }
        for row in rows
    ]


def keyword_search(
    question: str,
    top_k: int = 20,
) -> list[dict[str, Any]]:
    """Search Arabic and English keywords using PostgreSQL full-text search."""

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    terms = re.findall(
        r"[\u0600-\u06FF]+|[A-Za-z0-9]+",
        question,
    )

    meaningful_terms = []

    for term in terms:
        normalized_term = term.lower()

        if normalized_term in ARABIC_STOPWORDS:
            continue

        if normalized_term in ENGLISH_STOPWORDS:
            continue

        meaningful_terms.append(normalized_term)

    meaningful_terms = list(dict.fromkeys(meaningful_terms))

    if not meaningful_terms:
        return []

    # Escape characters with special meaning in PostgreSQL tsquery.
    safe_terms = [
        re.sub(r"[^a-zA-Z0-9_\u0600-\u06FF]", "", term)
        for term in meaningful_terms
    ]
    safe_terms = [term for term in safe_terms if term]

    if not safe_terms:
        return []

    and_tsquery = " & ".join(safe_terms)
    or_tsquery = " | ".join(safe_terms)

    query = """
        SELECT
            chunk_id,
            article_number,
            text_ar_normalized,
            text_en,
            citation,
            source_page,
            is_repealed,
            ts_rank_cd(
                search_vector,
                to_tsquery('simple', %s)
            ) AS score
        FROM legal_chunks
        WHERE search_vector @@ to_tsquery('simple', %s)
        ORDER BY score DESC
        LIMIT %s;
    """

    with get_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            query,
            (and_tsquery, and_tsquery, top_k),
        )
        rows = cursor.fetchall()

        if not rows:
            cursor.execute(
                query,
                (or_tsquery, or_tsquery, top_k),
            )
            rows = cursor.fetchall()

    return [
        {
            "chunk_id": row[0],
            "article_number": row[1],
            "text_ar_normalized": row[2],
            "text_en": row[3],
            "citation": row[4],
            "source_page": row[5],
            "is_repealed": row[6],
            "score": float(row[7]),
        }
        for row in rows
    ]


def reciprocal_rank_fusion(
    vector_results: list[dict[str, Any]],
    keyword_results: list[dict[str, Any]],
    top_k: int = 5,
    rrf_k: int = 60,
    vector_weight: float = 1.0,
    keyword_weight: float = 1.0,
) -> list[dict[str, Any]]:
    """Fuse vector and keyword rankings using weighted RRF."""

    fused: dict[str, dict[str, Any]] = {}

    for rank, result in enumerate(vector_results, start=1):
        chunk_id = result["chunk_id"]

        if chunk_id not in fused:
            fused[chunk_id] = {
                "result": result,
                "rrf_score": 0.0,
                "vector_rank": None,
                "keyword_rank": None,
            }

        fused[chunk_id]["rrf_score"] += (
            vector_weight / (rrf_k + rank)
        )
        fused[chunk_id]["vector_rank"] = rank

    for rank, result in enumerate(keyword_results, start=1):
        chunk_id = result["chunk_id"]

        if chunk_id not in fused:
            fused[chunk_id] = {
                "result": result,
                "rrf_score": 0.0,
                "vector_rank": None,
                "keyword_rank": None,
            }

        fused[chunk_id]["rrf_score"] += (
            keyword_weight / (rrf_k + rank)
        )
        fused[chunk_id]["keyword_rank"] = rank

    ranked = sorted(
        fused.values(),
        key=lambda item: item["rrf_score"],
        reverse=True,
    )

    final_results = []

    for item in ranked[:top_k]:
        result = dict(item["result"])
        result["rrf_score"] = item["rrf_score"]
        result["vector_rank"] = item["vector_rank"]
        result["keyword_rank"] = item["keyword_rank"]
        final_results.append(result)

    return final_results


def hybrid_search(
    question: str,
    top_k: int = 10,
    candidate_k: int = 20,
    rrf_k: int = 20,
    vector_weight: float = 1.0,
    keyword_weight: float = 1.0,
    embedder: ArabicEmbedder | None = None,
) -> list[dict[str, Any]]:
    """Retrieve legal chunks using multilingual weighted hybrid search."""

    vector_results = vector_search(
        question=question,
        top_k=candidate_k,
        embedder=embedder,
    )

    keyword_results = keyword_search(
        question=question,
        top_k=candidate_k,
    )

    return reciprocal_rank_fusion(
        vector_results=vector_results,
        keyword_results=keyword_results,
        top_k=top_k,
        rrf_k=rrf_k,
        vector_weight=vector_weight,
        keyword_weight=keyword_weight,
    )
