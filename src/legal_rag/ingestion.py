
from __future__ import annotations

from typing import Any

from legal_rag.database import get_connection
from legal_rag.embeddings import ArabicEmbedder


INSERT_QUERY = """
    INSERT INTO legal_chunks (
        chunk_id,
        article_number,
        text_ar,
        text_ar_normalized,
        text_en,
        book,
        chapter,
        section,
        topic,
        is_repealed,
        source_page,
        citation,
        embedding,
        search_vector
    )
    VALUES (
        %(chunk_id)s,
        %(article_number)s,
        %(text_ar)s,
        %(text_ar_normalized)s,
        %(text_en)s,
        %(book)s,
        %(chapter)s,
        %(section)s,
        %(topic)s,
        %(is_repealed)s,
        %(source_page)s,
        %(citation)s,
        %(embedding)s,
        to_tsvector(
            'simple',
            concat_ws(
                ' ',
                COALESCE(%(text_ar_normalized)s, ''),
                COALESCE(%(text_en)s, '')
            )
        )
    )
    ON CONFLICT (chunk_id)
    DO UPDATE SET
        article_number = EXCLUDED.article_number,
        text_ar = EXCLUDED.text_ar,
        text_ar_normalized = EXCLUDED.text_ar_normalized,
        text_en = EXCLUDED.text_en,
        book = EXCLUDED.book,
        chapter = EXCLUDED.chapter,
        section = EXCLUDED.section,
        topic = EXCLUDED.topic,
        is_repealed = EXCLUDED.is_repealed,
        source_page = EXCLUDED.source_page,
        citation = EXCLUDED.citation,
        embedding = EXCLUDED.embedding,
        search_vector = EXCLUDED.search_vector;
"""


def get_embedding_text(chunk: dict[str, Any]) -> str:
    """Combine Arabic and English content for multilingual embeddings."""

    text_ar = str(chunk.get("text_ar_normalized") or "").strip()
    text_en = str(chunk.get("text_en") or "").strip()

    parts = []

    if text_ar:
        parts.append(f"Arabic legal text: {text_ar}")

    if text_en:
        parts.append(f"English translation: {text_en}")

    if not parts:
        raise ValueError(
            f"Chunk {chunk.get('chunk_id')} has no text to embed."
        )

    return "\n".join(parts)


def insert_chunk(
    chunk: dict[str, Any],
    embedding: list[float],
) -> None:
    """Insert or update one legal chunk."""

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                INSERT_QUERY,
                {
                    **chunk,
                    "embedding": embedding,
                },
            )

        connection.commit()


def ingest_one_chunk(
    chunk: dict[str, Any],
    embedder: ArabicEmbedder | None = None,
) -> None:
    """Generate and store one multilingual embedding."""

    if embedder is None:
        embedder = ArabicEmbedder()

    text = get_embedding_text(chunk)
    embedding = embedder.embed_documents([text])[0]

    insert_chunk(chunk, embedding)


def ingest_chunks(
    chunks: list[dict[str, Any]],
    embedder: ArabicEmbedder,
    batch_size: int = 32,
) -> None:
    """Generate multilingual embeddings and store chunks in batches."""

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    with get_connection() as connection, connection.cursor() as cursor:
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]

            texts = [
                get_embedding_text(chunk)
                for chunk in batch
            ]

            embeddings = embedder.embed_documents(texts)

            rows = [
                {
                    **chunk,
                    "embedding": embedding,
                }
                for chunk, embedding in zip(
                    batch,
                    embeddings,
                    strict=True,
                )
            ]

            cursor.executemany(INSERT_QUERY, rows)
            connection.commit()

            end = start + len(batch)
            print(f"Ingested {end}/{len(chunks)} chunks")
