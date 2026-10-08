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
            COALESCE(%(text_ar_normalized)s, '')
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


def insert_chunk(
    chunk: dict[str, Any],
    embedding: list[float],
) -> None:
    """Insert one legal chunk into PostgreSQL."""

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


def ingest_one_chunk(chunk: dict[str, Any]) -> None:
    """Generate and store one chunk embedding."""

    embedder = ArabicEmbedder()

    embedding = embedder.embed_documents(
        [chunk["text_ar_normalized"]]
    )[0]

    insert_chunk(chunk, embedding)


def ingest_chunks(
    chunks: list[dict[str, Any]],
    embedder: ArabicEmbedder,
    batch_size: int = 32,
) -> None:
    """Generate embeddings and store legal chunks in batches."""

    with get_connection() as connection:
        with connection.cursor() as cursor:
            for start in range(0, len(chunks), batch_size):
                batch = chunks[start : start + batch_size]

                texts = [
                    chunk["text_ar_normalized"]
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

                cursor.executemany(
                    INSERT_QUERY,
                    rows,
                )

                connection.commit()

                end = start + len(batch)
                print(
                    f"Ingested {end}/{len(chunks)} chunks"
                )