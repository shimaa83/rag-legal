from legal_rag.ingestion import ingest_one_chunk
from legal_rag.database import get_connection


def test_insert_one_chunk() -> None:
    chunk = {
        "chunk_id": "test_article_999999",
        "article_number": 999999,
        "text_ar": "هذا نص تجريبي.",
        "text_ar_normalized": "هذا نص تجريبي.",
        "text_en": "Test text.",
        "book": "Test",
        "chapter": "Test",
        "section": "Test",
        "topic": None,
        "is_repealed": False,
        "source_page": 1,
        "citation": "Test Article 999999",
    }

    ingest_one_chunk(chunk)

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    article_number,
                    vector_dims(embedding),
                    search_vector IS NOT NULL
                FROM legal_chunks
                WHERE chunk_id = %s
                """,
                (chunk["chunk_id"],),
            )

            result = cursor.fetchone()

    assert result is not None
    assert result[0] == 999999
    assert result[1] == 384
    assert result[2] is True