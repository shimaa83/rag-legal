from __future__ import annotations

from typing import Any


def chunk_articles(
    articles: list[dict[str, Any]],
    max_chars: int | None = None,
    overlap: int = 0,
) -> list[dict[str, Any]]:
    """Create legal chunks while preserving article-level citations.

    Baseline strategy:
    - One legal article is one chunk.
    - Long articles can optionally be split by paragraphs.
    - Every resulting chunk keeps the original article number and citation.
    """
    chunks: list[dict[str, Any]] = []

    for article in articles:
        article_number = article["article_number"]
        text = (article.get("text_ar_normalized") or "").strip()

        if not text:
            continue

        if max_chars is None or len(text) <= max_chars:
            chunks.append(
                _build_chunk(
                    article=article,
                    text=text,
                    chunk_index=0,
                )
            )
            continue

        paragraphs = [
            paragraph.strip()
            for paragraph in text.split("\n")
            if paragraph.strip()
        ]

        current_parts: list[str] = []
        current_length = 0
        chunk_index = 0

        for paragraph in paragraphs:
            paragraph_length = len(paragraph)

            if (
                current_parts
                and current_length + paragraph_length + 1 > max_chars
            ):
                chunk_text = "\n".join(current_parts)

                chunks.append(
                    _build_chunk(
                        article=article,
                        text=chunk_text,
                        chunk_index=chunk_index,
                    )
                )

                chunk_index += 1

                if overlap > 0:
                    overlap_text = chunk_text[-overlap:]
                    current_parts = [overlap_text, paragraph]
                    current_length = len(overlap_text) + paragraph_length + 1
                else:
                    current_parts = [paragraph]
                    current_length = paragraph_length
            else:
                current_parts.append(paragraph)
                current_length += paragraph_length + 1

        if current_parts:
            chunks.append(
                _build_chunk(
                    article=article,
                    text="\n".join(current_parts),
                    chunk_index=chunk_index,
                )
            )

    return chunks


def _build_chunk(
    article: dict[str, Any],
    text: str,
    chunk_index: int,
) -> dict[str, Any]:
    """Build a chunk while preserving legal metadata."""

    article_number = article["article_number"]

    return {
        "chunk_id": f"article_{article_number}_{chunk_index}",
        "article_number": article_number,
        "text_ar": article.get("text_ar"),
        "text_ar_normalized": text,
        "text_en": article.get("text_en"),
        "book": article.get("book"),
        "chapter": article.get("chapter"),
        "section": article.get("section"),
        "topic": article.get("topic"),
        "is_repealed": article.get("is_repealed", False),
        "source_page": article.get("source_page"),
        "citation": article.get(
            "citation",
            f"Egyptian Civil Code, Article {article_number}",
        ),
    }