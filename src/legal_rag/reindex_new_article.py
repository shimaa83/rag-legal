

from __future__ import annotations

import argparse

from legal_rag.data_preprocessing import preprocess_article
from legal_rag.chunking import chunk_articles
from legal_rag.embeddings import ArabicEmbedder
from legal_rag.ingestion import ingest_chunks


def validate_article(article: dict) -> None:
    """Validate a single legal article."""

    if not isinstance(article, dict):
        raise TypeError("Article must be a dictionary.")

    number = article.get("article_number")

    if isinstance(number, bool) or not isinstance(number, int):
        raise ValueError("Article number must be an integer.")

    if number <= 0:
        raise ValueError("Article number must be positive.")

    text = article.get("text_ar", "")

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Arabic article text cannot be empty.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate, preprocess, embed, and ingest one legal article."
    )

    parser.add_argument("--article-number", type=int, required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--book", default=None)
    parser.add_argument("--chapter", default=None)
    parser.add_argument("--section", default=None)
    parser.add_argument("--topic", default=None)
    parser.add_argument("--text-en", default="")
    parser.add_argument("--source-page", type=int, default=None)
    parser.add_argument("--citation", default=None)
    parser.add_argument("--repealed", action="store_true")

    args = parser.parse_args()

    # 1. Build the raw article
    raw_article = {
        "article_number": args.article_number,
        "book": args.book,
        "chapter": args.chapter,
        "section": args.section,
        "topic": args.topic,
        "ar_text": args.text,
        "text_en": args.text_en,
        "source_page": args.source_page,
        "citation": args.citation,
        "is_repealed": args.repealed,
    }

    # 2. Validate input
    validate_article({
        "article_number": raw_article["article_number"],
        "text_ar": raw_article["ar_text"],
    })
    print("1/5 - Validation passed.")

    # 3. Preprocess the single article
    article = preprocess_article(raw_article)
    validate_article(article)
    print("2/5 - Preprocessing completed.")

    # Do not index repealed articles by accident.
    if article["is_repealed"]:
        raise ValueError(
            "This article is marked as repealed. "
            "Confirm the intended indexing policy before inserting it."
        )

    # 4. Generate chunks
    chunks = chunk_articles([article])

    if not chunks:
        raise ValueError("No chunks were generated.")

    print(f"3/5 - Generated {len(chunks)} chunk(s).")

    # 5. Generate embeddings and ingest
    embedder = ArabicEmbedder()

    ingest_chunks(
        chunks=chunks,
        embedder=embedder,
        batch_size=32,
    )

    print("4/5 - Ingestion function completed.")
    print("5/5 - Verify database insertion and retrieval.")


if __name__ == "__main__":
    main()


