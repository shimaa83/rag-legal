
from __future__ import annotations

import json

from legal_rag.chunking import chunk_articles
from legal_rag.config import PROCESSED_JSON_PATH
from legal_rag.embeddings import ArabicEmbedder
from legal_rag.ingestion import ingest_chunks


def main() -> None:
    """Rebuild multilingual embeddings and update PostgreSQL records."""

    if not PROCESSED_JSON_PATH.exists():
        raise FileNotFoundError(
            f"Processed corpus not found: {PROCESSED_JSON_PATH}"
        )

    with PROCESSED_JSON_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        articles = json.load(file)

    if not isinstance(articles, list):
        raise ValueError(
            "The processed corpus must contain a JSON list of articles."
        )

    if not articles:
        raise ValueError("The processed corpus is empty.")

    chunks = chunk_articles(articles)

    if not chunks:
        raise ValueError("No chunks were generated from the corpus.")

    print(f"Loaded {len(articles)} articles.")
    print(f"Generated {len(chunks)} chunks.")
    print("Preparing multilingual Arabic-English embeddings...")

    embedder = ArabicEmbedder()

    ingest_chunks(
        chunks=chunks,
        embedder=embedder,
        batch_size=32,
    )

    print("Multilingual ingestion completed successfully.")
    print("Arabic and English text were submitted for indexing.")


if __name__ == "__main__":
    main()
