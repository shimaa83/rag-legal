from legal_rag.chunking import chunk_articles
from legal_rag.config import PROCESSED_JSON_PATH
from legal_rag.embeddings import ArabicEmbedder
from legal_rag.ingestion import ingest_chunks

import json


with PROCESSED_JSON_PATH.open("r", encoding="utf-8") as file:
    articles = json.load(file)

chunks = chunk_articles(articles)

test_chunks = chunks
print(f"Testing with {len(test_chunks)} chunks...")

embedder = ArabicEmbedder()

ingest_chunks(
    chunks=test_chunks,
    embedder=embedder,
    batch_size=32,
)

print("Batch ingestion test completed successfully.")