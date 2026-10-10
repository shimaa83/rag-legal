
from __future__ import annotations

from sentence_transformers import SentenceTransformer

MODEL_NAME = "intfloat/multilingual-e5-small"


class ArabicEmbedder:
    """Generate multilingual Arabic-English legal embeddings using E5."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed legal documents using the E5 passage prefix."""

        passages = [
            f"passage: {text.strip()}"
            for text in texts
        ]

        embeddings = self.model.encode(
            passages,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Embed Arabic or English questions using the E5 query prefix."""

        query = query.strip()

        if not query:
            raise ValueError("Query must not be empty.")

        embedding = self.model.encode(
            f"query: {query}",
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        return embedding.tolist()
