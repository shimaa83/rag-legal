from __future__ import annotations

from sentence_transformers import SentenceTransformer


MODEL_NAME = "intfloat/multilingual-e5-small"


class ArabicEmbedder:
    """Generate Arabic text embeddings using multilingual E5."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed legal documents using the E5 passage prefix."""
        passages = [f"passage: {text}" for text in texts]

        embeddings = self.model.encode(
            passages,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Embed a user query using the E5 query prefix."""
        embedding = self.model.encode(
            f"query: {query}",
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        return embedding.tolist()