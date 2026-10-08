from __future__ import annotations

from typing import Any

from sentence_transformers import CrossEncoder


MODEL_NAME = "Horizon-Labs/multilingual-reranker-small"


class ArabicReranker:
    """Rerank retrieved legal chunks using a multilingual Cross-Encoder."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        question: str,
        results: list[dict[str, Any]],
        top_k: int = 5,
        batch_size: int = 4,
    ) -> list[dict[str, Any]]:
        """Rank candidate chunks by query-document relevance."""

        if not results:
            return []

        pairs = [
            (question, result["text_ar_normalized"])
            for result in results
        ]

        scores = self.model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=True,
        )

        reranked = []

        for result, score in zip(results, scores, strict=True):
            item = dict(result)
            item["rerank_score"] = float(score)
            reranked.append(item)

        reranked.sort(
            key=lambda item: item["rerank_score"],
            reverse=True,
        )

        return reranked[:top_k]