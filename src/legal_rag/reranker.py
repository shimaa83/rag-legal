from __future__ import annotations

from typing import Any

from sentence_transformers import CrossEncoder


MODEL_NAME = "Horizon-Labs/multilingual-reranker-small"


class ArabicReranker:
    """Rerank legal chunks using a multilingual Cross-Encoder."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    @staticmethod
    def _get_document_text(result: dict[str, Any]) -> str:
        """Combine Arabic legal text and English translation."""

        text_ar = (
            result.get("text_ar_normalized")
            or result.get("ar_text")
            or result.get("text_ar")
            or ""
        )

        text_en = result.get("text_en") or ""

        parts: list[str] = []

        if text_ar.strip():
            parts.append(f"Arabic legal text:\n{text_ar.strip()}")

        if text_en.strip():
            parts.append(f"English translation:\n{text_en.strip()}")

        return "\n\n".join(parts)

    def rerank(
        self,
        question: str,
        results: list[dict[str, Any]],
        top_k: int = 5,
        batch_size: int = 4,
    ) -> list[dict[str, Any]]:
        """Rerank candidate documents by query-document relevance."""

        if not question or not question.strip():
            raise ValueError("Question must not be empty.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

        if not results:
            return []

        valid_results: list[dict[str, Any]] = []
        pairs: list[tuple[str, str]] = []

        for result in results:
            document_text = self._get_document_text(result)

            if not document_text:
                continue

            valid_results.append(result)
            pairs.append((question.strip(), document_text))

        if not pairs:
            return []

        scores = self.model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=False,
        )

        reranked: list[dict[str, Any]] = []

        for result, score in zip(valid_results, scores, strict=True):
            item = dict(result)
            item["rerank_score"] = float(score)
            reranked.append(item)

        reranked.sort(
            key=lambda item: item["rerank_score"],
            reverse=True,
        )

        return reranked[:top_k]

