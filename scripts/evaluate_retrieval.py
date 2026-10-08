from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from legal_rag.embeddings import ArabicEmbedder
from legal_rag.reranker import ArabicReranker
from legal_rag.retrieval import hybrid_search, vector_search


PROJECT_ROOT = Path(__file__).resolve().parents[1]

QUESTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "retrieval_questions.json"
)

REPORTS_DIR = PROJECT_ROOT / "reports"

REPORT_PATH = (
    REPORTS_DIR / "retrieval_evaluation.json"
)


def recall_at_k(
    results: list[dict[str, Any]],
    relevant_articles: list[int],
    k: int,
) -> float:
    """Return 1 if any relevant article appears in top-k."""

    retrieved_articles = {
        result["article_number"]
        for result in results[:k]
    }

    return float(
        any(
            article in retrieved_articles
            for article in relevant_articles
        )
    )


def reciprocal_rank(
    results: list[dict[str, Any]],
    relevant_articles: list[int],
) -> float:
    """Return reciprocal rank of the first relevant article."""

    for rank, result in enumerate(
        results,
        start=1,
    ):
        if result["article_number"] in relevant_articles:
            return 1.0 / rank

    return 0.0


def evaluate_results(
    questions: list[dict[str, Any]],
    results_per_question: list[list[dict[str, Any]]],
) -> dict[str, float]:
    """Calculate retrieval metrics."""

    recall_values = []
    reciprocal_rank_values = []

    for question, results in zip(
        questions,
        results_per_question,
        strict=True,
    ):
        relevant_articles = question["relevant_articles"]

        recall_values.append(
            recall_at_k(
                results=results,
                relevant_articles=relevant_articles,
                k=5,
            )
        )

        reciprocal_rank_values.append(
            reciprocal_rank(
                results=results,
                relevant_articles=relevant_articles,
            )
        )

    return {
        "recall@5": (
            sum(recall_values)
            / len(recall_values)
        ),
        "mrr": (
            sum(reciprocal_rank_values)
            / len(reciprocal_rank_values)
        ),
    }


def evaluate_vector(
    questions: list[dict[str, Any]],
    embedder: ArabicEmbedder,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Evaluate vector search."""

    results_per_question = []
    details = []

    for question_data in questions:
        question = question_data["question"]
        relevant_articles = question_data[
            "relevant_articles"
        ]

        results = vector_search(
            question=question,
            top_k=10,
            embedder=embedder,
        )

        results_per_question.append(results)

        details.append(
            {
                "question": question,
                "expected_articles": relevant_articles,
                "top_5": [
                    result["article_number"]
                    for result in results[:5]
                ],
                "reciprocal_rank": reciprocal_rank(
                    results=results,
                    relevant_articles=relevant_articles,
                ),
            }
        )

    metrics = evaluate_results(
        questions=questions,
        results_per_question=results_per_question,
    )

    return metrics, details


def evaluate_hybrid(
    questions: list[dict[str, Any]],
    embedder: ArabicEmbedder,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Evaluate hybrid search with RRF."""

    results_per_question = []
    details = []

    for question_data in questions:
        question = question_data["question"]
        relevant_articles = question_data[
            "relevant_articles"
        ]

        results = hybrid_search(
            question=question,
            top_k=10,
            candidate_k=50,
            embedder=embedder,
        )

        results_per_question.append(results)

        details.append(
            {
                "question": question,
                "expected_articles": relevant_articles,
                "top_5": [
                    result["article_number"]
                    for result in results[:5]
                ],
                "reciprocal_rank": reciprocal_rank(
                    results=results,
                    relevant_articles=relevant_articles,
                ),
            }
        )

    metrics = evaluate_results(
        questions=questions,
        results_per_question=results_per_question,
    )

    return metrics, details


def evaluate_reranker(
    questions: list[dict[str, Any]],
    embedder: ArabicEmbedder,
    reranker: ArabicReranker,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Evaluate hybrid retrieval followed by reranking."""

    results_per_question = []
    details = []

    for question_data in questions:
        question = question_data["question"]
        relevant_articles = question_data[
            "relevant_articles"
        ]

        candidates = hybrid_search(
            question=question,
            top_k=10,
            candidate_k=50,
            embedder=embedder,
        )

        results = reranker.rerank(
            question=question,
            results=candidates,
            top_k=5,
            batch_size=4,
        )

        results_per_question.append(results)

        details.append(
            {
                "question": question,
                "expected_articles": relevant_articles,
                "top_5": [
                    result["article_number"]
                    for result in results[:5]
                ],
                "reciprocal_rank": reciprocal_rank(
                    results=results,
                    relevant_articles=relevant_articles,
                ),
            }
        )

    metrics = evaluate_results(
        questions=questions,
        results_per_question=results_per_question,
    )

    return metrics, details


def print_details(
    method_name: str,
    details: list[dict[str, Any]],
) -> None:
    """Print per-question evaluation results."""

    print("\n" + "=" * 80)
    print(method_name)
    print("=" * 80)

    for item in details:
        print(
            f"\nQuestion: {item['question']}"
        )

        print(
            f"Expected: "
            f"{item['expected_articles']}"
        )

        print(
            f"Top 5: "
            f"{item['top_5']}"
        )

        print(
            f"Reciprocal Rank: "
            f"{item['reciprocal_rank']:.3f}"
        )


def main() -> None:
    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with QUESTIONS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        questions = json.load(file)

    print(
        f"Evaluation questions: "
        f"{len(questions)}"
    )

    # Load E5 once and reuse it.
    embedder = ArabicEmbedder()

    # --------------------------------------------------
    # 1. Vector Search
    # --------------------------------------------------

    vector_metrics, vector_details = evaluate_vector(
        questions=questions,
        embedder=embedder,
    )

    print_details(
        "VECTOR SEARCH",
        vector_details,
    )

    # --------------------------------------------------
    # 2. Hybrid Search + RRF
    # --------------------------------------------------

    hybrid_metrics, hybrid_details = evaluate_hybrid(
        questions=questions,
        embedder=embedder,
    )

    print_details(
        "HYBRID SEARCH + RRF",
        hybrid_details,
    )

    # --------------------------------------------------
    # 3. Hybrid + Reranker
    # --------------------------------------------------

    print("\nLoading reranker...")

    reranker = ArabicReranker()

    reranker_metrics, reranker_details = (
        evaluate_reranker(
            questions=questions,
            embedder=embedder,
            reranker=reranker,
        )
    )

    print_details(
        "HYBRID + RRF + RERANKER",
        reranker_details,
    )

    # --------------------------------------------------
    # Final comparison
    # --------------------------------------------------

    comparison = {
        "vector": vector_metrics,
        "hybrid_rrf": hybrid_metrics,
        "hybrid_rrf_reranker": reranker_metrics,
    }

    print("\n" + "=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)

    print(
        f"\nVector Recall@5: "
        f"{vector_metrics['recall@5']:.3f}"
    )

    print(
        f"Vector MRR: "
        f"{vector_metrics['mrr']:.3f}"
    )

    print(
        f"\nHybrid Recall@5: "
        f"{hybrid_metrics['recall@5']:.3f}"
    )

    print(
        f"Hybrid MRR: "
        f"{hybrid_metrics['mrr']:.3f}"
    )

    print(
        f"\nReranker Recall@5: "
        f"{reranker_metrics['recall@5']:.3f}"
    )

    print(
        f"Reranker MRR: "
        f"{reranker_metrics['mrr']:.3f}"
    )

    # Save report.
    report = {
        "num_questions": len(questions),
        "metrics": comparison,
        "details": {
            "vector": vector_details,
            "hybrid_rrf": hybrid_details,
            "hybrid_rrf_reranker": (
                reranker_details
            ),
        },
    }

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"\nReport saved to: "
        f"{REPORT_PATH}"
    )


if __name__ == "__main__":
    main()