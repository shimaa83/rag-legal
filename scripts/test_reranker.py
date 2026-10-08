from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from legal_rag.embeddings import ArabicEmbedder
from legal_rag.reranker import ArabicReranker
from legal_rag.retrieval import hybrid_search


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

QUESTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "retrieval_questions.json"
)

REPORTS_DIR = PROJECT_ROOT / "reports"

REPORT_PATH = (
    REPORTS_DIR
    / "weighted_hybrid_reranker_evaluation.json"
)


# ---------------------------------------------------------
# Best Weighted RRF configuration
# ---------------------------------------------------------

RRF_K = 20
CANDIDATE_K = 20

VECTOR_WEIGHT = 1.0
KEYWORD_WEIGHT = 0.25

HYBRID_TOP_K = 10
RERANKER_TOP_K = 5


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

def recall_at_k(
    results: list[dict[str, Any]],
    relevant_articles: list[int],
    k: int,
) -> float:
    """Return 1 if a relevant article appears in top-k."""

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


def calculate_metrics(
    questions: list[dict[str, Any]],
    results_per_question: list[list[dict[str, Any]]],
) -> dict[str, float]:
    """Calculate Recall@5 and MRR."""

    recall_values: list[float] = []
    reciprocal_rank_values: list[float] = []

    for question_data, results in zip(
        questions,
        results_per_question,
        strict=True,
    ):
        relevant_articles = question_data[
            "relevant_articles"
        ]

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
        "recall_at_5": (
            sum(recall_values)
            / len(recall_values)
        ),
        "mrr": (
            sum(reciprocal_rank_values)
            / len(reciprocal_rank_values)
        ),
    }


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main() -> None:
    """Evaluate best weighted hybrid search with reranker."""

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # Load questions
    # -----------------------------------------------------

    with QUESTIONS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        questions = json.load(file)

    if not questions:
        raise RuntimeError(
            "No evaluation questions found."
        )

    print(
        f"Evaluation questions: "
        f"{len(questions)}"
    )

    # -----------------------------------------------------
    # Load models once
    # -----------------------------------------------------

    print(
        "\nLoading embedding model..."
    )

    embedder = ArabicEmbedder()

    print(
        "\nLoading reranker..."
    )

    reranker = ArabicReranker()

    # -----------------------------------------------------
    # Store results
    # -----------------------------------------------------

    hybrid_results_per_question = []
    reranked_results_per_question = []

    details = []

    # -----------------------------------------------------
    # Evaluate
    # -----------------------------------------------------

    for index, question_data in enumerate(
        questions,
        start=1,
    ):
        question = question_data["question"]

        relevant_articles = question_data[
            "relevant_articles"
        ]

        print(
            "\n"
            + "=" * 80
        )

        print(
            f"[{index}/{len(questions)}]"
        )

        print(
            f"Question: "
            f"{question}"
        )

        print(
            f"Expected articles: "
            f"{relevant_articles}"
        )

        # -------------------------------------------------
        # Best Weighted Hybrid + RRF
        # -------------------------------------------------

        hybrid_results = hybrid_search(
            question=question,
            top_k=HYBRID_TOP_K,
            candidate_k=CANDIDATE_K,
            rrf_k=RRF_K,
            vector_weight=VECTOR_WEIGHT,
            keyword_weight=KEYWORD_WEIGHT,
            embedder=embedder,
        )

        # -------------------------------------------------
        # Reranking
        # -------------------------------------------------

        reranked_results = reranker.rerank(
            question=question,
            results=hybrid_results,
            top_k=RERANKER_TOP_K,
            batch_size=4,
        )

        hybrid_results_per_question.append(
            hybrid_results
        )

        reranked_results_per_question.append(
            reranked_results
        )

        # -------------------------------------------------
        # Per-question metrics
        # -------------------------------------------------

        hybrid_rr = reciprocal_rank(
            results=hybrid_results,
            relevant_articles=relevant_articles,
        )

        reranked_rr = reciprocal_rank(
            results=reranked_results,
            relevant_articles=relevant_articles,
        )

        hybrid_top_5 = [
            result["article_number"]
            for result in hybrid_results[:5]
        ]

        reranked_top_5 = [
            result["article_number"]
            for result in reranked_results[:5]
        ]

        print(
            f"\nHybrid Top 5: "
            f"{hybrid_top_5}"
        )

        print(
            f"Hybrid Reciprocal Rank: "
            f"{hybrid_rr:.3f}"
        )

        print(
            f"Reranked Top 5: "
            f"{reranked_top_5}"
        )

        print(
            f"Reranked Reciprocal Rank: "
            f"{reranked_rr:.3f}"
        )

        # -------------------------------------------------
        # Detailed reranker results
        # -------------------------------------------------

        reranker_details = []

        for rank, result in enumerate(
            reranked_results,
            start=1,
        ):
            reranker_details.append(
                {
                    "rank": rank,
                    "article_number": (
                        result[
                            "article_number"
                        ]
                    ),
                    "citation": (
                        result[
                            "citation"
                        ]
                    ),
                    "rerank_score": (
                        result[
                            "rerank_score"
                        ]
                    ),
                    "rrf_score": (
                        result[
                            "rrf_score"
                        ]
                    ),
                    "vector_rank": (
                        result[
                            "vector_rank"
                        ]
                    ),
                    "keyword_rank": (
                        result[
                            "keyword_rank"
                        ]
                    ),
                }
            )

        details.append(
            {
                "question": question,
                "expected_articles": (
                    relevant_articles
                ),
                "hybrid_top_5": hybrid_top_5,
                "hybrid_reciprocal_rank": hybrid_rr,
                "reranked_top_5": reranked_top_5,
                "reranked_reciprocal_rank": (
                    reranked_rr
                ),
                "reranked_results": (
                    reranker_details
                ),
            }
        )

    # -----------------------------------------------------
    # Calculate final metrics
    # -----------------------------------------------------

    hybrid_metrics = calculate_metrics(
        questions=questions,
        results_per_question=(
            hybrid_results_per_question
        ),
    )

    reranked_metrics = calculate_metrics(
        questions=questions,
        results_per_question=(
            reranked_results_per_question
        ),
    )

    # -----------------------------------------------------
    # Final comparison
    # -----------------------------------------------------

    print(
        "\n"
        + "=" * 80
    )

    print(
        "FINAL COMPARISON"
    )

    print(
        "=" * 80
    )

    print(
        "\nBest Weighted Hybrid + RRF"
    )

    print(
        f"Recall@5: "
        f"{hybrid_metrics['recall_at_5']:.3f}"
    )

    print(
        f"MRR: "
        f"{hybrid_metrics['mrr']:.3f}"
    )

    print(
        "\nBest Weighted Hybrid + RRF + Reranker"
    )

    print(
        f"Recall@5: "
        f"{reranked_metrics['recall_at_5']:.3f}"
    )

    print(
        f"MRR: "
        f"{reranked_metrics['mrr']:.3f}"
    )

    # -----------------------------------------------------
    # Improvement
    # -----------------------------------------------------

    recall_improvement = (
        reranked_metrics["recall_at_5"]
        - hybrid_metrics["recall_at_5"]
    )

    mrr_improvement = (
        reranked_metrics["mrr"]
        - hybrid_metrics["mrr"]
    )

    print(
        "\nImprovement after reranking"
    )

    print(
        f"Recall@5 improvement: "
        f"{recall_improvement:+.3f}"
    )

    print(
        f"MRR improvement: "
        f"{mrr_improvement:+.3f}"
    )

    # -----------------------------------------------------
    # Save report
    # -----------------------------------------------------

    report = {
        "evaluation_questions": len(
            questions
        ),
        "configuration": {
            "rrf_k": RRF_K,
            "candidate_k": CANDIDATE_K,
            "vector_weight": VECTOR_WEIGHT,
            "keyword_weight": KEYWORD_WEIGHT,
            "hybrid_top_k": HYBRID_TOP_K,
            "reranker_top_k": RERANKER_TOP_K,
        },
        "hybrid_metrics": hybrid_metrics,
        "reranked_metrics": reranked_metrics,
        "improvement": {
            "recall_at_5": recall_improvement,
            "mrr": mrr_improvement,
        },
        "details": details,
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