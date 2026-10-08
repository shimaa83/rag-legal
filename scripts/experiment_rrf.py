from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import mlflow
from dotenv import load_dotenv

from legal_rag.embeddings import ArabicEmbedder
from legal_rag.retrieval import hybrid_search


# ---------------------------------------------------------
# Environment
# ---------------------------------------------------------

load_dotenv()


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

QUESTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "retrieval_questions.json"
)

REPORTS_DIR = PROJECT_ROOT / "reports"


# ---------------------------------------------------------
# PostgreSQL / MLflow configuration
# ---------------------------------------------------------

POSTGRES_HOST = os.getenv(
    "POSTGRES_HOST",
    "localhost",
)

POSTGRES_PORT = os.getenv(
    "POSTGRES_PORT",
    "5432",
)

POSTGRES_USER = os.getenv(
    "POSTGRES_USER",
)

POSTGRES_PASSWORD = os.getenv(
    "POSTGRES_PASSWORD",
)

MLFLOW_DB_NAME = os.getenv(
    "MLFLOW_DB_NAME",
    "mlflow",
)


def build_mlflow_tracking_uri() -> str:
    """Build the MLflow PostgreSQL tracking URI."""

    required_values = {
        "POSTGRES_USER": POSTGRES_USER,
        "POSTGRES_PASSWORD": POSTGRES_PASSWORD,
        "POSTGRES_HOST": POSTGRES_HOST,
        "POSTGRES_PORT": POSTGRES_PORT,
        "MLFLOW_DB_NAME": MLFLOW_DB_NAME,
    }

    missing = [
        key
        for key, value in required_values.items()
        if not value
    ]

    if missing:
        raise RuntimeError(
            "Missing environment variables: "
            + ", ".join(missing)
        )

    return (
        "postgresql+psycopg://"
        f"{POSTGRES_USER}:"
        f"{POSTGRES_PASSWORD}@"
        f"{POSTGRES_HOST}:"
        f"{POSTGRES_PORT}/"
        f"{MLFLOW_DB_NAME}"
    )


# ---------------------------------------------------------
# RRF experiment configurations
# ---------------------------------------------------------

CONFIGS = [
    {
        "name": "rrf_k20_candidates20",
        "rrf_k": 20,
        "candidate_k": 20,
    },
    {
        "name": "rrf_k40_candidates20",
        "rrf_k": 40,
        "candidate_k": 20,
    },
    {
        "name": "rrf_k60_candidates20",
        "rrf_k": 60,
        "candidate_k": 20,
    },
    {
        "name": "rrf_k60_candidates50",
        "rrf_k": 60,
        "candidate_k": 50,
    },
    {
        "name": "rrf_k100_candidates50",
        "rrf_k": 100,
        "candidate_k": 50,
    },
]


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


# ---------------------------------------------------------
# Evaluate one configuration
# ---------------------------------------------------------

def evaluate_configuration(
    questions: list[dict[str, Any]],
    embedder: ArabicEmbedder,
    candidate_k: int,
    rrf_k: int,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Evaluate one Hybrid + RRF configuration."""

    recall_values: list[float] = []
    reciprocal_rank_values: list[float] = []

    details: list[dict[str, Any]] = []

    for index, question_data in enumerate(
        questions,
        start=1,
    ):
        question = question_data["question"]

        relevant_articles = question_data[
            "relevant_articles"
        ]

        results = hybrid_search(
            question=question,
            top_k=10,
            candidate_k=candidate_k,
            rrf_k=rrf_k,
            embedder=embedder,
        )

        recall = recall_at_k(
            results=results,
            relevant_articles=relevant_articles,
            k=5,
        )

        rr = reciprocal_rank(
            results=results,
            relevant_articles=relevant_articles,
        )

        recall_values.append(recall)
        reciprocal_rank_values.append(rr)

        details.append(
            {
                "question": question,
                "expected_articles": relevant_articles,
                "top_5": [
                    result["article_number"]
                    for result in results[:5]
                ],
                "recall_at_5": recall,
                "reciprocal_rank": rr,
            }
        )

        print(
            f"[{index}/{len(questions)}] "
            f"{question} "
            f"-> RR={rr:.3f}"
        )

    metrics = {
        "recall_at_5": (
            sum(recall_values)
            / len(recall_values)
        ),
        "mrr": (
            sum(reciprocal_rank_values)
            / len(reciprocal_rank_values)
        ),
    }

    return metrics, details


# ---------------------------------------------------------
# MLflow configuration
# ---------------------------------------------------------

def configure_mlflow() -> str:
    """Configure MLflow to use PostgreSQL."""

    tracking_uri = build_mlflow_tracking_uri()

    mlflow.set_tracking_uri(
        tracking_uri
    )

    mlflow.set_experiment(
        "legal-rag-retrieval"
    )

    return tracking_uri


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main() -> None:
    """Run all RRF experiments and log them to MLflow."""

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # Load evaluation questions
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
    # Configure MLflow
    # -----------------------------------------------------

    tracking_uri = configure_mlflow()

    print(
        f"MLflow tracking URI: "
        f"{tracking_uri}"
    )

    # -----------------------------------------------------
    # Load embedding model once
    # -----------------------------------------------------

    print(
        "\nLoading embedding model..."
    )

    embedder = ArabicEmbedder()

    all_results: list[dict[str, Any]] = []

    # -----------------------------------------------------
    # Run experiments
    # -----------------------------------------------------

    for config in CONFIGS:
        config_name = config["name"]

        rrf_k = config["rrf_k"]

        candidate_k = config[
            "candidate_k"
        ]

        print(
            "\n"
            + "=" * 80
        )

        print(
            f"Configuration: "
            f"{config_name}"
        )

        print(
            f"RRF k: "
            f"{rrf_k}"
        )

        print(
            f"Candidate k: "
            f"{candidate_k}"
        )

        print(
            "=" * 80
        )

        with mlflow.start_run(
            run_name=config_name
        ):
            # ---------------------------------------------
            # Experiment parameters
            # ---------------------------------------------

            mlflow.log_params(
                {
                    "chunking_strategy": "article",
                    "chunk_size": "article",
                    "overlap": 0,
                    "embedding_model": (
                        embedder.model_name
                    ),
                    "retrieval_type": (
                        "hybrid_rrf"
                    ),
                    "rrf_k": rrf_k,
                    "candidate_k": candidate_k,
                    "retrieval_top_k": 10,
                    "evaluation_questions": (
                        len(questions)
                    ),
                }
            )

            # ---------------------------------------------
            # Evaluation
            # ---------------------------------------------

            metrics, details = (
                evaluate_configuration(
                    questions=questions,
                    embedder=embedder,
                    candidate_k=candidate_k,
                    rrf_k=rrf_k,
                )
            )

            # ---------------------------------------------
            # MLflow metrics
            # ---------------------------------------------

            mlflow.log_metrics(
                {
                    "recall_at_5": (
                        metrics[
                            "recall_at_5"
                        ]
                    ),
                    "mrr": (
                        metrics[
                            "mrr"
                        ]
                    ),
                }
            )

            # ---------------------------------------------
            # Save detailed report
            # ---------------------------------------------

            report = {
                "configuration": config,
                "metrics": metrics,
                "details": details,
            }

            report_path = (
                REPORTS_DIR
                / f"rrf_{config_name}.json"
            )

            with report_path.open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    report,
                    file,
                    ensure_ascii=False,
                    indent=2,
                )

            # ---------------------------------------------
            # Log artifact
            # ---------------------------------------------

            mlflow.log_artifact(
                str(report_path),
                artifact_path="evaluation",
            )

            # ---------------------------------------------
            # Store run information
            # ---------------------------------------------

            active_run = mlflow.active_run()

            if active_run is None:
                raise RuntimeError(
                    "MLflow active run is missing."
                )

            run_id = active_run.info.run_id

            result = {
                "run_id": run_id,
                "configuration": config,
                "metrics": metrics,
            }

            all_results.append(result)

            # ---------------------------------------------
            # Print result
            # ---------------------------------------------

            print(
                "\nResults:"
            )

            print(
                f"Recall@5: "
                f"{metrics['recall_at_5']:.3f}"
            )

            print(
                f"MRR: "
                f"{metrics['mrr']:.3f}"
            )

    # -----------------------------------------------------
    # Select best configuration
    # -----------------------------------------------------

    best_result = max(
        all_results,
        key=lambda item: (
            item["metrics"]["mrr"],
            item["metrics"]["recall_at_5"],
        ),
    )

    # -----------------------------------------------------
    # Save summary
    # -----------------------------------------------------

    summary_path = (
        REPORTS_DIR
        / "rrf_experiment_summary.json"
    )

    summary = {
        "num_questions": len(questions),
        "results": all_results,
        "best_configuration": best_result,
    }

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    # -----------------------------------------------------
    # Final comparison
    # -----------------------------------------------------

    print(
        "\n"
        + "=" * 80
    )

    print(
        "FINAL RRF EXPERIMENT RESULTS"
    )

    print(
        "=" * 80
    )

    for result in all_results:
        config = result[
            "configuration"
        ]

        metrics = result[
            "metrics"
        ]

        print(
            f"\n{config['name']}"
        )

        print(
            f"  RRF k: "
            f"{config['rrf_k']}"
        )

        print(
            f"  Candidate k: "
            f"{config['candidate_k']}"
        )

        print(
            f"  Recall@5: "
            f"{metrics['recall_at_5']:.3f}"
        )

        print(
            f"  MRR: "
            f"{metrics['mrr']:.3f}"
        )

    # -----------------------------------------------------
    # Best configuration
    # -----------------------------------------------------

    best_config = best_result[
        "configuration"
    ]

    best_metrics = best_result[
        "metrics"
    ]

    print(
        "\n"
        + "-" * 80
    )

    print(
        "BEST CONFIGURATION"
    )

    print(
        f"Name: "
        f"{best_config['name']}"
    )

    print(
        f"RRF k: "
        f"{best_config['rrf_k']}"
    )

    print(
        f"Candidate k: "
        f"{best_config['candidate_k']}"
    )

    print(
        f"Recall@5: "
        f"{best_metrics['recall_at_5']:.3f}"
    )

    print(
        f"MRR: "
        f"{best_metrics['mrr']:.3f}"
    )

    print(
        f"\nSummary saved to: "
        f"{summary_path}"
    )


if __name__ == "__main__":
    main()