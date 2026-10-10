from __future__ import annotations

import subprocess

import mlflow

from legal_rag.config import (
    CIVIL_CODE_DVC_PATH,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
)
from legal_rag.rag import ask


def get_git_commit() -> str:
    """Return the current Git commit hash."""
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()


def get_dvc_data_hash() -> str:
    """Read the tracked data hash from the DVC metadata file."""

    import yaml

    dvc_metadata = yaml.safe_load(
        CIVIL_CODE_DVC_PATH.read_text(encoding="utf-8")
    )

    return dvc_metadata["outs"][0]["md5"]

def run_rag_experiment(
    question: str,
    top_k: int = 5,
    candidate_k: int = 20,
) -> dict:
    """Run RAG and track the experiment in MLflow."""

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(
        run_name="rag-ollama-test"
    ):
        result = ask(
            question=question,
            top_k=top_k,
            candidate_k=candidate_k,
        )

        mlflow.log_params(
            {
                "model": "qwen2.5:3b",
                "top_k": top_k,
                "candidate_k": candidate_k,
                "retrieval": "hybrid_search",
                "fusion": "weighted_rrf",
                "generator": "ollama",
            }
        )

        mlflow.set_tags(
            {
                "dataset": "Egyptian Civil Code",
                "dataset_path": (
                    "data/raw/egyptian_civil_code.pdf"
                ),
                "dvc_data_hash": get_dvc_data_hash(),
                "git_commit": get_git_commit(),
            }
        )

        mlflow.log_text(
            question,
            "question.txt",
        )

        mlflow.log_text(
            result["answer"],
            "answer.txt",
        )

        mlflow.log_text(
            result["context"],
            "retrieved_context.txt",
        )

        return result


if __name__ == "__main__":
    result = run_rag_experiment(
        question="ما هي شروط العقد؟",
    )

    print("\n=== ANSWER ===")
    print(result["answer"])

    print("\n=== SOURCES ===")
    for source in result["sources"]:
        print(source)