from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from pathlib import Path

# Load GEMINI_API_KEY / DAGSHUB_USER_TOKEN / Langfuse keys from .env if present.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import mlflow
import pandas as pd
from openai import AsyncOpenAI
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

import legal_rag.config as legal_config
from legal_rag.config import (
    API_JUDGE_MODEL,
    CANDIDATE_K,
    DAGSHUB_REPO_NAME,
    DAGSHUB_REPO_OWNER,
    GENERATOR_MODEL,
    JUDGE_BASE_MODEL,
    JUDGE_NUM_CTX,
    JUDGE_PROVIDER,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_RUN_NAME,
    MLFLOW_TRACKING_URI,
    NUM_QUESTIONS,
    PROMPT_TEMPLATE,
    TOP_K,
    USE_DAGSHUB,
)
from legal_rag.mlflow_tracking import get_dvc_data_hash, get_git_commit
from legal_rag.rag import ask

# ============================================================
# Configuration (all main settings live in legal_rag/config.py)
# ============================================================

MODEL_SLUG = re.sub(r"[^A-Za-z0-9_.-]+", "-", GENERATOR_MODEL)
PROMPT_HASH = hashlib.sha256(PROMPT_TEMPLATE.encode("utf-8")).hexdigest()[:8]

if JUDGE_PROVIDER == "ollama":
    JUDGE_MODEL = f"{JUDGE_BASE_MODEL.replace(':', '-')}-ctx{JUDGE_NUM_CTX // 1024}k"
else:
    JUDGE_MODEL = API_JUDGE_MODEL

JUDGE_MAX_TOKENS = 4096
OLLAMA_BASE_URL = "http://localhost:11434/v1"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
SAMPLE_TIMEOUT_SECONDS = 900

# Free API tiers rate-limit hard: retry with backoff and pause between metrics.
METRIC_RETRIES = int(os.getenv("METRIC_RETRIES", "4"))
RETRY_BACKOFF_SECONDS = float(os.getenv("RETRY_BACKOFF_SECONDS", "15"))
PAUSE_BETWEEN_METRICS = float(
    os.getenv("PAUSE_BETWEEN_METRICS", "0" if JUDGE_PROVIDER == "ollama" else "3")
)

# Metric name -> the sample fields it needs.
METRIC_INPUTS = {
    "faithfulness": ("user_input", "response", "retrieved_contexts"),
    "answer_relevancy": ("user_input", "response"),
    "context_precision": ("user_input", "reference", "retrieved_contexts"),
    "context_recall": ("user_input", "reference", "retrieved_contexts"),
}
METRIC_NAMES = list(METRIC_INPUTS)

# ============================================================
# Paths (outputs are separate per generator model)
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = PROJECT_ROOT / "data" / "evaluation"
RUN_DIR = EVAL_DIR / "runs" / MODEL_SLUG / f"k{TOP_K}_p{PROMPT_HASH}"

DATASET_PATH = EVAL_DIR / "ragas_dataset.json"
RESULTS_PATH = RUN_DIR / "ragas_results.json"
SUMMARY_PATH = RUN_DIR / "ragas_summary.json"
SAMPLES_CACHE_PATH = RUN_DIR / "ragas_samples_cache.json"
DETAILS_CACHE_PATH = RUN_DIR / "ragas_details_cache.json"
CSV_PATH = RUN_DIR / "ragas_per_sample.csv"
ERRORS_PATH = RUN_DIR / "ragas_errors.json"


# ============================================================
# Helpers
# ============================================================

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, data) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def is_valid_number(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


# ============================================================
# MLflow / DagsHub setup
# ============================================================

def setup_mlflow() -> None:
    """
    Point MLflow at DagsHub (repo set in config.py).
    Auth: set DAGSHUB_USER_TOKEN. Set USE_LOCAL_MLFLOW=1 for local MLflow.
    """
    if USE_DAGSHUB:
        import dagshub

        dagshub.init(
            repo_owner=DAGSHUB_REPO_OWNER,
            repo_name=DAGSHUB_REPO_NAME,
            mlflow=True,
        )
    else:
        print("USE_LOCAL_MLFLOW=1 -> using local MLflow")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    print(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")


def ensure_judge_model() -> None:
    """Create the derived Ollama judge model (bigger context) if missing."""
    listing = subprocess.run(
        ["ollama", "list"], capture_output=True, text=True, check=True
    ).stdout

    if JUDGE_MODEL in listing:
        print(f"Judge model ready: {JUDGE_MODEL}")
        return

    print(f"Creating judge model {JUDGE_MODEL} (base={JUDGE_BASE_MODEL}, num_ctx={JUDGE_NUM_CTX})")
    subprocess.run(["ollama", "pull", JUDGE_BASE_MODEL], check=True)

    modelfile = (
        f"FROM {JUDGE_BASE_MODEL}\n"
        f"PARAMETER num_ctx {JUDGE_NUM_CTX}\n"
        "PARAMETER temperature 0\n"
    )
    with tempfile.NamedTemporaryFile(
        "w", suffix=".Modelfile", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(modelfile)
        tmp_path = tmp.name

    try:
        subprocess.run(["ollama", "create", JUDGE_MODEL, "-f", tmp_path], check=True)
    finally:
        os.unlink(tmp_path)


# ============================================================
# Retrieval metrics
# ============================================================

def article_overlap(relevant: list[int], retrieved: list[int], divide_by: str) -> float:
    """Share of overlapping articles, divided by 'relevant' (recall) or 'retrieved' (precision)."""
    relevant_set, retrieved_set = set(relevant), set(retrieved)
    base = relevant_set if divide_by == "relevant" else retrieved_set
    return len(relevant_set & retrieved_set) / len(base) if base else 0.0


# ============================================================
# Run the RAG system
# ============================================================

def build_ragas_dataset(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Run the RAG system for every evaluation question."""
    samples, details = [], []

    print("\n" + "=" * 70)
    print(f"Running RAG on evaluation dataset | generator = {GENERATOR_MODEL}")
    print("=" * 70)

    for index, record in enumerate(records, start=1):
        question = record["question"]
        reference = record["answer"]
        relevant = record["relevant_articles"]

        print(f"\n[{index}/{len(records)}] {question}")

        rag = ask(
            question=question,
            top_k=TOP_K,
            candidate_k=CANDIDATE_K,
            model=GENERATOR_MODEL,
            prompt_template=PROMPT_TEMPLATE,
        )

        contexts = [
            f"المادة {r['article_number']}\n{r['text_ar_normalized']}\nالمصدر: {r['citation']}"
            for r in rag["retrieved_results"]
        ]
        retrieved = [r["article_number"] for r in rag["retrieved_results"]]

        recall = article_overlap(relevant, retrieved, "relevant")
        precision = article_overlap(relevant, retrieved, "retrieved")

        details.append(
            {
                "question": question,
                "reference": reference,
                "response": rag["answer"],
                "relevant_articles": relevant,
                "retrieved_articles": retrieved,
                "retrieved_contexts": contexts,
                "article_recall_at_5": recall,
                "article_precision_at_5": precision,
                "sources": rag["sources"],
            }
        )
        samples.append(
            {
                "user_input": question,
                "response": rag["answer"],
                "reference": reference,
                "retrieved_contexts": contexts,
            }
        )

        print(f"Expected: {relevant} | Retrieved: {retrieved}")
        print(f"Article Recall@{TOP_K}: {recall:.3f} | Precision@{TOP_K}: {precision:.3f}")

    return samples, details


# ============================================================
# RAGAS evaluator
# ============================================================

def create_evaluator():
    """Create the RAGAS judge LLM + embeddings."""
    provider = "openai"

    if JUDGE_PROVIDER == "gemini":
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise SystemExit(
                "JUDGE_PROVIDER=gemini but GEMINI_API_KEY is not set.\n"
                'Get a free key from Google AI Studio, then: $env:GEMINI_API_KEY="your-key"'
            )
        client = AsyncOpenAI(
            api_key=api_key, base_url=GEMINI_BASE_URL, timeout=SAMPLE_TIMEOUT_SECONDS
        )

    elif JUDGE_PROVIDER == "anthropic":
        from anthropic import AsyncAnthropic

        client = AsyncAnthropic(timeout=SAMPLE_TIMEOUT_SECONDS)
        provider = "anthropic"

    elif JUDGE_PROVIDER == "openai":
        client = AsyncOpenAI(timeout=SAMPLE_TIMEOUT_SECONDS)

    else:  # ollama
        client = AsyncOpenAI(
            api_key="ollama", base_url=OLLAMA_BASE_URL, timeout=SAMPLE_TIMEOUT_SECONDS
        )

    try:
        llm = llm_factory(
            JUDGE_MODEL,
            provider=provider,
            client=client,
            max_tokens=JUDGE_MAX_TOKENS,
            temperature=0,
        )
    except TypeError:
        llm = llm_factory(JUDGE_MODEL, provider=provider, client=client)

    embeddings = HuggingFaceEmbeddings(
        model=EMBEDDING_MODEL, device="cpu", normalize_embeddings=True
    )
    return llm, embeddings


# ============================================================
# Metric evaluation (retries + backoff, never raises)
# ============================================================

async def safe_score(name: str, metric, errors: dict, **kwargs) -> float | None:
    last_error = None

    for attempt in range(1, METRIC_RETRIES + 1):
        try:
            result = await asyncio.wait_for(
                metric.ascore(**kwargs), timeout=SAMPLE_TIMEOUT_SECONDS
            )
            value = float(result.value)
            if not is_valid_number(value):
                raise ValueError(f"invalid metric value: {value}")
            return value

        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            print(f"  [{name}] attempt {attempt}/{METRIC_RETRIES} FAILED -> {last_error}")

            if attempt < METRIC_RETRIES:
                # Rate limits (429) need a real wait, not an instant retry.
                wait = RETRY_BACKOFF_SECONDS * attempt
                print(f"  waiting {wait:.0f}s before retrying...")
                await asyncio.sleep(wait)

    errors[name] = last_error
    return None


async def evaluate_one_sample(sample: dict, metrics: dict) -> dict:
    errors: dict[str, str] = {}
    result = dict(sample)

    for name, fields in METRIC_INPUTS.items():
        result[name] = await safe_score(
            name, metrics[name], errors, **{field: sample[field] for field in fields}
        )
        await asyncio.sleep(PAUSE_BETWEEN_METRICS)

    if errors:
        result["errors"] = errors
    return result


async def evaluate_all_samples(samples: list[dict], metrics: dict) -> list[dict]:
    results = []

    for index, sample in enumerate(samples, start=1):
        print("\n" + "-" * 70)
        print(f"RAGAS evaluation [{index}/{len(samples)}] | {sample['user_input']}")

        result = await evaluate_one_sample(sample, metrics)
        results.append(result)

        for name in METRIC_NAMES:
            value = result[name]
            print(f"{name}: {'None' if value is None else f'{value:.3f}'}")

            # Live per-sample logging: progress survives a later crash.
            if value is not None:
                try:
                    mlflow.log_metric(f"sample_{name}", value, step=index)
                except Exception as exc:
                    print(f"  [mlflow] log_metric failed: {exc}")

        if result["answer_relevancy"] == 0.0:
            preview = sample["response"][:300].replace("\n", " ")
            print(f"  [!] answer_relevancy=0 | response: {preview}")

    return results


def run_ragas_evaluation(samples: list[dict]) -> list[dict]:
    llm, embeddings = create_evaluator()

    metrics = {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),
        "context_precision": ContextPrecision(llm=llm),
        "context_recall": ContextRecall(llm=llm),
    }

    print("\n" + "=" * 70)
    print(f"Running RAGAS evaluation | judge = {JUDGE_MODEL} ({JUDGE_PROVIDER})")
    print("=" * 70)

    return asyncio.run(evaluate_all_samples(samples, metrics))


# ============================================================
# Aggregation
# ============================================================

def aggregate_scores(df: pd.DataFrame, total: int) -> tuple[dict, dict, dict]:
    """Return (averages, valid_counts, failed_counts) using valid values only."""
    scores, valid, failed = {}, {}, {}

    for name in METRIC_NAMES:
        values = (
            pd.to_numeric(df[name], errors="coerce").dropna()
            if name in df
            else pd.Series(dtype=float)
        )
        valid[name] = len(values)
        failed[name] = total - len(values)
        if not values.empty:
            scores[name] = float(values.mean())

    return scores, valid, failed


def merge_ragas_into_details(details: list[dict], ragas_results: list[dict]) -> list[dict]:
    """Attach RAGAS scores to the detailed results; return the list of errors."""
    all_errors = []

    for index, (detail, ragas) in enumerate(zip(details, ragas_results), start=1):
        detail["ragas"] = {name: ragas.get(name) for name in METRIC_NAMES}

        if "errors" in ragas:
            detail["ragas_errors"] = ragas["errors"]
            all_errors.append(
                {"sample": index, "question": detail["question"], "errors": ragas["errors"]}
            )

    return all_errors


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only the first N questions (overrides NUM_QUESTIONS).")
    parser.add_argument("--reuse-samples", action="store_true",
                        help="Skip running the RAG and reuse cached answers/contexts.")
    parser.add_argument("--run-name", default=None,
                        help="MLflow run name (overrides MLFLOW_RUN_NAME).")
    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    setup_mlflow()

    # CLI --limit wins over NUM_QUESTIONS in config.py
    limit = args.limit if args.limit is not None else NUM_QUESTIONS
    records = load_json(DATASET_PATH)
    if limit:
        records = records[:limit]
    num_questions = len(records)

    print(f"Loaded {num_questions} evaluation questions.")
    print(f"Generator: {GENERATOR_MODEL} | Judge: {JUDGE_MODEL} ({JUDGE_PROVIDER})")

    if JUDGE_PROVIDER == "ollama":
        ensure_judge_model()

    run_name = (
        args.run_name
        or MLFLOW_RUN_NAME
        or f"ragas-{MODEL_SLUG}-k{TOP_K}-{num_questions}q-p{PROMPT_HASH}"
    )

    with mlflow.start_run(run_name=run_name):
        dvc_hash = str(get_dvc_data_hash() or "unknown")
        git_commit = str(get_git_commit() or "unknown")

        mlflow.set_tags(
            {
                "dataset": "Egyptian Civil Code",
                "dataset_path": str(DATASET_PATH),
                "dvc_data_hash": dvc_hash,
                "git_commit": git_commit,
                "evaluation": "RAGAS",
                "evaluation_dataset_size": str(num_questions),
                "sweep_id": os.getenv("SWEEP_ID", ""),
                "prompt_hash": PROMPT_HASH,
            }
        )

        # The prompt itself + the config used for this run
        mlflow.log_text(PROMPT_TEMPLATE, "prompt/prompt_template.txt")
        mlflow.log_artifact(legal_config.__file__, artifact_path="config")

        run_config = {
            "top_k": TOP_K,
            "candidate_k": CANDIDATE_K,
            "generator_model": GENERATOR_MODEL,
            "judge_provider": JUDGE_PROVIDER,
            "judge_model": JUDGE_MODEL,
            "embedding_model": EMBEDDING_MODEL,
            "retrieval": "hybrid_search",
            "fusion": "weighted_rrf",
            "prompt_hash": PROMPT_HASH,
            "num_questions": num_questions,
        }
        mlflow.log_params(
            {
                **run_config,
                "prompt_preview": PROMPT_TEMPLATE[:400],
                "judge_base_model": JUDGE_BASE_MODEL,
                "judge_num_ctx": JUDGE_NUM_CTX,
                "judge_max_tokens": JUDGE_MAX_TOKENS,
                "evaluation_dataset": DATASET_PATH.name,
            }
        )

        # ---- RAG outputs (cached or fresh) ----
        if args.reuse_samples and SAMPLES_CACHE_PATH.exists() and DETAILS_CACHE_PATH.exists():
            print(f"Reusing cached RAG outputs from {RUN_DIR}")
            samples = load_json(SAMPLES_CACHE_PATH)[:num_questions]
            details = load_json(DETAILS_CACHE_PATH)[:num_questions]
        else:
            if args.reuse_samples:
                print(f"[!] No cache found in {RUN_DIR}; running the RAG instead.")
            samples, details = build_ragas_dataset(records)
            save_json(SAMPLES_CACHE_PATH, samples)
            save_json(DETAILS_CACHE_PATH, details)

        # ---- RAGAS + aggregation ----
        ragas_results = run_ragas_evaluation(samples)
        df = pd.DataFrame(ragas_results)

        scores, valid_counts, failed_counts = aggregate_scores(df, num_questions)

        retrieval_metrics = {
            key: sum(d[key] for d in details) / len(details)
            for key in ("article_recall_at_5", "article_precision_at_5")
        }
        all_errors = merge_ragas_into_details(details, ragas_results)

        # ---- MLflow metrics ----
        mlflow.log_metrics(scores)
        mlflow.log_metrics(
            {f"{n}_valid_count": valid_counts[n] for n in METRIC_NAMES}
            | {f"{n}_failed_count": failed_counts[n] for n in METRIC_NAMES}
        )
        mlflow.log_metrics(retrieval_metrics)

        for name in METRIC_NAMES:
            if name not in scores:
                print(
                    f"[WARNING] '{name}' has no valid values (all samples failed). "
                    f"See {ERRORS_PATH.name}"
                )

        # ---- Save outputs ----
        summary = {
            **run_config,
            "dvc_data_hash": dvc_hash,
            "git_commit": git_commit,
            "ragas": scores,
            "ragas_valid_counts": valid_counts,
            "ragas_failed_counts": failed_counts,
            "retrieval_metrics": retrieval_metrics,
        }

        save_json(RESULTS_PATH, details)
        save_json(SUMMARY_PATH, summary)
        save_json(ERRORS_PATH, all_errors)

        export_df = df.drop(columns=["retrieved_contexts"], errors="ignore")
        if "errors" in export_df:
            export_df["errors"] = export_df["errors"].apply(
                lambda e: json.dumps(e, ensure_ascii=False) if isinstance(e, dict) else ""
            )
        export_df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")

        # ---- MLflow artifacts + per-sample table ----
        for path in (RESULTS_PATH, SUMMARY_PATH, CSV_PATH, ERRORS_PATH):
            mlflow.log_artifact(str(path), artifact_path="evaluation")

        try:
            mlflow.log_table(
                data=export_df, artifact_file="evaluation/ragas_per_sample_table.json"
            )
        except Exception as exc:
            print(f"[mlflow] log_table skipped: {exc}")

        # ---- Console summary ----
        print("\n" + "=" * 70)
        print("RAGAS + MLFLOW SUMMARY")
        print("=" * 70)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"\nMLflow experiment: {MLFLOW_EXPERIMENT_NAME}")
        print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")


if __name__ == "__main__":
    main()