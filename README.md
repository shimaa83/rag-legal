# Arabic Legal RAG — Egyptian Civil Code

An MLOps-oriented Retrieval-Augmented Generation (RAG) system for the **Egyptian Civil Code**, with a focus on Arabic legal text processing, reproducible data pipelines, hybrid retrieval, and experiment tracking.

The project is being developed incrementally following an MLOps/RAG project checklist.

---

## 1. Project Overview

The system uses the Egyptian Civil Code as its primary knowledge source.

The current data pipeline is:

```text
Egyptian Civil Code PDF
        │
        ▼
PDF → JSON Conversion
        │
        ▼
Raw JSON
        │
        ▼
Arabic Text Preprocessing
        │
        ▼
Processed JSON
        │
        ▼
Validation
        │
        ▼
RAG / Retrieval Pipeline
```

The project is designed to support a future hybrid retrieval architecture using:

* PostgreSQL
* `pgvector` for vector search
* PostgreSQL `tsvector` for keyword search
* Reciprocal Rank Fusion (RRF)
* Reranking
* Generative LLM

---

## 2. Current Project Status

### Implemented

* Egyptian Civil Code PDF ingestion
* PDF → structured JSON conversion
* Arabic text preprocessing
* DVC data versioning
* DVC reproducible pipeline
* DagsHub as DVC remote storage
* PostgreSQL integration
* Vector embeddings
* Chunking
* Hybrid retrieval components
* RRF experiments
* Weighted RRF experiments
* Reranker component
* Retrieval evaluation scripts
* Automated unit tests

### Current test status

```text
8 passed
```

Tests currently cover:

* Chunking
* Embeddings
* Ingestion
* Validation

### Planned / In Progress

* MLflow experiment tracking
* RAGAS evaluation
* Linking every MLflow experiment to the exact DVC dataset version
* FastAPI `/ask`
* FastAPI `/health`
* Streaming responses
* BentoML serving
* vLLM inference serving
* Docker Compose production setup
* Monitoring and tracing
* Final RAG generation pipeline

---

## 3. Repository Structure

```text
legal_RAG/
│
├── data/
│   ├── raw/
│   │   ├── egyptian_civil_code.pdf.dvc
│   │   └── civil_code.json
│   │
│   ├── processed/
│   │   └── civil_code.json
│   │
│   └── evaluation/
│       └── retrieval_questions.json
│
├── reports/
│   ├── retrieval_evaluation.json
│   ├── rrf_experiment_summary.json
│   ├── weighted_rrf_experiment_summary.json
│   └── ...
│
├── scripts/
│   ├── evaluate_retrieval.py
│   ├── experiment_rrf.py
│   ├── experiment_weighted_rrf.py
│   ├── test_batch_ingestion.py
│   ├── test_chunking.py
│   ├── test_database.py
│   ├── test_hybrid_search.py
│   ├── test_reranker.py
│   └── test_vector_search.py
│
├── src/
│   └── legal_rag/
│       ├── chunking.py
│       ├── config.py
│       ├── database.py
│       ├── data_preprocessing.py
│       ├── embeddings.py
│       ├── ingestion.py
│       ├── pdf_to_json.py
│       ├── rag.py
│       ├── reranker.py
│       └── retrieval.py
│
├── tests/
│   ├── test_chunking.py
│   ├── test_embeddings.py
│   ├── test_ingestion.py
│   └── test_validation.py
│
├── dvc.yaml
├── dvc.lock
├── docker-compose.yml
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## 4. Requirements

The project currently requires:

* Python 3.13+
* `uv`
* Git
* Docker Desktop
* DVC
* Access to the DagsHub DVC remote

PostgreSQL is used for the retrieval layer and is provided through Docker Compose.

---

## 5. Installation

Clone the repository:

```bash
git clone https://github.com/shimaa83/rag-legal.git
cd rag-legal
```

Checkout the development branch:

```bash
git checkout legal-v1
```

Install the project dependencies:

```bash
uv sync
```

---

## 6. DVC Data Versioning

DVC is used as the source of truth for the project's datasets and reproducible data pipeline.

The original Egyptian Civil Code PDF is **not stored directly in Git**.

Instead, Git tracks:

```text
data/raw/egyptian_civil_code.pdf.dvc
```

while the actual PDF is stored in the configured DVC remote on DagsHub.

The DVC remote is:

```text
https://dagshub.com/shimaa83/rag-legal.dvc
```

Credentials are stored locally and are not committed to Git.

---

## 7. DVC Pipeline

The current pipeline contains two stages.

### Stage 1 — PDF Conversion

```text
data/raw/egyptian_civil_code.pdf
              │
              ▼
src/legal_rag/pdf_to_json.py
              │
              ▼
data/raw/civil_code.json
```

Command:

```bash
uv run python -m legal_rag.pdf_to_json \
    data/raw/egyptian_civil_code.pdf \
    data/raw/civil_code.json
```

### Stage 2 — Preprocessing

```text
data/raw/civil_code.json
              │
              ▼
src/legal_rag/data_preprocessing.py
              │
              ▼
data/processed/civil_code.json
```

Command:

```bash
uv run python -m legal_rag.data_preprocessing
```

### Complete pipeline

The complete pipeline is defined in:

```text
dvc.yaml
```

and its exact reproducible state is recorded in:

```text
dvc.lock
```

Run the complete pipeline with:

```bash
dvc repro
```

---

## 8. Reproducing the Dataset

A reviewer can reproduce the current data pipeline using:

```bash
git clone https://github.com/shimaa83/rag-legal.git
cd rag-legal
git checkout legal-v1
uv sync
dvc pull
dvc repro
```

`dvc pull` retrieves the versioned PDF from the DVC remote.

`dvc repro` reconstructs the derived JSON artifacts according to `dvc.yaml` and `dvc.lock`.

---

## 9. Data Processing

The preprocessing stage currently processes the Egyptian Civil Code into a structured article-level JSON dataset.

Each article is represented as a structured record containing information such as:

```json
{
  "article_number": 147,
  "book": "Obligations or Personal Rights",
  "chapter": "Sources of Obligations",
  "section": "Contracts",
  "topic": "The Effects of a Contract",
  "ar_text": "...",
  "text_en": "...",
  "is_repealed": false,
  "source_page": 42,
  "citation": "Egyptian Civil Code, Article 147"
}
```

The current preprocessing pipeline produces approximately:

```text
1149 articles
```

---

## 10. Testing

Pytest is configured to collect tests only from:

```text
tests/
```

Manual/integration scripts under:

```text
scripts/
```

are not automatically collected by pytest.

Run the test suite:

```bash
uv run pytest
```

Current result:

```text
8 passed
```

To inspect test collection without executing the tests:

```bash
uv run pytest --collect-only
```

---

## 11. PostgreSQL and Hybrid Retrieval

The retrieval architecture is being developed around PostgreSQL.

The planned hybrid search combines:

### Vector Search

Using:

```text
pgvector
```

to retrieve semantically similar legal articles/chunks.

### Keyword Search

Using PostgreSQL:

```text
tsvector
```

to retrieve documents based on lexical matching.

### Reciprocal Rank Fusion

The vector and keyword rankings are combined using:

```text
RRF
```

The project also contains experiments for weighted combinations of vector and keyword retrieval.

---

## 12. Reranking

After hybrid retrieval, a reranker is used to improve the ordering of retrieved candidates.

The intended retrieval flow is:

```text
User Query
    │
    ├──────────────► Vector Search
    │
    └──────────────► Keyword Search
                         │
                         ▼
                        RRF
                         │
                         ▼
                    Candidates
                         │
                         ▼
                      Reranker
                         │
                         ▼
                  Best Context
```

The retrieval and reranking experiments are stored under:

```text
reports/
```

---

## 13. MLflow and DVC Integration

MLflow will be used for experiment tracking.

DVC and MLflow have different responsibilities:

```text
DVC
│
├── Dataset versioning
├── Source PDF
├── Derived datasets
└── Reproducible data pipeline


MLflow
│
├── Experiment parameters
├── Metrics
├── Model / embedding configuration
├── RAG evaluation
└── Experiment artifacts
```

A key project requirement is that **every MLflow experiment must be traceable to the exact DVC data version used by that experiment**.

Future MLflow runs will therefore record information such as:

```text
dvc_revision
dataset
dataset_path
chunk_size
overlap
embedding_model
reranker
```

along with evaluation metrics such as:

```text
faithfulness
answer_relevancy
context_recall
precision
```

The PDF and datasets will remain managed by DVC rather than being duplicated inside MLflow.

---

## 14. Future RAG Serving Architecture

The planned production architecture is:

```text
                    User
                      │
                      ▼
                FastAPI /ask
                      │
                      ▼
                  RAG Service
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
     PostgreSQL                 Reranker
   Hybrid Retrieval                │
          │                        │
       Vector +                    │
       Keyword                     │
          │                        │
          └──────────► RRF ◄───────┘
                       │
                       ▼
                    Context
                       │
                       ▼
                     vLLM
                       │
                       ▼
                 Generative LLM
                       │
                       ▼
                  Final Answer
```

BentoML will later be used to package and serve the RAG application.

vLLM will serve the generative language model.

The `/ask` endpoint will eventually support streaming responses.

---

## 15. Planned API

The planned FastAPI interface is:

### `POST /ask`

Request:

```json
{
  "question": "ما هي شروط العقد؟"
}
```

Response:

```json
{
  "answer": "...",
  "sources": [
    "Egyptian Civil Code, Article ..."
  ]
}
```

Sources will reference legal articles rather than internal chunk IDs.

### `GET /health`

Planned response:

```json
{
  "status": "healthy",
  "documents_indexed": 1149
}
```

---

## 16. Reproducibility Philosophy

The project follows the principle:

```text
Data Versioning  → DVC
Experiment Tracking → MLflow
Code Versioning → Git
Containerization → Docker
Model Serving → vLLM / BentoML
Evaluation → RAGAS
```

This separation allows an experiment to be reproduced by identifying:

1. The Git code version.
2. The exact DVC dataset version.
3. The preprocessing pipeline version.
4. The chunking configuration.
5. The embedding model.
6. The retrieval configuration.
7. The reranker configuration.
8. The evaluation results.

---

## 17. Current Development Branch

Current development work is being performed on:

```text
legal-v1
```

The `main` branch is reserved for reviewed and approved changes.

The current branch should be reviewed before merging into `main`.

---

## 18. Roadmap

### Phase 1 — Data & Reproducibility

* [x] PDF ingestion
* [x] PDF → JSON conversion
* [x] Arabic preprocessing
* [x] DVC configuration
* [x] DagsHub DVC remote
* [x] Reproducible DVC pipeline
* [x] Data validation tests

### Phase 2 — Retrieval

* [x] Article chunking
* [x] Embeddings
* [x] PostgreSQL
* [x] Vector search
* [x] Keyword search
* [x] Hybrid retrieval
* [x] RRF experiments
* [x] Reranker
* [x] Retrieval evaluation

### Phase 3 — Experiment Tracking

* [ ] MLflow experiments
* [ ] DVC version recorded with every MLflow run
* [ ] RAGAS evaluation
* [ ] Compare multiple configurations
* [ ] Register best configuration

### Phase 4 — RAG API

* [ ] Complete RAG generation pipeline
* [ ] FastAPI `/ask`
* [ ] FastAPI `/health`
* [ ] Streaming responses
* [ ] Article-level source citations

### Phase 5 — Serving

* [ ] BentoML
* [ ] vLLM
* [ ] Generative model optimization
* [ ] Docker Compose production setup

### Phase 6 — Monitoring

* [ ] Request metrics
* [ ] Latency/error monitoring
* [ ] RAGAS monitoring
* [ ] Retrieval/cosine drift monitoring
* [ ] Langfuse tracing
* [ ] Token-cost monitoring

---

## 19. Quick Start

For a reviewer, the current reproducible workflow is:

```bash
git clone https://github.com/shimaa83/rag-legal.git
cd rag-legal
git checkout legal-v1
uv sync
dvc pull
dvc repro
uv run pytest
```

Expected test result:

```text
8 passed
```

---

## 20. License

This project is developed for educational and MLOps/RAG engineering purposes.
