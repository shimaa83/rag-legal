# Arabic Legal RAG — Egyptian Civil Code

An MLOps-oriented Retrieval-Augmented Generation (RAG) system for the Egyptian Civil Code, focused on Arabic legal text processing, reproducible data pipelines, PostgreSQL hybrid retrieval, RAG evaluation, and experiment tracking.

The project combines DVC, PostgreSQL, pgvector, hybrid search, Reciprocal Rank Fusion (RRF), Ollama, Gemini verification, MLflow/DagsHub, Langfuse, BentoML, Prometheus, and Grafana.

## 1. Project Overview

The system retrieves relevant articles from the Egyptian Civil Code and generates Arabic answers grounded in the retrieved legal context.

### Architecture

```text
Egyptian Civil Code PDF
          |
          v
   PDF-to-JSON Conversion
          |
          v
   Arabic Preprocessing
          |
          v
   Processed Article Dataset
          |
          v
    Chunking and Embeddings
          |
          v
      PostgreSQL
   +------------------+
   | pgvector          |
   | tsvector          |
   +------------------+
          |
          v
     Hybrid Retrieval
          |
          v
    Weighted RRF
          |
          v
    Retrieved Context
          |
          v
       Ollama LLM
          |
          v
    Gemini Verification
          |
          v
  Answer with Legal Sources
```

Langfuse provides tracing, MLflow tracks evaluation experiments, and DagsHub hosts the DVC data remote and MLflow tracking. Prometheus collects BentoML service metrics, and Grafana visualizes request activity and performance.

## 2. Project Status

### Implemented

* PDF-to-JSON conversion.
* Arabic text preprocessing.
* DVC data versioning and reproducible preprocessing pipeline.
* DagsHub DVC remote.
* PostgreSQL and pgvector integration.
* Article chunking and multilingual embeddings.
* Vector and keyword retrieval components.
* Hybrid search and RRF experiments.
* Weighted RRF experiments.
* Reranker component and retrieval evaluation scripts.
* RAG answer generation using Ollama.
* Optional answer verification using Gemini.
* Langfuse tracing integration.
* RAGAS evaluation and MLflow/DagsHub experiment tracking.
* BentoML serving endpoint.
* Docker image build and publishing workflow.
* Local Prometheus metrics collection from BentoML.
* Grafana monitoring dashboard for request counts, error counts, and request rate.

### Current serving setup

The BentoML service was successfully tested locally at:

`http://localhost:3002`

The API documentation is available at:

`http://localhost:3002/#/Service%20APIs/LegalRAG__ask`

### Current monitoring setup

The local monitoring stack is running through Docker Compose.

| Service | Local URL |
|---|---|
| BentoML | http://localhost:3002 |
| Prometheus | http://localhost:9093 |
| Prometheus Targets | http://localhost:9093/targets |
| Grafana | http://localhost:3003 |

The `legal-rag-bentoml` Prometheus target was verified as `UP`. The `bentoml_service_request_total` metric was also tested with successful `/ask` requests.

These addresses are local development URLs, not public links accessible to remote reviewers.

### Remaining work

* Larger and more consistent RAGAS evaluation runs.
* Recording the exact DVC dataset version in every evaluation run.
* Performance optimization and repeated load testing.
* Production deployment and production-grade monitoring.
* Further integration and benchmarking of the reranker.
* Verification of the complete Docker deployment on a clean machine.
* Additional monitoring for latency and retrieval quality where supported by available metrics.

The existence of a component does not necessarily mean it is integrated into the live serving path. In particular, the reranker should be verified against the current retrieval and serving code before assuming that every request passes through it.

## 3. Repository Structure

The main project files include:

```text
rag-legal/
├── data/
│   ├── raw/
│   │   ├── egyptian_civil_code.pdf.dvc
│   │   └── civil_code.json
│   ├── processed/
│   │   └── civil_code.json
│   └── evaluation/
│       ├── retrieval_questions.json
│       └── ragas_dataset.json
├── reports/
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
├── sql/
│   └── schema.sql
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
│       ├── retrieval.py
│       └── service.py
├── tests/
├── monitoring/
│   └── prometheus.yml
├── dvc.yaml
├── dvc.lock
├── docker-compose.yml
├── pyproject.toml
├── uv.lock
├── .env.example
└── README.md
```

This is a representative structure; consult the repository for the exact current file list.

## 4. Requirements

For running from source:

* Git
* Python 3.13+
* `uv`
* Docker Desktop
* Ollama
* Access to the DagsHub DVC remote if the source PDF must be downloaded
* The required credentials for any enabled external services

PostgreSQL with pgvector, Prometheus, and Grafana are provided through Docker Compose.

## 5. Quick Start — Run from GitHub

These steps describe the source-code workflow. They assume the reviewer has access to the required data and can configure the model services.

### Step 1 — Clone the repository

```bash
git clone https://github.com/shimaa83/rag-legal.git
cd rag-legal
git checkout legal-v1
```

### Step 2 — Install dependencies

```bash
uv sync --frozen
```

### Step 3 — Configure environment variables

Copy `.env.example` to `.env`.

Set the PostgreSQL connection details, DagsHub credentials, and any required model API keys.

Example PostgreSQL settings:

```dotenv
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=legal_rag
POSTGRES_USER=rag_user
POSTGRES_PASSWORD=replace_with_a_secure_password
```

For DagsHub tracking, configure the repository and token according to `.env.example`:

```dotenv
USE_LOCAL_MLFLOW=0
DAGSHUB_USER=shimaa83
DAGSHUB_REPO=rag-legal
DAGSHUB_USER_TOKEN=replace_with_your_dagshub_token
```

The exact setting for Gemini and Langfuse depends on whether those integrations are enabled.

**Security:** Never commit `.env`, access tokens, or API keys to Git.

### Step 4 — Start PostgreSQL

```bash
docker compose up -d postgres
```

### Step 5 — Retrieve and prepare the dataset

If the data files are not already available locally, configure DVC access to the DagsHub remote and retrieve the tracked source PDF:

```bash
uv run dvc pull
```

Rebuild the conversion and preprocessing pipeline:

```bash
uv run dvc repro
```

The pipeline generates the derived JSON files. It does not populate PostgreSQL automatically.

### Step 6 — Initialize the database schema

Apply the SQL schema:

```bash
psql -h localhost -p 5432 -U rag_user -d legal_rag -f sql/schema.sql
```

Ensure that the database contains the legal chunks and their embeddings before starting the RAG service.

If the database is empty, run the repository's ingestion script:

```bash
uv run python scripts/test_batch_ingestion.py
```

This script processes the chunks and inserts them into PostgreSQL. Despite its name, it ingests the full set of chunks loaded by the script, so do not rerun it unnecessarily when the database is already populated.

### Step 7 — Start Ollama

Download the configured generation model:

```bash
ollama pull qwen2.5:3b
```

Ensure Ollama is running and reachable by the application. If the application runs inside a container, configure its model-service URL appropriately; `localhost` inside a container refers to that container.

### Step 8 — Start BentoML

```bash
uv run bentoml serve legal_rag.service:LegalRAG --port 3002
```

### Step 9 — Test the service

Open:

`http://localhost:3002/#/Service%20APIs/LegalRAG__ask`

Submit the Arabic question:

```text
ما هي شروط العقد؟
```

Review the generated answer and its source citations.

### Step 10 — Start monitoring

In a separate terminal, from the project root, run:

```bash
docker compose up -d prometheus grafana
```

Open Prometheus Targets:

`http://localhost:9093/targets`

Verify that `legal-rag-bentoml` is `UP`.

Open Grafana:

`http://localhost:3003`

Select the configured Prometheus data source and open the saved **Arabic Legal RAG - Monitoring** dashboard.

The BentoML service must be running on port `3002` for Prometheus to collect its metrics.

### Step 11 — Run tests

```bash
uv run pytest
```

An earlier recorded test result was `8 passed`. Run the tests in your current checkout to confirm its actual result.

## 6. Run Using the Published Docker Image

The currently confirmed image reference is on GitHub Container Registry (GHCR):

```bash
docker pull ghcr.io/shimaa83/rag-legal:sha-e5d004f
```

The GitHub Actions workflow also contains a Docker Hub login step. Publishing to Docker Hub requires the image metadata to include the Docker Hub repository name and a successful push run.

The intended Docker Hub repository is:

`shimaa83/rag-legal`

Once the image and tag are confirmed to exist on Docker Hub, it can be pulled using:

```bash
docker pull shimaa83/rag-legal:sha-e5d004f
```

### Before running the container

1. Configure the required environment variables.
2. Ensure PostgreSQL is running and contains the legal chunks and embeddings.
3. Ensure Ollama is running with the configured generation model.
4. Configure the container to reach PostgreSQL and Ollama using addresses accessible from inside the container.
5. Use the image's documented startup command and publish its service port.

The Docker image alone does not guarantee that PostgreSQL, Ollama, the dataset, or the embeddings are included. Verify the image's Dockerfile and startup command before relying on a generic `docker run` command.

After startup, open:

`http://localhost:3002/#/Service%20APIs/LegalRAG__ask`

Then submit the same test question:

`ما هي شروط العقد؟`

## 7. DVC Data Versioning

DVC manages the source PDF and derived datasets, while Git tracks the code, DVC metadata, and pipeline definitions.

The original PDF is not stored directly in Git. Git tracks its DVC pointer, such as:

```text
data/raw/egyptian_civil_code.pdf.dvc
```

The configured DVC remote is:

`https://dagshub.com/shimaa83/rag-legal.dvc`

Credentials should be configured locally and never committed to the repository.

### Reproduce the data pipeline

```bash
uv run dvc pull
uv run dvc repro
```

`dvc pull` retrieves tracked data from the remote.

`dvc repro` executes the stages defined in `dvc.yaml` when required by the dependency state.

The pipeline currently contains two stages:

1. PDF conversion: source PDF to raw JSON.
2. Arabic preprocessing: raw JSON to processed JSON.

The exact pipeline state is recorded in `dvc.lock`.

DVC does not currently run the PostgreSQL ingestion stage, so database population is a separate step.

## 8. Data Processing

The preprocessing pipeline produces structured article-level records for the Egyptian Civil Code.

Example:

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

The project has previously produced approximately 1,149 articles. The exact count should be verified against the current processed dataset.

The pipeline is intended to preserve article metadata and source citations for retrieval and answer attribution.

## 9. PostgreSQL and Hybrid Retrieval

PostgreSQL provides the persistent retrieval store.

### Vector search

`pgvector` stores and searches embeddings generated using:

`intfloat/multilingual-e5-small`

The embedding dimension is 384.

The embedding implementation uses the `passage:` prefix for documents and the `query:` prefix for queries.

### Keyword search

PostgreSQL `tsvector` supports lexical retrieval.

This complements semantic retrieval, particularly when questions contain specific legal terms or wording.

### Reciprocal Rank Fusion

RRF combines the rankings returned by vector and keyword retrieval. The project also includes weighted RRF experiments.

The intended retrieval flow is:

```text
User Question
      |
      +--------------------+
      |                    |
      v                    v
  Vector Search       Keyword Search
      |                    |
      +---------+----------+
                |
                v
           RRF Fusion
                |
                v
       Ranked Candidates
                |
                v
          Best Context
```

The current RAG implementation uses the configured hybrid retrieval path. Consult the serving code to verify which ranking and reranking options are active for a specific run.

## 10. Reranking

The project includes a reranker component and evaluation scripts.

A reranker can rescore retrieved candidates using the query and candidate text, potentially improving the relevance of the final context.

The intended sequence is:

```text
Hybrid Retrieval
       |
       v
    RRF Fusion
       |
       v
    Candidates
       |
       v
    Reranking
       |
       v
  Final Context
```

The existence of the reranker module does not by itself establish that the BentoML serving path invokes it on every request.

Retrieval and ranking experiment outputs are stored under `reports/`.

## 11. RAG Generation Pipeline

The current RAG flow is:

1. **Question:** Receive an Arabic legal question.
2. **Retrieval:** Search the Egyptian Civil Code using PostgreSQL hybrid retrieval.
3. **Ranking:** Combine vector and keyword rankings using the configured RRF strategy.
4. **Context building:** Prepare retrieved legal text and article citations, excluding repealed articles where the implementation applies this filter.
5. **Generation:** Use Ollama, with `qwen2.5:3b` as the configured default, to draft an answer grounded in the retrieved context.
6. **Verification:** When configured, send the question, legal context, and draft answer to Gemini for review.
7. **Sources:** Return the final answer with the retrieved legal article citations.
8. **Tracing:** Record the operation through Langfuse when tracing is enabled.

The system is intended for legal research and experimentation. Generated answers should be checked against the original legislation.

## 12. RAGAS Evaluation

RAGAS is used to evaluate answer quality and the relevance of retrieved context.

The evaluation dataset is located at:

```text
data/evaluation/ragas_dataset.json
```

The evaluation workflow is:

1. Load the evaluation questions.
2. Run the RAG pipeline for each question.
3. Collect the generated answer, retrieved context, and reference information required by the evaluator.
4. Calculate Faithfulness, Answer Relevancy, Context Precision, and Context Recall.
5. Evaluate article retrieval using Article Recall@5 and Article Precision@5.
6. Save per-question results, summaries, and errors under `data/evaluation/runs/`.
7. Log experiment parameters, metrics, and artifacts to MLflow/DagsHub.

Evaluation scripts support configurable experiment settings, including the model, prompt, and retrieval parameters.

### What the metrics mean

| Metric | Purpose |
|---|---|
| Faithfulness | Measures whether the answer is supported by the retrieved context. |
| Answer Relevancy | Measures how relevant the answer is to the question. |
| Context Precision | Measures how relevant the retrieved context is. |
| Context Recall | Measures how much of the required reference information is captured. |
| Article Recall@5 | Measures whether the relevant article appears among the top five results. |
| Article Precision@5 | Measures the proportion of the top five results considered relevant. |

Metric values are meaningful only in relation to the evaluation dataset, references, and experiment configuration used to calculate them.

## 13. RAG Experiment Comparison

The following results are the experiment values recorded during development. The runs used different sample sizes and should not be interpreted as a controlled, same-dataset comparison.

| Experiment | Questions | Faithfulness | Answer Relevancy | Context Precision | Context Recall | Article Recall@5 | Article Precision@5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Successful baseline | 20 | 0.3067 | 0.0456 | 0.8857 | 1.0000 | 1.0000 | 0.2000 |
| Prompt 3 | 5 | 0.4667 | 0.5736 | ≈1.0000 | 1.0000 | 1.0000 | 0.2000 |
| Prompt 4 | 5 | 0.5500 | 0.8941 | ≈1.0000 | 1.0000 | 1.0000 | 0.2000 |
| Updated prompt | 5 | 0.8500 | 0.9356 | 1.0000 | 1.0000 | 1.0000 | 0.2000 |

A later five-question run recorded the following per-sample metrics:

| Metric | Recorded value |
|---|---:|
| Sample Faithfulness | 1.0000 |
| Sample Answer Relevancy | 0.8368 |
| Sample Context Precision | ≈1.0000 |
| Sample Context Recall | 1.0000 |

All four RAGAS metrics were valid for all five questions in that run, with zero failed evaluations.

### Interpretation

The updated prompt showed improved Faithfulness and Answer Relevancy in the recorded five-question experiment. Because the sample is small and the experiments may use different configurations, a larger evaluation on the same fixed dataset is required before concluding that the improvement generalizes.

Article Precision@5 remained 0.20 in the comparison table. This indicates that retrieval relevance and ranking should continue to be investigated even when Article Recall@5 is high.

## 14. MLflow and DagsHub Experiment Tracking

MLflow tracks experiment configurations, metrics, and artifacts. DagsHub provides the remote MLflow tracking backend when configured as the default.

The experiment name currently configured in the project is:

```text
legal-rag_a5
```

Tracking responsibilities:

| Tool | Responsibility |
|---|---|
| Git | Source code and version history |
| DVC | Source PDF, derived datasets, and reproducible data pipeline |
| MLflow | Experiment parameters, metrics, and artifacts |
| DagsHub | Remote DVC storage and MLflow experiment tracking |
| RAGAS | RAG quality evaluation |
| Langfuse | Request-level tracing and prompt/LLM observability |
| Prometheus | Service metrics collection |
| Grafana | Monitoring dashboards and visualization |

### Configuration

In `.env`, configure the DagsHub repository and token. For remote tracking, use the configuration expected by `src/legal_rag/config.py` and the evaluation script.

The tracking URI follows this pattern:

```text
https://dagshub.com/shimaa83/rag-legal.mlflow
```

If local MLflow is explicitly enabled, the project can instead use its configured local tracking URI. Remote DagsHub tracking is the intended default for shared experiments.

### Reproducible experiment metadata

Each experiment should record the configuration needed to reproduce it, including:

* Git revision.
* DVC revision or dataset version.
* Dataset path and evaluation sample size.
* Generator and embedding model.
* Prompt version.
* Chunking parameters where applicable.
* Retrieval and RRF configuration.
* Reranker configuration where used.
* RAGAS and article retrieval metrics.

The source PDF and datasets remain managed by DVC rather than being duplicated inside MLflow.

## 15. Langfuse Tracing

Langfuse is used for observability of the RAG execution path.

The project can trace operations such as:

* Question processing.
* Hybrid retrieval.
* LLM generation.
* Gemini verification when enabled.
* Final answer and source production.

Configure the required Langfuse credentials in `.env` if remote tracing is enabled. For local deployment, use the configured Langfuse host.

Tracing helps investigate latency and errors across retrieval, generation, and verification.

## 16. BentoML Service

The RAG service is exposed through BentoML.

### Start locally

```bash
uv run bentoml serve legal_rag.service:LegalRAG --port 3002
```

The service exposes an `ask` API accepting a question and an optional `top_k` parameter.

API documentation:

`http://localhost:3002/#/Service%20APIs/LegalRAG__ask`

Example question:

```text
ما هي شروط العقد؟
```

The response includes the question, generated answer, and sources.

This is a locally hosted development service, not a public deployment URL.

## 17. Monitoring and Observability

The project uses **Prometheus** and **Grafana** to monitor the locally hosted BentoML service.

### Monitoring Architecture

```text
BentoML RAG Service
   localhost:3002
         |
         | /metrics
         v
    Prometheus
   localhost:9093
         |
         v
      Grafana
   localhost:3003
```

* **BentoML:** Serves the RAG API and exposes service metrics.
* **Prometheus:** Scrapes metrics from BentoML at 15-second intervals.
* **Grafana:** Visualizes request counts, error counts, request rate, and latency when the corresponding metrics are available.

### Local Monitoring URLs

| Service | URL |
|---|---|
| BentoML API | http://localhost:3002 |
| BentoML metrics | http://localhost:3002/metrics |
| Prometheus | http://localhost:9093 |
| Prometheus Targets | http://localhost:9093/targets |
| Grafana | http://localhost:3003 |

These URLs are for local development. They are not public links that remote reviewers can access directly.

### Prometheus Configuration

The scrape configuration is stored in:

```text
monitoring/prometheus.yml
```

The BentoML target is configured as:

```yaml
- job_name: "legal-rag-bentoml"
  metrics_path: /metrics
  static_configs:
    - targets: ["host.docker.internal:3002"]
```

Prometheus uses `host.docker.internal` to reach the BentoML service running on the host machine from inside its Docker container.

### Grafana Dashboard

The saved dashboard is named:

`Arabic Legal RAG - Monitoring`

The dashboard includes the following panels:

| Panel | Purpose |
|---|---|
| Successful Requests | Displays successful `/ask` requests with HTTP 200 responses. |
| Failed Requests | Displays requests that returned HTTP 5xx responses. |
| Request Rate | Displays the request rate over a five-minute window. |
| Average Request Latency | Displays average request duration when the required duration metrics are exposed. |

The request counter used in the dashboard is:

```promql
bentoml_service_request_total
```

Example query for successful requests:

```promql
sum(bentoml_service_request_total{endpoint="/ask",http_response_code="200"})
```

Example query for server errors:

```promql
sum(bentoml_service_request_total{endpoint="/ask",http_response_code=~"5.."})
```

Example query for request rate:

```promql
sum(rate(bentoml_service_request_total{endpoint="/ask"}[5m]))
```

### Verification

The monitoring setup was tested locally by sending requests to `/ask` and checking the metrics in Grafana.

The following results were observed during testing:

* The `legal-rag-bentoml` Prometheus target was `UP`.
* `bentoml_service_request_total` reported one successful HTTP 200 request after a test request.
* The HTTP 500 counter remained at zero at that point.

These values represent the test session, not permanent or production-wide statistics. Request counters may reset when the service restarts.

### Start Monitoring

From the project root, run:

```bash
docker compose up -d prometheus grafana
```

Ensure that BentoML is running on port `3002`. Open Prometheus Targets and verify that `legal-rag-bentoml` is `UP`.

Then open Grafana, select the configured Prometheus data source, and open the saved dashboard.

For a reproducible review, send several test questions through the BentoML `/ask` API, refresh the dashboard, and inspect the request counters and rate.

### Dashboard Screenshot

To include a dashboard screenshot in the GitHub README, save the image at:

```text
docs/images/legal-rag-dashboard.png
```

Then the following Markdown can be used:

```markdown
![Arabic Legal RAG Monitoring Dashboard](docs/images/legal-rag-dashboard.png)
```

Add the image only after saving the actual screenshot at that path. The local Grafana URL itself is not accessible to remote reviewers.

### Current Limitations

* The current setup is local monitoring, not a publicly deployed monitoring service.
* Latency visualization depends on the duration metrics exposed by the running BentoML version.
* Request and service metrics do not replace RAGAS quality evaluation or Langfuse request tracing.
* Production alerting, deployment-wide monitoring, and retrieval-quality drift monitoring remain future work.

### Canary Rollout (Planned)

Canary rollout is a planned deployment strategy for gradually introducing a new version of the Arabic Legal RAG system while keeping the current stable version available.

The goal is to validate a new model, prompt, or retrieval configuration with a small portion of incoming traffic before making it the default version.

**Planned rollout strategy:**

1. **Stable version (v1):** Continue serving requests using the current stable RAG version.
2. **Canary version (v2):** Deploy a new version with an updated prompt, generator model, or retrieval configuration.
3. **Traffic splitting:** Initially route a small percentage of requests (for example, 10%) to v2 while keeping the remaining traffic on v1.
4. **Monitoring:** Use Prometheus and Grafana to monitor request counts, HTTP errors, and response latency where suitable metrics are available.
5. **Quality evaluation:** Use RAGAS to compare answer faithfulness, answer relevancy, and retrieval quality between versions.
6. **Promotion or rollback:** Gradually increase v2 traffic if the results are acceptable, or return traffic to v1 if the new version causes regressions.

**Implementation status:** Planned — traffic splitting, automated rollout decisions, and rollback mechanisms have not yet been implemented.

This section documents the intended deployment strategy. The current monitoring setup provides a foundation for observing the service, but monitoring alone does not implement a canary rollout.


### LLM Token Usage Monitoring (Planned)

Token usage monitoring is a planned enhancement for the Arabic Legal RAG system.

The project currently uses free-tier services and a locally hosted Ollama model. Therefore, monetary cost tracking is not a current requirement.

**Planned monitoring metrics:**
- Input tokens consumed by the LLM.
- Output tokens generated by the LLM.
- Total token usage over time.
- Token usage per request, where supported by the model or provider.

Prometheus and Grafana may be extended to visualize token consumption and usage trends. Token metrics will be collected only when reliable usage information is available from the model or provider.

**Cost tracking:** Not currently implemented, as monetary cost calculation is outside the project's current scope.

**Implementation status:** Planned — token instrumentation and Grafana panels have not yet been implemented.

## 18. Locust Load Testing

An initial Locust test was performed against the locally hosted BentoML RAG API.

### Test configuration

* Endpoint: `POST /ask`
* Target URL: `http://localhost:3002`
* Request payload: a selected legal question with `top_k=5`
* Timeout: 300 seconds
* Question set: six predefined Egyptian Civil Code questions

### Initial results

| Metric | Result |
|---|---:|
| Total Requests | 7 |
| Failed Requests | 0 |
| Median Response Time | 108,000 ms |
| Average Response Time | 99,630.56 ms |
| 95th Percentile (P95) | 151,000 ms |
| 99th Percentile (P99) | 151,000 ms |
| Minimum Response Time | 28,048 ms |
| Maximum Response Time | 150,746 ms |
| Average Response Size | 1,985.71 bytes |

### Observations

* All seven requests completed successfully, with no failures reported by Locust.
* Average latency was approximately 99.63 seconds, which is too high for a responsive interactive service.
* The sample is small and is only an initial local benchmark.
* Further profiling is needed to identify latency contributions from PostgreSQL retrieval, Ollama generation, and Gemini verification.

These results should not be interpreted as production capacity or a final performance benchmark.

## 19. Tests

The test suite is located under:

```text
tests/
```

Run it with:

```bash
uv run pytest
```

The suite covers components such as chunking, embeddings, ingestion, and validation.

To inspect test collection:

```bash
uv run pytest --collect-only
```

Manual and integration scripts under `scripts/` are not automatically collected as part of the normal pytest suite.

For CI, the workflow installs dependencies, prepares the test database, initializes the schema, runs code checks and compilation, and executes tests if the `tests/` directory exists.

The Docker job builds and publishes the image after the test job succeeds, except that pull-request builds do not push the image.

## 20. CI/CD and Docker Image Publishing

The GitHub Actions workflow is located under:

```text
.github/workflows/
```

It is triggered by pushes to `main` and `legal-v1`, pull requests, and manual dispatch.

### CI

The CI job includes:

* Dependency installation using `uv`.
* Dataset retrieval and preprocessing when the DagsHub token is available.
* PostgreSQL schema initialization.
* Ruff lint and format checks.
* Python compilation checks.
* Pytest execution.

The Ruff checks are configured with `continue-on-error: true`, so Ruff findings do not necessarily fail the job.

### CD

The Docker job builds the image and publishes it when the event is not a pull request.

The confirmed GHCR image reference is:

```text
ghcr.io/shimaa83/rag-legal:sha-e5d004f
```

The intended Docker Hub repository is:

```text
shimaa83/rag-legal
```

Docker Hub publication requires the workflow's image metadata to include that repository name and the push job to succeed. Verify the tag in the registry before instructing reviewers to pull it.

A successful image build does not, by itself, prove that the complete application can run on a clean machine with PostgreSQL, the dataset, and Ollama.

## 21. Planned Production Architecture

The intended production architecture is:

```text
User
 |
 v
RAG API / BentoML
 |
 v
PostgreSQL Hybrid Retrieval
 |
 v
RRF / Optional Reranker
 |
 v
Retrieved Legal Context
 |
 v
Generative Model
 |
 v
Answer + Article Citations
```

Future serving and deployment work includes:

* Streaming responses.
* Production-grade health checks.
* vLLM inference serving.
* Generative model optimization.
* Docker Compose deployment of the required services.
* Production alerting and monitoring.
* Retrieval and embedding drift monitoring.
* Ongoing RAGAS quality evaluation.
* Token-cost monitoring.

### vLLM and local development

The current development environment is Windows and CPU-only. vLLM is planned for the production serving stage and is not the current local inference engine.

The local RAG workflow uses Ollama for generation. A production vLLM deployment should use a compatible Linux environment and suitable GPU resources for the selected model.

## 22. Reproducibility

The project's reproducibility approach separates responsibilities:

```text
Git       -> Code versioning
DVC       -> Dataset versioning and preprocessing pipeline
MLflow    -> Experiment parameters, metrics, and artifacts
RAGAS     -> RAG quality evaluation
Langfuse  -> Request tracing
Prometheus-> Service metrics collection
Grafana   -> Monitoring dashboards
Docker    -> Container packaging
BentoML   -> RAG service serving
Ollama    -> Current local generation
vLLM      -> Planned production inference
```

To reproduce an experiment, record the code revision, DVC dataset version, model configuration, prompt, retrieval parameters, and evaluation results.

## 23. Roadmap

### Phase 1 — Data and reproducibility

* [x] PDF ingestion and conversion.
* [x] Arabic preprocessing.
* [x] DVC configuration and DagsHub remote.
* [x] Reproducible conversion and preprocessing pipeline.
* [x] Data validation tests.

### Phase 2 — Retrieval

* [x] Article chunking and embeddings.
* [x] PostgreSQL integration.
* [x] Vector and keyword retrieval components.
* [x] Hybrid search and RRF experiments.
* [x] Reranker component.
* [x] Retrieval evaluation scripts.

### Phase 3 — Evaluation and experiment tracking

* [x] RAGAS evaluation workflow.
* [x] MLflow/DagsHub experiment tracking.
* [ ] Consistent larger-scale comparison across configurations.
* [ ] Record exact DVC dataset version in every experiment.
* [ ] Select and register the best validated configuration.

### Phase 4 — Serving and optimization

* [x] Local BentoML service.
* [x] Local Ollama generation.
* [ ] Production streaming and health checks.
* [ ] vLLM deployment.
* [ ] Latency optimization and reranker benchmarking.

### Phase 5 — Monitoring and deployment

* [x] Langfuse tracing integration.
* [x] Docker image build workflow.
* [x] Local Prometheus metrics collection.
* [x] Grafana dashboard for BentoML request monitoring.
* [ ] Verify clean-machine Docker deployment.
* [ ] Complete service and database orchestration.
* [ ] Production alerting and monitoring.
* [ ] Retrieval drift and ongoing RAGAS monitoring.
* [ ] Token-cost monitoring.

## 24. License and Disclaimer

This project is developed for educational and MLOps/RAG engineering purposes.

Generated responses are intended to support legal research and must be verified against the applicable legislation and authoritative legal sources. The system is not a substitute for advice from a qualified lawyer.
