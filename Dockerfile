FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV PYTHONUNBUFFERED=1

# Install dependencies first to improve build caching

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev --no-install-project

# Copy project source and required runtime files

COPY . .

RUN uv sync --verbose --frozen --no-dev --no-install-project

EXPOSE 3002

CMD ["uv", "run", "--no-sync", "bentoml", "serve", "legal_rag.bento_service:LegalRAG", "--host", "0.0.0.0", "--port", "3002"]
