
#!/bin/bash
set -e

echo "Restoring the prebuilt Legal RAG database..."

pg_restore \
  --username="$POSTGRES_USER" \
  --dbname="$POSTGRES_DB" \
  --no-owner \
  --no-privileges \
  --exit-on-error \
  /docker-entrypoint-initdb.d/legal_rag.dump

echo "Legal RAG database restored successfully."
