#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

if [[ ! -f "Loan_default 2.csv" ]]; then
  echo "Missing Loan_default 2.csv in $project_root" >&2
  exit 1
fi

docker info >/dev/null

if ! docker network inspect smartflow-network >/dev/null 2>&1; then
  docker network create smartflow-network >/dev/null
fi

if ! docker container inspect cassandra-db >/dev/null 2>&1; then
  docker run -d --name cassandra-db --network smartflow-network \
    -p 9042:9042 -v loanlens-cassandra-data:/var/lib/cassandra cassandra:4.1 >/dev/null
elif [[ "$(docker inspect -f '{{.State.Running}}' cassandra-db)" != "true" ]]; then
  docker start cassandra-db >/dev/null
fi

network_members="$(docker network inspect smartflow-network -f '{{range .Containers}}{{.Name}} {{end}}')"
if [[ " $network_members " != *" cassandra-db "* ]]; then
  docker network connect smartflow-network cassandra-db
fi

echo "Waiting for Cassandra..."
ready=false
for ((attempt=1; attempt<=45; attempt++)); do
  if docker exec cassandra-db cqlsh -e 'DESCRIBE KEYSPACES' >/dev/null 2>&1; then
    ready=true
    break
  fi
  sleep 2
done
if [[ "$ready" != "true" ]]; then
  echo "Cassandra did not become ready." >&2
  exit 1
fi

if docker image inspect smartflow-flask:local >/dev/null 2>&1; then
  app_image="smartflow-flask:local"
else
  app_image="loanlens:local"
  if ! docker image inspect "$app_image" >/dev/null 2>&1; then
    docker build -f Dockerfile.loanlens -t "$app_image" .
  fi
fi

if [[ ! -d outputs/spark_preprocessor || ! -d outputs/spark_classifier || ! -f outputs/metrics.json ]]; then
  echo "Training PySpark models on 200,000 loan records..."
  docker run --rm --network smartflow-network -v "$project_root:/app" -w /app \
    "$app_image" python -m src.train --data 'Loan_default 2.csv' --rows 200000
fi

meta_rows="$(docker exec cassandra-db cqlsh -e \
  "SELECT dataset, total FROM loan_analytics.dataset_meta WHERE dataset = 'Loan_default 2.csv';" \
  2>/dev/null || true)"
if [[ "$meta_rows" != *"Loan_default 2.csv"* ]]; then
  echo "Importing loan records into Cassandra..."
  docker run --rm --network smartflow-network -v "$project_root:/app" -w /app \
    -e CASSANDRA_HOST=cassandra-db "$app_image" \
    python -m src.cassandra_store --data 'Loan_default 2.csv'
fi

if docker container inspect loanlens-app >/dev/null 2>&1; then
  if [[ "$(docker inspect -f '{{.State.Running}}' loanlens-app)" == "true" ]]; then
    docker restart loanlens-app >/dev/null
  else
    docker start loanlens-app >/dev/null
  fi
else
  docker run -d --name loanlens-app --network smartflow-network \
    -p 127.0.0.1:5000:5000 -v "$project_root:/app" -w /app \
    -e CASSANDRA_HOST=cassandra-db -e CASSANDRA_PORT=9042 \
    "$app_image" python -m flask --app app run --host 0.0.0.0 --port 5000 >/dev/null
fi

for ((attempt=1; attempt<=15; attempt++)); do
  if curl --fail --silent http://127.0.0.1:5000/health >/dev/null; then
    echo "LoanLens is ready at http://127.0.0.1:5000"
    exit 0
  fi
  sleep 1
done
echo "LoanLens container started, but its health endpoint is not responding." >&2
exit 1
