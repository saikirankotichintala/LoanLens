#!/usr/bin/env bash
set -euo pipefail

echo "Cassandra container"
docker ps --filter name=cassandra-db --format 'Name: {{.Names}} | Status: {{.Status}}'

echo
echo "Imported loan dataset (loan_analytics.dataset_meta)"
docker exec cassandra-db cqlsh -e \
  "SELECT dataset, total, defaults, ingested_at FROM loan_analytics.dataset_meta WHERE dataset = 'Loan_default 2.csv';"

echo
echo "Example stored loans (loan_analytics.loan_records)"
docker exec cassandra-db cqlsh -e \
  "SELECT loan_id, loan_amount, credit_score, defaulted FROM loan_analytics.loan_records LIMIT 5;"

echo
echo "Loan-purpose summary used by the dashboard (loan_analytics.category_counts)"
docker exec cassandra-db cqlsh -e \
  "SELECT value, total, defaults FROM loan_analytics.category_counts WHERE category = 'LoanPurpose';"

echo
echo "Open http://127.0.0.1:5000 and scroll to Apache Cassandra to see these saved totals and rates as graphs."
