# LoanLens — Loan Default Prediction with PySpark

LoanLens is a Flask dashboard that estimates loan default risk using PySpark ML. Enter applicant and loan details to see an estimated default probability and a prediction based on a threshold selected during validation. Apache Cassandra stores loan records and supplies the dashboard's dataset summaries.

This is an educational project, not a lending decision system. A prediction is not a guarantee of repayment or default.

## Features

- Applicant form with input validation and live predictions from a saved Spark model.
- Comparison of logistic regression, decision tree, and random forest models.
- Test metrics and a confusion-matrix chart.
- Cassandra-backed outcome distributions and default rates by loan purpose, employment, and education.
- A read-only terminal demonstration of stored Cassandra records.

## Requirements and dataset

Install and start Docker Desktop. The application container includes Python 3.11, Java 17, and PySpark 3.5.3; a separate container runs Cassandra 4.1.

Download the CSV from the [Loan Default Prediction Dataset by nikhil1e9 on Kaggle](https://www.kaggle.com/datasets/nikhil1e9/loan-default), then place it in the project root as:

```text
Loan_default 2.csv
```

Dataset files are excluded from Git. The supplied local dataset contains **255,347 rows** with 16 predictor fields, `LoanID`, and `Default`. It has not been independently verified as an exact match to the Kaggle download, and its real or synthetic origin has not been confirmed.

`Default = 1` means recorded default; `Default = 0` means no default recorded. The latter does not independently confirm full repayment. `LoanID` is excluded from model inputs.

The predictors are age, income, loan amount, credit score, months employed, number of credit lines, interest rate, loan term, debt-to-income ratio, education, employment type, marital status, mortgage, dependents, loan purpose, and co-signer. Prediction assumes proposed loan terms, including interest rate, are already known.

## Run the dashboard

From the project root:

```bash
./scripts/start-project.sh
```

Open <http://127.0.0.1:5000>. The first run can take time because it trains models and imports the dataset.

The startup script:

1. Creates the `smartflow-network` Docker network if needed.
2. Starts or reuses `cassandra-db` and waits for Cassandra to become ready.
3. Reuses `smartflow-flask:local` when available; otherwise builds or reuses `loanlens:local` from `Dockerfile.loanlens`.
4. Trains models on a 200,000-row sample if the saved artifacts are missing.
5. Imports the CSV into Cassandra if its dataset metadata is absent.
6. Starts or restarts `loanlens-app` and checks its health endpoint.

Cassandra data persists in a Docker volume. Existing containers and data are reused on subsequent runs.

Check the application and inspect stored records:

```bash
curl --fail http://127.0.0.1:5000/health
./scripts/show-cassandra.sh
```

The Cassandra demonstration shows the imported total, five loan records, and the loan-purpose counts used by the dashboard. It only reads the database.

## Retrain the models

After the initial setup, build the project image and retrain explicitly:

```bash
docker build -f Dockerfile.loanlens -t loanlens:local .
docker run --rm --network smartflow-network -v "$PWD:/app" -w /app \
  loanlens:local python -m src.train --data 'Loan_default 2.csv' --rows 200000
./scripts/start-project.sh
```

Training saves the preprocessing pipeline in `outputs/spark_preprocessor/`, the selected classifier in `outputs/spark_classifier/`, and evaluation results in `outputs/metrics.json`. Model directories are generated locally and excluded from Git.

## Project structure

| Path | Purpose |
| --- | --- |
| `app.py` | Flask routes, form validation, and Spark predictions |
| `src/train.py` | Feature preparation, training, model selection, and evaluation |
| `src/cassandra_store.py` | CSV ingestion and dashboard queries |
| `templates/` | Dashboard templates and graphs |
| `static/` | Styles and the LoanLens background image |
| `scripts/start-project.sh` | Docker startup, training, and ingestion |
| `scripts/show-cassandra.sh` | Read-only Cassandra demonstration |
| `tests/test_project.py` | Dataset, metrics, form, and application checks |
| `outputs/metrics.json` | Recorded model evaluation results |
| `LoanLens_Project_Report.docx` | Project report |
| `VERIFICATION.md` | Detailed verification results |

## Evaluation

The 200,000-row sample is split with PySpark `randomSplit` into approximately 70% training, 15% validation, and 15% test data. Model selection maximizes validation average precision. The decision threshold maximizes validation F1 over a fixed grid. The test set is reserved for final evaluation.

The recorded run selected logistic regression with a **16.5% default-risk threshold**:

| Held-out test metric | Result |
| --- | --- |
| Accuracy | 79.82% |
| Default precision | 28.82% |
| Default recall | 48.45% |
| F1 | 36.15% |
| ROC AUC | 0.7537 |
| Average precision | 0.3025 |

Always predicting no default achieves 88.21% accuracy on the same test set but detects no defaults. Read accuracy alongside default precision and recall. The CSV has no loan date, so a future-time evaluation is unavailable. See [VERIFICATION.md](VERIFICATION.md) for details.

## Local checks

With a configured Python environment, the dataset, and generated Spark artifacts available:

```bash
.venv/bin/python -m pytest -q
```

The five tests cover the dataset contract, metric helpers, form validation, dashboard and artifact checks, and prediction threshold behavior. Docker is the documented application runtime because PySpark requires Java.
