# LoanLens — Loan Default Prediction with PySpark

LoanLens predicts whether a loan is **likely to repay** or **at risk of not repaying**. It displays the estimated default probability and a comparison of three PySpark ML models. This is an educational project, not a lending decision system.

## Dataset and target

Kaggle dataset: [Loan Default Prediction Dataset by nikhil1e9](https://www.kaggle.com/datasets/nikhil1e9/loan-default). Its published dimensions and fields match the project dataset; the supplied local file has not been independently checked for an exact match. Download the CSV and place it in the project root as `Loan_default 2.csv` before training. Dataset files are excluded from this repository.

The project uses the user-provided `Loan_default 2.csv` in the repository root: **255,347 rows**, 16 predictor fields, `LoanID`, and `Default`. The original CSV and `Loan_default 2 copy.csv` are unchanged. The origin of the data, including whether records are real or synthetic, has not been independently verified.

`Default = 1` means a recorded default; `Default = 0` means no default recorded. The no-default label does not prove that every loan was fully repaid. `LoanID` is excluded from prediction. The 16 inputs are age, income, loan amount, credit score, months employed, number of credit lines, interest rate, loan term, debt-to-income ratio, education, employment type, marital status, mortgage, dependents, loan purpose, and co-signer. The interest rate input assumes proposed loan terms are already known.

## Architecture

1. **PySpark** reads, validates, samples, and splits the CSV.
2. **Spark ML** encodes categories, scales features, and trains logistic regression, decision tree, and random forest models.
3. The selected Spark model and preprocessing pipeline are saved in `outputs/spark_classifier/` and `outputs/spark_preprocessor/`. `outputs/metrics.json` stores validation and test results.
4. **Flask** loads those Spark artifacts and uses PySpark to predict each form submission. The displayed repayment label follows a threshold selected on validation data.
5. **Apache Cassandra** stores all 255,347 loan records in `loan_analytics.loan_records`. `category_counts` and `dataset_meta` supply the dashboard's dataset graphs and totals.

The dashboard has model comparison, outcome distribution, default rates by loan purpose, employment, and education, and a confusion-matrix chart. The supplied image at `static/loanlens-hero.png` is used as the full-page and hero background with pale text overlays.

To demonstrate Cassandra in a presentation, open the dashboard's **Apache Cassandra** section and run `./scripts/show-cassandra.sh` in another terminal. The script only reads the database. It shows the imported total, five stored loan records, and the same loan-purpose counts used by the dashboard graphs.

## Start the project

Docker Desktop must be running. From the project root:

```bash
./scripts/start-project.sh
```

The script reuses the existing `cassandra-db` and `loanlens-app` containers when present. Otherwise it creates them, builds the app image if needed, trains PySpark models if artifacts are missing, and imports the CSV if it is not yet in Cassandra. It does not delete existing Cassandra data. Open <http://127.0.0.1:5000>.

To retrain explicitly with PySpark:

```bash
docker run --rm --network smartflow-network -v "$PWD:/app" -w /app \
  smartflow-flask:local python -m src.train --data 'Loan_default 2.csv' --rows 200000
./scripts/start-project.sh
```

The first command assumes the existing `smartflow-flask:local` image. On a new setup, `start-project.sh` builds `loanlens:local` from `Dockerfile.loanlens`. The app runs in Docker because PySpark needs Java. Run local checks with `.venv/bin/python -m pytest -q`.

## Evaluation and limitations

The 200,000-row sample is split with PySpark `randomSplit` at roughly 70% training, 15% validation, and 15% test. The selected model maximizes validation average precision; its classification threshold maximizes validation F1 over a fixed grid. The test set is not used for model or threshold selection. The CSV has no loan date, so a future-time test is impossible. Accuracy must be read alongside default recall and precision because the majority of rows have no default. See `VERIFICATION.md` for measured results.
