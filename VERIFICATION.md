# LoanLens verification — PySpark version

## Dataset

- Active file: `Loan_default 2.csv` (255,347 rows, 18 columns). `LoanID` values are unique; the target is `Default` with 29,653 defaults and 225,694 no-default records.
- The original CSV and `Loan_default 2 copy.csv` remain unchanged. Dataset provenance has not been independently verified.
- Cassandra's existing `loan_analytics` keyspace still stores all 255,347 loan records and category summaries. Earlier keyspaces are unchanged.
- `scripts/show-cassandra.sh` was run successfully. Read-only CQL displayed the dataset total and default count, five rows from `loan_records`, and five loan-purpose groups from `category_counts` used by the page.

## PySpark pipeline

- Docker runtime: Python 3.11, Java 17, PySpark 3.5.3.
- `src/train.py` reads, validates, samples, prepares features, trains, and evaluates with PySpark and Spark ML. The app loads saved Spark models and uses PySpark for live predictions.
- Valid unique source rows: 255,347. Sample: 200,000. PySpark `randomSplit` partitions: 139,838 train, 30,016 validation, 30,146 held-out test.
- Validation average precision: logistic regression **0.3082**, decision tree **0.2144**, random forest **0.2953**. **Spark logistic regression** selected; validation F1 threshold **16.5%**.
- Held-out test: **79.82% accuracy**, **28.82% default precision**, **48.45% default recall**, **36.15% F1**, **0.7537 ROC AUC**, **0.3025 average precision**. Confusion matrix: 22,340 true negatives, 4,252 false positives, 1,832 false negatives, 1,722 true positives.
- Always predicting no default yields **88.21% accuracy** and **0% default recall** on the same test set. The selected threshold finds defaults at the cost of lower overall accuracy.
- Spark artifacts: `outputs/spark_preprocessor/`, `outputs/spark_classifier/`, `outputs/metrics.json`. The previous scikit-learn code and model are archived under `archive/sklearn-version-before-pyspark/` and are not used.

## App and notebook

- Live Flask app in `loanlens-app`: `/health` reported model ready, Cassandra connected, and dataset ingested. A live form submission returned HTTP 200, “Likely to repay,” and 13.2% estimated default risk through the saved Spark model.
- The page serves the full-page loan-themed image, model comparison graph, Cassandra-backed dataset graphs, and test prediction chart.
- `Loan_Default_Prediction.ipynb` is standalone for Google Colab: it prompts for a CSV upload and has no project-source or Cassandra dependency. In a Java 17 / PySpark 3.5.3 container, all seven core PySpark cells executed successfully and reproduced the selected logistic regression, 16.5% threshold, 79.82% test accuracy, and 48.45% default recall. Three matplotlib chart cells compiled; plotting dependencies were unavailable in that temporary container. Actual Google Colab execution has not been checked here.
- Local project tests: **6 passed**.
- The dataset has no loan date, so a future-time split cannot be evaluated. “No default” does not independently prove every loan was fully repaid. This model is for demonstration, not credit decisions.
