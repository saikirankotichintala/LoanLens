"""PySpark loan default training and evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

NUMERIC = ["Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed", "NumCreditLines", "InterestRate", "LoanTerm", "DTIRatio"]
CATEGORICAL = ["Education", "EmploymentType", "MaritalStatus", "HasMortgage", "HasDependents", "LoanPurpose", "HasCoSigner"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "Default"


def spark_session():
    from pyspark.sql import SparkSession
    return (SparkSession.builder.master("local[2]").appName("LoanLens PySpark")
            .config("spark.driver.memory", "2g")
            .config("spark.sql.shuffle.partitions", "8")
            .config("spark.ui.enabled", "false").getOrCreate())


def prepare_data(spark, path: Path, rows: int):
    from pyspark.sql import functions as F
    frame = spark.read.option("header", True).option("inferSchema", True).csv(str(path))
    required = set(FEATURES + [TARGET, "LoanID"])
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    frame = frame.dropna(subset=FEATURES + [TARGET, "LoanID"]).filter(F.col(TARGET).isin(0, 1))
    frame = frame.dropDuplicates(["LoanID"])
    available = frame.count()
    if available < rows:
        raise ValueError(f"Requested {rows:,} rows; found {available:,} valid unique loans")
    sample = frame.withColumn("_sample_order", F.rand(seed=42)).orderBy("_sample_order").limit(rows).drop("_sample_order")
    training, validation, testing = sample.randomSplit([0.70, 0.15, 0.15], seed=42)
    return training.cache(), validation.cache(), testing.cache(), available


def preprocessing_pipeline():
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler
    indexers = [StringIndexer(inputCol=field, outputCol=f"{field}_index", handleInvalid="keep") for field in CATEGORICAL]
    encoder = OneHotEncoder(inputCols=[f"{field}_index" for field in CATEGORICAL],
                            outputCols=[f"{field}_vector" for field in CATEGORICAL], handleInvalid="keep")
    assembler = VectorAssembler(inputCols=NUMERIC + [f"{field}_vector" for field in CATEGORICAL],
                                outputCol="raw_features", handleInvalid="skip")
    scaler = StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=False)
    return Pipeline(stages=indexers + [encoder, assembler, scaler])


def candidates():
    from pyspark.ml.classification import DecisionTreeClassifier, LogisticRegression, RandomForestClassifier
    return {
        "Logistic regression": LogisticRegression(featuresCol="features", labelCol=TARGET, maxIter=50, regParam=0.05),
        "Decision tree": DecisionTreeClassifier(featuresCol="features", labelCol=TARGET,
                                                 maxDepth=8, minInstancesPerNode=100, seed=42),
        "Random forest": RandomForestClassifier(featuresCol="features", labelCol=TARGET,
                                                 numTrees=60, maxDepth=8, minInstancesPerNode=25,
                                                 featureSubsetStrategy="sqrt", seed=42),
    }


def probability_rows(predictions):
    from pyspark.ml.functions import vector_to_array
    rows = predictions.select(TARGET, vector_to_array("probability")[1].alias("risk")).collect()
    y = np.fromiter((int(row[TARGET]) for row in rows), dtype=np.int8, count=len(rows))
    risk = np.fromiter((float(row["risk"]) for row in rows), dtype=float, count=len(rows))
    return y, risk


def choose_threshold(y, risk):
    thresholds = np.linspace(0.02, 0.50, 97)
    positive = y == 1
    prediction = risk[:, None] >= thresholds[None, :]
    tp = np.sum(prediction & positive[:, None], axis=0)
    fp = np.sum(prediction & ~positive[:, None], axis=0)
    fn = np.sum(~prediction & positive[:, None], axis=0)
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    return float(thresholds[np.argmax(f1)])


def rank_metrics(y, risk):
    order = np.argsort(-risk, kind="mergesort")
    sorted_risk = risk[order]
    sorted_y = y[order]
    positives = int(np.sum(sorted_y))
    negatives = len(sorted_y) - positives
    if not positives or not negatives:
        return 0.0, 0.0
    tps = np.cumsum(sorted_y)
    fps = np.cumsum(1 - sorted_y)
    last_at_score = np.r_[np.where(np.diff(sorted_risk) != 0)[0], len(sorted_risk) - 1]
    recall = tps[last_at_score] / positives
    precision = tps[last_at_score] / (tps[last_at_score] + fps[last_at_score])
    average_precision = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
    tpr = np.r_[0.0, recall]
    fpr = np.r_[0.0, fps[last_at_score] / negatives]
    roc_auc = float(np.trapezoid(tpr, fpr))
    return roc_auc, average_precision


def scores(y, risk, threshold):
    pred = risk >= threshold
    positive = y == 1
    tp = int(np.sum(pred & positive))
    tn = int(np.sum(~pred & ~positive))
    fp = int(np.sum(pred & ~positive))
    fn = int(np.sum(~pred & positive))
    roc_auc, average_precision = rank_metrics(y, risk)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"prevalence": float(np.mean(y)), "accuracy": (tp + tn) / len(y),
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "roc_auc": roc_auc, "average_precision": average_precision,
            "confusion_matrix": [[tn, fp], [fn, tp]]}


def train(data: Path, rows: int, output: Path):
    spark = spark_session()
    spark.sparkContext.setLogLevel("ERROR")
    training = validation = testing = train_features = val_features = test_features = None
    try:
        training, validation, testing, available = prepare_data(spark, data, rows)
        partitions = {"train": training.count(), "validation": validation.count(), "test": testing.count()}
        print("Data partitions:", partitions, flush=True)
        preprocessor = preprocessing_pipeline().fit(training)
        train_features = preprocessor.transform(training).select(TARGET, "features").cache()
        val_features = preprocessor.transform(validation).select(TARGET, "features").cache()
        test_features = preprocessor.transform(testing).select(TARGET, "features").cache()
        train_features.count(); val_features.count(); test_features.count()
        fitted = {}
        validation_report = {}
        for name, estimator in candidates().items():
            model = estimator.fit(train_features)
            y, risk = probability_rows(model.transform(val_features))
            threshold = choose_threshold(y, risk)
            validation_report[name] = {**scores(y, risk, threshold), "threshold": threshold}
            fitted[name] = model
            print(f"{name}: validation AP={validation_report[name]['average_precision']:.4f}", flush=True)
        winner = max(validation_report, key=lambda name: validation_report[name]["average_precision"])
        threshold = validation_report[winner]["threshold"]
        y_test, risk_test = probability_rows(fitted[winner].transform(test_features))
        test_report = scores(y_test, risk_test, threshold)
        report = {
            "source": str(data), "engine": "PySpark ML", "source_rows": available,
            "rows": sum(partitions.values()), "partitions": partitions,
            "split": "PySpark randomSplit 70/15/15, seed 42; dataset has no date field",
            "selection_metric": "validation average precision", "selected_model": winner,
            "threshold_selection": "validation maximum F1 on 0.02–0.50 grid",
            "decision_threshold": threshold, "validation": validation_report,
            "test": test_report, "always_no_default_accuracy": float(1 - np.mean(y_test)),
        }
        output.mkdir(parents=True, exist_ok=True)
        preprocessor.write().overwrite().save(str(output / "spark_preprocessor"))
        fitted[winner].write().overwrite().save(str(output / "spark_classifier"))
        (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
        return report
    finally:
        for frame in (train_features, val_features, test_features, training, validation, testing):
            if frame is not None:
                frame.unpersist()
        spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("Loan_default 2.csv"))
    parser.add_argument("--rows", type=int, default=200000)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    train(args.data, args.rows, args.output)
