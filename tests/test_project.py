import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app import app, DEFAULTS, parse_application, repayment_prediction
from src.train import FEATURES, choose_threshold, scores


def test_feature_contract_and_dataset():
    assert len(FEATURES) == 16
    assert 'LoanID' not in FEATURES and 'Default' not in FEATURES
    frame = pd.read_csv('Loan_default 2.csv', usecols=['LoanID', 'Default'])
    assert len(frame) == 255347
    assert frame.LoanID.is_unique
    assert set(frame.Default.unique()) == {0, 1}


def test_spark_metric_helpers_detect_defaults():
    actual = np.array([0, 0, 1, 1], dtype=np.int8)
    risk = np.array([0.05, 0.10, 0.75, 0.90])
    threshold = choose_threshold(actual, risk)
    result = scores(actual, risk, threshold)
    assert result['recall'] == 1.0
    assert result['precision'] == 1.0
    assert result['average_precision'] == 1.0
    assert result['roc_auc'] == 1.0


def test_form_validation():
    assert parse_application(DEFAULTS)['CreditScore'] == 650
    with pytest.raises(ValueError):
        parse_application({**DEFAULTS, 'CreditScore': '900'})
    with pytest.raises(ValueError):
        parse_application({**DEFAULTS, 'LoanTerm': '37'})


def test_page_and_spark_artifacts():
    metrics = json.loads(Path('outputs/metrics.json').read_text())
    assert metrics['engine'] == 'PySpark ML'
    assert set(metrics['validation']) == {'Logistic regression', 'Decision tree', 'Random forest'}
    assert Path('outputs/spark_preprocessor').is_dir()
    assert Path('outputs/spark_classifier').is_dir()
    client = app.test_client()
    page = client.get('/')
    assert page.status_code == 200
    assert b'Comparison graph' in page.data
    assert b'More graphs' in page.data
    assert b'Test predictions' in page.data
    assert b'Apache Cassandra' in page.data
    assert client.get('/health').json['model_ready'] is True


def test_repayment_prediction_uses_trained_threshold():
    threshold = json.loads(Path('outputs/metrics.json').read_text())['decision_threshold']
    assert repayment_prediction(threshold - 0.01, threshold) == 'Likely to repay'
    assert repayment_prediction(threshold + 0.01, threshold) == 'At risk of not repaying'
    assert repayment_prediction(threshold, threshold) == 'At risk of not repaying'
