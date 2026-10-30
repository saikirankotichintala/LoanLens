"""Educational loan default prediction interface."""
from __future__ import annotations
import json
from pathlib import Path
from threading import Lock
from flask import Flask, render_template, request
from src.train import FEATURES
from src.cassandra_store import dashboard

BASE = Path(__file__).resolve().parent
PREPROCESSOR_PATH = BASE / 'outputs/spark_preprocessor'
MODEL_PATH = BASE / 'outputs/spark_classifier'
METRICS_PATH = BASE / 'outputs/metrics.json'
app = Flask(__name__)
_model_lock = Lock()
_predictor = None
NUMERIC = {
    'Age': ('Age', 18, 69, 35), 'Income': ('Annual income ($)', 15000, 149999, 70000),
    'LoanAmount': ('Loan amount ($)', 5000, 249999, 50000), 'CreditScore': ('Credit score', 300, 849, 650),
    'MonthsEmployed': ('Months employed', 0, 119, 36), 'NumCreditLines': ('Credit lines', 1, 4, 2),
    'InterestRate': ('Interest rate (%)', 2, 25, 12), 'LoanTerm': ('Loan term (months)', 12, 60, 36),
    'DTIRatio': ('Debt-to-income ratio', 0.1, 0.9, 0.4),
}
CATEGORICAL = {
    'Education': ["High School", "Bachelor's", "Master's", 'PhD'],
    'EmploymentType': ['Full-time', 'Part-time', 'Self-employed', 'Unemployed'],
    'MaritalStatus': ['Single', 'Married', 'Divorced'],
    'HasMortgage': ['No', 'Yes'], 'HasDependents': ['No', 'Yes'],
    'LoanPurpose': ['Auto', 'Business', 'Education', 'Home', 'Other'], 'HasCoSigner': ['No', 'Yes'],
}
DEFAULTS = {name: str(spec[3]) for name, spec in NUMERIC.items()} | {name: options[0] for name, options in CATEGORICAL.items()}

def read_artifacts():
    if not PREPROCESSOR_PATH.exists() or not MODEL_PATH.exists() or not METRICS_PATH.exists():
        return None, None
    metrics = json.loads(METRICS_PATH.read_text())
    return {'threshold': metrics['decision_threshold']}, metrics


def predict_probability(application):
    """Run a single application through the saved PySpark preprocessing and model."""
    global _predictor
    from pyspark.ml import PipelineModel
    from pyspark.ml.classification import (
        DecisionTreeClassificationModel, LogisticRegressionModel,
        RandomForestClassificationModel,
    )
    from pyspark.ml.functions import vector_to_array
    from src.train import spark_session

    with _model_lock:
        if _predictor is None:
            metrics = json.loads(METRICS_PATH.read_text())
            model_type = {
                'Logistic regression': LogisticRegressionModel,
                'Decision tree': DecisionTreeClassificationModel,
                'Random forest': RandomForestClassificationModel,
            }[metrics['selected_model']]
            spark = spark_session()
            spark.sparkContext.setLogLevel('ERROR')
            preprocessor = PipelineModel.load(str(PREPROCESSOR_PATH))
            classifier = model_type.load(str(MODEL_PATH))
            _predictor = (spark, preprocessor, classifier)
        spark, preprocessor, classifier = _predictor
        row = spark.createDataFrame([application])
        prediction = classifier.transform(preprocessor.transform(row))
        return float(prediction.select(vector_to_array('probability')[1].alias('risk')).first()['risk'])

def parse_application(form):
    values = {}
    for field, (_, minimum, maximum, _) in NUMERIC.items():
        try:
            value = float(form[field])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f'Enter a valid {field}.') from None
        if not minimum <= value <= maximum:
            raise ValueError(f'{field} must be between {minimum} and {maximum}.')
        if field in ('Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed', 'NumCreditLines', 'LoanTerm') and not value.is_integer():
            raise ValueError(f'{field} must be a whole number.')
        if field == 'LoanTerm' and value not in (12, 24, 36, 48, 60):
            raise ValueError('Loan term must be 12, 24, 36, 48 or 60 months.')
        values[field] = value
    for field, options in CATEGORICAL.items():
        value = form.get(field)
        if value not in options:
            raise ValueError(f'Choose a valid {field}.')
        values[field] = value
    return values


def repayment_prediction(default_probability: float, threshold: float) -> str:
    """Translate the default classifier into a user-facing repayment prediction."""
    return 'At risk of not repaying' if default_probability >= threshold else 'Likely to repay'

@app.route('/', methods=['GET', 'POST'])
def index():
    artifact, metrics = read_artifacts()
    values = dict(DEFAULTS)
    result = error = outcome = None
    if request.method == 'POST':
        values.update(request.form.to_dict())
        if artifact is None:
            error = 'Train the model before requesting a prediction.'
        else:
            try:
                application = parse_application(request.form)
                probability = predict_probability(application)
                outcome = repayment_prediction(probability, artifact['threshold'])
                result = round(probability * 100, 1)
            except ValueError as exc:
                error = str(exc)
            except Exception:
                app.logger.exception('PySpark prediction failed')
                error = 'The prediction service is unavailable. Check the Spark model and try again.'
    return render_template('index.html', values=values, result=result, outcome=outcome, error=error,
                           ready=artifact is not None, metrics=metrics, numeric=NUMERIC,
                           categorical=CATEGORICAL, cassandra=dashboard())

@app.get('/health')
def health():
    database = dashboard()
    return {'status': 'ok', 'model_ready': PREPROCESSOR_PATH.exists() and MODEL_PATH.exists() and METRICS_PATH.exists(),
            'cassandra_connected': database['connected'], 'cassandra_ingested': database['ingested']}

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
