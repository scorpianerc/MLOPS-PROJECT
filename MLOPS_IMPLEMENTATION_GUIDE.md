# 🚀 MLOps Implementation Guide - IndoBERT Sentiment Analysis

## 📋 Daftar Isi
1. [Model Versioning & Registry (MLflow)](#1-model-versioning--registry-mlflow)
2. [Automated Testing (CI/CD ML)](#2-automated-testing-cicd-ml)
3. [Monitoring Model Production](#3-monitoring-model-production)
4. [Model Serving API (FastAPI)](#4-model-serving-api-fastapi)
5. [Automated Retraining Pipeline](#5-automated-retraining-pipeline)
6. [Feature Store](#6-feature-store)
7. [Experiment Tracking](#7-experiment-tracking)

---

## 1. Model Versioning & Registry (MLflow)

### 🎯 **Tujuan**
- Track semua training experiments dengan parameter, metrics, dan artifacts
- Version control untuk model dengan Model Registry
- Rollback capability untuk model yang bermasalah
- Compare multiple model versions

### 📂 **File Terkait**
- `src/mlops/mlflow_manager.py` - MLflow manager class
- `src/training/train_bert.py` - Integrasi dengan training

### 🚀 **Setup MLflow**

#### A. Start MLflow UI
```bash
# Start MLflow tracking server
mlflow ui --port 5000

# Akses di browser
http://localhost:5000
```

#### B. Training dengan MLflow
```python
from src.mlops.mlflow_manager import MLflowManager

# Initialize MLflow
mlflow_manager = MLflowManager(experiment_name="sentiment-analysis-indobert")

# Start run
mlflow_manager.start_run(
    run_name="training_20250111",
    tags={"model_type": "IndoBERT", "dataset": "pintu_reviews"}
)

# Log parameters
mlflow_manager.log_params({
    "batch_size": 16,
    "learning_rate": 2e-5,
    "epochs": 3
})

# Log metrics per epoch
mlflow_manager.log_metrics({
    "train_loss": 0.3,
    "train_accuracy": 0.85,
    "test_accuracy": 0.82
}, step=epoch)

# Log model
mlflow_manager.log_model_pytorch(
    model=model,
    registered_model_name="indobert-sentiment-analysis"
)

# End run
mlflow_manager.end_run()
```

#### C. Load Model dari Registry
```python
# Load production model
model = mlflow_manager.load_model_from_registry(
    model_name="indobert-sentiment-analysis",
    stage="Production"
)

# Load specific version
model = mlflow_manager.load_model_from_registry(
    model_name="indobert-sentiment-analysis",
    version=3
)
```

### 📊 **Model Stages**
- **None**: Model baru, belum di-promote
- **Staging**: Model untuk testing
- **Production**: Model aktif di production
- **Archived**: Model lama, sudah tidak dipakai

### 🔄 **Promote Model ke Production**
```python
# Transition model ke Production
mlflow_manager.transition_model_stage(
    model_name="indobert-sentiment-analysis",
    version=5,
    stage="Production",
    archive_existing_versions=True  # Archive model lama
)
```

### 📈 **Features**
- ✅ Auto-logging parameters, metrics, artifacts
- ✅ Confusion matrix visualization
- ✅ Training loss plot
- ✅ Model comparison
- ✅ Best run detection
- ✅ Model registry dengan versioning

---

## 2. Automated Testing (CI/CD ML)

### 🎯 **Tujuan**
- Automated validation untuk data quality
- Model performance testing
- Integration testing
- CI/CD pipeline dengan GitHub Actions

### 📂 **File Terkait**
- `tests/test_data_validation.py` - Data validation tests
- `tests/test_model_validation.py` - Model validation tests
- `tests/test_integration.py` - Integration tests
- `.github/workflows/ml-ci-cd.yml` - CI/CD pipeline

### 🧪 **Run Tests Locally**

#### A. Install pytest
```bash
pip install pytest pytest-cov
```

#### B. Run all tests
```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_data_validation.py -v

# Run dengan coverage report
pytest tests/ --cov=src --cov-report=html
```

#### C. Run specific test
```bash
# Test data validation
pytest tests/test_data_validation.py::TestDataValidation::test_required_columns -v

# Test model performance
pytest tests/test_model_validation.py::TestModelPerformance::test_minimum_accuracy -v
```

### 📋 **Test Categories**

#### 1. Data Validation Tests
```python
# Required columns check
def test_required_columns(df):
    assert 'review_text' in df.columns
    assert 'sentiment_label' in df.columns

# No null values
def test_no_null_values(df):
    assert df['review_text'].notna().all()

# Valid sentiment labels
def test_sentiment_labels_valid(df):
    valid_labels = ['positive', 'negative', 'neutral']
    assert df['sentiment_label'].isin(valid_labels).all()

# Data balance
def test_balanced_distribution(df):
    counts = df['sentiment_label'].value_counts()
    max_ratio = counts.max() / counts.min()
    assert max_ratio < 10  # Not too imbalanced
```

#### 2. Model Validation Tests
```python
# Model minimum accuracy
def test_minimum_accuracy():
    accuracy = get_model_accuracy()
    assert accuracy >= 0.70  # 70% minimum

# No severe overfitting
def test_no_severe_overfitting():
    train_acc = get_train_accuracy()
    test_acc = get_test_accuracy()
    gap = train_acc - test_acc
    assert gap <= 0.15  # Max 15% gap

# Model handles edge cases
def test_handles_empty_string():
    prediction = predict("")
    assert prediction is not None

def test_handles_long_text():
    long_text = "Bagus " * 1000
    prediction = predict(long_text)
    assert prediction is not None
```

#### 3. Integration Tests
```python
# Database connection
def test_database_connection():
    conn = connect_to_db()
    assert conn is not None

# End-to-end prediction
def test_end_to_end_prediction():
    text = "Aplikasi bagus"
    prediction = predict_sentiment(text)
    assert prediction['sentiment'] in ['positive', 'negative', 'neutral']
```

### 🔄 **CI/CD Pipeline (GitHub Actions)**

Pipeline otomatis berjalan saat:
- Push ke branch `main` atau `develop`
- Pull Request dibuat
- Manual trigger

**Stages:**
1. ✅ Data Validation
2. ✅ Model Validation
3. ✅ Integration Tests
4. ✅ Code Quality (linting, security)
5. ✅ Model Training (production only)
6. ✅ Test Report Generation

**View Pipeline Results:**
- GitHub Actions tab di repository
- Artifacts download (test reports)

---

## 3. Monitoring Model Production

### 🎯 **Tujuan**
- Detect data drift (perubahan distribusi data)
- Detect model drift (penurunan performance)
- Log predictions untuk audit
- Monitor model health

### 📂 **File Terkait**
- `src/mlops/drift_detection.py` - Drift detection classes

### 📊 **Data Drift Detection**

#### A. Initialize Drift Detector
```python
from src.mlops.drift_detection import DataDriftDetector
import pandas as pd

# Load reference data (training data)
reference_data = pd.read_csv('data/processed/processed_reviews.csv')

# Initialize detector
drift_detector = DataDriftDetector(
    reference_data=reference_data,
    significance_level=0.05  # 5% significance
)
```

#### B. Detect Categorical Drift
```python
# Load current production data
current_data = pd.read_csv('data/production/recent_reviews.csv')

# Detect drift pada sentiment distribution
drift_result = drift_detector.detect_categorical_drift(
    feature_name='sentiment_label',
    current_data=current_data
)

print(f"Drift detected: {drift_result['drift_detected']}")
print(f"P-value: {drift_result['p_value']}")
print(f"Distribution shift: {drift_result['max_distribution_shift']}")
```

#### C. Detect All Drifts
```python
# Detect drift untuk semua features
results = drift_detector.detect_all_drifts(
    current_data=current_data,
    numerical_features=['rating', 'text_length'],
    categorical_features=['sentiment_label']
)

if results['overall_drift_detected']:
    print(f"⚠️  Drift detected in {results['summary']['features_with_drift']} features")
    for feature in results['features']:
        if feature['drift_detected']:
            print(f"  - {feature['feature']}: p-value={feature['p_value']:.4f}")
```

### 📈 **Model Drift Monitoring**

#### A. Initialize Monitor
```python
from src.mlops.drift_detection import ModelDriftMonitor

monitor = ModelDriftMonitor(db_config)
```

#### B. Check Performance Degradation
```python
# Calculate current metrics
current_metrics = {
    'accuracy': 0.78,  # Current accuracy
    'precision': 0.77,
    'recall': 0.76,
    'f1': 0.76
}

# Detect drift
drift_result = monitor.detect_performance_drift(
    current_metrics=current_metrics,
    threshold=0.05  # 5% degradation threshold
)

if drift_result['drift_detected']:
    print("⚠️  Performance degradation detected!")
    for metric in drift_result['degraded_metrics']:
        print(f"  {metric['metric']}: {metric['baseline']:.2%} → {metric['current']:.2%}")
```

#### C. Analyze Trend
```python
# Get 30-day trend
trend = monitor.analyze_trend(days=30)

print(f"Overall trend: {trend['overall_trend']}")
for metric, info in trend['trends'].items():
    print(f"  {metric}: {info['trend']} (slope: {info['slope']:.4f})")
```

### 📝 **Prediction Logging**

#### A. Create Logs Table
```python
from src.mlops.drift_detection import create_prediction_logs_table

create_prediction_logs_table(db_config)
```

#### B. Log Predictions
```python
from src.mlops.drift_detection import PredictionLogger

logger = PredictionLogger(db_config)

# Log prediction
logger.log_prediction(
    review_id=12345,
    text="Aplikasi bagus",
    predicted_sentiment="positive",
    confidence=0.92,
    model_version="1.0.0"
)
```

#### C. Analyze Recent Predictions
```python
# Get predictions last 24 hours
recent_preds = logger.get_recent_predictions(hours=24)

print(f"Total predictions: {len(recent_preds)}")
print(f"Average confidence: {recent_preds['confidence'].mean():.2%}")
print("\nSentiment distribution:")
print(recent_preds['predicted_sentiment'].value_counts())
```

### 🎯 **Monitoring Metrics di Grafana**

Dashboard metrics:
- ✅ Total predictions per hour
- ✅ Average confidence score
- ✅ Sentiment distribution over time
- ✅ Error rate
- ✅ Prediction latency
- ✅ Data drift score
- ✅ Model performance trend

---

## 4. Model Serving API (FastAPI)

### 🎯 **Tujuan**
- REST API untuk model inference
- Terpisah dari Streamlit dashboard
- Production-ready dengan monitoring
- Scalable architecture

### 📂 **File Terkait**
- `src/api/api_server.py` - FastAPI server

### 🚀 **Start API Server**

#### A. Local Development
```bash
# Run server
python src/api/api_server.py

# Or dengan uvicorn
uvicorn src.api.api_server:app --host 0.0.0.0 --port 8080 --reload
```

#### B. Docker
```bash
# Start API container
docker-compose up -d api

# View logs
docker logs -f sentiment_api
```

### 📡 **API Endpoints**

#### 1. Health Check
```bash
curl http://localhost:8080/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "1.0.0",
  "uptime_seconds": 3600.5,
  "timestamp": "2025-01-11T10:30:00"
}
```

#### 2. Single Prediction
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Aplikasi ini sangat bagus dan mudah digunakan!",
    "review_id": 12345
  }'
```

Response:
```json
{
  "sentiment": "positive",
  "confidence": 0.95,
  "probabilities": {
    "positive": 0.95,
    "negative": 0.03,
    "neutral": 0.02
  },
  "review_id": 12345,
  "model_version": "1.0.0",
  "timestamp": "2025-01-11T10:30:00"
}
```

#### 3. Batch Prediction
```bash
curl -X POST http://localhost:8080/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Aplikasi bagus",
      "Sangat buruk",
      "Lumayan"
    ]
  }'
```

Response:
```json
{
  "predictions": [
    {
      "sentiment": "positive",
      "confidence": 0.92,
      "probabilities": {...},
      "model_version": "1.0.0",
      "timestamp": "2025-01-11T10:30:00"
    },
    ...
  ],
  "total": 3,
  "model_version": "1.0.0"
}
```

#### 4. Model Info
```bash
curl http://localhost:8080/model/info
```

#### 5. Reload Model (Hot Reload)
```bash
curl http://localhost:8080/model/reload
```

#### 6. Prometheus Metrics
```bash
curl http://localhost:8080/metrics
```

### 📊 **Prometheus Metrics**

API expose metrics untuk monitoring:
- `sentiment_predictions_total{sentiment="positive"}` - Total predictions per sentiment
- `sentiment_prediction_latency_seconds` - Prediction latency histogram
- `sentiment_prediction_confidence` - Average confidence gauge
- `sentiment_prediction_errors_total{error_type="..."}` - Error counter

### 📖 **Interactive API Documentation**

Buka di browser:
- **Swagger UI**: http://localhost:8080/docs
- **ReDoc**: http://localhost:8080/redoc

### 🐍 **Python Client Example**

```python
import requests

# Predict
response = requests.post(
    'http://localhost:8080/predict',
    json={
        'text': 'Aplikasi ini sangat bagus!',
        'review_id': 123
    }
)

result = response.json()
print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']:.2%}")
```

---

## 5. Automated Retraining Pipeline

### 🎯 **Tujuan**
- Automated retraining berdasarkan triggers
- Data-driven retraining decisions
- Performance-based retraining
- Scheduled retraining

### 📂 **File Terkait**
- `src/mlops/retraining_pipeline.py` - Retraining pipeline

### 🔄 **Retraining Triggers**

#### 1. **Time-Based Trigger**
```python
# Maximum 30 hari sejak training terakhir
if days_since_training >= 30:
    trigger_retraining("Monthly scheduled retraining")
```

#### 2. **Data-Based Trigger**
```python
# 500+ review baru sejak training terakhir
if new_reviews_count >= 500:
    trigger_retraining(f"New data: {new_reviews_count} reviews")
```

#### 3. **Performance-Based Trigger**
```python
# Accuracy drop > 3%
if accuracy_drop > 0.03:
    trigger_retraining("Performance degradation detected")
```

#### 4. **User Feedback Trigger**
```python
# Error rate > 12% dari user feedback
if error_rate > 0.12:
    trigger_retraining(f"High error rate: {error_rate:.1%}")
```

### 🚀 **Run Retraining Pipeline**

#### A. Manual Run
```bash
python src/mlops/retraining_pipeline.py
```

#### B. Check Triggers Only
```python
from src.mlops.retraining_pipeline import RetrainingTrigger

trigger = RetrainingTrigger(db_config)
evaluation = trigger.evaluate_triggers()

print(f"Should retrain: {evaluation['should_retrain']}")
print(f"Triggers: {evaluation['summary']['total_triggers']}")

for t in evaluation['triggers']:
    print(f"  - {t['type']}: {t['reason']} ({t['priority']})")
```

#### C. Full Pipeline Run
```python
from src.mlops.retraining_pipeline import RetrainingPipeline

pipeline = RetrainingPipeline(db_config)
results = pipeline.run()

print(f"Status: {results['status']}")
if results['status'] == 'success':
    print("✅ Retraining completed!")
else:
    print(f"❌ Retraining failed: {results.get('reason')}")
```

### 📅 **Scheduled Retraining**

#### A. Add to Scheduler (src/scheduler/main.py)
```python
from apscheduler.schedulers.background import BackgroundScheduler
from src.mlops.retraining_pipeline import RetrainingPipeline

scheduler = BackgroundScheduler()

@scheduler.scheduled_job('cron', hour=2, minute=0)  # Daily at 2 AM
def check_retraining():
    pipeline = RetrainingPipeline(db_config)
    results = pipeline.run()
    
    # Send notification
    if results['should_retrain']:
        send_notification(f"Retraining status: {results['status']}")

scheduler.start()
```

### 🔔 **Notifications**

Pipeline sends notifications untuk:
- ✅ Retraining triggered
- ✅ Training completed successfully
- ⚠️ Training failed
- ⚠️ Model validation failed

### 📝 **Pipeline Results**

Results disimpan di `logs/retraining/`:
```json
{
  "started_at": "2025-01-11T02:00:00",
  "trigger_evaluation": {
    "should_retrain": true,
    "triggers": [
      {
        "type": "new_data",
        "reason": "New data threshold reached (600 reviews)",
        "priority": "medium"
      }
    ]
  },
  "training_result": {
    "status": "success"
  },
  "validation_result": {
    "passed": true
  },
  "status": "success",
  "completed_at": "2025-01-11T04:30:00"
}
```

---

## 6. Feature Store

### 🎯 **Tujuan**
- Consistent feature extraction antara train/serve
- Feature versioning
- Reusable features
- Performance optimization

### 📂 **File Terkait**
- `src/mlops/feature_store.py` - Feature store implementation

### 🏗️ **Initialize Feature Store**

```bash
python src/mlops/feature_store.py
```

### 📝 **Feature Extraction**

#### A. Extract Features untuk Single Text
```python
from src.mlops.feature_store import TextFeatureExtractor

extractor = TextFeatureExtractor()

# Extract features
features = extractor.extract_features("Aplikasi ini sangat bagus!")

print(features)
# {
#   'original_text': 'Aplikasi ini sangat bagus!',
#   'cleaned_text': 'aplikasi ini sangat bagus',
#   'text_no_stopwords': 'aplikasi sangat bagus',
#   'stemmed_text': 'aplikasi sangat bagus',
#   'original_length': 26,
#   'cleaned_length': 25,
#   'word_count': 4,
#   'char_count': 25,
#   'avg_word_length': 6.25
# }
```

#### B. Batch Feature Extraction
```python
texts = [
    "Aplikasi bagus",
    "Sangat buruk",
    "Lumayan"
]

features_df = extractor.batch_extract_features(texts)
print(features_df)
```

### 💾 **Store Features to Database**

```python
from src.mlops.feature_store import FeatureStore

feature_store = FeatureStore(db_config)

# Store features untuk single review
features = feature_store.store_features(
    review_id=12345,
    text="Aplikasi ini bagus"
)

# Batch store untuk multiple reviews
reviews_df = pd.read_sql("SELECT id, review_text FROM reviews", conn)
feature_store.batch_store_features(reviews_df)
```

### 🔍 **Retrieve Features**

```python
# Get features untuk specific review
features = feature_store.get_features(review_id=12345)

# Get training features dengan labels
train_features = feature_store.get_training_features(limit=1000)
print(train_features.columns)
# ['cleaned_text', 'stemmed_text', 'word_count', 'avg_word_length', 'sentiment_label', 'rating']
```

### 📊 **Feature Store Tables**

**text_features table:**
```sql
CREATE TABLE text_features (
    id SERIAL PRIMARY KEY,
    review_id INTEGER REFERENCES reviews(id),
    original_text TEXT,
    cleaned_text TEXT,
    text_no_stopwords TEXT,
    stemmed_text TEXT,
    original_length INTEGER,
    cleaned_length INTEGER,
    word_count INTEGER,
    char_count INTEGER,
    avg_word_length FLOAT,
    created_at TIMESTAMP
);
```

### ⚙️ **Feature Configuration**

Config disimpan di `models/feature_store/feature_config.json`:
```json
{
  "max_length": 128,
  "min_length": 3,
  "version": "1.0.0",
  "created_at": "2025-01-11T10:00:00"
}
```

### 🔄 **Use in Training & Serving**

#### Training:
```python
# Get consistent features untuk training
feature_store = FeatureStore(db_config)
train_df = feature_store.get_training_features()

# Train model dengan consistent features
model.fit(train_df['cleaned_text'], train_df['sentiment_label'])
```

#### Serving:
```python
# Extract features dengan same extractor
extractor = TextFeatureExtractor.load_config('models/feature_store/feature_config.json')
features = extractor.extract_features(user_input)

# Predict dengan consistent features
prediction = model.predict(features['cleaned_text'])
```

---

## 7. Experiment Tracking

### 🎯 **Tujuan**
- Track all training experiments
- Compare hyperparameters
- Visualize training progress
- Reproduce results

### 📊 **MLflow Experiment Tracking**

Sudah fully integrated di `train_bert.py`:

#### A. Auto-logged Information
- ✅ **Parameters**: batch_size, learning_rate, epochs, model_type
- ✅ **Metrics**: accuracy, precision, recall, F1 (per epoch)
- ✅ **Train/Test Metrics**: Separated metrics
- ✅ **Loss Plot**: Training loss visualization
- ✅ **Confusion Matrix**: Classification performance
- ✅ **Model**: Saved model dengan versioning

#### B. View Experiments
```bash
# Start MLflow UI
mlflow ui --port 5000

# Browse to
http://localhost:5000
```

#### C. Compare Runs
1. Open MLflow UI
2. Select multiple runs
3. Click "Compare"
4. View parameter/metric differences

#### D. Get Best Run
```python
from src.mlops.mlflow_manager import MLflowManager

manager = MLflowManager()
best_run = manager.get_best_run(metric_name='test_accuracy', maximize=True)

print(f"Best run ID: {best_run.info.run_id}")
print(f"Test accuracy: {best_run.data.metrics['test_accuracy']:.2%}")
```

---

## 🎯 Summary Checklist

### ✅ Implemented Features

| # | Feature | Status | Files |
|---|---------|--------|-------|
| 1 | Model Versioning & Registry | ✅ Done | `src/mlops/mlflow_manager.py` |
| 2 | Automated Testing | ✅ Done | `tests/*.py`, `.github/workflows/ml-ci-cd.yml` |
| 3 | Monitoring (Drift Detection) | ✅ Done | `src/mlops/drift_detection.py` |
| 4 | Model Serving API | ✅ Done | `src/api/api_server.py` |
| 5 | Automated Retraining | ✅ Done | `src/mlops/retraining_pipeline.py` |
| 6 | Feature Store | ✅ Done | `src/mlops/feature_store.py` |
| 7 | Experiment Tracking | ✅ Done | Integrated in `train_bert.py` |

### 🚀 Quick Start Commands

```bash
# 1. Start MLflow UI
mlflow ui --port 5000

# 2. Run tests
pytest tests/ -v

# 3. Start API server
docker-compose up -d api

# 4. Initialize feature store
python src/mlops/feature_store.py

# 5. Check retraining triggers
python src/mlops/retraining_pipeline.py

# 6. Train with MLflow tracking
python src/training/train_bert.py
```

### 📖 Documentation URLs

- **MLflow UI**: http://localhost:5000
- **API Docs**: http://localhost:8080/docs
- **API Metrics**: http://localhost:8080/metrics
- **Grafana**: http://localhost:3000
- **Streamlit**: http://localhost:8501

---

## 📞 Support

Jika ada pertanyaan atau issue:
1. Check dokumentasi di `docs/`
2. Review test cases di `tests/`
3. Check logs di `logs/`
4. Review MLflow experiments

**Happy MLOps! 🚀**
