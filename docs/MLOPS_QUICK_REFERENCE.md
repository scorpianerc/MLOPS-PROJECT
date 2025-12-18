# 🎯 MLOps Features - Quick Reference

## ✅ Implemented Features (7/7)

| # | Feature | Status | Key File | Quick Command |
|---|---------|--------|----------|---------------|
| 1 | **Model Versioning & Registry** | ✅ | `src/mlops/mlflow_manager.py` | `mlflow ui --port 5000` |
| 2 | **Automated Testing (CI/CD)** | ✅ | `tests/*.py`, `.github/workflows/ml-ci-cd.yml` | `pytest tests/ -v` |
| 3 | **Monitoring (Drift Detection)** | ✅ | `src/mlops/drift_detection.py` | Auto-monitoring via Prometheus |
| 4 | **Model Serving API** | ✅ | `src/api/api_server.py` | `docker-compose up -d api` |
| 5 | **Automated Retraining** | ✅ | `src/mlops/retraining_pipeline.py` | `python src/mlops/retraining_pipeline.py` |
| 6 | **Feature Store** | ✅ | `src/mlops/feature_store.py` | `python src/mlops/feature_store.py` |
| 7 | **Experiment Tracking** | ✅ | Integrated in `train_bert.py` | Auto-logged during training |

---

## 🚀 Quick Start

### 1. Test All MLOps Features
```bash
python scripts/test_mlops_features.py
```

### 2. Start Services
```bash
# Start all services (including API)
docker-compose up -d

# Start MLflow UI
mlflow ui --port 5000
```

### 3. Access Dashboards
- **MLflow UI**: http://localhost:5000 - Model versioning & experiments
- **API Docs**: http://localhost:8080/docs - Interactive API documentation
- **Grafana**: http://localhost:3000 - Monitoring dashboards
- **Streamlit**: http://localhost:8501 - User dashboard

---

## 📖 Documentation

| Document | Description |
|----------|-------------|
| [LOCAL_DEPLOYMENT_GUIDE.md](LOCAL_DEPLOYMENT_GUIDE.md) | 📘 Panduan deployment lokal dengan Docker |
| [MONITORING_GUIDE.md](MONITORING_GUIDE.md) | 📘 Monitoring dengan Prometheus & Grafana |
| [GITHUB_ACTIONS_GUIDE.md](GITHUB_ACTIONS_GUIDE.md) | 📘 CI/CD dengan GitHub Actions |

---

## 🔧 Common Operations

### Training dengan MLflow Tracking
```bash
# Training akan auto-log ke MLflow
python src/training/train_bert.py

# View results di MLflow UI
mlflow ui --port 5000
```

### API Prediction
```bash
# Single prediction
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Aplikasi bagus!"}'

# Health check
curl http://localhost:8080/health

# Metrics
curl http://localhost:8080/metrics
```

### Run Tests
```bash
# All tests
pytest tests/ -v

# Specific test
pytest tests/test_data_validation.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

### Check Retraining Triggers
```bash
python -c "
from src.mlops.retraining_pipeline import RetrainingTrigger
import os
from dotenv import load_dotenv

load_dotenv()
db_config = {
    'host': os.getenv('POSTGRES_HOST', 'localhost'),
    'port': int(os.getenv('POSTGRES_PORT', 5432)),
    'database': os.getenv('POSTGRES_DB', 'sentiment_db'),
    'user': os.getenv('POSTGRES_USER', 'sentiment_user'),
    'password': os.getenv('POSTGRES_PASSWORD', 'password')
}

trigger = RetrainingTrigger(db_config)
result = trigger.evaluate_triggers()
print(f'Should retrain: {result[\"should_retrain\"]}')
for t in result['triggers']:
    print(f'  - {t[\"type\"]}: {t[\"reason\"]}')
"
```

### Initialize Feature Store
```bash
python src/mlops/feature_store.py
```

---

## 🎯 MLOps Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                    1. Data Collection                        │
│         src/data_collection/scraper.py                      │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│                2. Feature Engineering                        │
│         src/mlops/feature_store.py (NEW!)                   │
│         - Consistent preprocessing                          │
│         - Feature versioning                                │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│                  3. Model Training                           │
│         src/training/train_bert.py                          │
│         + MLflow Tracking (NEW!)                            │
│         + Experiment logging                                │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│                4. Automated Testing                          │
│         tests/test_*.py (NEW!)                              │
│         .github/workflows/ml-ci-cd.yml                      │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│                  5. Model Registry                           │
│         MLflow Model Registry (NEW!)                        │
│         - Version control                                   │
│         - Staging → Production                              │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│                 6. Model Serving                             │
│         src/api/api_server.py (NEW!)                        │
│         - REST API dengan FastAPI                           │
│         - Health check & metrics                            │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│              7. Monitoring & Drift Detection                 │
│         src/mlops/drift_detection.py (NEW!)                 │
│         - Data drift detection                              │
│         - Model drift monitoring                            │
│         - Prediction logging                                │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│              8. Automated Retraining                         │
│         src/mlops/retraining_pipeline.py (NEW!)             │
│         - Trigger evaluation                                │
│         - Auto-retraining                                   │
│         - Model validation                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Monitoring Metrics

### API Metrics (Prometheus)
- `sentiment_predictions_total` - Total predictions
- `sentiment_prediction_latency_seconds` - Latency
- `sentiment_prediction_confidence` - Average confidence
- `sentiment_prediction_errors_total` - Error count

### Model Metrics (Grafana)
- Train vs Test Accuracy
- Precision, Recall, F1 Score
- Overfitting Gap
- Sentiment Distribution

### Drift Metrics
- Data Drift Score
- Performance Degradation
- Error Rate from User Feedback

---

## 🔄 Retraining Triggers

| Trigger | Threshold | Priority |
|---------|-----------|----------|
| **Time-based** | Max 30 days | High |
| **New data** | 500+ new reviews | Medium |
| **User feedback** | Error rate > 12% | High |
| **Performance** | Accuracy drop > 3% | High |
| **Data drift** | Drift score > 0.3 | Medium |

---

## 📝 File Structure (MLOps Components)

```
SentimentProjek/
├── src/
│   ├── mlops/                    # NEW! MLOps modules
│   │   ├── mlflow_manager.py    # Model versioning & tracking
│   │   ├── drift_detection.py   # Data/Model drift detection
│   │   ├── retraining_pipeline.py  # Auto-retraining
│   │   └── feature_store.py     # Feature management
│   ├── api/                      # NEW! Model serving
│   │   └── api_server.py        # FastAPI server
│   ├── training/
│   │   └── train_bert.py        # With MLflow integration
│   └── ...
├── tests/                        # NEW! Automated tests
│   ├── test_data_validation.py
│   ├── test_model_validation.py
│   └── test_integration.py
├── .github/
│   └── workflows/
│       └── ml-ci-cd.yml         # NEW! CI/CD pipeline
├── scripts/
│   └── test_mlops_features.py   # NEW! Test script
├── mlruns/                       # MLflow tracking data
└── logs/
    └── retraining/              # Retraining logs
```

---

## ✅ Testing Checklist

Before deploying to production:

- [ ] Run `python scripts/test_mlops_features.py` - All tests pass
- [ ] Run `pytest tests/ -v` - All unit tests pass
- [ ] Check MLflow UI - Latest model logged with metrics
- [ ] Test API - Health check returns healthy
- [ ] Check Grafana - Dashboards showing correct data
- [ ] Verify drift detection - No critical drifts
- [ ] Test retraining triggers - Evaluation works correctly
- [ ] Check feature store - Features extracted correctly

---

## 🆘 Troubleshooting

### MLflow UI not starting
```bash
# Check port is free
netstat -an | findstr "5000"

# Start with specific host
mlflow ui --host 127.0.0.1 --port 5000
```

### API server errors
```bash
# Check logs
docker logs sentiment_api

# Rebuild container
docker-compose stop api
docker-compose build api
docker-compose up -d api
```

### Tests failing
```bash
# Check database connection
docker exec -it sentiment_postgres psql -U sentiment_user -d sentiment_db -c "\dt"

# Install missing dependencies
pip install -r requirements.txt
```

### Feature store initialization fails
```bash
# Check if tables exist
python -c "from src.mlops.drift_detection import create_prediction_logs_table; create_prediction_logs_table(db_config)"
```

---

## 🎓 Learning Resources

- **MLflow**: https://mlflow.org/docs/latest/index.html
- **FastAPI**: https://fastapi.tiangolo.com/
- **Pytest**: https://docs.pytest.org/
- **GitHub Actions**: https://docs.github.com/en/actions

---

## 📈 Next Steps

### Short-term (Week 1-2)
- [ ] Setup user feedback table untuk retraining
- [ ] Configure Slack/Email notifications
- [ ] Setup scheduled retraining (daily check)

### Medium-term (Month 1)
- [ ] Implement A/B testing framework
- [ ] Add model explainability (SHAP/LIME)
- [ ] Setup production monitoring alerts

### Long-term (Month 2-3)
- [ ] Multi-model ensemble
- [ ] Automated hyperparameter tuning
- [ ] Advanced drift detection algorithms

---

**Made with ❤️ for MLOps Best Practices**

Last updated: 2025-01-11
