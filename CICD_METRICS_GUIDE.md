# 🔄 CI/CD Integration - Auto Update Metrics

## ✅ Apa yang Sudah Diimplementasikan

### 1. **Auto-Save Metrics Setelah Training**

Script `train_bert.py` **OTOMATIS** menyimpan metrics ke database setelah training selesai:

```python
# ✅ AUTO-SAVE - Tidak perlu konfigurasi tambahan
python src/training/train_bert.py
```

**Yang Disimpan**:
- ✅ **Test Metrics**: Accuracy, Precision, Recall, F1
- ✅ **Train Metrics**: Train Accuracy, Train Precision, Train Recall, Train F1
- ✅ **Timestamp**: created_at (untuk tracking history)
- ✅ **Model Name**: Nama model untuk comparison

### 2. **Separate Train & Test Metrics di Dashboard**

Dashboard sekarang menampilkan **8 panels** untuk model metrics:

#### **Row 1: Test Metrics** (warna gelap)
- 🔵 Test Accuracy
- 🟣 Test Precision  
- 🟠 Test Recall
- 🟢 Test F1 Score

#### **Row 2: Train Metrics** (warna terang)
- 🔵 Train Accuracy (light-blue)
- 🟣 Train Precision (light-purple)
- 🟠 Train Recall (light-orange)
- 🟢 Train F1 Score (light-green)

**Benefit**:
- ✅ **Detect Overfitting**: Jika train metrics jauh lebih tinggi dari test
- ✅ **Model Quality**: Monitor generalization capability
- ✅ **CI/CD Tracking**: Track performance di setiap deployment

---

## 🚀 CI/CD Pipeline Integration

### GitHub Actions Example

```yaml
name: Train and Deploy Model

on:
  push:
    branches: [ main ]
  schedule:
    - cron: '0 0 * * 0'  # Weekly training

jobs:
  train:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v2
      
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      
      - name: Train Model
        env:
          POSTGRES_HOST: ${{ secrets.POSTGRES_HOST }}
          POSTGRES_USER: ${{ secrets.POSTGRES_USER }}
          POSTGRES_PASSWORD: ${{ secrets.POSTGRES_PASSWORD }}
          POSTGRES_DB: ${{ secrets.POSTGRES_DB }}
        run: |
          python src/training/train_bert.py
          # ✅ Metrics akan otomatis tersimpan ke database!
      
      - name: Upload Model Artifact
        uses: actions/upload-artifact@v2
        with:
          name: trained-model
          path: models/bert_model/
      
      - name: Notify Success
        run: |
          echo "✅ Model trained successfully!"
          echo "📊 Check Grafana: http://your-grafana-url/d/sentiment-dashboard-v3"
```

### GitLab CI Example

```yaml
stages:
  - train
  - deploy

train_model:
  stage: train
  image: python:3.9
  
  variables:
    POSTGRES_HOST: $POSTGRES_HOST
    POSTGRES_USER: $POSTGRES_USER
    POSTGRES_PASSWORD: $POSTGRES_PASSWORD
    POSTGRES_DB: $POSTGRES_DB
  
  script:
    - pip install -r requirements.txt
    - python src/training/train_bert.py
    # ✅ Auto-save metrics ke database
  
  artifacts:
    paths:
      - models/bert_model/
      - models/bert_metrics.json
    expire_in: 30 days
  
  only:
    - main
    - schedules
```

---

## 📊 Monitoring Train vs Test Metrics

### Apa yang Harus Dicek?

#### 1. **Overfitting Detection**
```sql
SELECT 
    model_name,
    ROUND((train_accuracy - accuracy)::numeric, 4) as overfitting_gap,
    ROUND(train_accuracy::numeric, 4) as train_acc,
    ROUND(accuracy::numeric, 4) as test_acc,
    created_at
FROM model_metrics
ORDER BY created_at DESC
LIMIT 5;
```

**Interpretation**:
- `overfitting_gap < 0.05` (5%) → ✅ Good generalization
- `overfitting_gap 0.05-0.10` (5-10%) → ⚠️ Slight overfitting
- `overfitting_gap > 0.10` (>10%) → ❌ Significant overfitting

#### 2. **Model Improvement Over Time**
```sql
SELECT 
    model_name,
    ROUND(accuracy::numeric, 4) as test_accuracy,
    ROUND(f1_score::numeric, 4) as test_f1,
    created_at
FROM model_metrics
ORDER BY created_at DESC
LIMIT 10;
```

#### 3. **Current Performance**
```sql
SELECT 
    'Test' as dataset,
    ROUND(accuracy::numeric, 4) as accuracy,
    ROUND(precision_score::numeric, 4) as precision,
    ROUND(recall_score::numeric, 4) as recall,
    ROUND(f1_score::numeric, 4) as f1
FROM model_metrics
ORDER BY created_at DESC LIMIT 1

UNION ALL

SELECT 
    'Train' as dataset,
    ROUND(train_accuracy::numeric, 4),
    ROUND(train_precision::numeric, 4),
    ROUND(train_recall::numeric, 4),
    ROUND(train_f1::numeric, 4)
FROM model_metrics
ORDER BY created_at DESC LIMIT 1;
```

---

## 🎯 Grafana Alerts (Optional)

### Alert 1: Overfitting Detection

**Condition**: `train_accuracy - test_accuracy > 0.10`

```yaml
alert:
  name: Model Overfitting Detected
  condition: |
    SELECT 
      CASE WHEN (train_accuracy - accuracy) > 0.10 THEN 1 ELSE 0 END as alert
    FROM model_metrics
    ORDER BY created_at DESC LIMIT 1
  notify: slack, email
  message: "⚠️ Model overfitting detected! Train accuracy significantly higher than test accuracy."
```

### Alert 2: Performance Degradation

**Condition**: `test_accuracy < 0.85`

```yaml
alert:
  name: Low Model Performance
  condition: |
    SELECT 
      CASE WHEN accuracy < 0.85 THEN 1 ELSE 0 END as alert
    FROM model_metrics
    ORDER BY created_at DESC LIMIT 1
  notify: slack, email
  message: "❌ Model performance below threshold! Test accuracy < 85%"
```

---

## 🔄 Automated Workflow

```
┌─────────────────────┐
│  GitHub/GitLab      │
│  Push/Schedule      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  CI/CD Pipeline     │
│  - Install deps     │
│  - Train model      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  train_bert.py      │
│  - Train BERT       │
│  - Calculate metrics│
└──────────┬──────────┘
           │
           ▼ 
┌─────────────────────┐
│  PostgreSQL         │
│  AUTO-SAVE:         │
│  - Test metrics     │
│  - Train metrics    │
│  - Timestamp        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Grafana Dashboard  │
│  AUTO-REFRESH:      │
│  - Test metrics     │
│  - Train metrics    │
│  - Overfitting gap  │
└─────────────────────┘
```

---

## 📝 Manual Testing

### 1. Train Model Locally
```bash
python src/training/train_bert.py
```

### 2. Check Database
```powershell
docker exec sentiment_postgres psql -U sentiment_user -d sentiment_db -c "
SELECT 
    model_name,
    ROUND(accuracy::numeric, 4) as test_acc,
    ROUND(train_accuracy::numeric, 4) as train_acc,
    ROUND((train_accuracy - accuracy)::numeric, 4) as overfitting,
    created_at
FROM model_metrics
ORDER BY created_at DESC
LIMIT 3;
"
```

### 3. View Dashboard
http://localhost:3000/d/sentiment-dashboard-v3/sentiment-analysis-dashboard

**Expected Output**:
- ✅ Test Accuracy: ~0.9234 (92.34%)
- ✅ Train Accuracy: ~0.9856 (98.56%)
- ✅ Overfitting Gap: ~0.0622 (6.22%) - Acceptable

---

## ✅ Summary

| Feature | Status | Description |
|---------|--------|-------------|
| **Auto-save after training** | ✅ Implemented | Tidak perlu kode tambahan |
| **Train metrics tracking** | ✅ Implemented | Accuracy, Precision, Recall, F1 |
| **Test metrics tracking** | ✅ Implemented | Accuracy, Precision, Recall, F1 |
| **Separate dashboard panels** | ✅ Implemented | 8 panels (4 test + 4 train) |
| **Overfitting detection** | ✅ SQL Query | Compare train vs test |
| **CI/CD ready** | ✅ Ready | Works dengan GitHub/GitLab |
| **History tracking** | ✅ Implemented | Timestamp-based |

---

## 🎉 CI/CD Siap Digunakan!

**Workflow**:
1. Push code → CI/CD trigger training
2. Training selesai → **Auto-save** train & test metrics
3. Grafana **auto-refresh** (30s) → Dashboard update
4. Team monitor **real-time** performance & overfitting

**No manual intervention needed!** 🚀
