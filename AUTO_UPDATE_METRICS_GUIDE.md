# 🔄 Auto-Update Model Metrics di Grafana

## 📌 Overview

Setelah training ulang model, metrics (accuracy, precision, recall, F1) akan **otomatis tersimpan ke database** dan **langsung muncul di Grafana dashboard**.

**🆕 NEW**: Dashboard sekarang menampilkan **Train & Test metrics terpisah** untuk detect overfitting!

---

## 🎯 Cara Kerja

```
Training Model → Calculate Metrics → Save ke Database → Grafana Auto-Refresh → Dashboard Update
```

### Flow Detail:
1. **Training**: Model ditraining dengan data terbaru
2. **Evaluation**: Metrics dihitung dari test set
3. **Save to DB**: Metrics disimpan ke tabel `model_metrics`
4. **Grafana Query**: Dashboard query metrics terbaru (ORDER BY created_at DESC LIMIT 1)
5. **Auto-Refresh**: Grafana refresh setiap 30 detik

---

## 🚀 Cara Training dengan Auto-Save Metrics

### Option 1: Gunakan Script BERT Training (RECOMMENDED)

```bash
python src/training/train_bert.py
```

Script ini **SUDAH OTOMATIS**:
- ✅ Load data dari `data/processed/processed_reviews.csv`
- ✅ Split train/test (dari params.yaml)
- ✅ Train BERT model (IndoBERT/multilingual)
- ✅ Calculate metrics on test set
- ✅ **Auto-save metrics ke database PostgreSQL** ⬅️ BARU DITAMBAHKAN!
- ✅ Save model ke `models/bert_model/`
- ✅ Save metrics ke `models/bert_metrics.json`

**Output Example:**
```
FINAL METRICS
==================================================
Test Accuracy: 0.9234
Test Precision: 0.9156
Test Recall: 0.9301
Test F1-Score: 0.9228

💾 Saving metrics to database for Grafana...
✅ Metrics saved to database!
📊 Dashboard akan menampilkan metrics terbaru dalam 30 detik
🔗 Dashboard: http://localhost:3000/d/sentiment-dashboard-v3/sentiment-analysis-dashboard
```

### Option 2: Manual Integration ke Training Script Lain

Tambahkan di akhir script training Anda:

```python
import psycopg2
from datetime import datetime
import os

def save_model_metrics(accuracy, precision, recall, f1_score):
    conn = psycopg2.connect(
        host=os.getenv('POSTGRES_HOST', 'localhost'),
        database=os.getenv('POSTGRES_DB', 'sentiment_db'),
        user=os.getenv('POSTGRES_USER', 'sentiment_user'),
        password=os.getenv('POSTGRES_PASSWORD', 'password')
    )
    
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO model_metrics 
        (model_name, accuracy, precision_score, recall_score, f1_score, created_at)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, ('your-model-name', accuracy, precision, recall, f1_score, datetime.now()))
    
    conn.commit()
    cursor.close()
    conn.close()

# Setelah evaluasi model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Auto-save ke database
save_model_metrics(accuracy, precision, recall, f1)
print(f"✅ Metrics saved! Accuracy: {accuracy:.4f}")
```

---

## 📊 Cek Metrics di Dashboard

### Via Grafana UI
1. Buka: http://localhost:3000/d/sentiment-dashboard-v3/sentiment-analysis-dashboard
2. Lihat panel:
   - **Model Accuracy** (biru)
   - **Precision** (ungu)
   - **Recall** (orange)
   - **F1 Score** (hijau)
3. Dashboard auto-refresh setiap 30 detik

### Via Database Query
```powershell
docker exec sentiment_postgres psql -U sentiment_user -d sentiment_db -c "
SELECT 
    model_name,
    ROUND(accuracy::numeric, 4) as accuracy,
    ROUND(precision_score::numeric, 4) as precision,
    ROUND(recall_score::numeric, 4) as recall,
    ROUND(f1_score::numeric, 4) as f1,
    created_at
FROM model_metrics
ORDER BY created_at DESC
LIMIT 5;
"
```

---

## 🔄 Workflow Training Lengkap

### Step 1: Training Model BERT
```bash
python src/training/train_bert.py
```
✅ Model training + **Auto-save metrics ke database**

### Step 2: Predict Reviews dengan Model Baru
```bash
python src/monitoring/batch_predict.py
```

### Step 3: Cek Dashboard
- Dashboard akan otomatis menampilkan:
  - ✅ Metrics terbaru (dari training baru)
  - ✅ Sentiment counts terbaru (dari batch predict)
  - ✅ Auto-refresh setiap 30 detik

---

## 📈 Track Metrics Over Time

### View Metrics History
```sql
SELECT 
    created_at,
    model_name,
    accuracy,
    precision_score,
    recall_score,
    f1_score
FROM model_metrics
ORDER BY created_at DESC;
```

### Compare dengan Training Sebelumnya
```sql
WITH latest AS (
    SELECT * FROM model_metrics ORDER BY created_at DESC LIMIT 1
),
previous AS (
    SELECT * FROM model_metrics ORDER BY created_at DESC LIMIT 1 OFFSET 1
)
SELECT 
    'Current' as version,
    latest.accuracy,
    latest.f1_score,
    latest.created_at
FROM latest
UNION ALL
SELECT 
    'Previous' as version,
    previous.accuracy,
    previous.f1_score,
    previous.created_at
FROM previous;
```

---

## 🎓 Example: Complete Training Flow

```bash
# 1. Training BERT model baru
python src/training/train_bert.py

# Output:
# FINAL METRICS
# ==================================================
# Test Accuracy: 0.9234
# Test Precision: 0.9156
# Test Recall: 0.9301
# Test F1-Score: 0.9228
# 
# 💾 Saving metrics to database for Grafana...
# ✅ Metrics saved to database!
# 📊 Dashboard akan menampilkan metrics terbaru dalam 30 detik

# 2. Predict unpredicted reviews
python src/monitoring/batch_predict.py

# Output:
# ✅ Updated 1000 sentiment predictions
# 💡 Note: Model metrics will auto-update in Grafana

# 3. Verify di Grafana (refresh browser)
# Dashboard akan menampilkan metrics terbaru:
# - Accuracy: 92.34%
# - Precision: 91.56%
# - Recall: 93.01%
# - F1 Score: 92.28%
```

---

## 🔧 Troubleshooting

### Metrics tidak update di Grafana

1. **Cek data di database**:
```powershell
docker exec sentiment_postgres psql -U sentiment_user -d sentiment_db -c "SELECT * FROM model_metrics ORDER BY created_at DESC LIMIT 1;"
```

2. **Refresh Grafana** (F5) atau tunggu 30 detik

3. **Cek query di panel**:
- Panel query: `SELECT accuracy FROM model_metrics ORDER BY created_at DESC LIMIT 1`
- Pastikan ada data di tabel

### Metrics masih menampilkan nilai lama

Dashboard **selalu menampilkan metrics terbaru**:
```sql
ORDER BY created_at DESC LIMIT 1
```

Jika masih nilai lama:
- Pastikan training berhasil save ke DB
- Refresh browser (Ctrl+Shift+R untuk hard refresh)
- Cek Grafana datasource connection

---

## 📝 Manual Update (Jika Perlu)

Jika ingin update manual tanpa training:

```bash
python src/monitoring/update_model_metrics.py \
  --model "bert-base-multilingual" \
  --accuracy 0.95 \
  --precision 0.94 \
  --recall 0.96 \
  --f1 0.95
```

Atau via SQL:
```powershell
docker exec sentiment_postgres psql -U sentiment_user -d sentiment_db -c "
INSERT INTO model_metrics (model_name, accuracy, precision_score, recall_score, f1_score)
VALUES ('my-new-model', 0.95, 0.94, 0.96, 0.95);
"
```

---

## ✅ Summary

| Action | Command | Result |
|--------|---------|--------|
| **Training Baru** | `python src/monitoring/train_model_with_metrics.py` | Model + Metrics auto-saved |
| **Predict Reviews** | `python src/monitoring/batch_predict.py` | Sentiment predictions updated |
| **View Dashboard** | http://localhost:3000/d/sentiment-dashboard-v3 | Metrics auto-displayed |
| **Check DB** | `docker exec ... psql -c "SELECT * FROM model_metrics"` | View metrics history |

**Metrics akan otomatis update di Grafana setiap kali Anda training ulang model!** 🎉
