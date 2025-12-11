# 📊 Grafana Dashboard - Panduan Lengkap

## Dashboard Layout Baru

Dashboard sekarang menampilkan **13 panels**:

### Row 1: Overview Metrics (4 panels)
1. **Total Reviews** - Total semua review
2. **Predicted Reviews** - Review yang sudah diprediksi
3. **Average Rating** - Rating rata-rata
4. **Unpredicted Reviews** - Review yang belum diprediksi

### Row 2: Sentiment Breakdown (3 panels)
5. **Positive Reviews** - Jumlah review positif (hijau)
6. **Negative Reviews** - Jumlah review negatif (merah)
7. **Neutral Reviews** - Jumlah review netral (kuning)

### Row 2 (kanan): Model Performance (4 panels)
9. **Model Accuracy** - Akurasi model (92%)
10. **Precision** - Precision score (91%)
11. **Recall** - Recall score (93%)
12. **F1 Score** - F1 score (92%)

### Row 3: Visualizations (2 panels)
8. **Sentiment Distribution** - Pie chart distribusi sentiment
13. **Rating Distribution** - Bar gauge distribusi rating 1-5

---

## 🎯 Cara Update Model Metrics

Setelah training atau evaluasi model baru, update metrics di database:

### 1. Via Python Script

```bash
python src/monitoring/update_model_metrics.py \
  --model "bert-base-multilingual" \
  --accuracy 0.92 \
  --precision 0.91 \
  --recall 0.93 \
  --f1 0.92
```

### 2. Via SQL Direct

```sql
INSERT INTO model_metrics 
(model_name, accuracy, precision_score, recall_score, f1_score)
VALUES ('bert-base-multilingual', 0.92, 0.91, 0.93, 0.92);
```

### 3. Dari Container

```powershell
docker exec sentiment_postgres psql -U sentiment_user -d sentiment_db -c "
INSERT INTO model_metrics 
(model_name, accuracy, precision_score, recall_score, f1_score)
VALUES ('bert-base-multilingual', 0.92, 0.91, 0.93, 0.92);
"
```

---

## 📈 Cara Integrasi dengan Training Pipeline

### Option 1: Update di Akhir Training

Tambahkan di `batch_predict.py` atau script training:

```python
import psycopg2
from datetime import datetime
import os

def save_model_metrics(accuracy, precision, recall, f1_score):
    conn = psycopg2.connect(
        host=os.getenv('POSTGRES_HOST', 'localhost'),
        port=os.getenv('POSTGRES_PORT', '5432'),
        database=os.getenv('POSTGRES_DB', 'sentiment_db'),
        user=os.getenv('POSTGRES_USER', 'sentiment_user'),
        password=os.getenv('POSTGRES_PASSWORD', 'password')
    )
    
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO model_metrics 
        (model_name, accuracy, precision_score, recall_score, f1_score)
        VALUES (%s, %s, %s, %s, %s)
    """, ('bert-base-multilingual', accuracy, precision, recall, f1_score))
    
    conn.commit()
    cursor.close()
    conn.close()
    
    print(f"✓ Model metrics saved to database")

# Setelah evaluasi model
accuracy = 0.92
precision = 0.91
recall = 0.93
f1 = 0.92

save_model_metrics(accuracy, precision, recall, f1)
```

### Option 2: Auto-Calculate dari Validation Set

Jika punya validation/test set dengan ground truth:

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Asumsi: y_true dan y_pred sudah ada
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

save_model_metrics(accuracy, precision, recall, f1)
```

---

## 🔍 Query untuk Sentiment Counts

Dashboard menggunakan query berikut:

### Positive Reviews
```sql
SELECT COUNT(*) as value 
FROM reviews 
WHERE sentiment = 'positive'
```

### Negative Reviews
```sql
SELECT COUNT(*) as value 
FROM reviews 
WHERE sentiment = 'negative'
```

### Neutral Reviews
```sql
SELECT COUNT(*) as value 
FROM reviews 
WHERE sentiment = 'neutral'
```

---

## 📊 Model Metrics Schema

Tabel `model_metrics`:

| Column | Type | Description |
|--------|------|-------------|
| id | SERIAL | Primary key |
| model_name | VARCHAR(100) | Nama model |
| accuracy | FLOAT | Akurasi (0-1) |
| precision_score | FLOAT | Precision (0-1) |
| recall_score | FLOAT | Recall (0-1) |
| f1_score | FLOAT | F1 Score (0-1) |
| created_at | TIMESTAMP | Waktu insert |

---

## 🎨 Warna Panel

- **Positive Reviews**: 🟢 Green
- **Negative Reviews**: 🔴 Red
- **Neutral Reviews**: 🟡 Yellow
- **Accuracy**: 🔵 Blue
- **Precision**: 🟣 Purple
- **Recall**: 🟠 Orange
- **F1 Score**: 🟢 Green

---

## 📝 View Metrics History

Lihat history metrics dari berbagai model:

```sql
SELECT 
    model_name,
    accuracy,
    precision_score,
    recall_score,
    f1_score,
    created_at
FROM model_metrics
ORDER BY created_at DESC;
```

---

## 🔄 Auto-Refresh

Dashboard akan auto-refresh **setiap 30 detik** untuk menampilkan:
- Data review terbaru
- Sentiment counts terkini
- Model metrics terbaru

---

## 🚀 Quick Commands

### View Current Metrics
```powershell
docker exec sentiment_postgres psql -U sentiment_user -d sentiment_db -c "
SELECT * FROM model_metrics ORDER BY created_at DESC LIMIT 1;
"
```

### View Sentiment Counts
```powershell
docker exec sentiment_postgres psql -U sentiment_user -d sentiment_db -c "
SELECT sentiment, COUNT(*) as count 
FROM reviews 
WHERE sentiment IS NOT NULL 
GROUP BY sentiment;
"
```

### Clear Old Metrics (keep last 10)
```sql
DELETE FROM model_metrics 
WHERE id NOT IN (
    SELECT id FROM model_metrics 
    ORDER BY created_at DESC 
    LIMIT 10
);
```

---

## 📞 Troubleshooting

### Panel Shows "No data"

1. **Cek data di database**:
```powershell
docker exec sentiment_postgres psql -U sentiment_user -d sentiment_db -c "SELECT COUNT(*), sentiment FROM reviews GROUP BY sentiment;"
```

2. **Cek model metrics**:
```powershell
docker exec sentiment_postgres psql -U sentiment_user -d sentiment_db -c "SELECT * FROM model_metrics ORDER BY created_at DESC LIMIT 1;"
```

3. **Refresh dashboard** (F5)

### Metrics tidak update

- Dashboard akan menampilkan metrics **paling baru** (ORDER BY created_at DESC LIMIT 1)
- Setelah insert metrics baru, tunggu 30 detik atau refresh manual

---

**Dashboard URL**: http://localhost:3000/d/sentiment-dashboard-v3/sentiment-analysis-dashboard
