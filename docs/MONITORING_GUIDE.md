# 📊 Monitoring & Dashboard Guide

## 1. Grafana Dashboard Setup

### 📥 Import Dashboard

1. **Buka Grafana**
   ```
   http://localhost:3000
   ```

2. **Login**
   - Username: `admin`
   - Password: `admin123`

3. **Import Sentiment Dashboard**
   - Klik icon **"+"** di sidebar kiri
   - Pilih **"Import dashboard"**
   - Klik **"Upload JSON file"**
   - Pilih file: `grafana/dashboards/sentiment-dashboard.json`
   - Pada dropdown "Datasource", pilih **"PostgreSQL"**
   - Klik **"Import"**

4. **Import Prometheus Dashboard (Optional)**
   - Ulangi langkah 3
   - Pilih file: `grafana/dashboards/prometheus-dashboard.json`
   - Datasource: **"Prometheus"**

---

## 2. Dashboard yang Tersedia

### 📈 Sentiment Dashboard (`sentiment-dashboard.json`)
**Gunakan ini untuk monitoring sentiment analysis (SQL queries)**

**Panels:**
- **Total Reviews**: Jumlah total review di database
- **Sentiment Distribution**: Pie chart Positive/Neutral/Negative
- **Average Rating**: Rating rata-rata
- **Reviews Over Time**: Timeline jumlah review per hari
- **Sentiment Timeline**: Trend sentiment positif vs negatif
- **Rating Distribution**: Bar chart rating 1-5 bintang
- **Top Positive Reviews**: Review dengan rating tertinggi
- **Top Negative Reviews**: Review dengan rating terendah

**Data Source**: PostgreSQL (`sentiment_db`) - Direct SQL queries

---

### 🔥 Prometheus Dashboard (`prometheus-dashboard.json`)
**Gunakan ini untuk monitoring metrics real-time (Time Series)**

**Panels:**
- **Total Reviews**: Real-time count dari Prometheus
- **Positive/Negative/Neutral Reviews**: Live counters
- **Average Rating**: Current average rating
- **Sentiment Percentage**: Positive/Negative/Neutral %
- **Predicted vs Unpredicted**: Reviews dengan/tanpa prediksi
- **Average Thumbs Up**: Rata-rata like count
- **Review Trends**: Time series graph

**Data Source**: Prometheus - Metrics dari exporter (port 8000)

---

### 🎯 Perbedaan Kedua Dashboard

| Feature | Sentiment Dashboard | Prometheus Dashboard |
|---------|-------------------|---------------------|
| Data Source | PostgreSQL | Prometheus |
| Query Type | SQL | PromQL |
| Update Method | Query on refresh | Time series metrics |
| Best For | Detailed queries, raw data | Real-time monitoring, trends |
| Data Retention | Unlimited (PostgreSQL) | 15 days (Prometheus) |
| Use Case | Analysis, reports | Live monitoring, alerts |

**Rekomendasi**: Import **KEDUA DASHBOARD** untuk monitoring lengkap!

---

## 3. Automated Pipeline Flow

### 🔄 Scheduler Automation

**File**: `src/scheduler/main.py`

```
┌─────────────────────────────────────────────────┐
│           SCHEDULER (Docker Container)          │
└─────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│  STEP 1: scrape_and_store()                     │
│  - Scrape reviews dari Google Play Store        │
│  - Save ke PostgreSQL via DatabaseManager       │
│  - Save ke MongoDB                               │
│  - Save CSV backup (data/raw/reviews.csv)       │
│  - **NEW**: Run load_to_db.py untuk sync CSV    │
└─────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│  STEP 2: run_predictions()                      │
│  - **NEW**: Run batch_predict.py                │
│  - Predict sentiment untuk unpredicted reviews  │
│  - Update PostgreSQL dengan hasil prediksi      │
└─────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│  STEP 3: Grafana Auto-Update                    │
│  - Grafana query PostgreSQL                     │
│  - Dashboard refresh otomatis setiap 5 detik    │
│  - Metrics ter-update real-time                 │
└─────────────────────────────────────────────────┘
```

---

### ⏰ Schedule Configuration

**File**: `params.yaml`

```yaml
scheduler:
  scraping_interval_hours: 24     # Scrape setiap 24 jam
  prediction_interval_hours: 6    # Predict setiap 6 jam
  model_retrain_days: 7           # Check retraining setiap 7 hari
```

**Customize schedule:**
- Edit `params.yaml`
- Restart scheduler: `docker-compose restart scheduler`

---

## 4. Manual Operations

### 🔧 Run Tasks Manually

**Load CSV to Database:**
```bash
python src/data_collection/load_to_db.py
```

**Predict Sentiments:**
```bash
python src/monitoring/batch_predict.py
```

**Check Stats:**
```bash
docker-compose logs scheduler -f
```

---

## 5. Arsitektur Data Storage

### 📊 Mengapa Training Results Tidak di Database?

```
┌─────────────────────────────────────────────────┐
│           MLOps Arsitektur Terpisah             │
└─────────────────────────────────────────────────┘

┌──────────────────────────┐  ┌──────────────────────────┐
│     MLflow (Training)    │  │  PostgreSQL (Production) │
├──────────────────────────┤  ├──────────────────────────┤
│ • Model accuracy         │  │ • Review data            │
│ • Precision, recall, F1  │  │ • User name, text        │
│ • Hyperparameters        │  │ • Rating, thumbs_up      │
│ • Model versions         │  │ • Sentiment predictions  │
│ • Experiment tracking    │  │ • Predicted_at timestamp │
│ • Model artifacts (.pkl) │  │                          │
└──────────────────────────┘  └──────────────────────────┘
         ▲                              ▲
         │                              │
    Development                    Production
    (Evaluation)                   (Serving)
```

**Alasan Pemisahan:**

1. **MLflow - Experiment Tracking**
   - Training metrics hanya relevan saat development
   - Untuk compare model performance
   - Version control model
   - Tidak dipakai oleh aplikasi production

2. **PostgreSQL - Production Data**
   - Data yang dipakai oleh Grafana
   - Data yang dipakai oleh Streamlit
   - Data untuk monitoring real-time
   - Data untuk API serving

3. **Separation of Concerns**
   - Development ≠ Production
   - Training metrics ≠ Prediction results
   - Model evaluation ≠ Model inference

**Jika Anda Ingin Melihat Training Results:**
```bash
# Start MLflow UI
mlflow ui --port 5000

# Open browser
http://localhost:5000
```

---

## 6. Database Schema

### 🗄️ PostgreSQL `reviews` Table

```sql
CREATE TABLE reviews (
    id SERIAL PRIMARY KEY,
    review_id VARCHAR(255) UNIQUE,
    app_id VARCHAR(100),
    user_name VARCHAR(255),
    review_text TEXT,
    rating INTEGER,
    thumbs_up INTEGER,
    app_version VARCHAR(50),
    review_date TIMESTAMP,
    scraped_at TIMESTAMP,
    
    -- Sentiment Analysis Results
    sentiment VARCHAR(20),        -- 'positive', 'negative', 'neutral'
    sentiment_score FLOAT,        -- Confidence score
    predicted_at TIMESTAMP        -- When prediction was made
);
```

---

## 7. Monitoring Checklist

### ✅ Verifikasi Pipeline Berjalan

1. **Check Scheduler Logs**
   ```bash
   docker-compose logs scheduler -f
   ```

2. **Check Database**
   ```bash
   docker exec -it sentiment_postgres psql -U sentiment_user -d sentiment_db -c "SELECT COUNT(*) FROM reviews;"
   ```

3. **Check Grafana Dashboard**
   - Buka http://localhost:3000
   - Pastikan "Total Reviews" > 0
   - Pastikan "Sentiment Distribution" terisi

4. **Check Prometheus Metrics**
   - Buka http://localhost:9090
   - Query: `sentiment_total_reviews`

---

## 8. Troubleshooting

### ❌ Dashboard Shows "No Data"

1. **Check Database Connection**
   ```bash
   docker-compose logs grafana | grep -i error
   ```

2. **Check Datasource Configuration**
   - Grafana → Configuration → Data Sources
   - PostgreSQL should be green (default)

3. **Check Data Exists**
   ```bash
   python src/monitoring/batch_predict.py
   ```

### ❌ Scheduler Not Running

1. **Check Container Status**
   ```bash
   docker-compose ps
   ```

2. **Restart Scheduler**
   ```bash
   docker-compose restart scheduler
   ```

3. **Check Logs**
   ```bash
   docker-compose logs scheduler --tail=100
   ```

---

## 9. Next Steps

### 🚀 Production Recommendations

1. **Add Authentication**
   - Secure Grafana with real credentials
   - Use secrets management

2. **Add Alerting**
   - Grafana alerts for anomalies
   - Email/Slack notifications

3. **Add Data Retention**
   - Archive old reviews
   - Clean up old predictions

4. **Add Model Versioning**
   - Track model changes in database
   - Add model_version column

5. **Add API Monitoring**
   - Track API response times
   - Monitor error rates

---

## 📞 Support

**Dokumentasi Terkait:**
- `README.md` - Setup guide
- `docs/ARCHITECTURE.md` - System architecture
- `params.yaml` - Configuration

**Logs Location:**
- Scheduler: `logs/scheduler.log`
- Grafana: `docker-compose logs grafana`
- PostgreSQL: `docker-compose logs postgres`
