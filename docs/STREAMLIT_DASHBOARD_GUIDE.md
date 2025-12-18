# Streamlit Dashboard Guide

## Overview
Dashboard Streamlit yang telah diperbaiki dengan koneksi langsung ke PostgreSQL, responsive design, dan tanpa emoji/emoticon.

## Fitur Perbaikan

### 1. **Database Connection**
- ✓ Koneksi langsung ke PostgreSQL (bukan file CSV)
- ✓ Load data dari tabel `reviews`
- ✓ Load metrics dari tabel `model_metrics`
- ✓ Real-time data dengan caching 30 detik

### 2. **Icons & UI**
- ✓ Emoji diganti dengan simbol text-based:
  - `★` untuk rating (bukan ⭐)
  - `◉ ▲ ≡ ◐` untuk tabs (bukan 📊📈📝☁️)
  - `↻` untuk refresh (bukan 🔄)
  - `⬇` untuk download (bukan 📥)
  - `⚡` untuk predict (bukan 🔮)
  - `←` `→` untuk pagination (bukan ◀▶)

### 3. **Refresh Data - FIXED**
```python
# Tombol refresh sekarang berfungsi dengan benar:
if st.sidebar.button("↻ Refresh Data"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.session_state.last_refresh = datetime.now()
    st.success("Data refreshed successfully!")
    time.sleep(0.5)
    st.rerun()
```

### 4. **Responsive Design**
- ✓ Mobile-friendly (< 480px)
- ✓ Tablet-friendly (< 768px)
- ✓ Desktop optimized (> 768px)
- ✓ Adaptive columns dan font sizes
- ✓ Review cards responsive

## Running Dashboard

### Metode 1: Direct Run (Recommended)
```bash
# Pastikan dalam direktori project
cd d:\MLOPS\SentimentProjek

# Jalankan streamlit
streamlit run app_streamlit.py
```

### Metode 2: Docker
```bash
# Dashboard sudah running di container
docker-compose up -d

# Akses: http://localhost:8501
```

### Metode 3: Python Environment
```bash
# Activate environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install streamlit pandas plotly psycopg2-binary python-dotenv

# Run
streamlit run app_streamlit.py
```

## Environment Variables
Dashboard membaca dari `.env`:
```env
POSTGRES_HOST=localhost  # atau 'postgres' jika di Docker
POSTGRES_PORT=5432
POSTGRES_DB=sentiment_db
POSTGRES_USER=sentiment_user
POSTGRES_PASSWORD=password
```

## Database Schema

### Table: `reviews`
```sql
- id (integer)
- review_text (text)
- rating (integer 1-5)
- review_date (timestamp)
- sentiment (text: positive/negative/neutral)
- confidence (float)
- created_at (timestamp)
```

### Table: `model_metrics`
```sql
- id (serial)
- model_name (varchar)
- accuracy (float) -- test accuracy
- precision_score (float) -- test precision
- recall_score (float) -- test recall
- f1_score (float) -- test f1
- train_accuracy (float)
- train_precision (float)
- train_recall (float)
- train_f1 (float)
- created_at (timestamp)
```

## Features

### 1. System Status Sidebar
- Database connection status
- Metrics loading status
- Last update timestamp

### 2. Filters
- Sentiment filter (multi-select)
- Rating range slider
- Export to CSV/JSON

### 3. Analytics Tabs

#### Tab 1: Distribution
- Key metrics cards (Total, Avg Rating, Accuracy, F1)
- Sentiment pie chart
- Rating bar chart

#### Tab 2: Trends
- Reviews over time line chart
- Daily review counts

#### Tab 3: Details
- Paginated review cards (8 per page)
- Star ratings display
- Sentiment badges with colors

#### Tab 4: Insights
- Word frequency analysis (top 20)
- Sentiment timeline area chart
- Sentiment vs Rating heatmap

### 4. Model Performance
- Test metrics display
- Train vs Test comparison chart
- Raw metrics JSON viewer
- Model type badge

### 5. Real-Time Prediction
- Text input for Indonesian reviews
- BERT model prediction
- Confidence gauge
- Sentiment result display

## Troubleshooting

### Error: Database Connection Failed
```bash
# Check Docker containers
docker ps

# Check PostgreSQL
docker exec -it sentiment_postgres psql -U sentiment_user -d sentiment_db

# Test connection
\dt
SELECT COUNT(*) FROM reviews;
```

### Error: Metrics Not Loading
```sql
-- Check if model_metrics table exists
SELECT * FROM model_metrics ORDER BY created_at DESC LIMIT 1;

-- If empty, run training
python src/training/train_bert.py
```

### Error: Refresh Button Not Working
- Sudah diperbaiki di versi terbaru
- Gunakan `st.rerun()` bukan `st.experimental_rerun()`

### Error: Import Error
```bash
# Install missing packages
pip install psycopg2-binary python-dotenv
```

## Performance Tips

1. **Caching**: Data di-cache 30 detik
   ```python
   @st.cache_data(ttl=30, show_spinner=False)
   ```

2. **Pagination**: Review cards dibatasi 8 per page

3. **Lazy Loading**: Charts hanya render saat tab dibuka

4. **Connection Pooling**: Database connection di-cache
   ```python
   @st.cache_resource
   def get_db_connection()
   ```

## Browser Compatibility
- ✓ Chrome/Edge (Recommended)
- ✓ Firefox
- ✓ Safari
- ✓ Mobile browsers

## Mobile Optimization
- Font sizes scaled down
- Columns collapse to single column
- Metrics cards responsive
- Charts full-width with touch gestures

## Color Scheme
- Positive: `#4CAF50` (Green)
- Negative: `#F44336` (Red)
- Neutral: `#9E9E9E` (Gray)
- Accent: `#2196F3` (Blue)

## Next Steps
1. Jalankan dashboard: `streamlit run app_streamlit.py`
2. Buka browser: http://localhost:8501
3. Test refresh button
4. Test filtering
5. Test prediction dengan review baru

## Notes
- Dashboard otomatis reload setiap 30 detik (cache TTL)
- Klik "↻ Refresh Data" untuk manual refresh
- Metrics berasal dari training terakhir di database
- Responsive design tested di Chrome DevTools
