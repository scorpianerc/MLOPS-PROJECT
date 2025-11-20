# Sentiment Analysis MLOps Project

Project MLOps untuk analisis sentiment review aplikasi Pintu dari Google Play Store dengan monitoring real-time menggunakan Grafana.

## Fitur

- 🔄 **Auto Data Collection**: Scraping otomatis review dari Google Play Store
- 🤖 **ML Pipeline**: Training dan prediction dengan tracking DVC
- 📊 **Real-time Dashboard**: Grafana dashboard yang update otomatis
- ⏰ **Scheduler**: APScheduler untuk menjalankan pipeline secara berkala
- 🐳 **Docker**: Containerized application untuk easy deployment
- 📈 **Monitoring**: Prometheus metrics untuk monitoring performa

## Struktur Project

```
SentimentProjek/
├── data/
│   ├── raw/              # Raw data dari scraping
│   ├── processed/        # Data setelah preprocessing
│   └── predictions/      # Hasil prediksi
├── models/               # Trained models
├── notebooks/            # Jupyter notebooks untuk eksperimen
├── src/
│   ├── data_collection/  # Scraping scripts
│   ├── preprocessing/    # Data cleaning & feature engineering
│   ├── training/         # Model training
│   ├── prediction/       # Inference pipeline
│   └── monitoring/       # Metrics & monitoring
├── config/               # Configuration files
├── docker/               # Dockerfile dan docker-compose
├── grafana/              # Grafana dashboards & datasources
└── prometheus/           # Prometheus configuration

```

## Quick Start

1. **Setup Environment**
```bash
cp .env.example .env
# Edit .env dengan konfigurasi Anda
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Initialize DVC**
```bash
dvc init
```

4. **Run with Docker**
```bash
docker-compose up -d
```

5. **Access Dashboard**
- Grafana: http://localhost:3000 (admin/admin)
- Prometheus: http://localhost:9090

## Usage

### Manual Scraping
```bash
python src/data_collection/scraper.py
```

### Train Model
```bash
python src/training/train.py
```

### Run Prediction Pipeline
```bash
python src/prediction/predict.py
```

### Start Scheduler
```bash
python src/scheduler/main.py
```

## Architecture

```
Google Play Store → Scraper → MongoDB → Preprocessing → Model → PostgreSQL → Grafana
                                                          ↓
                                                         DVC
```

## Monitoring

Dashboard Grafana menampilkan:
- Total reviews processed
- Sentiment distribution (Positive/Negative/Neutral)
- Sentiment trend over time
- Word clouds
- Model performance metrics

## License

MIT License
