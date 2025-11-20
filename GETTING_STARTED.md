# 🎯 Sentiment Analysis MLOps - Proyek Lengkap

Selamat! Anda telah berhasil membuat **Sentiment Analysis MLOps Project** yang lengkap! 🎉

## 📋 Yang Telah Dibuat

### 1. ✅ **Data Collection**
- `src/data_collection/scraper.py` - Google Play Store scraper
- `src/data_collection/database.py` - Database manager (PostgreSQL & MongoDB)
- Otomatis scraping review dari app Pintu

### 2. ✅ **Data Preprocessing**
- `src/preprocessing/preprocess.py` - Text preprocessing untuk Bahasa Indonesia
- Support stemming, stopwords removal, slang normalization
- Feature engineering

### 3. ✅ **Model Training**
- `src/training/train.py` - Training pipeline dengan MLflow tracking
- `src/training/evaluate.py` - Model evaluation
- Support multiple models: Logistic Regression, Naive Bayes, SVM, Random Forest
- DVC integration untuk versioning

### 4. ✅ **Prediction Pipeline**
- `src/prediction/predict.py` - Batch prediction
- Auto-prediction untuk review baru
- Database integration

### 5. ✅ **Scheduler**
- `src/scheduler/main.py` - APScheduler untuk automated tasks
- Periodic scraping
- Auto-prediction
- Model retraining check

### 6. ✅ **Monitoring & Dashboard**
- Grafana dashboard configuration
- Real-time sentiment visualization
- Prometheus metrics
- PostgreSQL datasource

### 7. ✅ **Docker & Deployment**
- `Dockerfile` - Application container
- `docker-compose.yml` - Multi-service orchestration
  - PostgreSQL
  - MongoDB
  - Grafana
  - Prometheus
  - App container

### 8. ✅ **MLOps Tools**
- `dvc.yaml` - DVC pipeline definition
- `params.yaml` - Centralized parameters
- Pipeline tracking dan versioning

### 9. ✅ **CLI & Utilities**
- `cli.py` - Command-line interface
- `quickstart.py` - Easy setup script
- `Makefile` - Common tasks automation

### 10. ✅ **Documentation**
- `README.md` - Project overview
- `SETUP.md` - Detailed setup instructions
- Comprehensive comments dalam code

## 🚀 Cara Memulai

### Opsi 1: Quick Start (Paling Mudah)
```powershell
python quickstart.py
```
Pilih mode yang sesuai dan ikuti instruksi.

### Opsi 2: Step-by-Step
```powershell
# 1. Setup environment
cp .env.example .env
# Edit .env dengan konfigurasi Anda

# 2. Install dependencies
pip install -r requirements.txt

# 3. Initialize project
python cli.py init

# 4. Scrape data
python cli.py scrape

# 5. Preprocess
python cli.py preprocess

# 6. Train model
python cli.py train

# 7. Start scheduler (auto-pilot)
python src/scheduler/main.py
```

### Opsi 3: Docker (Production)
```powershell
# 1. Setup .env
cp .env.example .env

# 2. Start containers
docker-compose up -d --build

# 3. Access Grafana
# http://localhost:3000 (admin/admin)
```

## 📊 Dashboard Grafana

Dashboard akan menampilkan:
- **Total reviews** yang telah diproses
- **Sentiment distribution** (Positive/Negative/Neutral)
- **Rating distribution** dari 1-5 bintang
- **Sentiment trend over time** - grafik time series
- **Recent reviews** dengan sentiment prediction
- **Average sentiment score** per hari

Dashboard akan **auto-refresh setiap 30 detik** dan menampilkan data terbaru!

## 🔄 Workflow Otomatis

Scheduler akan menjalankan:
1. **Scraping** - Setiap 6 jam (default)
2. **Prediction** - Setiap 1 jam (default)
3. **Stats logging** - Setiap 1 jam
4. **Retraining check** - Setiap 7 hari (default)

Semua interval bisa diubah di `params.yaml`

## 📁 Struktur Project

```
SentimentProjek/
├── src/                    # Source code
│   ├── data_collection/    # Scraping & database
│   ├── preprocessing/      # Text preprocessing
│   ├── training/          # Model training
│   ├── prediction/        # Prediction pipeline
│   ├── scheduler/         # Automated tasks
│   └── utils.py           # Utility functions
├── data/                  # Data storage
│   ├── raw/              # Raw data
│   ├── processed/        # Processed data
│   └── predictions/      # Predictions
├── models/               # Trained models
├── notebooks/            # Jupyter notebooks
├── grafana/             # Grafana config
├── prometheus/          # Prometheus config
├── logs/                # Application logs
├── docker-compose.yml   # Docker services
├── Dockerfile          # App container
├── dvc.yaml           # DVC pipeline
├── params.yaml        # Parameters
├── cli.py             # CLI tool
├── quickstart.py      # Quick start
└── Makefile          # Task automation
```

## 🎯 Fitur Utama

### 1. **Automated Data Collection**
- Scraping otomatis dari Google Play Store
- Incremental update (tidak duplicate)
- Error handling dan retry logic

### 2. **Smart Preprocessing**
- Bahasa Indonesia support
- Slang normalization
- Stopwords removal
- Stemming dengan Sastrawi

### 3. **ML Pipeline dengan DVC**
- Version control untuk data & models
- Reproducible experiments
- Metrics tracking

### 4. **Real-time Dashboard**
- Live sentiment visualization
- Auto-refresh setiap 30 detik
- Multiple visualization types

### 5. **Production-Ready**
- Docker containerization
- Database untuk persistence
- Monitoring dengan Prometheus
- Logging comprehensive

## 🛠 Customization

### Ubah Model
Edit `params.yaml`:
```yaml
training:
  model_type: logistic_regression  # atau: naive_bayes, svm, random_forest
```

### Ubah Schedule
Edit `params.yaml`:
```yaml
scheduler:
  scraping_interval_hours: 6
  prediction_interval_hours: 1
  model_retrain_days: 7
```

### Ubah Preprocessing
Edit `params.yaml`:
```yaml
preprocessing:
  min_text_length: 10
  remove_stopwords: true
  stem: true
```

## 📈 Monitoring

### CLI Commands
```powershell
# Check statistics
python cli.py stats

# View project info
python cli.py --help
```

### Docker Logs
```powershell
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f app
docker-compose logs -f grafana
```

### Grafana Dashboard
Akses: http://localhost:3000
- Username: `admin`
- Password: dari `.env` (default: `admin`)

## 🔧 Development

### Experiment dengan Jupyter
```powershell
jupyter notebook notebooks/sentiment_analysis_experiment.ipynb
```

### Run Tests
```powershell
# Test individual components
python src/data_collection/scraper.py
python src/preprocessing/preprocess.py
python src/training/train.py
python src/prediction/predict.py --mode test
```

### DVC Pipeline
```powershell
# Run full pipeline
dvc repro

# Check status
dvc status

# View metrics
dvc metrics show

# Compare experiments
dvc params diff
dvc metrics diff
```

## 🎓 Learning Resources

Project ini mengimplementasikan best practices untuk:
- **MLOps**: Pipeline automation, versioning, monitoring
- **Data Engineering**: ETL, batch processing, database design
- **Machine Learning**: Sentiment analysis, NLP for Indonesian
- **DevOps**: Docker, containerization, CI/CD ready
- **Software Engineering**: Clean code, modular design, CLI tools

## 🐛 Troubleshooting

Lihat `SETUP.md` untuk detailed troubleshooting guide.

Common issues:
- Database connection → Check if containers are running
- NLTK data missing → Run NLTK downloads
- Out of memory → Reduce batch size
- Grafana not showing data → Check datasource connection

## 📞 Next Steps

1. ✅ **Setup project** - Gunakan `quickstart.py`
2. ✅ **Scrape initial data** - Run scraper
3. ✅ **Train model** - Train sentiment model
4. ✅ **Start scheduler** - Enable auto-pilot
5. ✅ **Monitor dashboard** - Watch in real-time
6. 🚀 **Scale & optimize** - Tune untuk production

## 🎉 Selamat!

Anda sekarang memiliki **production-ready sentiment analysis MLOps system** yang:
- ✅ Otomatis scrape data dari Google Play Store
- ✅ Preprocess teks Bahasa Indonesia
- ✅ Train dan evaluate ML models
- ✅ Predict sentiment secara real-time
- ✅ Visualize hasil di Grafana dashboard
- ✅ Track dengan DVC
- ✅ Deploy dengan Docker
- ✅ Monitor dengan Prometheus

**Happy analyzing! 🚀📊🎯**
