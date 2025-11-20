# Panduan Setup - Sentiment Analysis MLOps Project

## Prerequisites

- Python 3.10+
- Docker & Docker Compose (untuk production)
- Git
- DVC (optional, akan di-install otomatis)

## Quick Start (Tercepat)

### Windows (PowerShell):
```powershell
# 1. Clone atau buka project
cd d:\MLOPS\SentimentProjek

# 2. Jalankan quick start script
python quickstart.py
```

Pilih mode yang sesuai:
- **Mode 1**: Local Development (tanpa Docker, untuk development)
- **Mode 2**: Production (dengan Docker, untuk deployment)
- **Mode 3**: Scrape data saja
- **Mode 4**: Train model saja

## Setup Manual

### 1. Install Dependencies

```powershell
# Install Python packages
pip install -r requirements.txt
pip install -r requirements-cli.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### 2. Setup Environment

```powershell
# Copy environment file
cp .env.example .env

# Edit .env dengan konfigurasi Anda
notepad .env
```

Isi minimal yang harus diatur:
```
POSTGRES_PASSWORD=your_strong_password
GRAFANA_ADMIN_PASSWORD=your_admin_password
```

### 3. Initialize DVC

```powershell
dvc init
```

### 4. Create Directories

```powershell
python cli.py init
```

## Menjalankan Project

### A. Mode Local (Development)

#### 1. Scrape Data
```powershell
# Scrape 500 reviews (default)
python cli.py scrape

# Atau dengan custom jumlah
python cli.py scrape --max-reviews 1000
```

#### 2. Preprocess Data
```powershell
python cli.py preprocess
```

#### 3. Train Model
```powershell
# Via CLI
python cli.py train

# Atau langsung
python src/training/train.py
```

#### 4. Run Predictions
```powershell
python cli.py predict --batch-size 100
```

#### 5. Check Statistics
```powershell
python cli.py stats
```

#### 6. Start Scheduler (Auto-pilot)
```powershell
python src/scheduler/main.py
```

### B. Mode Docker (Production)

#### 1. Build dan Start Containers
```powershell
docker-compose up -d --build
```

#### 2. Check Logs
```powershell
# Semua services
docker-compose logs -f

# Specific service
docker-compose logs -f app
docker-compose logs -f grafana
```

#### 3. Stop Containers
```powershell
docker-compose down
```

#### 4. Stop dan Hapus Data
```powershell
docker-compose down -v
```

## Menggunakan DVC Pipeline

### Run Full Pipeline
```powershell
dvc repro
```

### Run Specific Stage
```powershell
dvc repro training
```

### Check Pipeline Status
```powershell
dvc status
```

### View Metrics
```powershell
dvc metrics show
```

### Compare Experiments
```powershell
dvc params diff
dvc metrics diff
```

## Accessing Services

Setelah Docker containers berjalan:

- **Grafana Dashboard**: http://localhost:3000
  - Username: `admin`
  - Password: dari `.env` (default: `admin`)

- **Prometheus**: http://localhost:9090

- **PostgreSQL**: 
  - Host: `localhost`
  - Port: `5432`
  - Database: `sentiment_db`
  - Username/Password: dari `.env`

- **MongoDB**:
  - Host: `localhost`
  - Port: `27017`

## Development Workflow

### 1. Experiment dengan Jupyter
```powershell
jupyter notebook notebooks/sentiment_analysis_experiment.ipynb
```

### 2. Modify Parameters
Edit `params.yaml` untuk mengubah:
- Model parameters
- Preprocessing options
- Scraping configuration
- Scheduler intervals

### 3. Test Changes
```powershell
# Test scraping
python src/data_collection/scraper.py

# Test preprocessing
python src/preprocessing/preprocess.py

# Test training
python src/training/train.py

# Test prediction
python src/prediction/predict.py --mode test
```

### 4. Run DVC Pipeline
```powershell
dvc repro
```

## Monitoring

### View Dashboard
1. Buka browser: http://localhost:3000
2. Login dengan kredensial dari `.env`
3. Navigate ke "Sentiment Analysis Dashboard"

### Dashboard Panels:
- Total reviews processed
- Sentiment distribution (pie chart)
- Rating distribution
- Sentiment trend over time
- Recent reviews table
- Average sentiment score

## Troubleshooting

### Issue: DVC not found
```powershell
pip install dvc
```

### Issue: Database connection error
```powershell
# Check if containers are running
docker-compose ps

# Restart containers
docker-compose restart postgres mongodb
```

### Issue: Out of memory during scraping
Solution: Reduce `max_reviews` in scraping command atau dalam `params.yaml`

### Issue: NLTK data not found
```powershell
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### Issue: Grafana dashboard tidak muncul
```powershell
# Restart Grafana container
docker-compose restart grafana

# Check logs
docker-compose logs grafana
```

## Project Structure

```
SentimentProjek/
├── src/
│   ├── data_collection/    # Scraping & database
│   ├── preprocessing/       # Text preprocessing
│   ├── training/           # Model training
│   ├── prediction/         # Prediction pipeline
│   ├── scheduler/          # Automated tasks
│   └── utils.py            # Utility functions
├── data/
│   ├── raw/                # Raw scraped data
│   ├── processed/          # Preprocessed data
│   └── predictions/        # Prediction results
├── models/                 # Trained models
├── notebooks/              # Jupyter notebooks
├── grafana/                # Grafana configs
├── prometheus/             # Prometheus configs
├── docker-compose.yml      # Docker orchestration
├── Dockerfile             # App container
├── dvc.yaml               # DVC pipeline
├── params.yaml            # Parameters
├── cli.py                 # CLI tool
└── quickstart.py          # Quick start script
```

## Advanced Usage

### Custom Scraping Schedule
Edit `params.yaml`:
```yaml
scheduler:
  scraping_interval_hours: 6  # Ubah sesuai kebutuhan
  prediction_interval_hours: 1
```

### Change Model Type
Edit `params.yaml`:
```yaml
training:
  model_type: logistic_regression  # Pilih: naive_bayes, svm, random_forest
```

### Add DVC Remote Storage
```powershell
# Local remote
dvc remote add -d myremote /path/to/storage

# S3
dvc remote add -d myremote s3://mybucket/path

# Google Drive
dvc remote add -d myremote gdrive://folder_id
```

## Support

Jika ada pertanyaan atau issue:
1. Check logs di folder `logs/`
2. Review dokumentasi di `README.md`
3. Check Docker logs: `docker-compose logs`

## Next Steps

1. ✅ Setup project dengan `quickstart.py`
2. ✅ Scrape initial data
3. ✅ Train model
4. ✅ Start scheduler untuk auto-update
5. ✅ Monitor via Grafana dashboard
6. 🚀 Scale dan optimize sesuai kebutuhan!
