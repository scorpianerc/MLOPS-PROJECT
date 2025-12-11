# 📁 Project File Structure

## ✅ File yang Tersisa (Clean & Organized)

### 📄 Dokumentasi Utama
- `README.md` - Project overview dan quick start
- `SETUP.md` - Installation guide
- `GETTING_STARTED.md` - Tutorial untuk pemula
- `MONITORING_GUIDE.md` - Panduan monitoring dengan Prometheus & Grafana
- `DATABASE_GUIDE.md` - Database schema dan query guide
- `AUTO_UPDATE_METRICS_GUIDE.md` - **Cara auto-update model metrics di Grafana**
- `GRAFANA_DASHBOARD_GUIDE.md` - Dashboard layout dan panel guide
- `ARCHITECTURE_FLOW.md` - System architecture
- `DVC_GITHUB_ACTIONS_SETUP.md` - DVC setup
- `DockerCommand.md` - Docker commands reference
- `HASIL.md` - Project results

### 🔧 Scripts & Tools
- `import_dashboard.ps1` - Script untuk import Grafana dashboard
- `setup_grafana_datasources.ps1` - Script untuk setup datasources
- `docker-compose.yml` - Container orchestration
- `Dockerfile` - Docker image definition
- `params.yaml` - Model parameters
- `requirements.txt` - Python dependencies

### 📊 Grafana Configuration
**Dashboards** (`grafana/dashboards/`):
- `sentiment-dashboard-v3.json` - **Main dashboard** (13 panels)
- `prometheus-dashboard.json` - Prometheus metrics dashboard
- `dashboard-provider.yml` - Dashboard provisioning config

**Datasources** (`grafana/datasources/`):
- (Empty - datasources dibuat via API)

### 🐍 Source Code (`src/`)
**Training** (`src/training/`):
- `train_bert.py` - **BERT training dengan auto-save metrics ke DB**

**Monitoring** (`src/monitoring/`):
- `prometheus_exporter.py` - Expose metrics untuk Prometheus
- `batch_predict.py` - Batch prediction untuk reviews
- `update_model_metrics.py` - Manual update metrics (optional)
- `simple_monitor.py` - Simple monitoring script

**Data Processing** (`src/data/`):
- Data preprocessing scripts

**API** (`src/api/`):
- API endpoints

---

## ❌ File yang Dihapus (Redundant)

### Dokumentasi Redundant
- ❌ `TROUBLESHOOTING.md` - Info sudah ada di MONITORING_GUIDE.md
- ❌ `METRICS_GUIDE.md` - Digabung ke GRAFANA_DASHBOARD_GUIDE.md
- ❌ `GRAFANA_READY.md` - Temporary guide
- ❌ `GRAFANA_QUICK_START.md` - Temporary guide

### Scripts Redundant
- ❌ `create_grafana_dashboard.py` - Tidak diperlukan (pakai import_dashboard.ps1)
- ❌ `fix_datasource.ps1` - Temporary troubleshooting script
- ❌ `test_query.ps1` - Temporary testing script
- ❌ `src/monitoring/train_model_with_metrics.py` - **Diganti dengan train_bert.py**

### Backup Files
- ❌ `grafana/dashboards/sentiment-dashboard.json.backup`
- ❌ `grafana/dashboards/sentiment-dashboard.json.old`
- ❌ `grafana/datasources/datasource.yml.backup2`
- ❌ `grafana/datasources/datasource.yml.disabled`

---

## 🎯 Key Files untuk Development

### Training Model
```bash
python src/training/train_bert.py
```
✅ Auto-save metrics ke database untuk Grafana

### Batch Prediction
```bash
python src/monitoring/batch_predict.py
```
✅ Predict sentiment untuk reviews baru

### Import Dashboard
```powershell
powershell -ExecutionPolicy Bypass -File import_dashboard.ps1
```
✅ Import dashboard ke Grafana

### Setup Datasources
```powershell
powershell -ExecutionPolicy Bypass -File setup_grafana_datasources.ps1
```
✅ Create PostgreSQL & Prometheus datasources

---

## 📚 Documentation Reading Order

1. **README.md** - Start here
2. **SETUP.md** - Installation
3. **GETTING_STARTED.md** - Tutorial
4. **AUTO_UPDATE_METRICS_GUIDE.md** - Training workflow
5. **GRAFANA_DASHBOARD_GUIDE.md** - Dashboard guide
6. **MONITORING_GUIDE.md** - Monitoring setup

---

## 🧹 File Cleanup Summary

**Dihapus**: 11 file redundant
**Tersisa**: Clean & organized structure

**Benefit**:
- ✅ Lebih mudah navigate
- ✅ Tidak ada konfusi dengan file duplicate
- ✅ Dokumentasi terpusat
- ✅ Scripts yang benar-benar dipakai

---

**Project sekarang lebih clean dan organized!** 🎉
