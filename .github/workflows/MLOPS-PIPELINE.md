# MLOps Unified Pipeline

## Overview

Pipeline MLOps terpadu yang mengotomatisasi seluruh siklus machine learning dari data collection hingga deployment.

## 🔄 Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     MLOPS UNIFIED PIPELINE                      │
└─────────────────────────────────────────────────────────────────┘

1️⃣ DATA COLLECTION & VALIDATION
   ├─ Scrape new reviews (scheduled)
   ├─ Validate data quality
   ├─ Check for data drift
   ├─ Preprocess data
   └─ Upload artifacts
              ↓
2️⃣ DVC VERSION CONTROL
   ├─ Download validated data
   ├─ Add to DVC tracking
   ├─ Commit to Git
   └─ Push to DVC remote
              ↓
3️⃣ MODEL RETRAINING (if new data or forced)
   ├─ Pull latest data
   ├─ Train BERT model
   ├─ Train Traditional ML model
   ├─ Extract & compare metrics
   ├─ Version models with DVC
   └─ Upload model artifacts
              ↓
4️⃣ DOCKER BUILD & DEPLOY (if models updated)
   ├─ Download latest models
   ├─ Build Docker image
   ├─ Push to registry
   ├─ Deploy to production
   └─ Run health checks
              ↓
5️⃣ NOTIFICATION
   └─ Send pipeline summary
```

## 🚀 Triggers

### Automatic Triggers

1. **Scheduled** (setiap 6 jam)
   - Scrape data baru dari Play Store
   - Validasi data
   - Retrain jika ada data baru
   - Deploy otomatis jika model update

2. **Push ke master/main**
   - Ketika ada perubahan di `data/`, `src/`, `dvc.yaml`, atau `params.yaml`
   - Validasi data
   - Retrain jika diperlukan
   - Deploy otomatis

### Manual Trigger

Buka: https://github.com/scorpianerc/MLOPS-PROJECT/actions

Parameters:
- **force_retrain**: 
  - `true` = Force retraining meskipun tidak ada data baru
  - `false` = Retrain hanya jika ada data baru (default)
  
- **skip_deploy**: 
  - `true` = Skip deployment step
  - `false` = Deploy jika model berhasil di-train (default)

## 📊 Pipeline Stages Detail

### Stage 1: Data Collection & Validation

**Outputs:**
- `new_data_available`: `true` jika ada data baru
- `data_valid`: `true` jika data lolos validasi

**Validations:**
- ✅ Dataset tidak kosong
- ✅ Kolom wajib tersedia (`review_text`, `rating`)
- ✅ Tidak ada null values
- ✅ Distribusi sentiment normal (30%-90% positive)

**Artifacts:**
- `validated-data-{run_number}`
  - `data/raw/*.csv`
  - `data/processed/*.csv`
  - Retention: 30 hari

### Stage 2: DVC Version Control

**Kondisi:** Data valid dari Stage 1

**Outputs:**
- `dvc_committed`: `true` jika ada perubahan di-commit

**Actions:**
- Track data dengan DVC
- Commit `.dvc` files ke Git
- Push data ke DVC remote storage
- Push Git commits

### Stage 3: Model Retraining

**Kondisi:** 
- Data baru tersedia ATAU `force_retrain=true`
- Data valid

**Outputs:**
- `models_trained`: `true` jika minimal 1 model berhasil
- `bert_accuracy`: Akurasi BERT model
- `traditional_accuracy`: Akurasi Traditional ML

**Models:**
1. **BERT Model** (`train_bert.py`)
   - IndoBERT pre-trained
   - Fine-tuning untuk sentiment analysis
   - Output: `models/bert_model.pth`, `models/bert_metrics.json`

2. **Traditional ML** (`train.py`)
   - TF-IDF + Logistic Regression
   - Ensemble dengan Random Forest
   - Output: `models/sentiment_model.pkl`, `models/metrics.json`

**Artifacts:**
- `trained-models-{run_number}`
  - `models/*.pkl`
  - `models/*.pth`
  - `models/*.json`
  - Retention: 90 hari

### Stage 4: Docker Build & Deploy

**Kondisi:**
- Models berhasil di-train
- `skip_deploy != true`
- Branch adalah `main` atau `master`

**Actions:**
1. Download latest models
2. Build Docker image dengan multi-stage build
3. Push ke GitHub Container Registry (ghcr.io)
4. Deploy ke production environment
5. Run health checks

**Image Tags:**
- `latest` (default branch)
- `{branch}-{sha}` (specific commit)
- `v{version}` (semver tags)

### Stage 5: Notification

**Always runs** untuk memberikan summary eksekusi pipeline.

**Summary includes:**
- Status setiap stage (✅/❌/⏭️)
- Model metrics jika available
- Pipeline run number & timestamp
- Trigger event

## 🎯 Use Cases

### Use Case 1: Scheduled Automatic Retraining
```
Trigger: Cron (setiap 6 jam)
Flow: Scrape → Validate → DVC → Retrain → Deploy
Result: Model selalu up-to-date dengan review terbaru
```

### Use Case 2: Manual Retraining dengan Data Baru
```
Trigger: Manual (force_retrain=true)
Flow: Skip scraping → Use existing data → Retrain → Deploy
Result: Retrain dengan data yang sudah ada
```

### Use Case 3: Code Changes Only (No Deploy)
```
Trigger: Manual (skip_deploy=true)
Flow: Validate → DVC → Retrain → Skip deploy
Result: Test model training tanpa deploy
```

### Use Case 4: Data Update dari External Source
```
Trigger: Push to master (after manual data commit)
Flow: Validate → DVC → Retrain → Deploy
Result: Retrain dengan data yang di-commit manual
```

## 🔐 Required Secrets

Configure di: Settings → Secrets and variables → Actions

| Secret | Description | Required |
|--------|-------------|----------|
| `GITHUB_TOKEN` | Otomatis tersedia | ✅ Yes |
| `DVC_REMOTE_URL` | URL remote storage DVC (opsional) | ⚠️ Recommended |
| `POSTGRES_USER` | PostgreSQL username | ✅ Yes (for deploy) |
| `POSTGRES_PASSWORD` | PostgreSQL password | ✅ Yes (for deploy) |
| `POSTGRES_DB` | Database name | ✅ Yes (for deploy) |
| `MONGO_DB` | MongoDB database | ✅ Yes (for deploy) |
| `GRAFANA_ADMIN_USER` | Grafana admin username | ✅ Yes (for deploy) |
| `GRAFANA_ADMIN_PASSWORD` | Grafana admin password | ✅ Yes (for deploy) |

## 📈 Monitoring

### GitHub Actions Dashboard
- View all pipeline runs: `/actions`
- Check artifacts: Each successful run includes data & model artifacts
- Job summaries: Detailed metrics dan status untuk setiap stage

### DVC Metrics Tracking
```bash
# View metrics history
dvc metrics show

# Compare across commits
dvc metrics diff HEAD~1 HEAD

# Plot metrics
dvc plots show
```

### Grafana Dashboards
- Real-time inference metrics
- Model performance over time
- System health monitoring

## 🐛 Troubleshooting

### Pipeline Tidak Trigger Retraining

**Check:**
1. Apakah ada data baru? → `new_data_available` output
2. Apakah data valid? → `data_valid` output
3. Set `force_retrain=true` untuk bypass

### DVC Push Failed

**Solutions:**
1. Pastikan `DVC_REMOTE_URL` secret configured
2. Check remote storage credentials
3. Verifikasi network connectivity

### Model Training Failed

**Debug:**
1. Check data artifacts dari Stage 1
2. Review training logs di job output
3. Verify dependencies dalam `requirements.txt`
4. Check GPU/memory requirements

### Deployment Failed

**Check:**
1. Apakah Docker image berhasil di-build?
2. Verify secrets configuration
3. Check target environment availability
4. Review health check logs

## 🔄 Migration from Old Workflows

### Before (3 separate workflows)
```
❌ dvc-pipeline.yml
❌ model-training.yml  
❌ data-collection.yml
```

### After (1 unified pipeline)
```
✅ mlops-unified-pipeline.yml
```

**Benefits:**
- ✅ Single source of truth
- ✅ Automatic orchestration
- ✅ Better visibility & tracking
- ✅ Simplified maintenance
- ✅ Conditional execution (save resources)

## 📝 Best Practices

1. **Always review data validation results** sebelum retraining
2. **Monitor model metrics** untuk detect degradation
3. **Use DVC remote storage** untuk team collaboration
4. **Set up notifications** untuk critical failures
5. **Test dengan `skip_deploy=true`** sebelum production
6. **Keep artifacts** untuk rollback capabilities
7. **Regular schedule review** untuk optimization

## 🚦 Pipeline Status Badges

Add ke README.md:

```markdown
![MLOps Pipeline](https://github.com/scorpianerc/MLOPS-PROJECT/actions/workflows/mlops-unified-pipeline.yml/badge.svg)
```

## 📚 Additional Resources

- [DVC Documentation](https://dvc.org/doc)
- [GitHub Actions Docs](https://docs.github.com/en/actions)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [MLOps Maturity Model](https://learn.microsoft.com/en-us/azure/architecture/example-scenario/mlops/mlops-maturity-model)

---

**Created:** December 2025  
**Last Updated:** December 10, 2025  
**Maintained by:** MLOps Team
