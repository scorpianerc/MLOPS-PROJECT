# Utility Scripts

Folder ini berisi utility scripts untuk setup, testing, dan automation.

## PowerShell Scripts

### `import_dashboard.ps1`
**Import Grafana dashboards automatically**

Mengimport dashboard JSON configurations ke Grafana instance.

**Usage:**
```powershell
.\scripts\import_dashboard.ps1
```

**What it does:**
- Reads dashboard JSON from `grafana/dashboards/`
- Creates datasources if not exist
- Imports dashboards via Grafana API
- Configures dashboard permissions

**Prerequisites:**
- Grafana must be running (http://localhost:3000)
- Default credentials: admin/admin

---

### `setup_grafana_datasources.ps1`
**Setup Grafana datasources (PostgreSQL & Prometheus)**

Automatically configures datasources untuk Grafana monitoring.

**Usage:**
```powershell
.\scripts\setup_grafana_datasources.ps1
```

**Datasources configured:**
1. **PostgreSQL** - Sentiment database
   - Host: postgres:5432
   - Database: sentiment_db
   - User: sentiment_user
   
2. **Prometheus** - Metrics collection
   - Host: prometheus:9090
   - Scrape interval: 15s

**Notes:**
- Run after first Grafana startup
- Credentials from .env file
- Idempotent (safe to run multiple times)

---

### `test-github-actions-locally.ps1`
**Test GitHub Actions workflows locally before pushing**

Simulates GitHub Actions environment untuk testing workflows.

**Usage:**
```powershell
.\scripts\test-github-actions-locally.ps1
```

**Features:**
- Validates workflow YAML syntax
- Runs workflow steps in local environment
- Checks for common issues
- Generates test report

**Workflows tested:**
- `ml-ci-cd.yml` - ML testing pipeline
- `mlops-pipeline.yml` - Automated retraining
- `docker-test.yml` - Docker validation

**Output:**
- Console output dengan color-coded results
- Log file: `logs/workflow-test.log`

---

## Python Scripts

### `test_mlops_features.py`
**Comprehensive MLOps features testing**

Tests all 7 MLOps features implementation.

**Usage:**
```bash
python scripts/test_mlops_features.py
```

**Tests performed:**
1. ✅ Model Versioning (MLflow)
2. ✅ Automated Testing (pytest)
3. ✅ Monitoring (Prometheus/Grafana)
4. ✅ Model Serving (FastAPI)
5. ✅ Automated Retraining
6. ✅ Feature Store
7. ✅ Experiment Tracking

**Output:**
```
Testing MLOps Features...
✅ Feature 1: Model Versioning - PASSED
✅ Feature 2: Automated Testing - PASSED
...
All tests passed! (7/7)
```

**Report generated:**
- `logs/mlops-features-test.json`
- Contains detailed test results
- Used for CI/CD validation

---

## Running Scripts

### PowerShell Scripts (Windows)
```powershell
# Navigate to project root
cd d:\MLOPS\SentimentProjek

# Run script
.\scripts\script-name.ps1
```

### Python Scripts
```bash
# From project root
python scripts/test_mlops_features.py
```

---

## Script Development Guidelines

1. **Add documentation header**
   - Script purpose
   - Author
   - Last updated
   - Usage examples

2. **Error handling**
   - Check prerequisites
   - Validate inputs
   - Provide helpful error messages

3. **Logging**
   - Log to console and file
   - Use appropriate log levels
   - Include timestamps

4. **Testing**
   - Test on clean environment
   - Handle edge cases
   - Document expected behavior

---

## Related Documentation

- [GETTING_STARTED.md](../docs/GETTING_STARTED.md) - Initial setup
- [MONITORING_GUIDE.md](../docs/MONITORING_GUIDE.md) - Grafana setup
- [GITHUB_ACTIONS_GUIDE.md](../docs/GITHUB_ACTIONS_GUIDE.md) - CI/CD workflows
