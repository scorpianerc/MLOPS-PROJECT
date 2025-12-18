# 🚀 GitHub Actions CI/CD Pipeline Guide

Complete guide untuk menggunakan GitHub Actions dengan Docker deployment untuk MLOps pipeline.

---

## 📋 Overview Workflows

Project ini memiliki **3 automated workflows**:

### 1. **ML CI/CD Pipeline** (`ml-ci-cd.yml`)
**Trigger**: Push/PR ke `main` atau `develop`  
**Purpose**: Testing & Quality Assurance

**Jobs**:
- ✅ Data Validation
- ✅ Model Validation  
- ✅ Integration Tests
- ✅ Code Quality (flake8, black, isort, bandit)
- ✅ Model Training
- ✅ Test Report Generation

### 2. **MLOps Pipeline** (`mlops-pipeline.yml`)
**Trigger**: Schedule (setiap 6 jam), Push ke `main`, Manual  
**Purpose**: Production MLOps dengan automated retraining

**Jobs**:
- ✅ Data Collection & Validation
- ✅ DVC Version Control
- ✅ Model Retraining
- ✅ Docker Build & Push
- ✅ Deployment Instructions
- ✅ Notifications

### 3. **Docker Stack Test** (`docker-test.yml`) ⭐ NEW!
**Trigger**: PR/Push yang mengubah Docker files, Manual  
**Purpose**: Validate Docker stack sebelum deployment

**Jobs**:
- ✅ Build & Test Docker Stack (7 services)
- ✅ Security Scanning (Trivy)
- ✅ Docker Compose Validation

---

## 🔧 Setup GitHub Actions

### **Step 1: Enable GitHub Actions**

1. Go to your repository on GitHub
2. Click **Settings** → **Actions** → **General**
3. Enable **"Allow all actions and reusable workflows"**
4. Enable **"Read and write permissions"** for GITHUB_TOKEN

### **Step 2: Setup Secrets**

Tidak ada secrets wajib untuk basic deployment, tapi opsional:

| Secret | Purpose | Required? |
|--------|---------|-----------|
| `GITHUB_TOKEN` | Auto-generated, untuk push Docker images | ✅ Auto |
| `SLACK_WEBHOOK` | Notifikasi Slack | ❌ Optional |
| `DISCORD_WEBHOOK` | Notifikasi Discord | ❌ Optional |

**Setup secrets**:
1. Repository → Settings → Secrets and variables → Actions
2. Click "New repository secret"
3. Add name & value

### **Step 3: Configure Container Registry**

GitHub Container Registry (GHCR) sudah otomatis aktif!

**Verify**:
```bash
# Login to GHCR
echo $GITHUB_TOKEN | docker login ghcr.io -u YOUR_USERNAME --password-stdin

# Pull image
docker pull ghcr.io/YOUR_USERNAME/REPO_NAME:latest
```

---

## 🎯 How Workflows Work

### **Workflow 1: ML CI/CD Pipeline**

Triggered on every push/PR untuk ensure code quality:

```yaml
on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
```

**What it does**:
1. ✅ Validates data quality
2. ✅ Tests model performance
3. ✅ Runs integration tests with PostgreSQL
4. ✅ Checks code quality (linting, formatting)
5. ✅ Trains models and reports metrics
6. ✅ Generates test coverage report

**View Results**:
- Go to **Actions** tab
- Click on workflow run
- See detailed logs and test reports

### **Workflow 2: MLOps Production Pipeline**

Runs automatically every 6 hours or manually:

```yaml
on:
  schedule:
    - cron: '0 */6 * * *'  # Every 6 hours
  workflow_dispatch:        # Manual trigger
```

**What it does**:

#### **Job 1: Data Collection**
```bash
# Scrapes new reviews (scheduled only)
python src/data_collection/scraper.py

# Validates data quality
- Checks dataset not empty
- Validates required columns
- Checks for duplicates
```

#### **Job 2: DVC Version Control**
```bash
# Commits data to DVC
dvc add data/
dvc push

# Creates Git commit
git add data.dvc .dvc/
git commit -m "Update data"
```

#### **Job 3: Model Retraining**
```bash
# Only runs if new data available
if [ "${{ new_data_available }}" == "true" ]; then
  python src/training/train.py
  
  # Reports metrics:
  # - BERT Accuracy
  # - Traditional ML Accuracy
fi
```

#### **Job 4: Docker Build & Push**
```bash
# Builds Docker image
docker build -t ghcr.io/username/repo:latest .

# Pushes to GitHub Container Registry
docker push ghcr.io/username/repo:latest

# Creates deployment instructions
```

#### **Job 5: Notifications**
```bash
# Sends pipeline summary
# - All job statuses
# - Model metrics
# - Deployment info
```

### **Workflow 3: Docker Stack Test** ⭐ NEW!

Comprehensive Docker testing before deployment:

```yaml
on:
  pull_request:
    paths:
      - 'docker-compose.yml'
      - 'Dockerfile'
      - 'src/**'
```

**What it does**:

#### **Job 1: Build & Test**
```bash
# 1. Build all images
docker-compose build --parallel

# 2. Start all services
docker-compose up -d

# 3. Wait for services to be healthy
# Waits up to 120 seconds

# 4. Test all endpoints
curl http://localhost:8080/health          # API
curl http://localhost:8501                 # Streamlit
curl http://localhost:9090/-/healthy       # Prometheus
curl http://localhost:3000                 # Grafana

# 5. Test database connections
docker exec sentiment_postgres pg_isready
docker exec sentiment_mongodb mongosh --eval "db.runCommand('ping')"

# 6. Test sentiment prediction
curl -X POST http://localhost:8080/predict \
  -d '{"text": "Aplikasi bagus"}'
```

#### **Job 2: Security Scan**
```bash
# Trivy vulnerability scanner
trivy fs . --format sarif

# Hadolint Dockerfile linter
hadolint Dockerfile
```

#### **Job 3: Docker Compose Validation**
```bash
# Validate syntax
docker-compose config --quiet

# Check environment variables
grep POSTGRES_USER .env.example
```

---

## 🎬 Manual Workflow Triggers

### **Trigger MLOps Pipeline Manually**

1. Go to **Actions** tab
2. Click **MLOps Pipeline**
3. Click **Run workflow** dropdown
4. Select options:
   - `force_retrain`: Force retraining even without new data
   - `skip_deploy`: Skip deployment step
5. Click **Run workflow**

**Via GitHub CLI**:
```bash
# Install gh CLI
gh workflow run "MLOps Pipeline" \
  --ref main \
  -f force_retrain=true \
  -f skip_deploy=false
```

### **Trigger Docker Test Manually**

```bash
gh workflow run "Docker Stack Test & Validation"
```

---

## 📊 Viewing Results

### **In GitHub UI**

1. **Actions Tab**: See all workflow runs
2. **Click Run**: View detailed logs
3. **Summary**: See job summaries with metrics
4. **Artifacts**: Download test results

### **Job Summaries Include**:

**ML CI/CD Pipeline**:
```
✅ Data Validation: Passed
✅ Model Validation: Passed  
✅ Integration Tests: 6/6 Passed
✅ Code Quality: No issues
✅ Model Training: Complete
   - BERT Accuracy: 95.2%
   - Training Time: 45s
```

**MLOps Pipeline**:
```
🔄 MLOps Pipeline Execution Summary

| Stage              | Status        |
|--------------------|---------------|
| Data Validation    | ✅ Passed     |
| New Data Available | ✅ Yes        |
| DVC Committed      | ✅ Yes        |
| Models Trained     | ✅ Yes        |
| Deployment         | ✅ Success    |

📦 Docker Image: ghcr.io/username/repo:latest
🎯 BERT Accuracy: 96.5%
```

**Docker Stack Test**:
```
🐳 Docker Stack Test Summary

| Service      | Status      |
|--------------|-------------|
| API Server   | ✅ Tested   |
| Streamlit    | ✅ Tested   |
| Prometheus   | ✅ Tested   |
| Grafana      | ✅ Tested   |
| PostgreSQL   | ✅ Tested   |
| MongoDB      | ✅ Tested   |
```

---

## 🚀 Deployment Workflow

### **Automated Deployment Flow**

```
┌─────────────────────────────────────────┐
│  1. Code Push / Schedule Trigger        │
└─────────────┬───────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│  2. Data Collection & Validation        │
│     - Scrape new reviews (if scheduled) │
│     - Validate data quality             │
└─────────────┬───────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│  3. DVC Version Control                 │
│     - Commit data to DVC                │
│     - Push to remote storage            │
└─────────────┬───────────────────────────┘
              │
              ▼ (if new data)
┌─────────────────────────────────────────┐
│  4. Model Retraining                    │
│     - Train BERT model                  │
│     - Train traditional ML              │
│     - Save artifacts                    │
└─────────────┬───────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│  5. Docker Build & Push                 │
│     - Build image with new models       │
│     - Push to ghcr.io                   │
│     - Tag with version                  │
└─────────────┬───────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│  6. Deployment (Manual/Automated)       │
│     - Pull latest image                 │
│     - Update docker-compose             │
│     - Restart services                  │
└─────────────┬───────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│  7. Health Checks & Notifications       │
│     - Verify services healthy           │
│     - Send success/failure alerts       │
└─────────────────────────────────────────┘
```

### **Option 1: Local Deployment from CI**

After workflow completes:

```powershell
# Pull latest image from GitHub Container Registry
docker pull ghcr.io/YOUR_USERNAME/REPO_NAME:latest

# Start services
docker-compose pull
docker-compose up -d

# Verify
docker-compose ps
curl http://localhost:8080/health
```

### **Option 2: Automated Server Deployment**

Add to workflow (mlops-pipeline.yml):

```yaml
- name: Deploy to Production Server
  uses: appleboy/ssh-action@master
  with:
    host: ${{ secrets.PROD_SERVER_HOST }}
    username: ${{ secrets.PROD_SERVER_USER }}
    key: ${{ secrets.PROD_SERVER_SSH_KEY }}
    script: |
      cd /path/to/app
      docker-compose pull
      docker-compose up -d
      docker-compose ps
```

**Required Secrets**:
- `PROD_SERVER_HOST`: Server IP/hostname
- `PROD_SERVER_USER`: SSH username
- `PROD_SERVER_SSH_KEY`: SSH private key

---

## 📦 Docker Image Registry
docker-compose pull
docker-compose up -d
```

---

## 🔔 Notifications Setup

### **Option 1: Slack Notifications**

Add to notification job:

```yaml
- name: Send Slack notification
  uses: 8398a7/action-slack@v3
  with:
    status: ${{ job.status }}
    text: |
      🚀 MLOps Pipeline Complete!
      
      • Models Trained: ${{ needs.model-retraining.outputs.models_trained }}
      • BERT Accuracy: ${{ needs.model-retraining.outputs.bert_accuracy }}
      • Docker Image: ghcr.io/${{ github.repository }}:latest
    webhook_url: ${{ secrets.SLACK_WEBHOOK }}
  if: always()
```

**Setup**:
1. Create Slack Incoming Webhook
2. Add `SLACK_WEBHOOK` secret to GitHub

### **Option 2: Discord Notifications**

```yaml
- name: Send Discord notification
  uses: sarisia/actions-status-discord@v1
  with:
    webhook: ${{ secrets.DISCORD_WEBHOOK }}
    title: "MLOps Pipeline"
    description: |
      Pipeline completed successfully!
      Models trained with 96.5% accuracy
```

### **Option 3: Email Notifications**

```yaml
- name: Send email notification
  uses: dawidd6/action-send-mail@v3
  with:
    server_address: smtp.gmail.com
    server_port: 465
    username: ${{ secrets.EMAIL_USERNAME }}
    password: ${{ secrets.EMAIL_PASSWORD }}
    subject: "MLOps Pipeline Status"
    to: your-email@example.com
    from: GitHub Actions
    body: Pipeline completed successfully!
```

---

## 🧪 Testing Workflows Locally

### **Using Act (GitHub Actions Runner)**

Install Act:
```powershell
# Using Chocolatey
choco install act-cli

# Or using Scoop
scoop install act
```

Run workflows locally:
```bash
# Test ML CI/CD Pipeline
act -j data-validation

# Test MLOps Pipeline
act -j data-collection-and-validation --secret-file .env

# Test Docker Stack
act -j docker-build-test
```

### **Manual Docker Testing**

Before pushing, test Docker stack locally:

```powershell
# Run the same tests as GitHub Actions
docker-compose build --parallel
docker-compose up -d

# Wait for services
Start-Sleep -Seconds 30

# Test API
curl http://localhost:8080/health

# Test prediction
$body = @{ text = "Test review" } | ConvertTo-Json
Invoke-WebRequest -Uri http://localhost:8080/predict -Method POST -Body $body -ContentType "application/json"

# Cleanup
docker-compose down -v
```

---

## 📈 Monitoring Workflows

### **GitHub Actions Insights**

View workflow metrics:
1. Go to **Insights** → **Actions**
2. See:
   - Workflow run history
   - Success/failure rates
   - Average run duration
   - Billable time (minutes)

### **Workflow Status Badge**

Add to README.md:

```markdown
![MLOps Pipeline](https://github.com/USERNAME/REPO/actions/workflows/mlops-pipeline.yml/badge.svg)
![ML CI/CD](https://github.com/USERNAME/REPO/actions/workflows/ml-ci-cd.yml/badge.svg)
![Docker Test](https://github.com/USERNAME/REPO/actions/workflows/docker-test.yml/badge.svg)
```

### **Workflow Artifacts**

Download test results:
1. Actions → Workflow Run → **Artifacts**
2. Download:
   - Test reports
   - Coverage reports
   - Model artifacts
   - Logs

---

## 🔧 Troubleshooting

### **Workflow Fails to Start**

```bash
# Check workflow syntax
gh workflow view "MLOps Pipeline"

# Check repository settings
# Settings → Actions → Enable workflows
```

### **Docker Build Fails**

```yaml
# Add debug steps
- name: Debug Docker build
  run: |
    docker version
    docker-compose version
    cat Dockerfile
    cat docker-compose.yml
```

### **Service Health Check Fails**

```yaml
# Increase timeout
timeout=300  # 5 minutes instead of 2

# Add more detailed logging
docker-compose logs --tail=100 api
```

### **Permission Denied Pushing to GHCR**

```bash
# Check GITHUB_TOKEN permissions
# Settings → Actions → General
# ✅ Enable "Read and write permissions"
```

### **Out of Disk Space**

```yaml
# Add cleanup steps
- name: Clean Docker cache
  run: |
    docker system prune -af
    docker volume prune -f
```

---

## 📊 Best Practices

### **1. Use Caching**

```yaml
- name: Cache Python dependencies
  uses: actions/cache@v4
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}
```

### **2. Fail Fast**

```yaml
strategy:
  fail-fast: true  # Stop all jobs if one fails
```

### **3. Conditional Jobs**

```yaml
jobs:
  deploy:
    if: github.ref == 'refs/heads/main' && success()
```

### **4. Secrets Management**

```yaml
# Never hardcode secrets
password: ${{ secrets.DB_PASSWORD }}  # ✅ Good
password: "my-password"               # ❌ Bad
```

### **5. Matrix Testing**

```yaml
strategy:
  matrix:
    python-version: [3.9, 3.10, 3.11]
    os: [ubuntu-latest, windows-latest]
```

---

## 🎯 Quick Reference

### **Common Commands**

```bash
# List workflows
gh workflow list

# View workflow
gh workflow view "MLOps Pipeline"

# Run workflow
gh workflow run "MLOps Pipeline"

# Check run status
gh run list --workflow="MLOps Pipeline"

# View run logs
gh run view <run-id> --log

# Download artifacts
gh run download <run-id>
```

### **Workflow Files Location**

```
.github/
└── workflows/
    ├── ml-ci-cd.yml           # Testing & QA
    ├── mlops-pipeline.yml     # Production MLOps
    └── docker-test.yml        # Docker validation
```

---

## 📚 Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [GitHub Container Registry](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry)
- [Act - Local Testing](https://github.com/nektos/act)

---

**🎉 Your CI/CD pipeline is now fully automated!**

Every code push triggers:
- ✅ Automated testing
- ✅ Code quality checks
- ✅ Model retraining (if needed)
- ✅ Docker image building
- ✅ Deployment instructions

**Total Cost**: $0 (GitHub Actions free tier: 2,000 minutes/month)
