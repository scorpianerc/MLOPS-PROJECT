# 🚀 Quick Setup: Push to GitHub & Enable Actions

## Step 1: Initialize Git (if not done)

```bash
git init
git add .
git commit -m "Initial commit: Complete MLOps stack with Docker"
```

## Step 2: Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `SentimentProjek` (or your choice)
3. **Important**: Keep repository **Public** or enable GitHub Container Registry for private repos
4. Don't initialize with README (already have one)
5. Click "Create repository"

## Step 3: Push to GitHub

```bash
# Add remote
git remote add origin https://github.com/YOUR_USERNAME/SentimentProjek.git

# Push
git branch -M main
git push -u origin main
```

## Step 4: Enable GitHub Actions

1. Go to your repository on GitHub
2. Click **Settings** → **Actions** → **General**
3. Under "Actions permissions":
   - ✅ Select "Allow all actions and reusable workflows"
4. Under "Workflow permissions":
   - ✅ Select "Read and write permissions"
   - ✅ Check "Allow GitHub Actions to create and approve pull requests"
5. Click **Save**

## Step 5: Enable GitHub Container Registry

1. Settings → **Packages**
2. Ensure packages can be public
3. No additional setup needed (auto-configured)

## Step 6: Verify Workflows

1. Go to **Actions** tab
2. You should see 3 workflows:
   - ML CI/CD Pipeline
   - MLOps Pipeline  
   - Docker Stack Test & Validation

## Step 7: Trigger First Workflow Run

### Option A: Manual Trigger
1. Actions → **MLOps Pipeline**
2. Click "Run workflow"
3. Select branch: `main`
4. Click "Run workflow"

### Option B: Via Command Line
```bash
# Install GitHub CLI
# Windows: choco install gh
# Or download from: https://cli.github.com/

# Login
gh auth login

# Run workflow
gh workflow run "MLOps Pipeline"

# Check status
gh run list --workflow="MLOps Pipeline"
```

## Step 8: View Results

1. Actions tab → Click on running workflow
2. Watch real-time logs
3. See job summaries after completion
4. Download artifacts if needed

## Expected Results

### First Run (ML CI/CD Pipeline)
```
✅ Data Validation: PASSED
✅ Model Validation: PASSED  
✅ Integration Tests: 6/6 PASSED
✅ Code Quality: NO ISSUES
✅ Model Training: COMPLETED
```

### First Run (MLOps Pipeline)
```
✅ Data Collection: COMPLETED
✅ DVC Version Control: SKIPPED (no remote storage)
✅ Model Retraining: COMPLETED
✅ Docker Build & Push: COMPLETED
✅ Deployment: INSTRUCTIONS PROVIDED
```

### First Run (Docker Test - on PR)
```
✅ Build & Test: ALL SERVICES HEALTHY
✅ Security Scan: NO CRITICAL ISSUES
✅ Docker Compose: VALIDATED
```

## Step 9: Access Docker Image

After workflow completes:

```bash
# Login to GitHub Container Registry
echo YOUR_GITHUB_TOKEN | docker login ghcr.io -u YOUR_USERNAME --password-stdin

# Pull image
docker pull ghcr.io/YOUR_USERNAME/sentimentprojek:latest

# Run locally
docker-compose pull
docker-compose up -d
```

## Step 10: Setup Automated Retraining

**Already configured!** MLOps Pipeline runs automatically:
- Every 6 hours (cron schedule)
- On push to main branch
- Manual trigger anytime

## Troubleshooting

### Workflow not appearing?
- Check `.github/workflows/` folder exists
- Verify YAML syntax: `gh workflow list`
- Refresh Actions tab

### Docker push fails?
- Settings → Actions → General
- Enable "Read and write permissions"
- Re-run workflow

### Tests fail?
- Run local test first: `.\test-github-actions-locally.ps1`
- Check logs in Actions tab
- Verify .env.example is complete

## Next Steps

1. ✅ **Monitor workflows** in Actions tab
2. ✅ **Add status badges** to README
3. ✅ **Setup notifications** (Slack/Discord)
4. ✅ **Deploy to Oracle Cloud** (optional)
5. ✅ **Create pull requests** to trigger tests

---

**🎉 Congratulations!** Your CI/CD pipeline is now fully automated on GitHub Actions!
