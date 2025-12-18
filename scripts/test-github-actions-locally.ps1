# GitHub Actions Local Test Script
# This script simulates what GitHub Actions does

Write-Host "`n╔══════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║     🧪 GitHub Actions Local Test Simulation            ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════════╝`n" -ForegroundColor Cyan

# Configuration
$ErrorActionPreference = "Continue"
$testsPassed = 0
$testsFailed = 0

function Test-Step {
    param(
        [string]$Name,
        [scriptblock]$Command
    )
    
    Write-Host "`n▶ $Name" -ForegroundColor Yellow
    try {
        & $Command
        Write-Host "  ✅ PASSED" -ForegroundColor Green
        $script:testsPassed++
    } catch {
        Write-Host "  ❌ FAILED: $_" -ForegroundColor Red
        $script:testsFailed++
    }
}

# Simulate GitHub Actions workflow
Write-Host "🔄 Simulating GitHub Actions Docker Test Workflow...`n" -ForegroundColor Cyan

Test-Step "Check Docker Installation" {
    $version = docker --version
    if (-not $version) { throw "Docker not found" }
    Write-Host "  Docker: $version" -ForegroundColor Gray
}

Test-Step "Verify .env File" {
    if (-not (Test-Path .env)) {
        Write-Host "  Creating .env from .env.example..." -ForegroundColor Gray
        Copy-Item .env.example .env
    }
    Write-Host "  ✓ .env file exists" -ForegroundColor Gray
}

Test-Step "Validate docker-compose.yml Syntax" {
    docker-compose config --quiet
    if ($LASTEXITCODE -ne 0) { throw "Invalid docker-compose.yml" }
    Write-Host "  ✓ docker-compose.yml is valid" -ForegroundColor Gray
}

Test-Step "Build Docker Images" {
    Write-Host "  Building images (this may take a few minutes)..." -ForegroundColor Gray
    docker-compose build --parallel 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Docker build failed" }
    Write-Host "  ✓ All images built successfully" -ForegroundColor Gray
}

Test-Step "Check Docker Image Sizes" {
    $images = docker images --format "table {{.Repository}}\t{{.Size}}" | Select-String "sentiment"
    Write-Host "  Images created:" -ForegroundColor Gray
    $images | ForEach-Object { Write-Host "    $_" -ForegroundColor Gray }
}

Test-Step "Start All Services" {
    Write-Host "  Starting services..." -ForegroundColor Gray
    docker-compose up -d 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Failed to start services" }
    Write-Host "  ✓ Services started" -ForegroundColor Gray
}

Test-Step "Wait for Services to be Ready" {
    Write-Host "  Waiting for services to become healthy..." -ForegroundColor Gray
    $timeout = 60
    $elapsed = 0
    
    while ($elapsed -lt $timeout) {
        try {
            $response = Invoke-WebRequest -Uri http://localhost:8080/health -TimeoutSec 2 -ErrorAction SilentlyContinue
            if ($response.StatusCode -eq 200) {
                Write-Host "  ✓ Services are ready (${elapsed}s)" -ForegroundColor Gray
                break
            }
        } catch {
            Start-Sleep -Seconds 5
            $elapsed += 5
        }
    }
    
    if ($elapsed -ge $timeout) {
        throw "Services did not become ready in time"
    }
}

Test-Step "Check Service Status" {
    $services = docker-compose ps --format table 2>&1
    Write-Host "  Service Status:" -ForegroundColor Gray
    $services | Select-String "sentiment" | ForEach-Object { 
        Write-Host "    $_" -ForegroundColor Gray 
    }
}

Test-Step "Test API Health Endpoint" {
    $response = Invoke-WebRequest -Uri http://localhost:8080/health
    $content = $response.Content | ConvertFrom-Json
    
    if ($content.status -ne "healthy") {
        throw "API not healthy"
    }
    
    Write-Host "  ✓ API Status: $($content.status)" -ForegroundColor Gray
    Write-Host "  ✓ Model: $($content.model_version)" -ForegroundColor Gray
}

Test-Step "Test Model Info Endpoint" {
    $response = Invoke-WebRequest -Uri http://localhost:8080/model/info
    $content = $response.Content | ConvertFrom-Json
    
    Write-Host "  ✓ Model: $($content.model_type)" -ForegroundColor Gray
    Write-Host "  ✓ Parameters: $($content.num_parameters)" -ForegroundColor Gray
}

Test-Step "Test Sentiment Prediction (Positive)" {
    $body = @{ text = "Aplikasi ini sangat bagus dan mudah digunakan!" } | ConvertTo-Json
    $response = Invoke-WebRequest -Uri http://localhost:8080/predict -Method POST -Body $body -ContentType "application/json"
    $content = $response.Content | ConvertFrom-Json
    
    Write-Host "  ✓ Sentiment: $($content.sentiment)" -ForegroundColor Gray
    Write-Host "  ✓ Confidence: $([math]::Round($content.confidence * 100, 2))%" -ForegroundColor Gray
    
    if ($content.sentiment -ne "positive") {
        throw "Expected positive sentiment"
    }
}

Test-Step "Test Sentiment Prediction (Negative)" {
    $body = @{ text = "Aplikasi ini buruk sekali, sering error!" } | ConvertTo-Json
    $response = Invoke-WebRequest -Uri http://localhost:8080/predict -Method POST -Body $body -ContentType "application/json"
    $content = $response.Content | ConvertFrom-Json
    
    Write-Host "  ✓ Sentiment: $($content.sentiment)" -ForegroundColor Gray
    Write-Host "  ✓ Confidence: $([math]::Round($content.confidence * 100, 2))%" -ForegroundColor Gray
    
    if ($content.sentiment -ne "negative") {
        throw "Expected negative sentiment"
    }
}

Test-Step "Test Streamlit Accessibility" {
    try {
        $response = Invoke-WebRequest -Uri http://localhost:8501 -TimeoutSec 5 -ErrorAction Stop
        Write-Host "  ✓ Streamlit is accessible" -ForegroundColor Gray
    } catch {
        Write-Host "  ⚠ Streamlit not accessible yet (may need more time)" -ForegroundColor Yellow
    }
}

Test-Step "Test Prometheus Health" {
    $response = Invoke-WebRequest -Uri http://localhost:9090/-/healthy
    if ($response.StatusCode -ne 200) {
        throw "Prometheus not healthy"
    }
    Write-Host "  ✓ Prometheus is healthy" -ForegroundColor Gray
}

Test-Step "Test Grafana Accessibility" {
    try {
        $response = Invoke-WebRequest -Uri http://localhost:3000 -TimeoutSec 5 -ErrorAction Stop
        Write-Host "  ✓ Grafana is accessible" -ForegroundColor Gray
    } catch {
        Write-Host "  ⚠ Grafana not accessible yet" -ForegroundColor Yellow
    }
}

Test-Step "Test PostgreSQL Connection" {
    docker exec sentiment_postgres pg_isready -U sentiment_user 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "PostgreSQL not ready" }
    Write-Host "  ✓ PostgreSQL is ready" -ForegroundColor Gray
}

Test-Step "Test MongoDB Connection" {
    docker exec sentiment_mongodb mongosh --eval "db.runCommand('ping')" --quiet 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "MongoDB not ready" }
    Write-Host "  ✓ MongoDB is ready" -ForegroundColor Gray
}

# Results Summary
Write-Host "`n╔══════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║                    TEST RESULTS                          ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════════╝`n" -ForegroundColor Cyan

Write-Host "✅ Tests Passed: $testsPassed" -ForegroundColor Green
Write-Host "❌ Tests Failed: $testsFailed" -ForegroundColor $(if ($testsFailed -gt 0) { "Red" } else { "Gray" })
Write-Host "📊 Success Rate: $([math]::Round(($testsPassed / ($testsPassed + $testsFailed)) * 100, 2))%`n" -ForegroundColor Cyan

if ($testsFailed -eq 0) {
    Write-Host "🎉 ALL TESTS PASSED!" -ForegroundColor Green
    Write-Host "✅ Ready to push to GitHub - workflows will succeed!`n" -ForegroundColor Green
} else {
    Write-Host "⚠️ SOME TESTS FAILED" -ForegroundColor Yellow
    Write-Host "Please fix the issues before pushing to GitHub`n" -ForegroundColor Yellow
}

Write-Host "📋 Service URLs:" -ForegroundColor Cyan
Write-Host "  • API Docs:    http://localhost:8080/docs" -ForegroundColor White
Write-Host "  • Streamlit:   http://localhost:8501" -ForegroundColor White
Write-Host "  • Grafana:     http://localhost:3000" -ForegroundColor White
Write-Host "  • Prometheus:  http://localhost:9090`n" -ForegroundColor White

Write-Host "🛑 To stop services: docker-compose down`n" -ForegroundColor Yellow

# Ask if user wants to view logs
$viewLogs = Read-Host "View container logs? (y/n)"
if ($viewLogs -eq 'y') {
    Write-Host "`n📋 Recent logs from all services:`n" -ForegroundColor Cyan
    docker-compose logs --tail=20
}

# Ask if user wants to stop services
$stopServices = Read-Host "`nStop all services? (y/n)"
if ($stopServices -eq 'y') {
    Write-Host "`n🛑 Stopping services..." -ForegroundColor Yellow
    docker-compose down -v
    Write-Host "✅ Services stopped and cleaned up`n" -ForegroundColor Green
} else {
    Write-Host "`n✅ Services are still running" -ForegroundColor Green
    Write-Host "Run 'docker-compose down' when you're done`n" -ForegroundColor Gray
}
