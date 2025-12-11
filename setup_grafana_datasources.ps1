# Setup Grafana Datasources via API
$grafanaUrl = "http://localhost:3000"
$username = "admin"
$password = "admin123"

# Create Basic Auth header
$pair = "${username}:${password}"
$bytes = [System.Text.Encoding]::ASCII.GetBytes($pair)
$base64 = [System.Convert]::ToBase64String($bytes)
$headers = @{
    Authorization = "Basic $base64"
    "Content-Type" = "application/json"
}

Write-Host "Creating PostgreSQL datasource..." -ForegroundColor Cyan

# Create PostgreSQL datasource
$postgresBody = @{
    name = "PostgreSQL"
    type = "postgres"
    uid = "postgres-sentiment"
    url = "postgres:5432"
    database = "sentiment_db"
    user = "sentiment_user"
    isDefault = $true
    jsonData = @{
        sslmode = "disable"
        postgresVersion = 1500
    }
    secureJsonData = @{
        password = "password"
    }
} | ConvertTo-Json -Depth 10

try {
    $response = Invoke-RestMethod -Uri "$grafanaUrl/api/datasources" -Method Post -Headers $headers -Body $postgresBody
    Write-Host "✓ PostgreSQL datasource created successfully!" -ForegroundColor Green
    Write-Host "  UID: $($response.uid)" -ForegroundColor Gray
} catch {
    if ($_.Exception.Response.StatusCode -eq 409) {
        Write-Host "PostgreSQL datasource already exists" -ForegroundColor Yellow
    } else {
        Write-Host "Error creating PostgreSQL datasource: $_" -ForegroundColor Red
    }
}

Write-Host "`nCreating Prometheus datasource..." -ForegroundColor Cyan

# Create Prometheus datasource
$prometheusBody = @{
    name = "Prometheus"
    type = "prometheus"
    uid = "prometheus-sentiment"
    url = "http://prometheus:9090"
    access = "proxy"
    isDefault = $false
} | ConvertTo-Json -Depth 10

try {
    $response = Invoke-RestMethod -Uri "$grafanaUrl/api/datasources" -Method Post -Headers $headers -Body $prometheusBody
    Write-Host "✓ Prometheus datasource created successfully!" -ForegroundColor Green
    Write-Host "  UID: $($response.uid)" -ForegroundColor Gray
} catch {
    if ($_.Exception.Response.StatusCode -eq 409) {
        Write-Host "Prometheus datasource already exists" -ForegroundColor Yellow
    } else {
        Write-Host "Error creating Prometheus datasource: $_" -ForegroundColor Red
    }
}

Write-Host "`n=== Datasource Setup Complete ===" -ForegroundColor Green
Write-Host "You can now import the dashboard at: $grafanaUrl" -ForegroundColor Cyan
Write-Host "Login: $username / $password" -ForegroundColor Gray
