# Import Grafana Dashboard
$grafanaUrl = "http://localhost:3000"
$username = "admin"
$password = "admin123"
$dashboardFile = "d:\MLOPS\SentimentProjek\grafana\dashboards\sentiment-dashboard-v3.json"

# Create Basic Auth header
$pair = "${username}:${password}"
$bytes = [System.Text.Encoding]::ASCII.GetBytes($pair)
$base64 = [System.Convert]::ToBase64String($bytes)
$headers = @{
    Authorization = "Basic $base64"
    "Content-Type" = "application/json"
}

Write-Host "Reading dashboard JSON..." -ForegroundColor Cyan
$dashboardJson = Get-Content $dashboardFile -Raw | ConvertFrom-Json

Write-Host "Importing dashboard..." -ForegroundColor Cyan
Write-Host "  Title: $($dashboardJson.title)" -ForegroundColor Gray

# Wrap dashboard in required format
$importBody = @{
    dashboard = $dashboardJson
    overwrite = $true
    message = "Imported via API"
} | ConvertTo-Json -Depth 100

try {
    $response = Invoke-RestMethod -Uri "$grafanaUrl/api/dashboards/db" -Method Post -Headers $headers -Body $importBody
    Write-Host "SUCCESS - Dashboard imported!" -ForegroundColor Green
    Write-Host "  View at: $grafanaUrl$($response.url)" -ForegroundColor Yellow
} catch {
    Write-Host "ERROR - Failed to import" -ForegroundColor Red
    Write-Host $_.Exception.Message
}
