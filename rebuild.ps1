# Deepiri Clean Rebuild Script (PowerShell)
# Removes old images, rebuilds fresh, and starts services
# Usage: .\rebuild.ps1 [docker-compose-file]

param(
    [string]$ComposeFile = "docker-compose.dev.yml"
)

$env:BUILD_TIMESTAMP = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()

Write-Host "🧹 Stopping containers and removing old images..." -ForegroundColor Yellow
docker compose -f $ComposeFile down --rmi all --volumes --remove-orphans

Write-Host "🔨 Rebuilding containers (no cache)..." -ForegroundColor Yellow
docker compose -f $ComposeFile build --no-cache --pull

Write-Host "🚀 Starting services..." -ForegroundColor Yellow
docker compose -f $ComposeFile up -d

Write-Host "✅ Rebuild complete!" -ForegroundColor Green
Write-Host ""
Write-Host "View logs: docker compose -f $ComposeFile logs -f" -ForegroundColor Cyan
Write-Host "Check status: docker compose -f $ComposeFile ps" -ForegroundColor Cyan

