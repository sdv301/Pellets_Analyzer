# ============================================================
# Pellets Analyzer - Sync to Server
# Server: 178.250.158.115
# Usage: .\sync-to-server.ps1
# ============================================================

$SERVER = "root@178.250.158.115"
$REMOTE_DIR = "/opt/pellets-analyzer"
$LOCAL_DIR = $PSScriptRoot

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Pellets Analyzer - Sync to Server" -ForegroundColor Cyan
Write-Host "  Server: 178.250.158.115" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Create archive
Write-Host "[1/3] Creating archive..." -ForegroundColor Yellow

$TEMP_ZIP = Join-Path $env:TEMP "pellets-sync.zip"

$allFiles = @(
    'app', 'templates', 'static', 'Uploads',
    'main.py', 'requirements.txt', 'Dockerfile',
    'docker-compose.yml', 'docker-compose.prod.yml', 'nginx.conf', 'nginx.prod.conf',
    '.dockerignore',
    'db_config.json', 'mail_config.json'
)
# .env НЕ включаем — никогда не перезаписываем серверный .env

# Фильтруем — только существующие файлы/директории
$files = @()
$missing = @()
foreach ($f in $allFiles) {
    $fullPath = Join-Path $LOCAL_DIR $f
    if (Test-Path $fullPath) {
        $files += $f
    } else {
        $missing += $f
    }
}

if ($missing.Count -gt 0) {
    Write-Host "  WARNING: Skipping (not found): $($missing -join ', ')" -ForegroundColor Yellow
}

Write-Host "  Archiving $($files.Count) items..." -ForegroundColor Gray
Compress-Archive -Path $files -DestinationPath $TEMP_ZIP -Force

if (-not (Test-Path $TEMP_ZIP)) {
    Write-Host "  ERROR: Failed to create archive" -ForegroundColor Red
    exit 1
}

$zipSize = [math]::Round((Get-Item $TEMP_ZIP).Length / 1MB, 2)
Write-Host "  Archive created: $zipSize MB" -ForegroundColor Green

# Step 2: Copy to server
Write-Host "[2/3] Copying to server..." -ForegroundColor Yellow
Write-Host "  (SSH password will be requested)" -ForegroundColor Gray

scp "$TEMP_ZIP" "${SERVER}:/tmp/pellets-sync.zip"
if ($LASTEXITCODE -ne 0) {
    Write-Host "  ERROR: Copy failed!" -ForegroundColor Red
    Remove-Item $TEMP_ZIP -Force -ErrorAction SilentlyContinue
    exit 1
}

Write-Host "  Copied!" -ForegroundColor Green

# Step 3: Unpack and restart on server
Write-Host "[3/3] Backing up, unpacking and restarting..." -ForegroundColor Yellow

$remoteCmd = @"
cd $REMOTE_DIR &&

# 1) Бэкап текущей БД и конфигов
echo '>>> Creating backup...' &&
BACKUP_DIR=backups/`$(date +%Y%m%d_%H%M%S) &&
mkdir -p `\$BACKUP_DIR &&
if [ -f data/pellets_data.db ]; then cp data/pellets_data.db `\$BACKUP_DIR/; fi &&
if [ -f .env ]; then cp .env `\$BACKUP_DIR/; fi &&
if [ -f db_config.json ]; then cp db_config.json `\$BACKUP_DIR/; fi &&
if [ -f mail_config.json ]; then cp mail_config.json `\$BACKUP_DIR/; fi &&
echo ">>> Backup saved to `\$BACKUP_DIR" &&

# 2) Graceful остановка контейнеров (сохраняет volumes)
echo '>>> Stopping containers gracefully...' &&
docker compose -f docker-compose.prod.yml down --timeout 30 &&

# 3) Распаковка (исключая .env чтобы не перезаписать серверный)
echo '>>> Unpacking new version...' &&
unzip -o /tmp/pellets-sync.zip -x '.env' 'db_config.json' 'mail_config.json' &&
rm -f /tmp/pellets-sync.zip &&

# 4) Убедимся что директория data существует
mkdir -p data &&

# 5) Пересборка образа без кэша и запуск
echo '>>> Building new image (no cache)...' &&
docker compose -f docker-compose.prod.yml build --no-cache &&
echo '>>> Starting new version...' &&
docker compose -f docker-compose.prod.yml up -d &&

# 6) Проверка статуса
echo '>>> Deployment complete!' &&
docker compose -f docker-compose.prod.yml ps
"@

ssh $SERVER $remoteCmd

# Cleanup
Remove-Item $TEMP_ZIP -Force -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "  Sync completed!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "Check: http://178.250.158.115" -ForegroundColor Cyan
