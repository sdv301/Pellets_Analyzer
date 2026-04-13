# ============================================================
# Pellets Analyzer — Деплой на сервер (Windows PowerShell)
# Сервер: 185.210.154.20
# Запуск: .\deploy-windows.ps1
# ============================================================

$ErrorActionPreference = "Stop"

$SERVER = "root@185.210.154.20"
$REMOTE_DIR = "/opt/pellets-analyzer"
$LOCAL_DIR = $PSScriptRoot

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Pellets Analyzer — Деплой на сервер" -ForegroundColor Cyan
Write-Host "  Сервер: 185.210.154.20" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# --- 1. Проверка SSH ---
Write-Host "[1/4] Проверка SSH-подключения..." -ForegroundColor Yellow
try {
    $testResult = ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no $SERVER "echo OK" 2>&1
    if ($testResult -eq "OK") {
        Write-Host "  SSH подключён!" -ForegroundColor Green
    } else {
        Write-Host "  SSH требует пароль — будет запрошен при подключении" -ForegroundColor Yellow
    }
} catch {
    Write-Host "  SSH не доступен. Убедитесь, что OpenSSH клиент установлен." -ForegroundColor Red
    Write-Host "  Установите: Add-WindowsCapability -Online -Name OpenSSH.Client~~~~0.0.1.0" -ForegroundColor Red
    exit 1
}

# --- 2. Настройка сервера ---
Write-Host "[2/4] Настройка сервера (Docker, firewall)..." -ForegroundColor Yellow
scp -o StrictHostKeyChecking=no "$LOCAL_DIR\setup-server.sh" "${SERVER}:/tmp/setup-server.sh" 2>&1
ssh -o StrictHostKeyChecking=no $SERVER "bash /tmp/setup-server.sh" 2>&1

# --- 3. Копирование файлов ---
Write-Host "[3/4] Копирование файлов проекта..." -ForegroundColor Yellow

# Исключения для rsync-аналога через robocopy + scp
$excludeDirs = @('.git', '__pycache__', 'sessions', 'data', 'backups', 'node_modules', 'venv', '.venv')
$excludeFiles = @('.env', '*.log', '*.pyc', '*.db', '*.sqlite3', '.env.production')

# Создаём временную директорию для синхронизации
$TEMP_DIR = Join-Path $env:TEMP "pellets-deploy-$(Get-Date -Format 'yyyyMMddHHmmss')"
New-Item -ItemType Directory -Path $TEMP_DIR -Force | Out-Null

# Копируем файлы без исключений
robocopy $LOCAL_DIR $TEMP_DIR /E /NFL /NDL /NJH /NJS /nc /ns /np /XD @excludeDirs /XF @excludeFiles 2>&1 | Out-Null

# Копируем на сервер
Write-Host "  Копирование на сервер (может занять время)..." -ForegroundColor Gray
scp -o StrictHostKeyChecking=no -r "${TEMP_DIR}\*" "${SERVER}:${REMOTE_DIR}/" 2>&1

# Чистим временную директорию
Remove-Item -Recurse -Force $TEMP_DIR -ErrorAction SilentlyContinue

# --- 4. Копирование конфигов ---
Write-Host "[4/4] Копирование production конфигов..." -ForegroundColor Yellow
if (Test-Path "$LOCAL_DIR\.env.production") {
    scp -o StrictHostKeyChecking=no "$LOCAL_DIR\.env.production" "${SERVER}:${REMOTE_DIR}/.env.production" 2>&1
}

# --- 5. Запуск деплоя ---
Write-Host ""
Write-Host "  Запуск деплоя на сервере..." -ForegroundColor Yellow
ssh -o StrictHostKeyChecking=no $SERVER "cd $REMOTE_DIR && chmod +x deploy.sh && bash deploy.sh" 2>&1

Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "  ДЕПЛОЙ ЗАВЕРШЁН!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "Приложение доступно: http://185.210.154.20" -ForegroundColor Cyan
Write-Host ""
