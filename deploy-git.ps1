# ============================================================
# Pellets Analyzer - Deploy via Git Pull
# Server: 178.250.158.115
# Usage: .\deploy-git.ps1
# ============================================================

$SERVER = "root@178.250.158.115"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Pellets Analyzer - Deploy via Git Pull" -ForegroundColor Cyan
Write-Host "  Server: 178.250.158.115" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "[1/2] Pushing local changes to GitHub..." -ForegroundColor Yellow

# Check for uncommitted changes
$hasChanges = git status --porcelain
if ($hasChanges) {
    Write-Host "  Found uncommitted changes. Committing..." -ForegroundColor Yellow
    git add -A
    git commit -m "Auto-commit before deploy"
}

git push origin main
if ($LASTEXITCODE -ne 0) {
    Write-Host "  ERROR: Git push failed!" -ForegroundColor Red
    exit 1
}
Write-Host "  Pushed!" -ForegroundColor Green

Write-Host "[2/2] Deploying on server via SSH..." -ForegroundColor Yellow
Write-Host "  (SSH password will be requested)" -ForegroundColor Gray

# Read deploy script and pipe to ssh
$deployScript = Get-Content -Path "$PSScriptRoot\deploy-git.sh" -Raw
$deployScript | ssh $SERVER "bash -s"

Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "  Deploy completed!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "Check: http://178.250.158.115" -ForegroundColor Cyan
