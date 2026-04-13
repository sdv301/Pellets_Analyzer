#!/bin/bash
# ============================================================
# Pellets Analyzer - Deploy via Git Pull
# Server: 178.250.158.115
# Usage: Run locally, it SSH-es to server and does git pull + rebuild
# ============================================================

set -e

SERVER="root@178.250.158.115"
REMOTE_DIR="/opt/pellets-analyzer"

echo "============================================"
echo "  Pellets Analyzer - Deploy via Git Pull"
echo "  Server: 178.250.158.115"
echo "============================================"
echo ""

# Remote commands executed via SSH on the server
ssh "$SERVER" bash -s << 'REMOTE_SCRIPT'
set -e

cd /opt/pellets-analyzer

# 1) Backup
echo '>>> Creating backup...'
BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
[ -f data/pellets_data.db ] && cp data/pellets_data.db "$BACKUP_DIR/"
[ -f .env ] && cp .env "$BACKUP_DIR/"
[ -f db_config.json ] && cp db_config.json "$BACKUP_DIR/"
[ -f mail_config.json ] && cp mail_config.json "$BACKUP_DIR/"
echo ">>> Backup saved to $BACKUP_DIR"

# 2) Git pull
echo '>>> Pulling latest changes from git...'
git stash --include-untracked 2>/dev/null || true
git pull origin main
echo ">>> Current commit:"
git log -1 --oneline

# 3) Stop containers gracefully
echo '>>> Stopping containers gracefully...'
docker compose -f docker-compose.prod.yml down --timeout 30

# 4) Rebuild without cache
echo '>>> Building new image (no cache)...'
docker compose -f docker-compose.prod.yml build --no-cache

# 5) Start
echo '>>> Starting new version...'
docker compose -f docker-compose.prod.yml up -d

# 6) Status
echo '>>> Deployment complete!'
docker compose -f docker-compose.prod.yml ps
REMOTE_SCRIPT

echo ""
echo "============================================"
echo "  Deploy completed!"
echo "============================================"
echo ""
echo "Check: http://178.250.158.115"
