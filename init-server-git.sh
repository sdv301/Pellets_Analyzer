#!/bin/bash
# ============================================================
# Pellets Analyzer - Initialize Git on Server
# Run once: ssh root@178.250.158.115 "bash -s" < init-server-git.sh
# ============================================================

set -e

REPO_URL="https://github.com/sdv301/Pellets_Analyzer.git"
PROJECT_DIR="/opt/pellets-analyzer"

echo "============================================"
echo "  Initializing Git on Server"
echo "============================================"

# 1) Save important server-only files
echo "[1/4] Saving server-only files (.env, configs, data)..."
mkdir -p /tmp/pellets-backup
[ -f "$PROJECT_DIR/.env" ] && cp "$PROJECT_DIR/.env" /tmp/pellets-backup/
[ -f "$PROJECT_DIR/db_config.json" ] && cp "$PROJECT_DIR/db_config.json" /tmp/pellets-backup/
[ -f "$PROJECT_DIR/mail_config.json" ] && cp "$PROJECT_DIR/mail_config.json" /tmp/pellets-backup/
echo "  Saved to /tmp/pellets-backup/"

# 2) Stop containers
echo "[2/4] Stopping containers..."
cd "$PROJECT_DIR"
docker compose -f docker-compose.prod.yml down 2>/dev/null || true

# 3) Remove old .git and clone fresh
echo "[3/4] Cloning repository..."
rm -rf "$PROJECT_DIR/.git"

cd /tmp
rm -rf pellets-new
git clone "$REPO_URL" pellets-new

# Restore server-only files
for f in .env db_config.json mail_config.json; do
    [ -f "/tmp/pellets-backup/$f" ] && cp "/tmp/pellets-backup/$f" "pellets-new/$f"
done

# Replace old project with new clone
rm -rf "$PROJECT_DIR"
mv pellets-new "$PROJECT_DIR"

# 4) Verify
echo "[4/4] Verifying..."
cd "$PROJECT_DIR"
echo "  Remote: $(git remote -v | head -1)"
echo "  Branch: $(git branch --show-current)"
echo "  Last commit: $(git log -1 --oneline)"

echo ""
echo "============================================"
echo "  Git initialized successfully!"
echo "============================================"
echo ""
echo "Now run: docker compose -f docker-compose.prod.yml up -d --build"
echo "And for future deploys: .\deploy-git.ps1"
