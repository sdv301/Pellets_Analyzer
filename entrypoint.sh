#!/bin/bash
# ============================================================
# Pellets Analyzer - Docker Entrypoint Script
# Fixes permissions on mounted volumes before starting the app
# ============================================================

set -e

# Ensure data directories exist and are writable
mkdir -p /app/data /app/Uploads /app/sessions /app/app/services/models
chown -R appuser:appuser /app/data /app/Uploads /app/sessions /app/app/services/models

# Run the main command
exec "$@"
