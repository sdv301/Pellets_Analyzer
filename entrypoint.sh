#!/bin/bash
# ============================================================
# Pellets Analyzer - Docker Entrypoint Script
# Runs as root, fixes permissions, then drops to appuser
# ============================================================

set -e

# Ensure data directories exist and are writable by appuser
mkdir -p /app/data /app/Uploads /app/sessions /app/app/services/models /app/models/v2
chown -R appuser:appuser /app/data /app/Uploads /app/sessions /app/app/services/models /app/models/v2

# Drop privileges and run the main command
exec gosu appuser "$@"
