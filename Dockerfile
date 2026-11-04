# ============================================================
# Pellets Analyzer — Dockerfile
# ============================================================
# Многоэтапная сборка для оптимизации размера образа
# ============================================================

FROM python:3.11-slim AS base

# Переменные окружения
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Рабочая директория
WORKDIR /app

# Системные зависимости (для scipy, numpy, paramiko, gosu)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    pkg-config \
    gosu \
    && rm -rf /var/lib/apt/lists/*

# Копирование зависимостей продукта и пакета pellet_ml (субмодуль lab/).
# Перед сборкой образа на хосте: git submodule update --init lab
COPY requirements.txt .
COPY lab/requirements-ml.txt lab/requirements-ml.txt
RUN pip install --upgrade pip && \
    pip install -r requirements.txt -r lab/requirements-ml.txt

# Копирование приложения и зафиксированной лаборатории (без .git).
COPY main.py .
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY lab/ ./lab/
COPY templates/ ./templates/
COPY static/ ./static/

ENV PELLET_LAB_PATH=/app/lab \
    PYTHONPATH=/app/lab \
    PELLET_MODEL_DIR=/app/models/v2 \
    MODEL_VERSION=legacy

# Артефакт v2 собирается внутри Python 3.11 и не коммитится.
# lab_commit берётся из app/services/ml_v2/PIN.
RUN python scripts/build_v2_artifact.py --out /app/models/v2

# Создание необходимых директорий
RUN mkdir -p data Uploads sessions static/assets/images app/services/models /app/models/v2

# Копирование entrypoint скрипта
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Непривилегированный пользователь
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app

# Порт
EXPOSE 5000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/', timeout=5)" || exit 1

# Запуск через entrypoint (root -> su-exec appuser)
ENTRYPOINT ["/entrypoint.sh"]
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--threads", "2", "--timeout", "120", "main:app"]

# Для разработки (с автоперезагрузкой):
# CMD ["python", "main.py"]
