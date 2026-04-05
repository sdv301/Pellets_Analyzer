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

# Системные зависимости (для scipy, numpy, paramiko)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Копирование зависимостей
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Копирование приложения
COPY . .

# Создание необходимых директорий
RUN mkdir -p Uploads sessions static/assets/images

# Непривилегированный пользователь
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Порт
EXPOSE 5000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/', timeout=5)" || exit 1

# Запуск через gunicorn для продакшена
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--threads", "2", "--timeout", "120", "main:app"]
