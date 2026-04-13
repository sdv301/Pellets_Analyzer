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

# Системные зависимости (для scipy, numpy, paramiko, su-exec)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    pkg-config \
    su-exec \
    && rm -rf /var/lib/apt/lists/*

# Копирование зависимостей
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Копирование приложения (исключая .git, venv, __pycache__)
COPY main.py .
COPY app/ ./app/
COPY templates/ ./templates/
COPY static/ ./static/

# Создание необходимых директорий
RUN mkdir -p data Uploads sessions static/assets/images app/services/models

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
