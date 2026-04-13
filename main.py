# main.py — Точка входа в приложение
import os
from dotenv import load_dotenv

# Явно загружаем переменные окружения из .env до инициализации приложения
load_dotenv()

from app import create_app

app = create_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
