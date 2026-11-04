# main.py — Точка входа в приложение
import os
from dotenv import load_dotenv

# Явно загружаем переменные окружения из .env до инициализации приложения
load_dotenv()

from app import create_app

app = create_app()

if __name__ == '__main__':
    debug = os.getenv('FLASK_DEBUG', '0').lower() in ('1', 'true', 'yes')
    host = os.getenv('FLASK_HOST', '127.0.0.1')
    port = int(os.getenv('FLASK_PORT', '5000'))
    app.run(host=host, port=port, debug=debug)
