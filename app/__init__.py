# app/__init__.py
"""
Pellets Analyzer — Flask Application Package.

Структура проекта:
    app/
    ├── __init__.py          — Фабрика приложения (create_app)
    ├── auth/                — Аутентификация и авторизация
    │   ├── __init__.py
    │   ├── auth.py          — Хеширование, CRUD пользователей, декораторы
    │   └── routes.py        — Роуты /login, /register, /profile и т.д.
    ├── models/              — Работа с данными (БД)
    │   ├── __init__.py
    │   └── database.py      — SQLite, миграции, ORM-хелперы
    ├── services/            — Бизнес-логика
    │   ├── __init__.py
    │   ├── data_processor.py    — Загрузка и обработка CSV/Excel
    │   ├── gui.py               — Генерация графиков (matplotlib, plotly, seaborn)
    │   ├── ml_optimizer.py      — ML-модели, оптимизация составов
    │   ├── ai_integration.py    — Интеграция с внешними ИИ API
    │   └── ai_ml_analyzer.py    — Локальный ML-анализатор
    ├── routes/              — HTTP-роуты (Blueprints)
    │   ├── __init__.py
    │   ├── main.py          — Основные роуты (dashboard, upload, search)
    │   ├── compare.py       — Сравнение составов
    │   ├── economics.py     — Экономический расчёт
    │   ├── graphs.py        — Создание графиков
    │   ├── ml.py            — ML Dashboard, обучение, оптимизация
    │   └── admin.py         — Админ-панель
    └── utils/               — Утилиты
        ├── __init__.py
        ├── security.py      — Rate limiting, XSS, CSRF, brute force
        └── ssh_tunnel.py    — SSH-туннель для удалённой БД
"""

from flask import Flask, session, render_template
from flask_session import Session
from cachelib.file import FileSystemCache
import secrets
import os
import logging


def create_app(config=None):
    """
    Фабрика Flask-приложения.

    Args:
        config: Опциональный словарь конфигурации для переопределения

    Returns:
        Настроенное Flask-приложение
    """
    app = Flask(
        __name__,
        template_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates'),
        static_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static')
    )

    # ============================================================
    # БАЗОВАЯ КОНФИГУРАЦИЯ
    # ============================================================
    app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Uploads')
    app.config['SESSION_TYPE'] = 'filesystem'
    # Используем новый подход с FileSystemCache вместо устаревшего SESSION_FILE_DIR
    sessions_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'sessions')
    app.config['SESSION_CACHELIB'] = FileSystemCache(threshold=500, default_timeout=3600, cache_dir=sessions_dir)
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', secrets.token_hex(24))
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

    if config:
        app.config.update(config)

    # ============================================================
    # ЛОГИРОВАНИЕ
    # ============================================================
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(
                os.path.join(os.path.dirname(os.path.dirname(__file__)), 'app_errors.log'),
                encoding='utf-8'
            ),
            logging.StreamHandler()
        ]
    )

    # ============================================================
    # РАСШИРЕНИЯ
    # ============================================================
    Session(app)

    # ============================================================
    # ОБРАБОТЧИКИ ОШИБОК
    # ============================================================
    def _build_error_context(e):
        """Строит контекст ошибки для отображения на странице."""
        import traceback
        from flask import request
        from datetime import datetime

        error_type = type(e).__name__
        error_str = str(e)
        traceback_str = traceback.format_exc()

        # Извлекаем первую строку traceback как основную причину
        cause = error_str
        if traceback_str:
            lines = traceback_str.strip().split('\n')
            for line in reversed(lines):
                line = line.strip()
                if line and not line.startswith('Traceback') and not line.startswith('File'):
                    # Берём последнюю строку с описанием ошибки
                    if ':' in line:
                        cause = line.split(':', 1)[-1].strip()
                        break

        return {
            'error_type': error_type,
            'error_message': error_str,
            'traceback': traceback_str,
            'cause': cause if cause else 'Неизвестная ошибка',
            'url': request.url if request else 'N/A',
            'method': request.method if request else 'N/A',
            'endpoint': request.endpoint if request else 'N/A',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'ip': request.remote_addr if request else 'N/A',
        }

    @app.errorhandler(404)
    def handle_not_found(e):
        ctx = _build_error_context(e)
        app.logger.warning(f"404 Not Found: {ctx['url']} (метод: {ctx['method']})")
        return render_template('errors/404.html', **ctx), 404

    @app.errorhandler(403)
    def handle_forbidden(e):
        ctx = _build_error_context(e)
        app.logger.warning(f"403 Forbidden: {ctx['url']} (пользователь: {session.get('username', 'anon')})")
        return render_template('errors/403.html', **ctx), 403

    @app.errorhandler(500)
    def handle_server_error(e):
        ctx = _build_error_context(e)
        app.logger.error(f"КРИТИЧЕСКАЯ ОШИБКА 500:\n{ctx['traceback']}")
        return render_template('errors/500.html', **ctx), 500

    @app.errorhandler(Exception)
    def handle_exception(e):
        import traceback
        ctx = _build_error_context(e)
        if hasattr(e, 'code'):
            if e.code == 404:
                return render_template('errors/404.html', **ctx), 404
            if e.code == 403:
                return render_template('errors/403.html', **ctx), 403
            if e.code == 500:
                app.logger.error(f"КРИТИЧЕСКАЯ ОШИБКА:\n{ctx['traceback']}")
                return render_template('errors/500.html', **ctx), 500
            return str(e), e.code
        app.logger.error(f"НЕОБРАБОТАННОЕ ИСКЛЮЧЕНИЕ:\n{ctx['traceback']}")
        return render_template('errors/500.html', **ctx), 500

    # ============================================================
    # КОНТЕКСТНЫЙ ПРОЦЕССОР
    # ============================================================
    @app.context_processor
    def inject_user():
        if 'user_id' in session:
            return {
                'current_user': {
                    'id': session.get('user_id'),
                    'username': session.get('username'),
                    'email': session.get('email'),
                    'role_name': session.get('role_name'),
                    'full_name': session.get('full_name'),
                    'is_authenticated': True,
                }
            }
        return {'current_user': {'is_authenticated': False}}

    # ============================================================
    # ЗАГРУЗКА BLUEPRINTS
    # ============================================================
    _register_blueprints(app)

    # ============================================================
    # ИНИЦИАЛИЗАЦИЯ (БД, email, ML)
    # ============================================================
    _initialize_extensions(app)

    return app


def _register_blueprints(app):
    """Регистрирует все Blueprint'ы."""
    from app.routes.main import main_bp
    from app.routes.compare import compare_bp
    from app.routes.economics import economics_bp
    from app.routes.graphs import graphs_bp
    from app.routes.ml import ml_bp
    from app.routes.admin import admin_bp
    from app.auth.routes import auth_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(main_bp)
    app.register_blueprint(compare_bp)
    app.register_blueprint(economics_bp)
    app.register_blueprint(graphs_bp)
    app.register_blueprint(ml_bp)
    app.register_blueprint(admin_bp)


def _initialize_extensions(app):
    """Инициализирует БД, email, ML-систему."""
    import os
    from app.auth.auth import init_auth_tables, init_mail
    from app.database.database import init_db
    from app.services.ai_ml_analyzer import AIMLAnalyzer

    # Устанавливаем путь к БД во всех blueprints
    # БД хранится в /app/data/ — это смонтированный Docker volume для сохранения данных
    db_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, 'pellets_data.db')

    # Инициализация БД
    init_db(db_path)
    init_auth_tables(db_path)

    # Инициализация email
    init_mail(app)

    # Устанавливаем db_path во все blueprints
    from app.routes import main, compare, economics, graphs, ml, admin
    from app.auth import routes as auth_routes

    for module in [main, compare, economics, graphs, ml, admin, auth_routes]:
        if hasattr(module, 'set_db_path'):
            module.set_db_path(db_path)

    # Глобальный экземпляр ML-анализатора
    app.config['AI_ML_ANALYZER'] = AIMLAnalyzer(db_path)
