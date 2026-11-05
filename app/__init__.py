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
    # Проверка и установка SECRET_KEY (D3)
    secret_key = None
    if config and 'SECRET_KEY' in config:
        secret_key = config['SECRET_KEY']
    else:
        secret_key = os.getenv('SECRET_KEY')

    if not secret_key:
        if config and config.get('TESTING'):
            secret_key = 'test-secret-key-fallback'
        else:
            raise RuntimeError(
                "SECRET_KEY не установлен! Задайте надёжный SECRET_KEY в файле .env или переменных окружения перед запуском."
            )

    app.config['SECRET_KEY'] = secret_key
    app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_CONTENT_LENGTH', 10 * 1024 * 1024))  # 10 МБ лимит (D4)
    # Настройки куки сессии (D6)
    app.config['SESSION_COOKIE_HTTPONLY'] = True
    app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
    app.config['SESSION_COOKIE_SECURE'] = os.getenv('SESSION_COOKIE_SECURE', '0').lower() in ('1', 'true', 'yes')
    app.config['PERMANENT_SESSION_LIFETIME'] = int(os.getenv('PERMANENT_SESSION_LIFETIME', 7 * 24 * 3600))
    # Флаг AstroWind UI v2 (Фаза E)
    app.config['UI_V2'] = os.getenv('UI_V2', '0').lower() in ('1', 'true', 'yes')
    default_static_cache_age = int(os.getenv('STATIC_CACHE_MAX_AGE', '3600'))
    app.config['TEMPLATES_AUTO_RELOAD'] = False
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = default_static_cache_age
    app.config['SESSION_REFRESH_EACH_REQUEST'] = False

    if config:
        app.config.update(config)

    # В dev/testing оставляем быструю обратную связь, в остальных режимах —
    # кэш статики для ускорения переходов между вкладками.
    if app.config.get('DEBUG') or app.config.get('TESTING'):
        app.config['TEMPLATES_AUTO_RELOAD'] = True
        if app.config.get('TESTING'):
            app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

    app.jinja_env.auto_reload = bool(app.config['TEMPLATES_AUTO_RELOAD'])

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

        is_debug = bool(app.config.get('DEBUG'))
        safe_traceback = traceback_str if is_debug else None
        safe_message = error_str if is_debug else 'Внутренняя ошибка сервера. Пожалуйста, обратитесь к администратору.'
        safe_cause = cause if (is_debug and cause) else 'Внутренняя ошибка сервиса'

        return {
            'error_type': error_type if is_debug else 'ServerError',
            'error_message': safe_message,
            'traceback': safe_traceback,
            'cause': safe_cause,
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


    @app.errorhandler(400)
    def handle_bad_request(e):
        from flask import request, jsonify
        msg = getattr(e, 'description', 'Неверный запрос (Bad Request)')
        if request and (request.is_json or request.path.startswith('/api/')):
            return jsonify({'success': False, 'error': msg}), 400
        return msg, 400

    @app.errorhandler(413)
    def handle_request_entity_too_large(e):
        from flask import jsonify, request, flash, render_template
        app.logger.warning(f"413 Payload Too Large: {request.url if request else 'N/A'}")
        if request and (request.is_json or request.path.startswith('/api/')):
            return jsonify({
                'success': False,
                'message': 'Размер загружаемого файла превышает допустимый лимит (10 МБ).',
                'error': 'Payload Too Large'
            }), 413
        flash('Размер загружаемого файла превышает допустимый лимит (10 МБ).', 'danger')
        ctx = _build_error_context(e)
        return render_template('errors/500.html', **ctx), 413

    # ============================================================
    # БЕЗОПАСНОСТЬ: CSRF И ЗАГОЛОВКИ (D5)
    # ============================================================
    @app.before_request
    def check_csrf_protection():
        from app.utils.security import check_csrf
        check_csrf()

    @app.after_request
    def apply_security_headers(response):
        from app.utils.security import add_security_headers
        return add_security_headers(response)

    # ============================================================
    # КОНТЕКСТНЫЙ ПРОЦЕССОР
    # ============================================================
    @app.context_processor
    def inject_csrf():
        from app.utils.security import generate_csrf_token

        class _CSRFTokenString(str):
            def __call__(self):
                return str(self)

        token = generate_csrf_token()
        return {'csrf_token': _CSRFTokenString(token)}

    @app.context_processor
    def inject_ui_version():
        from flask import current_app
        return {'ui_v2': current_app.config.get('UI_V2', False)}

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
    _check_pellet_lab(app)
    _register_cli(app)

    return app


def _check_pellet_lab(app):
    """Проверяет мажор API лаборатории. Ошибка v2 не останавливает legacy."""
    try:
        from app.services.ml_v2.lab_import import check_lab_at_startup

        check_lab_at_startup()
    except Exception:
        app.logger.exception(
            "Проверка pellet_ml не удалась. Приложение запущено, режим v2 выключен, legacy работает."
        )


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

    # Устанавливаем db_path во все blueprints и сервисы
    from app.routes import main, compare, economics, graphs, ml, admin
    from app.auth import routes as auth_routes
    from app.services.ml_v2 import switch

    for module in [main, compare, economics, graphs, ml, admin, auth_routes, switch]:
        if hasattr(module, 'set_db_path'):
            module.set_db_path(db_path)

    # Глобальный экземпляр ML-анализатора
    app.config['AI_ML_ANALYZER'] = AIMLAnalyzer(db_path)


def _register_cli(app):
    """Регистрирует CLI-команды приложения."""
    import click

    @app.cli.command('create-admin')
    @click.option('--username', prompt=True, default='admin', help='Логин администратора')
    @click.option('--email', prompt=True, default='admin@example.com', help='Email администратора')
    @click.option('--password', prompt=True, hide_input=True, confirmation_prompt=True, help='Пароль администратора')
    def create_admin_command(username, email, password):
        """Создать или обновить учётную запись администратора."""
        from app.auth.auth import create_admin_user
        db_path = app.config.get('DATABASE_PATH') or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 'data', 'pellets_data.db'
        )
        try:
            user_id, created = create_admin_user(db_path, username, email, password)
            if created:
                click.echo(f"Администратор {username} ({email}) успешно создан (id={user_id}).")
            else:
                click.echo(f"Данные администратора {username} ({email}) успешно обновлены (id={user_id}).")
        except Exception as err:
            click.echo(f"Ошибка создания администратора: {err}", err=True)
