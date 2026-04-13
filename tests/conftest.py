"""
Pytest конфигурация и фикстуры для Pellets Analyzer.
"""
import os
import sys
import tempfile
import pytest

# Добавляем корень проекта в path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from app.database.database import init_db, insert_data
import pandas as pd


@pytest.fixture
def app():
    """Создаёт тестовое Flask-приложение."""
    db_fd, db_path = tempfile.mkstemp(suffix='.db')

    app = create_app({
        'TESTING': True,
        'SECRET_KEY': 'test-secret-key',
        'WTF_CSRF_ENABLED': False,
    })

    # Переопределяем путь к БД — патчим все модули через monkeypatch
    app.config['DATABASE_PATH'] = db_path

    # Инициализируем БД
    init_db(db_path)

    # Патчим _db_path во всех blueprints
    import app.routes.main as main_mod
    import app.routes.compare as compare_mod
    import app.routes.economics as economics_mod
    import app.routes.graphs as graphs_mod
    import app.routes.ml as ml_mod
    import app.routes.admin as admin_mod
    import app.auth.routes as auth_routes_mod

    for mod in [main_mod, compare_mod, economics_mod, graphs_mod, ml_mod, admin_mod, auth_routes_mod]:
        if hasattr(mod, '_db_path'):
            mod._db_path = db_path
        if hasattr(mod, 'set_db_path'):
            mod.set_db_path(db_path)

    yield app

    # Cleanup
    os.close(db_fd)
    os.unlink(db_path)


@pytest.fixture
def client(app):
    """Тестовый клиент Flask."""
    return app.test_client()


@pytest.fixture
def runner(app):
    """Тестовый CLI-раннер."""
    return app.test_cli_runner()


@pytest.fixture
def db_path(app):
    """Путь к тестовой БД."""
    return app.config['DATABASE_PATH']


@pytest.fixture
def sample_measured_data():
    """Пример данных измеренных параметров."""
    return pd.DataFrame([
        {
            'composition': '60% Опилки, 30% Солома, 10% Картон',
            'density': 1050.0, 'kf': 95.0, 'kt': 90.0, 'h': 5.0,
            'mass_loss': 2.0, 'tign': 300.0, 'tb': 800.0,
            'tau_d1': 10.0, 'tau_d2': 15.0, 'tau_b': 120.0,
            'co2': 12.0, 'co': 0.5, 'so2': 0.1, 'nox': 0.2,
            'q': 18.0, 'ad': 3.0
        },
        {
            'composition': '50% Опилки, 40% Солома, 10% Торф',
            'density': 980.0, 'kf': 88.0, 'kt': 85.0, 'h': 7.0,
            'mass_loss': 3.0, 'tign': 280.0, 'tb': 750.0,
            'tau_d1': 12.0, 'tau_d2': 18.0, 'tau_b': 110.0,
            'co2': 11.0, 'co': 0.6, 'so2': 0.15, 'nox': 0.25,
            'q': 16.5, 'ad': 5.0
        },
        {
            'composition': '70% Опилки, 20% Картон, 10% Пластик',
            'density': 1100.0, 'kf': 97.0, 'kt': 92.0, 'h': 4.0,
            'mass_loss': 1.5, 'tign': 320.0, 'tb': 850.0,
            'tau_d1': 8.0, 'tau_d2': 12.0, 'tau_b': 130.0,
            'co2': 13.0, 'co': 0.4, 'so2': 0.08, 'nox': 0.18,
            'q': 19.5, 'ad': 2.0
        },
    ])


@pytest.fixture
def sample_components_data():
    """Пример данных компонентов."""
    return pd.DataFrame([
        {
            'component': 'Опилки', 'war': 10.8, 'ad': 0.18, 'vd': 80.0,
            'q': 19.79, 'cd': 50.0, 'hd': 6.82, 'nd': 0.5, 'sd': 0.01, 'od': 42.0,
            'ro': 1048.0, 'cost_raw': 1.0, 'cost_crush': 0.2, 'cost_granule': 1.2
        },
        {
            'component': 'Солома', 'war': 9.83, 'ad': 4.33, 'vd': 75.0,
            'q': 16.75, 'cd': 45.0, 'hd': 6.54, 'nd': 0.8, 'sd': 0.05, 'od': 40.0,
            'ro': 977.0, 'cost_raw': 1.4, 'cost_crush': 0.9, 'cost_granule': 2.8
        },
        {
            'component': 'Картон', 'war': 5.02, 'ad': 3.16, 'vd': 78.0,
            'q': 16.73, 'cd': 44.0, 'hd': 6.50, 'nd': 0.3, 'sd': 0.03, 'od': 41.0,
            'ro': 1040.0, 'cost_raw': 2.0, 'cost_crush': 1.1, 'cost_granule': 2.4
        },
    ])


@pytest.fixture
def authenticated_client(client, db_path):
    """Клиент с авторизованным пользователем."""
    from app.auth.auth import init_auth_tables, hash_password

    init_auth_tables(db_path)

    # Создаём тестового пользователя напрямую в БД
    import sqlite3
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    password_hash = hash_password('testpassword123')
    cursor.execute('''
        INSERT OR IGNORE INTO users (username, email, password_hash, role_id, is_active, is_verified, full_name)
        VALUES (?, ?, ?, ?, 1, 1, ?)
    ''', ('testuser', 'test@example.com', password_hash, 1, 'Test User'))
    conn.commit()

    user_id = cursor.lastrowid or 1
    conn.close()

    # Логинимся через клиент
    with client.session_transaction() as sess:
        sess['user_id'] = user_id
        sess['username'] = 'testuser'
        sess['email'] = 'test@example.com'
        sess['role_name'] = 'user'
        sess['full_name'] = 'Test User'

    return client


@pytest.fixture
def admin_client(client, db_path):
    """Клиент с администратором."""
    from app.auth.auth import init_auth_tables, hash_password

    init_auth_tables(db_path)

    # Создаём админа
    import sqlite3
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    password_hash = hash_password('adminpass123')

    # Убедимся что роль admin существует
    cursor.execute('SELECT id FROM roles WHERE name = ?', ('admin',))
    admin_role = cursor.fetchone()
    role_id = admin_role[0] if admin_role else 2

    cursor.execute('''
        INSERT OR IGNORE INTO users (username, email, password_hash, role_id, is_active, is_verified, full_name)
        VALUES (?, ?, ?, ?, 1, 1, ?)
    ''', ('adminuser', 'admin@example.com', password_hash, role_id, 'Admin User'))
    conn.commit()

    user_id = cursor.lastrowid or 1
    conn.close()

    with client.session_transaction() as sess:
        sess['user_id'] = user_id
        sess['username'] = 'adminuser'
        sess['email'] = 'admin@example.com'
        sess['role_name'] = 'admin'
        sess['full_name'] = 'Admin User'

    return client
