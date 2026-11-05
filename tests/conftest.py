import sqlite3
from pathlib import Path

import pandas as pd
import pytest

import app as app_module
from app.auth.auth import hash_password


@pytest.fixture
def test_db_path(tmp_path: Path) -> str:
    return str(tmp_path / "test_pellets.db")


@pytest.fixture
def app(test_db_path, monkeypatch, tmp_path):
    def _light_initialize_extensions(flask_app):
        from app.auth.auth import init_auth_tables
        from app.database.database import init_db
        from app.routes import admin, compare, economics, graphs, main, ml
        from app.auth import routes as auth_routes
        from app.services.ml_v2 import switch
        from app.services.ai_ml_analyzer import AIMLAnalyzer

        init_db(test_db_path)
        init_auth_tables(test_db_path)

        for module in [main, compare, economics, graphs, ml, admin, auth_routes, switch]:
            if hasattr(module, "set_db_path"):
                module.set_db_path(test_db_path)

        flask_app.config["AI_ML_ANALYZER"] = AIMLAnalyzer(test_db_path)

    monkeypatch.setattr(app_module, "_initialize_extensions", _light_initialize_extensions)

    flask_app = app_module.create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "test-secret-key",
            "UPLOAD_FOLDER": str(tmp_path / "Uploads"),
            "DATABASE_PATH": test_db_path,
        }
    )
    Path(flask_app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)

    yield flask_app


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def runner(app):
    return app.test_cli_runner()


@pytest.fixture
def db_path(app):
    return app.config["DATABASE_PATH"]


@pytest.fixture
def authed_client(client):
    with client.session_transaction() as sess:
        sess["user_id"] = 1
        sess["username"] = "tester"
        sess["role_name"] = "admin"
        sess["email"] = "tester@example.com"
    return client


@pytest.fixture
def sample_measured_data():
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
def seeded_components(test_db_path, sample_components_data):
    conn = sqlite3.connect(test_db_path)
    sample_components_data.to_sql("components", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()
    return test_db_path


@pytest.fixture
def authenticated_client(client, db_path):
    from app.auth.auth import init_auth_tables
    init_auth_tables(db_path)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    pwd_hash = hash_password('testpassword123')
    cursor.execute('''
        INSERT OR IGNORE INTO users (username, email, password_hash, role_id, is_active, is_verified, full_name)
        VALUES (?, ?, ?, ?, 1, 1, ?)
    ''', ('testuser', 'test@example.com', pwd_hash, 1, 'Test User'))
    conn.commit()
    user_id = cursor.lastrowid or 1
    conn.close()

    with client.session_transaction() as sess:
        sess['user_id'] = user_id
        sess['username'] = 'testuser'
        sess['email'] = 'test@example.com'
        sess['role_name'] = 'user'
        sess['full_name'] = 'Test User'

    return client


@pytest.fixture
def admin_client(client, db_path):
    from app.auth.auth import init_auth_tables
    init_auth_tables(db_path)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    pwd_hash = hash_password('adminpass123')
    
    cursor.execute('SELECT id FROM roles WHERE name = ?', ('admin',))
    admin_role = cursor.fetchone()
    role_id = admin_role[0] if admin_role else 2

    cursor.execute('''
        INSERT OR IGNORE INTO users (username, email, password_hash, role_id, is_active, is_verified, full_name)
        VALUES (?, ?, ?, ?, 1, 1, ?)
    ''', ('adminuser', 'admin@example.com', pwd_hash, role_id, 'Admin User'))
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
