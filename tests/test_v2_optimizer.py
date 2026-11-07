"""Тесты для Фазы B4: Оптимизатор состава для v2 и B5: синхронизация состояния.

Проверяет:
1. Оптимизацию свойств пеллет (максимизация q, минимизация ad).
2. Нахождение составов строго внутри области применимости (in_domain=True).
3. Соблюдение пользовательских ограничений по компонентам.
4. Детерминированную воспроизводимость с фиксированным seed.
5. Неприкосновенность таблицы measured_parameters (прогнозы не пишутся в обучение).
6. Запись в журнал аудита ml_optimizations.
7. Работу HTTP-эндпоинта /ml_optimize в режиме v2.
8. Синхронизацию статуса обучения через system_settings (Phase B5).
"""
import json
import sqlite3
import pytest
from app import create_app
from app.services.ml_v2.optimizer import optimize_composition_v2
from app.services.ml_v2.switch import set_runtime_override


@pytest.fixture(autouse=True)
def clean_switch_state(monkeypatch):
    from app.services.ml_v2.predictor import reset_predictor
    monkeypatch.delenv("PELLET_MODEL_DIR", raising=False)
    reset_predictor()
    set_runtime_override(None, None)
    yield
    monkeypatch.delenv("PELLET_MODEL_DIR", raising=False)
    reset_predictor()
    set_runtime_override(None, None)


def _init_test_db(db_path: str):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS system_settings (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_by TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS ml_optimizations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            target_property TEXT,
            maximize INTEGER,
            optimal_composition_json TEXT,
            optimal_value REAL,
            constraints_json TEXT,
            algorithm TEXT,
            model_metrics_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS measured_parameters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            composition TEXT,
            source TEXT DEFAULT 'laboratory',
            is_questionable INTEGER DEFAULT 0
        )
    """)
    # 3 эталонных замера
    cur.execute("INSERT INTO measured_parameters (composition, source) VALUES ('100% Опилки', 'laboratory')")
    cur.execute("INSERT INTO measured_parameters (composition, source) VALUES ('90% Опилки, 10% Торф', 'laboratory')")
    cur.execute("INSERT INTO measured_parameters (composition, source) VALUES ('85% Опилки, 15% Картон', 'laboratory')")
    conn.commit()
    conn.close()


def test_v2_optimizer_maximizes_calorific_value_in_domain():
    res = optimize_composition_v2("q", maximize=True, seed=42)
    assert res["success"] is True
    assert res["model_version"] == "v2"
    assert res["in_domain"] is True
    assert res["optimal_value"] > 16.0
    comp = res["optimal_composition"]
    assert isinstance(comp, dict)
    assert len(comp) >= 1
    total = sum(comp.values())
    assert abs(total - 100.0) < 0.2
    assert "contract" in res
    assert res["contract"]["in_domain"] is True


def test_v2_optimizer_minimizes_ash_in_domain():
    res = optimize_composition_v2("ad", maximize=False, seed=42)
    assert res["success"] is True
    assert res["model_version"] == "v2"
    assert res["in_domain"] is True
    assert res["optimal_value"] < 15.0
    comp = res["optimal_composition"]
    total = sum(comp.values())
    assert abs(total - 100.0) < 0.2


def test_v2_optimizer_deterministic_reproducibility():
    res1 = optimize_composition_v2("q", maximize=True, seed=42)
    res2 = optimize_composition_v2("q", maximize=True, seed=42)
    assert res1["optimal_composition"] == res2["optimal_composition"]
    assert res1["optimal_value"] == res2["optimal_value"]
    assert res1["contract"]["value"] == res2["contract"]["value"]
    assert res1["contract"]["std"] == res2["contract"]["std"]


def test_v2_optimizer_respects_constraints():
    constraints = {
        "Опилки": (50.0, 95.0),
        "Пластик": (0.0, 5.0),
    }
    res = optimize_composition_v2("q", maximize=True, constraints=constraints, seed=42)
    assert res["success"] is True
    comp = res["optimal_composition"]
    if "Опилки" in comp:
        assert comp["Опилки"] >= 49.9
        assert comp["Опилки"] <= 95.1
    if "Пластик" in comp:
        assert comp["Пластик"] <= 5.1


def test_v2_optimizer_incompatible_constraints():
    constraints = {
        "Опилки": (70.0, 80.0),
        "Торф": (40.0, 50.0),
    }
    res = optimize_composition_v2("q", maximize=True, constraints=constraints, seed=42)
    assert res["success"] is False
    assert "Ограничения несовместимы" in res["error"]


def test_v2_optimizer_does_not_contaminate_training_data(tmp_path):
    test_db = str(tmp_path / "test_pellets.db")
    _init_test_db(test_db)

    # До оптимизации: 3 строки
    conn = sqlite3.connect(test_db)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM measured_parameters")
    before_count = cur.fetchone()[0]
    assert before_count == 3
    conn.close()

    res = optimize_composition_v2("q", maximize=True, db_path=test_db, seed=42)
    assert res["success"] is True

    # После оптимизации: В measured_parameters должно остаться ровно 3 строки!
    conn = sqlite3.connect(test_db)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM measured_parameters")
    after_count = cur.fetchone()[0]
    assert after_count == 3, "Оптимизация v2 НЕ должна добавлять записи в measured_parameters!"

    # В ml_optimizations должна появиться ровно 1 запись для истории
    cur.execute("SELECT COUNT(*), target_property, algorithm FROM ml_optimizations")
    row = cur.fetchone()
    assert row[0] == 1
    assert row[1] == "q"
    assert "v2" in row[2]
    conn.close()


def test_training_status_persistence(tmp_path):
    test_db = str(tmp_path / "test_pellets.db")
    _init_test_db(test_db)

    from app.routes.ml import _update_training_status, _read_training_status
    _update_training_status(
        db_path=test_db,
        is_training=True,
        progress=42,
        message="Тестовый прогресс",
    )

    status = _read_training_status(db_path=test_db)
    assert status["is_training"] is True
    assert status["progress"] == 42
    assert status["message"] == "Тестовый прогресс"


def test_route_ml_optimize_v2(tmp_path):
    test_db = str(tmp_path / "test_pellets.db")
    _init_test_db(test_db)

    set_runtime_override("v2", None)

    app = create_app({
        'TESTING': True,
        'WTF_CSRF_ENABLED': False,
        'SECRET_KEY': 'test_secret_b4',
        'DATABASE_PATH': test_db,
    })

    with app.test_client() as client:
        with client.session_transaction() as sess:
            sess['user_id'] = 1
            sess['username'] = 'researcher'
            sess['role'] = 'user'

        resp = client.post('/ml_optimize', data={
            'target_property': 'q',
            'maximize': 'true',
            'min_Опилки': '50',
            'max_Опилки': '95',
        })
        assert resp.status_code == 200
        payload = resp.get_json()
        assert payload['success'] is True
        assert payload['model_version'] == 'v2'
        assert payload['in_domain'] is True
        assert payload['optimal_value'] > 15.0
        assert 'contract' in payload
