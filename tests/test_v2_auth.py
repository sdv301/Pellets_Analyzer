"""ML-роуты закрыты login_required. v2 отдаёт интервал и не пишет прогнозы в БД."""
import pytest

ML_ROUTES = [
    ("GET", "/ml_dashboard"),
    ("POST", "/ml_system_train"),
    ("GET", "/ml_train_status"),
    ("POST", "/ml_augment_database"),
    ("POST", "/ml_optimize"),
    ("POST", "/ml_predict"),
    ("GET", "/ml_system_status"),
    ("POST", "/api/predict_composition"),
    ("POST", "/ml_auto_retrain"),
    ("POST", "/ml_load_excel"),
    ("GET", "/ai_ml_recommendations"),
    ("GET", "/ai_ml_history"),
]


@pytest.fixture()
def isolated_app(tmp_path, monkeypatch):
    import app as app_module

    def _light(flask_app):
        from app.auth.auth import init_auth_tables
        from app.database.database import init_db
        from app.routes import ml

        db_path = str(tmp_path / "auth.db")
        init_db(db_path)
        init_auth_tables(db_path)
        ml.set_db_path(db_path)

    monkeypatch.setattr(app_module, "_initialize_extensions", _light)
    flask_app = app_module.create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "test-secret",
            "UPLOAD_FOLDER": str(tmp_path / "Uploads"),
        }
    )
    (tmp_path / "Uploads").mkdir(parents=True, exist_ok=True)
    return flask_app


@pytest.fixture()
def anon(isolated_app):
    return isolated_app.test_client()


@pytest.fixture()
def authed(anon):
    with anon.session_transaction() as sess:
        sess["user_id"] = 1
        sess["username"] = "tester"
        sess["role_name"] = "user"
        sess["email"] = "tester@example.com"
    return anon


def test_anonymous_ml_routes_redirect_to_login(anon):
    for method, path in ML_ROUTES:
        response = anon.open(path, method=method, json={})
        assert response.status_code == 302, path
        assert "/login" in response.headers["Location"]


def test_ml_views_are_wrapped_with_login_required(isolated_app):
    protected = [name for name in isolated_app.view_functions if name.startswith("ml.")]
    assert protected
    for name in protected:
        view = isolated_app.view_functions[name]
        assert hasattr(view, "__wrapped__"), name


def test_v2_predict_keeps_value_and_adds_interval(authed, monkeypatch):
    monkeypatch.setenv("MODEL_VERSION", "v2")
    writes = []
    monkeypatch.setattr(
        "app.database.database.insert_ml_model_metrics",
        lambda *args, **kwargs: writes.append("metrics"),
    )
    monkeypatch.setattr(
        "app.database.database.insert_data",
        lambda *args, **kwargs: writes.append("rows"),
    )
    monkeypatch.setattr(
        "app.auth.auth.save_ml_model",
        lambda *args, **kwargs: writes.append("model"),
    )

    status = authed.get("/ml_system_status")
    assert status.status_code == 200
    body = status.get_json()
    assert body["success"] is True
    assert body["system_status"]["model_version"] == "v2"
    assert body["system_status"]["metrics_kind"] == "nested_cv"
    q_metrics = body["system_status"]["model_metrics"]["q"]["training_metrics"]
    assert "r2_train" not in q_metrics
    assert q_metrics["metrics_kind"] == "nested_cv"
    assert q_metrics["r2_score"] == pytest.approx(q_metrics["r2_test_nested"])
    assert q_metrics["r2_score"] < 0.99
    density = body["system_status"]["model_metrics"]["density"]["training_metrics"]
    assert density["informative"] is False

    predicted = authed.post(
        "/ml_predict",
        json={"composition": {"Опилки": 95, "Рапсовое масло": 5}, "target_property": "q"},
    )
    assert predicted.status_code == 200
    payload = predicted.get_json()
    assert payload["success"] is True
    assert "prediction" in payload
    contract = payload["contract"]
    assert contract["lo90"] < contract["value"] < contract["hi90"]
    assert contract["in_domain"] is True
    assert payload["prediction"] == pytest.approx(contract["value"])

    outside = authed.post(
        "/ml_predict",
        json={"composition": {"Опилки": 80, "Древесная мука": 20}, "target_property": "q"},
    )
    assert outside.get_json()["contract"]["in_domain"] is False

    sandbox = authed.post(
        "/api/predict_composition",
        json={"composition": {"Опилки": 95, "Рапсовое масло": 5}},
    )
    results = sandbox.get_json()["results"]
    assert "value" in results["q"] and "display_name" in results["q"]
    assert "lo90" in results["q"] and "hi90" in results["q"]
    assert results["density"]["informative"] is False
    assert results["density"]["in_domain"] is True

    train = authed.post("/ml_system_train", data={})
    assert train.get_json()["success"] is True
    assert train.get_json()["model_version"] == "v2"
    blocked = authed.post("/ml_optimize", data={"target_property": "q"})
    assert blocked.get_json()["success"] is False
    retrain = authed.post("/ml_auto_retrain")
    assert retrain.get_json()["success"] is False
    assert writes == []


def test_legacy_response_stays_numeric_without_contract(authed, monkeypatch):
    monkeypatch.setenv("MODEL_VERSION", "legacy")
    monkeypatch.delenv("ML_SHADOW_V2", raising=False)

    class _System:
        class predictor:
            is_trained = True
            main_target_properties = ["q"]
            target_properties_mapping = {"q": "Теплота сгорания"}

            @staticmethod
            def predict(composition, target_property):
                return 17.5

    monkeypatch.setattr("app.routes.ml._get_ml_system", lambda: _System())
    response = authed.post(
        "/ml_predict",
        json={"composition": {"Опилки": 100}, "target_property": "q"},
    )
    body = response.get_json()
    assert body["success"] is True
    assert body["prediction"] == pytest.approx(17.5)
    assert body["property"] == "q"
    assert "contract" not in body
    assert body["model_version"] == "legacy"

    many = authed.post("/api/predict_composition", json={"composition": {"Опилки": 100}})
    row = many.get_json()["results"]["q"]
    assert set(row) == {"value", "display_name"}
    assert row["value"] == pytest.approx(17.5)


def test_dashboard_renders_for_authenticated_user(authed, monkeypatch):
    monkeypatch.setenv("MODEL_VERSION", "v2")
    response = authed.get("/ml_dashboard")
    assert response.status_code == 200
    page = response.get_data(as_text=True)
    assert "avgAccuracyLabel" in page
    assert "Вне области применимости" in page
    assert "lo90" in page
