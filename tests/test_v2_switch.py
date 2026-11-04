"""MODEL_VERSION и теневой режим: лог разницы, без записи прогнозов в БД."""
import logging

import pytest

from app.services.ml_v2.switch import model_version, predict_one, shadow_enabled


class _V2:
    def __init__(self, value=2.5, in_domain=False):
        self.value = value
        self.in_domain = in_domain

    def predict_contract(self, composition, target_property):
        return {
            "value": self.value,
            "std": 0.4,
            "lo90": self.value - 0.6,
            "hi90": self.value + 0.6,
            "in_domain": self.in_domain,
            "informative": True,
            "model_name": "bayes_ridge",
            "note": "",
        }


def test_default_version_is_legacy(monkeypatch):
    monkeypatch.delenv("MODEL_VERSION", raising=False)
    assert model_version() == "legacy"
    monkeypatch.setenv("MODEL_VERSION", "nope")
    assert model_version() == "legacy"
    monkeypatch.setenv("MODEL_VERSION", "v2")
    assert model_version() == "v2"


def test_v2_returns_contract_and_keeps_numeric_prediction(monkeypatch):
    monkeypatch.setenv("MODEL_VERSION", "v2")
    monkeypatch.setattr("app.services.ml_v2.predictor.get_predictor", lambda: _V2(value=3.0, in_domain=True))
    result = predict_one({"Опилки": 100.0}, "q", legacy_predict=lambda *_args: 1.0)
    assert result["model_version"] == "v2"
    assert result["prediction"] == 3.0
    assert result["contract"]["value"] == 3.0
    assert "lo90" in result["contract"]


def test_shadow_logs_delta_and_does_not_touch_storage(monkeypatch, caplog):
    monkeypatch.setenv("MODEL_VERSION", "legacy")
    monkeypatch.setenv("ML_SHADOW_V2", "1")
    assert shadow_enabled() is True
    writes = []
    monkeypatch.setattr(
        "app.database.database.insert_ml_model_metrics",
        lambda *args, **kwargs: writes.append("metrics"),
    )
    monkeypatch.setattr(
        "app.database.database.insert_data",
        lambda *args, **kwargs: writes.append("rows"),
    )
    monkeypatch.setattr("app.services.ml_v2.predictor.get_predictor", lambda: _V2(value=1.5, in_domain=True))
    caplog.set_level(logging.INFO, logger="app.services.ml_v2.switch")
    result = predict_one({"Опилки": 100.0}, "q", legacy_predict=lambda *_args: 1.0)
    assert result["model_version"] == "legacy"
    assert result["prediction"] == 1.0
    assert result["contract"] is None
    assert result["shadow_logged"] is True
    assert "ml_shadow" in caplog.text
    assert "delta=0.5" in caplog.text
    assert writes == []


def test_shadow_is_off_when_v2_is_active(monkeypatch):
    monkeypatch.setenv("MODEL_VERSION", "v2")
    monkeypatch.setenv("ML_SHADOW_V2", "1")
    assert shadow_enabled() is False


def test_shadow_off_by_default(monkeypatch, caplog):
    monkeypatch.setenv("MODEL_VERSION", "legacy")
    monkeypatch.delenv("ML_SHADOW_V2", raising=False)
    caplog.set_level(logging.INFO, logger="app.services.ml_v2.switch")
    result = predict_one({"Опилки": 100.0}, "q", legacy_predict=lambda *_args: 4.0)
    assert result["prediction"] == 4.0
    assert "shadow_logged" not in result
    assert "ml_shadow" not in caplog.text
