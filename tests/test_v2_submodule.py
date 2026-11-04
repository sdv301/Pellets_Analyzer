"""Субмодуль: gitlink, старт без lab/, отказ при чужом мажоре API, допуск снимка."""
import json
import os
from pathlib import Path

import pytest

from app.services.ml_v2.lab_import import check_lab_at_startup, read_pin_file
from scripts.compare_v2_baseline import compare_payload
from tests.lab_support import ensure_lab

ROOT = Path(__file__).resolve().parents[1]
PIN = "f001dead23564d5cae97bb8b42ac0b81b0bf2791"


def test_gitmodules_and_pin_file():
    text = (ROOT / ".gitmodules").read_text(encoding="utf-8")
    assert 'path = lab' in text
    assert "https://github.com/sdv301/Tolko_model.git" in text
    assert read_pin_file() == PIN


def test_gitlink_is_pinned():
    import subprocess

    completed = subprocess.run(
        ["git", "ls-files", "-s", "lab"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    parts = completed.stdout.split()
    assert parts[0] == "160000"
    assert parts[1] == PIN
    assert parts[-1] == "lab"


def test_app_starts_when_submodule_is_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("PELLET_LAB_PATH", str(tmp_path))
    monkeypatch.delenv("LAB_PATH", raising=False)
    state = check_lab_at_startup()
    assert state["available"] is False
    assert "legacy" in state["reason"]

    import app as app_module

    def _light(flask_app):
        flask_app.config["SKIP_DB"] = True

    monkeypatch.setattr(app_module, "_initialize_extensions", _light)
    flask_app = app_module.create_app({"TESTING": True, "SECRET_KEY": "test-secret"})
    assert flask_app is not None


def test_wrong_api_major_refuses_v2(monkeypatch):
    pellet_ml = ensure_lab()
    monkeypatch.setattr(pellet_ml, "API_VERSION", "2.0.0")
    state = check_lab_at_startup()
    assert state["available"] is False
    assert "мажор" in state["reason"]
    monkeypatch.setattr(pellet_ml, "API_VERSION", "1.0.0")
    restored = check_lab_at_startup()
    assert restored["available"] is True


def test_prediction_drift_is_reported():
    payload = json.loads((ROOT / "tests" / "fixtures" / "v2_baseline.json").read_text(encoding="utf-8"))
    assert payload["lab_commit"] == PIN
    assert payload["api_version"] == "1.0.0"
    same = compare_payload(payload["recipes"], payload)
    assert same == []
    drifted = json.loads(json.dumps(payload))
    drifted["recipes"][0]["predictions"]["q"]["value"] = 0.0
    drifted["recipes"][0]["predictions"]["q"]["model_name"] = "other"
    problems = compare_payload(drifted["recipes"], payload)
    assert any("model_name" in line for line in problems)
    assert any("value" in line for line in problems)


def test_adapter_import_does_not_require_a_database(monkeypatch):
    ensure_lab()
    monkeypatch.delenv("PELLET_MODEL_DIR", raising=False)
    from app.services.ml_v2.switch import model_version

    monkeypatch.delenv("MODEL_VERSION", raising=False)
    assert model_version() == "legacy"
    assert os.environ.get("MODEL_VERSION") in (None, "", "legacy")
