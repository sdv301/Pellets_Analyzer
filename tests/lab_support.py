"""Общий доступ тестов к распакованному субмодулю или к каталогу lab/."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def lab_root() -> Path:
    for key in ("PELLET_LAB_PATH", "LAB_PATH"):
        raw = os.environ.get(key, "").strip()
        if raw and (Path(raw) / "pellet_ml" / "__init__.py").is_file():
            os.environ["PELLET_LAB_PATH"] = str(Path(raw).resolve())
            return Path(raw).resolve()
    bundled = REPO / "lab"
    if (bundled / "pellet_ml" / "__init__.py").is_file():
        os.environ["PELLET_LAB_PATH"] = str(bundled)
        return bundled
    pytest.fail(
        "Нет пакета pellet_ml. Выполните git submodule update --init lab "
        "или задайте PELLET_LAB_PATH на распакованный снимок лаборатории."
    )


def ensure_lab():
    root = lab_root()
    os.environ["PELLET_LAB_PATH"] = str(root)
    from app.services.ml_v2.lab_import import check_lab_at_startup

    state = check_lab_at_startup()
    if not state["available"]:
        pytest.fail(state["reason"])
    import pellet_ml

    return pellet_ml


def small_artifact(tmp_path_factory) -> Path:
    """Короткий артефакт q/density/kf/ad для тестов API, без полного прогона GP."""
    pellet_ml = ensure_lab()
    from app.services.ml_v2.lab_import import expected_lab_commit
    from app.services.ml_v2.predictor import stamp_manifest

    destination = Path(tmp_path_factory.mktemp("v2-artifact"))
    pellet_ml.train_and_save(
        lab_root() / "data" / "recipes.csv",
        destination,
        targets=["q", "density", "kf", "ad"],
        models=["ridge"],
    )
    stamp_manifest(destination, expected_lab_commit())
    from app.services.ml_v2.predictor import reset_predictor

    reset_predictor()
    return destination
