"""Загрузка артефакта v2, собранного `python -m pellet_ml build`.

Каталог по умолчанию — models/v2 (PELLET_MODEL_DIR его подменяет).
joblib в git не хранится: его собирает образ или scripts/update_lab.sh.
Прогнозы в БД отсюда не пишутся.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Dict, Optional

from app.services.ml_v2.lab_import import (
    EXPECTED_API_MAJOR,
    LabUnavailable,
    api_major,
    expected_lab_commit,
    require_lab,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
_LOCK = threading.Lock()
_PREDICTOR: Optional["V2Predictor"] = None


def model_dir() -> Path:
    raw = os.environ.get("PELLET_MODEL_DIR", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return REPO_ROOT / "models" / "v2"


class V2Predictor:
    def __init__(self, directory: Optional[Path] = None):
        self.model_dir = Path(directory) if directory is not None else model_dir()
        self.loaded = None
        self.manifest: Dict = {}
        self.is_loaded = False

    def load(self) -> bool:
        pellet_ml = require_lab()
        directory = self.model_dir
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file() or not (directory / "pellet_models.joblib").is_file():
            logger.error("Отказ загрузки v2: в %s нет manifest.json и pellet_models.joblib", directory)
            return False
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        saved_api = str(manifest.get("api_version") or "")
        if api_major(saved_api) != EXPECTED_API_MAJOR or api_major(saved_api) != api_major(pellet_ml.API_VERSION):
            logger.error(
                "Отказ загрузки v2: api_version артефакта %s несовместим с пакетом %s (нужен мажор %s). Legacy работает.",
                saved_api or "пусто",
                pellet_ml.API_VERSION,
                EXPECTED_API_MAJOR,
            )
            return False
        pinned = expected_lab_commit()
        saved_commit = str(manifest.get("lab_commit") or "")
        if not pinned or not saved_commit or saved_commit != pinned:
            logger.error(
                "Отказ загрузки v2: lab_commit артефакта %s не совпал с пином %s. "
                "Пересоберите артефакт scripts/build_v2_artifact.py. Legacy работает.",
                saved_commit or "пусто",
                pinned or "не задан",
            )
            return False
        try:
            loaded = pellet_ml.load_models(directory)
        except Exception as exc:
            logger.error("Отказ загрузки v2: %s. Legacy работает.", exc)
            return False
        self.loaded = loaded
        self.manifest = loaded.manifest
        self.is_loaded = True
        return True

    def predict_all(self, composition) -> Dict[str, dict]:
        if not self.is_loaded and not self.load():
            raise LabUnavailable(
                f"артефакт v2 не загружен из {self.model_dir}. "
                "Соберите его: python -m pellet_ml build --data lab/data/recipes.csv --out models/v2"
            )
        pellet_ml = require_lab()
        return pellet_ml.predict(composition, models=self.loaded)

    def predict_contract(self, composition, target_property: str) -> Optional[dict]:
        pellet_ml = require_lab()
        target_property = pellet_ml.properties.resolve_property_id(target_property)
        contracts = self.predict_all(composition)
        return contracts.get(target_property)


def get_predictor() -> V2Predictor:
    global _PREDICTOR
    with _LOCK:
        if _PREDICTOR is None or not _PREDICTOR.is_loaded:
            predictor = V2Predictor()
            if not predictor.load():
                from app.services.ml_v2.lab_import import _state

                reason = _state.get("reason") or f"не удалось загрузить v2 из {predictor.model_dir}"
                raise LabUnavailable(reason)
            _PREDICTOR = predictor
        return _PREDICTOR


def reset_predictor() -> None:
    global _PREDICTOR
    with _LOCK:
        _PREDICTOR = None


def public_metrics(metrics: Dict) -> Dict:
    """Поля для UI. In-sample R² сюда не входит."""
    r2_test = metrics.get("r2_test")
    return {
        "metrics_kind": "nested_cv",
        "r2_test_nested": r2_test,
        "r2_score": r2_test,
        "cv_r2": r2_test,
        "baseline_r2": metrics.get("baseline_r2"),
        "n": metrics.get("n"),
        "informative": bool(metrics.get("informative")),
        "cv_scheme": metrics.get("cv_scheme") or "nested_loro_group",
        "coverage90": metrics.get("coverage90"),
    }


def v2_system_status(predictor: Optional[V2Predictor] = None) -> Dict:
    pellet_ml = require_lab()
    predictor = predictor or get_predictor()
    manifest = predictor.manifest or {}
    metrics_map = manifest.get("metrics") or {}
    model_metrics = {}
    for prop, metrics in metrics_map.items():
        shown = public_metrics(metrics)
        model_metrics[prop] = {
            "display_name": pellet_ml.properties.display_name(prop),
            "algorithm_used": metrics.get("model"),
            "algorithm_reason": (
                "Модель выбрана по nested leave-one-recipe-out CV пакета pellet_ml. "
                "В интерфейсе показан R² test, не in-sample."
            ),
            "training_metrics": shown,
            "informative": shown["informative"],
        }
    components = list(manifest.get("components") or [])
    return {
        "model_version": "v2",
        "is_trained": bool(predictor.is_loaded),
        "trained_models": list(metrics_map.keys()),
        "available_components": components,
        "training_data_size": int(manifest.get("n_real") or 0),
        "metrics_kind": "nested_cv",
        "lab_commit": manifest.get("lab_commit"),
        "api_version": manifest.get("api_version"),
        "model_metrics": model_metrics,
        "target_properties_mapping": {prop: pellet_ml.properties.display_name(prop) for prop in metrics_map},
        "ml_optimizations_count": 0,
    }


def stamp_manifest(directory: Path, lab_commit: str) -> dict:
    """Дописывает lab_commit в manifest.json. sha256 joblib не меняется."""
    path = Path(directory) / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    pellet_ml = require_lab()
    saved_api = str(payload.get("api_version") or "")
    if api_major(saved_api) != api_major(pellet_ml.API_VERSION):
        raise LabUnavailable(
            f"сборка записала api_version {saved_api}, пакет сообщает {pellet_ml.API_VERSION}"
        )
    if not lab_commit:
        raise LabUnavailable("нечем записать lab_commit: нет пина субмодуля")
    payload["lab_commit"] = lab_commit
    payload["api_version"] = pellet_ml.API_VERSION
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
