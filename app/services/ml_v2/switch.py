"""Переключатель MODEL_VERSION=legacy|v2 и теневой прогон v2.

legacy — значение по умолчанию. Откат: вернуть переменную в legacy (или убрать её).
Поддерживает переключение из веб-админки (хранение в system_settings и runtime).
Теневой режим логирует разницу в консоль и сохраняет в таблицу ml_shadow_logs.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
from typing import Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_TRUE = ("1", "true", "yes", "on")

_db_path: str = "data/pellets_data.db"
_RUNTIME_MODEL_VERSION: Optional[str] = None
_RUNTIME_SHADOW_ENABLED: Optional[bool] = None


def set_db_path(path: str) -> None:
    """Устанавливает путь к базе данных."""
    global _db_path
    _db_path = path


def get_runtime_override() -> Tuple[Optional[str], Optional[bool]]:
    """Возвращает текущие runtime-переопределения (версия, тень)."""
    return _RUNTIME_MODEL_VERSION, _RUNTIME_SHADOW_ENABLED


def set_runtime_override(version: Optional[str] = None, shadow: Optional[bool] = None) -> None:
    """Устанавливает runtime-переопределение версии модели и тени (или сбрасывает в None)."""
    global _RUNTIME_MODEL_VERSION, _RUNTIME_SHADOW_ENABLED
    if version in ("legacy", "v2", None):
        _RUNTIME_MODEL_VERSION = version
    _RUNTIME_SHADOW_ENABLED = None if shadow is None else bool(shadow)


def _get_setting_from_db(key: str) -> Optional[str]:
    """Считывает настройку из system_settings с защитой от ошибок."""
    try:
        if not os.path.exists(_db_path):
            return None
        conn = sqlite3.connect(_db_path)
        cur = conn.cursor()
        cur.execute("SELECT value FROM system_settings WHERE key = ?", (key,))
        row = cur.fetchone()
        conn.close()
        return str(row[0]) if row and row[0] is not None else None
    except Exception:
        return None


def model_version() -> str:
    # 1. Runtime override (если явно переключено администратором через веб-панель)
    if _RUNTIME_MODEL_VERSION in ("legacy", "v2"):
        return _RUNTIME_MODEL_VERSION

    # 2. Переменная окружения (если явно задана пользователем или тестом)
    raw = os.environ.get("MODEL_VERSION")
    if raw is not None and str(raw).strip():
        value = str(raw).strip().lower()
        if value in ("legacy", "v2"):
            return value
        logger.warning("Неизвестное MODEL_VERSION=%s, остаётся legacy", raw)
        return "legacy"

    # 3. Сохранённое значение в БД system_settings
    db_val = _get_setting_from_db("MODEL_VERSION")
    if db_val and db_val.strip().lower() in ("legacy", "v2"):
        return db_val.strip().lower()

    return "legacy"


def shadow_enabled() -> bool:
    """Тень включается только пока активен legacy."""
    if model_version() != "legacy":
        return False

    # 1. Runtime override
    if _RUNTIME_SHADOW_ENABLED is not None:
        return _RUNTIME_SHADOW_ENABLED

    # 2. Переменная окружения
    raw = os.environ.get("ML_SHADOW_V2")
    if raw is not None and str(raw).strip():
        return str(raw).strip().lower() in _TRUE

    # 3. Сохранённое значение в БД system_settings
    db_val = _get_setting_from_db("ML_SHADOW_V2")
    if db_val is not None:
        return db_val.strip().lower() in _TRUE

    return False


def predict_one(
    composition: Dict[str, float],
    target_property: str,
    legacy_predict: Callable[[Dict[str, float], str], Optional[float]],
) -> Dict:
    """Единая точка прогноза для роутов.

    В режиме v2 число `prediction` — это `contract['value']` (совместимость со старым JSON).
    В режиме legacy контракт не подмешивается в ответ; при ML_SHADOW_V2 пишется только лог.
    """
    if model_version() == "v2":
        from app.services.ml_v2.predictor import get_predictor

        contract = get_predictor().predict_contract(composition, target_property)
        value = None if contract is None else contract["value"]
        return {"model_version": "v2", "prediction": value, "contract": contract}

    value = legacy_predict(composition, target_property)
    payload = {"model_version": "legacy", "prediction": value, "contract": None}
    if shadow_enabled():
        payload["shadow_logged"] = _log_shadow(composition, target_property, value)
    return payload


def _log_shadow(composition: Dict[str, float], target_property: str, legacy_value) -> bool:
    try:
        from app.services.ml_v2.predictor import get_predictor

        contract = get_predictor().predict_contract(composition, target_property)
    except Exception as exc:
        logger.warning("ml_shadow property=%s failed: %s", target_property, exc)
        return False

    v2_val = None
    delta = None
    in_domain = None
    informative = None

    if contract is not None:
        v2_val = contract.get("value")
        in_domain = contract.get("in_domain")
        informative = contract.get("informative")
        if legacy_value is not None and v2_val is not None:
            try:
                delta = float(v2_val) - float(legacy_value)
            except (TypeError, ValueError):
                delta = None

    logger.info(
        "ml_shadow property=%s legacy=%s v2=%s delta=%s in_domain=%s informative=%s",
        target_property,
        legacy_value,
        v2_val,
        delta,
        in_domain,
        informative,
    )

    # Сохраняем в таблицу ml_shadow_logs для админ-панели (G2)
    _record_shadow_log_to_db(target_property, legacy_value, v2_val, delta, in_domain, informative, composition)
    return True


def _record_shadow_log_to_db(
    target_property: str,
    legacy_value,
    v2_value,
    delta,
    in_domain,
    informative,
    composition: Optional[Dict[str, float]] = None,
) -> None:
    """Записывает расхождение теневого прогона в SQLite."""
    try:
        if not os.path.exists(_db_path):
            return
        conn = sqlite3.connect(_db_path)
        cur = conn.cursor()
        cur.execute('''
            INSERT INTO ml_shadow_logs (target_property, legacy_value, v2_value, delta, in_domain, informative, details)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            target_property,
            float(legacy_value) if legacy_value is not None else None,
            float(v2_value) if v2_value is not None else None,
            float(delta) if delta is not None else None,
            1 if in_domain else (0 if in_domain is False else None),
            1 if informative else (0 if informative is False else None),
            json.dumps(composition, ensure_ascii=False) if composition else None,
        ))
        conn.commit()
        conn.close()
    except Exception as exc:
        logger.debug("Не удалось записать теневой лог в БД: %s", exc)
