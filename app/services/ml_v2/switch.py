"""Переключатель MODEL_VERSION=legacy|v2 и теневой прогон v2.

legacy — значение по умолчанию. Откат: вернуть переменную в legacy (или убрать её).
Теневой режим только логирует разницу и ничего не пишет в БД.
"""
from __future__ import annotations

import logging
import os
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)

_TRUE = ("1", "true", "yes", "on")


def model_version() -> str:
    raw = os.environ.get("MODEL_VERSION", "legacy")
    value = str(raw).strip().lower() if raw is not None else "legacy"
    if value in ("legacy", "v2"):
        return value
    logger.warning("Неизвестное MODEL_VERSION=%s, остаётся legacy", raw)
    return "legacy"


def shadow_enabled() -> bool:
    """Тень включается только пока активен legacy."""
    if model_version() != "legacy":
        return False
    raw = os.environ.get("ML_SHADOW_V2", "")
    return str(raw).strip().lower() in _TRUE


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
    if contract is None:
        logger.info(
            "ml_shadow property=%s legacy=%s v2=None delta=None in_domain=None informative=None",
            target_property,
            legacy_value,
        )
        return True
    delta = None
    if legacy_value is not None:
        try:
            delta = float(contract["value"]) - float(legacy_value)
        except (TypeError, ValueError):
            delta = None
    logger.info(
        "ml_shadow property=%s legacy=%s v2=%s delta=%s in_domain=%s informative=%s",
        target_property,
        legacy_value,
        contract.get("value"),
        delta,
        contract.get("in_domain"),
        contract.get("informative"),
    )
    return True
