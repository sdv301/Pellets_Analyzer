"""Справочник свойств пеллеты: id, подпись, единица, можно ли обучать."""
from __future__ import annotations

from typing import Dict, List

# id в данных и в модели. td1/td2 — устаревшие алиасы.
PROPERTY_ALIASES = {
    "td1": "tau_d1",
    "td2": "tau_d2",
}

PROPERTY_CATALOG: Dict[str, Dict[str, str]] = {
    "density": {"label": "Плотность", "unit": "кг/м³", "kind": "measured"},
    "kf": {"label": "Ударопрочность", "unit": "%", "kind": "measured"},
    "strength_limit": {"label": "Предел прочности", "unit": "МПа", "kind": "measured"},
    "tau_d1": {"label": "Время задержки горения 1", "unit": "с", "kind": "measured"},
    "tau_d2": {"label": "Время задержки горения 2", "unit": "с", "kind": "measured"},
    "tau_b": {"label": "Время выгорания", "unit": "с", "kind": "measured"},
    "q": {"label": "Теплота сгорания", "unit": "МДж/кг", "kind": "measured_or_computed"},
    "ad": {"label": "Зольность на сухую массу", "unit": "%", "kind": "measured_or_computed"},
    "tign": {"label": "Температура зажигания", "unit": "°C", "kind": "measured"},
    "tb": {"label": "Температура выгорания", "unit": "°C", "kind": "measured"},
    "mass_loss": {"label": "Потеря массы", "unit": "%", "kind": "measured"},
    "kt": {"label": "Устойчивость к вибрациям", "unit": "%", "kind": "measured"},
    "h": {"label": "Гигроскопичность", "unit": "%", "kind": "measured"},
    "co2": {"label": "Выбросы CO2", "unit": "%", "kind": "measured"},
    "co": {"label": "Выбросы CO", "unit": "%", "kind": "measured"},
    "so2": {"label": "Выбросы SO2", "unit": "%", "kind": "measured"},
    "nox": {"label": "Выбросы NOx", "unit": "ppm", "kind": "measured"},
    # В measured_parameters пусты на 100% — это свойства сырья, не цели пеллеты.
    "war": {"label": "Влажность на аналитическую массу", "unit": "%", "kind": "component_only"},
    "vd": {"label": "Содержание летучих на сухую массу", "unit": "%", "kind": "component_only"},
    "cd": {"label": "Содержание углерода на сухую массу", "unit": "%", "kind": "component_only"},
    "hd": {"label": "Содержание водорода на сухую массу", "unit": "%", "kind": "component_only"},
    "nd": {"label": "Содержание азота на сухую массу", "unit": "%", "kind": "component_only"},
    "sd": {"label": "Содержание серы на сухую массу", "unit": "%", "kind": "component_only"},
    "od": {"label": "Содержание кислорода на сухую массу", "unit": "%", "kind": "component_only"},
    "hardness": {"label": "Твердость", "unit": "", "kind": "component_only"},
}

TRAINABLE_TARGETS: List[str] = [
    "density",
    "kf",
    "tau_d1",
    "tau_d2",
    "tau_b",
    "strength_limit",
    "q",
    "ad",
    "tign",
    "tb",
    "mass_loss",
    "kt",
    "h",
    "co2",
    "co",
    "so2",
    "nox",
]

EMPTY_TARGETS = ["war", "vd", "cd", "hd", "nd", "sd", "od", "hardness"]

# На 48 рецептурах ни одна модель не лучше среднего в пределах шума.
NON_INFORMATIVE_TARGETS = {"density", "kf", "ad"}

# Происхождение q не установлено: в двух проектах числа разные, правило
# смешения по таблице components даёт R²≈0.40. Обучаем сохранённое значение.
Q_ORIGIN_NOTE = (
    "Колонка q в выгрузке Tolko_model сохранена как есть (q_origin=stored_unverified). "
    "Совпадает ли она с лабораторным измерением или с расчётом — по репозиторию не установить: "
    "правило смешения по таблице components объясняет её лишь с R²≈0.40, а значения q "
    "отличаются от выгрузки Pellets_Analyzer у всех 48 рецептур. "
    "Высокий R² модели по q не доказывает, что теплота «измерена»."
)

STRENGTH_ORIGIN_NOTE = (
    "strength_limit хранится в МПа (диапазон реальных строк примерно 1.4–13, "
    "делений >20 нет, эвристика кг/см²→МПа по умолчанию выключена). "
    "31 значение на real-строках; происхождение (прямой замер или калибровочная кривая) "
    "по коду не доказывается. Лабораторный Excel есть (artifacts/Характеристики Данилу.xlsx) "
    "и используется только как внешняя проверка, не как обучающая выборка."
)


def resolve_property_id(name: str) -> str:
    key = str(name).strip()
    return PROPERTY_ALIASES.get(key, key)


def is_trainable(name: str) -> bool:
    return resolve_property_id(name) in TRAINABLE_TARGETS


def display_name(name: str) -> str:
    key = resolve_property_id(name)
    meta = PROPERTY_CATALOG.get(key)
    if not meta:
        return key
    unit = meta.get("unit") or ""
    return f"{meta['label']}, {unit}" if unit else meta["label"]


def target_properties_mapping() -> Dict[str, str]:
    return {key: display_name(key) for key in PROPERTY_CATALOG}
