"""Единый реестр компонентов и парсер состава.

Паттерны повторяют проверенный `parse_fixed` из независимой оценки:
масла и листья не отбрасываются. Неизвестный фрагмент — ошибка, а не
тихая нормировка остатка до 100%.
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

# Порядок важен: более специфичные шаблоны раньше общих («уголь» не должен
# перехватывать «угольный шлам», «моторн» — раньше прочих).
COMPONENT_PATTERNS: Tuple[Tuple[str, str], ...] = (
    ("Отработанное моторное масло", r"отработанн|моторн"),
    ("Рапсовое масло", r"рапс"),
    ("Подсолнечный жмых", r"жмых"),
    ("Рисовая шелуха", r"рисов|шелух"),
    ("Угольный шлам", r"шлам"),
    ("Бурый уголь", r"бурый|уголь"),
    ("СМС", r"смс|cmc"),
    ("Опилки", r"опил"),
    ("Солома", r"солом"),
    ("Картон", r"картон"),
    ("Торф", r"торф"),
    ("Пластик", r"пластик"),
    ("Листья", r"лист"),
    ("Древесная мука", r"древесн\w*\s*мук|мук\w*\s*древесн"),
    ("Щепа", r"щеп"),
    ("Лигнин", r"лигнин"),
)

CANONICAL_COMPONENTS: Tuple[str, ...] = tuple(name for name, _ in COMPONENT_PATTERNS)

# Компоненты, которые встречаются в 48 реальных рецептурах.
RECIPE_COMPONENTS: Tuple[str, ...] = (
    "Опилки",
    "Картон",
    "Солома",
    "Рисовая шелуха",
    "Торф",
    "Угольный шлам",
    "Пластик",
    "Бурый уголь",
    "СМС",
    "Подсолнечный жмых",
    "Листья",
    "Рапсовое масло",
    "Отработанное моторное масло",
)

_PREFIX = re.compile(
    r"^\s*(?:ML-опыт|Опыт|УЧИТЕЛЬ|ЛАБ(?:-гор)?|РНФ-гор|Предложено|синт)\s*:\s*",
    re.IGNORECASE,
)
_TOKEN = re.compile(r"(\d+(?:\.\d+)?)\s*%\s*([^%,]+)")


def component_patterns_dict() -> Dict[str, List[str]]:
    return {name: [pat] for name, pat in COMPONENT_PATTERNS}


def canonical_name(name: str) -> str:
    """Пробелы и подчёркивания — один канонический вид («Рисовая шелуха»)."""
    raw = str(name).strip().replace("_", " ")
    raw = re.sub(r"\s+", " ", raw)
    matched = match_component(raw)
    return matched if matched else raw


def match_component(name: str) -> str | None:
    text = str(name).strip().replace("_", " ").lower()
    text = re.sub(r"\s+", " ", text)
    if not text:
        return None
    for comp, pat in COMPONENT_PATTERNS:
        if re.search(pat, text):
            return comp
    return None


def parse_composition(text, *, strict: bool = True) -> Dict[str, float]:
    """Разбирает строку состава в доли, %.

    strict=True: неизвестный фрагмент → ValueError, без домножения
    потерянной части на оставшиеся компоненты.
    """
    if text is None:
        return {}
    try:
        import math

        if isinstance(text, float) and math.isnan(text):
            return {}
    except TypeError:
        pass
    raw = str(text).strip()
    if not raw or raw.lower() == "nan":
        return {}

    cleaned = re.sub(r"\(.*?\)", "", raw)
    # Префикс может повторяться после очистки скобок.
    while True:
        nxt = _PREFIX.sub("", cleaned, count=1)
        if nxt == cleaned:
            break
        cleaned = nxt

    out: Dict[str, float] = {}
    unknown: List[str] = []
    for match in _TOKEN.finditer(cleaned):
        pct = float(match.group(1))
        names = [part.strip() for part in match.group(2).split("+") if part.strip()]
        if not names or pct <= 0:
            continue
        share = pct / len(names)
        for nm in names:
            comp = match_component(nm)
            if comp is None:
                unknown.append(nm)
                continue
            out[comp] = out.get(comp, 0.0) + share

    if strict and unknown:
        raise ValueError("неизвестный компонент: " + ", ".join(unknown))

    total = sum(out.values())
    if total > 0 and abs(total - 100.0) > 0.5 and not unknown:
        out = {k: v / total * 100.0 for k, v in out.items()}
    elif total > 0 and abs(total - 100.0) > 0.5 and unknown and not strict:
        out = {k: v / total * 100.0 for k, v in out.items()}

    return {k: round(v, 4) for k, v in out.items() if v > 1e-9}


def canonicalize_composition(composition: Dict[str, float]) -> Dict[str, float]:
    """Сводит синонимы ключей к каноническим именам. Неизвестные ключи сохраняются."""
    merged: Dict[str, float] = {}
    for key, value in composition.items():
        name = str(key)
        if name.startswith("phys_") or name.startswith("synergy_"):
            merged[name] = float(value)
            continue
        canon = match_component(name) or re.sub(r"\s+", " ", name.strip().replace("_", " "))
        merged[canon] = merged.get(canon, 0.0) + float(value)
    return merged
