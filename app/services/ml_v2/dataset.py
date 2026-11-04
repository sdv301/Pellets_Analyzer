"""Сборка обучающей таблицы v2: только source=real, без IQR и без дублей «Опыт:»/«УЧИТЕЛЬ:».

Прогнозы модели в эту таблицу не дописываются.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from app.services.ml_v2.components_registry import parse_composition

PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_DIR / "data"
RECIPES_CSV = DATA_DIR / "recipes.csv"
COMPONENTS_CSV = DATA_DIR / "components.csv"
HONEST_METRICS_CSV = DATA_DIR / "honest_metrics.csv"

SOURCES = ("real", "lab_experiment", "lab_teacher", "synthetic", "ml_pred")

_ML = re.compile(r"^\s*ML-опыт", re.IGNORECASE)
_TEACHER = re.compile(r"^\s*УЧИТЕЛЬ", re.IGNORECASE)
_EXPERIMENT = re.compile(r"^\s*Опыт\s*:", re.IGNORECASE)
_LAB = re.compile(r"^\s*(ЛАБ|РНФ-гор|ЛАБ-гор)\s*:", re.IGNORECASE)
_SYNTH = re.compile(r"\(aug|синт|synthetic|augment", re.IGNORECASE)


def classify_source(name: str) -> str:
    text = str(name).strip()
    if _ML.match(text):
        return "ml_pred"
    if _TEACHER.match(text):
        return "lab_teacher"
    if _EXPERIMENT.match(text):
        return "lab_experiment"
    if _LAB.match(text):
        return "lab_experiment"
    if _SYNTH.search(text):
        return "synthetic"
    return "real"


def canonical_recipe_id(composition: Dict[str, float]) -> str:
    payload = {
        k: round(float(v), 1)
        for k, v in sorted(composition.items())
        if float(v) > 0 and not str(k).startswith(("phys_", "synergy_"))
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def recipe_id_from_text(text: str, *, strict: bool = True) -> str:
    return canonical_recipe_id(parse_composition(text, strict=strict))


def correct_kf_value(value) -> Tuple[Optional[float], bool]:
    """kf в (0, 1) трактуется как доля и переводится в проценты. Исходное значение остаётся в kf_raw."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None, False
    try:
        if pd.isna(value):
            return None, False
    except TypeError:
        pass
    number = float(value)
    if 0 < number < 1:
        return number * 100.0, True
    return number, False


def validate_kf(value) -> List[str]:
    warnings: List[str] = []
    if value is None:
        return warnings
    try:
        if pd.isna(value):
            return warnings
    except TypeError:
        pass
    number = float(value)
    if number < 0 or number > 100:
        raise ValueError(f"kf должен быть в диапазоне [0, 100], получено {number}")
    if number < 1:
        warnings.append(
            f"kf={number} < 1: похоже на долю, а не на проценты. "
            "Ожидаемый диапазон остальных рецептур — около 92–100."
        )
    return warnings


def annotate_frame(df: pd.DataFrame, *, strict_parse: bool = False) -> pd.DataFrame:
    """Добавляет source, recipe_id и поправку kf. Строки не удаляет и IQR не применяет."""
    if df is None or df.empty:
        return df
    out = df.copy()
    if "composition" not in out.columns:
        return out
    sources = []
    recipe_ids = []
    for comp in out["composition"].astype(str):
        guessed = classify_source(comp)
        sources.append(guessed)
        try:
            recipe_ids.append(recipe_id_from_text(comp, strict=strict_parse))
        except ValueError:
            recipe_ids.append("")
    if "source" not in out.columns:
        out["source"] = sources
    else:
        filled = []
        for existing, guessed in zip(out["source"], sources):
            if existing is None or (isinstance(existing, float) and math.isnan(existing)) or str(existing).strip() == "":
                filled.append(guessed)
            elif str(existing).strip() == "real" and guessed != "real":
                filled.append(guessed)
            else:
                filled.append(str(existing).strip())
        out["source"] = filled
    out["recipe_id"] = recipe_ids
    if "kf" in out.columns:
        raw = []
        corrected = []
        changed = []
        for value in out["kf"]:
            new, flag = correct_kf_value(value)
            raw.append(value)
            corrected.append(new if flag or new is not None else value)
            changed.append(flag)
        if "kf_raw" not in out.columns:
            out["kf_raw"] = raw
        out["kf"] = corrected
        out["kf_corrected_flag"] = changed
    return out


def filter_sources(df: pd.DataFrame, allowed: Iterable[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    allowed_set = set(allowed)
    if "source" not in df.columns:
        df = annotate_frame(df)
    return df.loc[df["source"].isin(allowed_set)].reset_index(drop=True)


def load_real_recipes(path: Optional[Path] = None) -> pd.DataFrame:
    """Только source=real. «Опыт:», «УЧИТЕЛЬ:» и ML-опыт в обучение не попадают."""
    csv_path = Path(path) if path else RECIPES_CSV
    frame = annotate_frame(pd.read_csv(csv_path), strict_parse=True)
    real = filter_sources(frame, ["real"])
    leaked = real["composition"].astype(str).str.match(r"^\s*(Опыт|УЧИТЕЛЬ|ML-опыт)")
    if leaked.any():
        raise RuntimeError("в обучающую выборку попали дубли с префиксом опыта или учителя")
    if real["recipe_id"].eq("").any():
        raise RuntimeError("в recipes.csv нет recipe_id")
    if real["recipe_id"].nunique() != len(real):
        raise RuntimeError("recipe_id не уникален внутри source=real")
    return real.reset_index(drop=True)
