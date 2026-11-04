"""Парсер v2 сохраняет масла и листья; обучающая выборка не содержит утечек."""
from pathlib import Path

import pandas as pd
import pytest

from app.services.ml_v2.components_registry import parse_composition
from app.services.ml_v2.dataset import (
    correct_kf_value,
    load_real_recipes,
    validate_kf,
)

ROOT = Path(__file__).resolve().parents[1]


def test_unknown_component_raises():
    with pytest.raises(ValueError, match="неизвестный компонент"):
        parse_composition("90% Опилки, 10% банан")


def test_oils_and_leaves_survive():
    oil = parse_composition("95% Опилки, 5% рапсовое масло")
    assert oil["Рапсовое масло"] == pytest.approx(5.0)
    assert oil["Опилки"] == pytest.approx(95.0)
    motor = parse_composition("90% Опилки, 10% отработанное моторное масло")
    assert motor["Отработанное моторное масло"] == pytest.approx(10.0)
    leaves = parse_composition("70% Опилки, 30% листья")
    assert leaves["Листья"] == pytest.approx(30.0)
    assert abs(sum(leaves.values()) - 100) <= 0.5


def test_all_real_recipes_parse_without_leakage():
    frame = load_real_recipes()
    assert len(frame) == 48
    assert set(frame["source"]) == {"real"}
    text = frame["composition"].astype(str)
    assert not text.str.startswith("УЧИТЕЛЬ").any()
    assert not text.str.startswith("Опыт").any()
    assert not text.str.startswith("ML-опыт").any()
    assert frame["recipe_id"].nunique() == len(frame)
    for recipe in text:
        parsed = parse_composition(recipe, strict=True)
        assert parsed, recipe
        assert abs(sum(parsed.values()) - 100) <= 0.5
        assert all(value > 0 for value in parsed.values())
    leaves = frame[frame["composition"].str.contains("листья", case=False)]
    assert len(leaves) == 2
    row = leaves.loc[leaves["composition"].str.contains("30%"), "kf"].iloc[0]
    assert float(row) == pytest.approx(98.508, abs=0.01)


def test_kf_validator_and_correction():
    corrected, changed = correct_kf_value(0.98508)
    assert changed
    assert corrected == pytest.approx(98.508)
    assert validate_kf(0.5)
    with pytest.raises(ValueError):
        validate_kf(120)


def test_legacy_baseline_shows_dropped_oils():
    import json

    payload = json.loads((ROOT / "tests" / "fixtures" / "legacy_baseline.json").read_text(encoding="utf-8"))
    by_text = {row["composition"]: row for row in payload["recipes"]}
    sawdust = by_text["100% Опилки"]
    assert sawdust["legacy_parsed"] == {"Опилки": 100.0}
    for text in (
        "95% Опилки, 5% рапсовое масло",
        "90% Опилки, 10% отработанное моторное масло",
        "70% Опилки, 30% листья",
    ):
        assert by_text[text]["legacy_parsed"] == {"Опилки": 100.0}
        assert by_text[text]["predictions"] == sawdust["predictions"]
        fixed = parse_composition(text)
        assert fixed != by_text[text]["legacy_parsed"]
        assert abs(sum(fixed.values()) - 100) <= 0.5


def test_shipped_recipes_match_loader():
    raw = pd.read_csv(ROOT / "app" / "services" / "ml_v2" / "data" / "recipes.csv")
    assert (raw["source"] == "real").sum() == 48
