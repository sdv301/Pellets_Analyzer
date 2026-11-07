"""Парсер лаборатории сохраняет масла и листья; снимок legacy это по-прежнему показывает."""
import json
from pathlib import Path

import pandas as pd
import pytest

from tests.lab_support import ensure_lab, lab_root

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module", autouse=True)
def _lab():
    ensure_lab()


def test_unknown_component_raises():
    import pellet_ml

    with pytest.raises(ValueError, match="неизвестный компонент"):
        pellet_ml.parse_composition("90% Опилки, 10% банан")


def test_oils_and_leaves_survive():
    import pellet_ml

    oil = pellet_ml.parse_composition("95% Опилки, 5% рапсовое масло")
    assert oil["Рапсовое масло"] == pytest.approx(5.0)
    assert oil["Опилки"] == pytest.approx(95.0)
    motor = pellet_ml.parse_composition("90% Опилки, 10% отработанное моторное масло")
    assert motor["Отработанное моторное масло"] == pytest.approx(10.0)
    leaves = pellet_ml.parse_composition("70% Опилки, 30% листья")
    assert leaves["Листья"] == pytest.approx(30.0)
    assert abs(sum(leaves.values()) - 100) <= 0.5


def test_real_recipes_have_no_teacher_rows():
    import pellet_ml

    frame = pellet_ml.dataset.build_training_frame(lab_root() / "data" / "recipes.csv")
    assert len(frame) == 48
    assert set(frame["source"]) == {"real"}
    text = frame["composition"].astype(str)
    assert not text.str.startswith("УЧИТЕЛЬ").any()
    assert not text.str.startswith("Опыт").any()
    assert not text.str.startswith("ML-опыт").any()
    leaves = frame[frame["composition"].str.contains("30% листья", case=False)]
    assert float(leaves["kf"].iloc[0]) == pytest.approx(98.508, abs=0.01)


def test_kf_validator_and_correction():
    import pellet_ml

    corrected, changed = pellet_ml.dataset.correct_kf_value(0.98508)
    assert changed
    assert corrected == pytest.approx(98.508)
    assert pellet_ml.dataset.validate_kf(0.5)
    with pytest.raises(ValueError):
        pellet_ml.dataset.validate_kf(120)


def test_legacy_baseline_shows_dropped_oils():
    import pellet_ml

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
        fixed = pellet_ml.parse_composition(text)
        assert fixed != by_text[text]["legacy_parsed"]


def test_lab_csv_is_not_copied_into_the_product_package():
    assert not (ROOT / "app" / "services" / "ml_v2" / "data" / "recipes.csv").exists()
    raw = pd.read_csv(lab_root() / "data" / "recipes.csv")
    assert (raw["source"] == "real").sum() == 48
