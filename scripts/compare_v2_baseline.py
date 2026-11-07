#!/usr/bin/env python3
"""Сравнить прогнозы текущего артефакта v2 со снимком tests/fixtures/v2_baseline.json.

Расхождение model_name, in_domain, informative или числа за допуском — ненулевой код.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from app.services.ml_v2.predictor import V2Predictor, reset_predictor

FIXTURE = REPO / "tests" / "fixtures" / "v2_baseline.json"


def values_close(actual, expected, *, abs_tol: float, rel_tol: float) -> bool:
    if actual is None or expected is None:
        return actual is expected
    return math.isclose(float(actual), float(expected), rel_tol=rel_tol, abs_tol=abs_tol)


def compare_payload(actual_rows, expected_payload) -> list[str]:
    abs_tol = float(expected_payload.get("abs_tol", 0.05))
    rel_tol = float(expected_payload.get("rel_tol", 0.01))
    expected = {row["composition"]: row["predictions"] for row in expected_payload["recipes"]}
    problems = []
    for row in actual_rows:
        text = row["composition"]
        if text not in expected:
            problems.append(f"в снимке нет рецепта: {text}")
            continue
        for prop, pred in expected[text].items():
            got = row["predictions"].get(prop)
            if got is None:
                problems.append(f"{text} / {prop}: нет прогноза")
                continue
            if got.get("model_name") != pred.get("model_name"):
                problems.append(
                    f"{text} / {prop}: model_name {got.get('model_name')} != {pred.get('model_name')}"
                )
            if bool(got.get("in_domain")) != bool(pred.get("in_domain")):
                problems.append(f"{text} / {prop}: in_domain изменился")
            if bool(got.get("informative")) != bool(pred.get("informative")):
                problems.append(f"{text} / {prop}: informative изменился")
            if not values_close(got.get("value"), pred.get("value"), abs_tol=abs_tol, rel_tol=rel_tol):
                problems.append(
                    f"{text} / {prop}: value {got.get('value')} != {pred.get('value')} "
                    f"(abs_tol={abs_tol}, rel_tol={rel_tol})"
                )
    return problems


def predictions_for(predictor: V2Predictor, compositions: list[str]) -> list[dict]:
    rows = []
    for text in compositions:
        contracts = predictor.predict_all(text)
        rows.append(
            {
                "composition": text,
                "predictions": {
                    prop: {
                        "value": contract["value"],
                        "std": contract["std"],
                        "lo90": contract["lo90"],
                        "hi90": contract["hi90"],
                        "in_domain": contract["in_domain"],
                        "informative": contract["informative"],
                        "model_name": contract["model_name"],
                    }
                    for prop, contract in contracts.items()
                },
            }
        )
    return rows


def main(argv=None) -> int:
    del argv
    if not FIXTURE.is_file():
        print(f"нет снимка {FIXTURE}", file=sys.stderr)
        return 1
    expected = json.loads(FIXTURE.read_text(encoding="utf-8"))
    reset_predictor()
    predictor = V2Predictor()
    if not predictor.load():
        print("артефакт v2 не загрузился, сравнение не выполнено", file=sys.stderr)
        return 1
    actual = predictions_for(predictor, [row["composition"] for row in expected["recipes"]])
    problems = compare_payload(actual, expected)
    if problems:
        print("прогнозы v2 разошлись со снимком:")
        for line in problems:
            print(f"  {line}")
        return 1
    print(f"прогнозы v2 совпали со снимком ({len(actual)} рецептов)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
