import sqlite3

import pandas as pd
import pytest

from app.database.database import init_db, insert_data, update_ml_optimization_with_actual


@pytest.mark.integration
def test_insert_components_with_partial_columns_fills_contract(test_db_path):
    init_db(test_db_path)
    df = pd.DataFrame(
        [
            {
                "component": "Тестовый компонент",
                "q": 12.5,
            }
        ]
    )

    insert_data(test_db_path, "components", df)

    conn = sqlite3.connect(test_db_path)
    row = conn.execute(
        "SELECT component, q, ro, cost_raw, cost_crush, cost_granule FROM components WHERE component = ?",
        ("Тестовый компонент",),
    ).fetchone()
    conn.close()

    assert row is not None
    assert row[0] == "Тестовый компонент"
    assert row[1] == pytest.approx(12.5)
    assert row[2] is None
    assert row[3] is None


@pytest.mark.integration
def test_update_ml_optimization_uses_json_composition_column(test_db_path):
    init_db(test_db_path)
    conn = sqlite3.connect(test_db_path)
    conn.execute(
        """
        INSERT INTO ml_optimizations
        (target_property, maximize, optimal_composition_json, optimal_value, constraints_json, algorithm, model_metrics_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        ("q", 1, '{"Опилки": 100}', 19.5, "{}", "xgboost", "{}"),
    )
    optimization_id = conn.execute("SELECT id FROM ml_optimizations ORDER BY id DESC LIMIT 1").fetchone()[0]
    conn.commit()
    conn.close()

    ok = update_ml_optimization_with_actual(
        test_db_path,
        optimization_id,
        {
            "q": 19.5,
            "ad": 0.2,
            "density": 1000.0,
        },
    )
    assert ok is True

    conn = sqlite3.connect(test_db_path)
    composition = conn.execute(
        "SELECT composition FROM measured_parameters ORDER BY composition DESC LIMIT 1"
    ).fetchone()[0]
    conn.close()
    assert "Опилки" in composition
