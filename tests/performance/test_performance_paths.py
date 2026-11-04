import time

import pandas as pd
import pytest

from app.database.database import init_db, insert_data
from app.routes.economics import _compute_economics_result


@pytest.mark.performance
def test_economics_compute_path_is_fast():
    db_map = {}
    components = {"Опилки": 70, "Солома": 30}

    start = time.perf_counter()
    for _ in range(300):
        _compute_economics_result(80, 720, 258, 0.8, 0.9, components, db_map)
    elapsed = time.perf_counter() - start

    # Лимит намеренно мягкий, задача smoke-профилирования.
    assert elapsed < 2.5


@pytest.mark.performance
def test_batch_insert_components_is_reasonable(test_db_path):
    init_db(test_db_path)
    payload = pd.DataFrame(
        [
            {
                "component": f"comp_{idx}",
                "q": 10 + idx,
                "war": 1.0,
                "ad": 0.1,
                "vd": 0.2,
                "cd": 0.3,
                "hd": 0.4,
                "nd": 0.5,
                "sd": 0.6,
                "od": 0.7,
            }
            for idx in range(250)
        ]
    )

    start = time.perf_counter()
    insert_data(test_db_path, "components", payload)
    elapsed = time.perf_counter() - start

    assert elapsed < 2.5
