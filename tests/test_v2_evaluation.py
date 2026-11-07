"""Вложенная групповая CV живёт в pellet_ml, не в продукте."""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from tests.lab_support import ensure_lab


def test_group_folds_do_not_overlap():
    pellet_ml = ensure_lab()
    groups = ["a", "a", "b", "c", "d", "d"]
    splits = pellet_ml.evaluation.group_folds(groups, k=10**6, seed=42)
    pellet_ml.evaluation.assert_groups_disjoint(splits, groups)
    seen = []
    for _, test_idx in splits:
        seen.extend(test_idx.tolist())
    assert sorted(seen) == list(range(len(groups)))


def test_same_recipe_stays_together():
    pellet_ml = ensure_lab()
    groups = ["r1", "r1", "r2"]
    splits = pellet_ml.evaluation.group_folds(groups, k=2, seed=0)
    pellet_ml.evaluation.assert_groups_disjoint(splits, groups)
    for train_idx, test_idx in splits:
        assert not ({groups[i] for i in train_idx} & {groups[i] for i in test_idx})


def test_noise_nested_r2_near_zero():
    pellet_ml = ensure_lab()
    rng = np.random.RandomState(pellet_ml.constants.SEED)
    count = 24
    features = rng.normal(size=(count, 3))
    target = rng.normal(size=count)
    result = pellet_ml.evaluation.nested_cv(
        "ridge", features, target, [f"g{i}" for i in range(count)], seed=pellet_ml.constants.SEED
    )
    assert result["r2_test"] <= 0.2
    assert result["cv_scheme"].startswith("nested")


def test_insample_can_look_good_while_nested_does_not():
    pellet_ml = ensure_lab()
    rng = np.random.RandomState(pellet_ml.constants.SEED)
    count = 24
    features = rng.normal(size=(count, 4))
    target = rng.normal(size=count)
    scaler = StandardScaler().fit(features)
    model = RandomForestRegressor(n_estimators=30, max_depth=6, random_state=pellet_ml.constants.SEED)
    model.fit(scaler.transform(features), target)
    insample = model.score(scaler.transform(features), target)
    result = pellet_ml.evaluation.nested_cv(
        "ridge",
        features,
        target,
        [f"g{i}" for i in range(count)],
        seed=pellet_ml.constants.SEED,
    )
    assert insample > 0.5
    assert result["r2_test"] < insample
