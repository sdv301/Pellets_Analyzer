"""Вложенная групповая CV: шум не получает высокий test R², группы не пересекаются."""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from app.services.ml_v2.evaluation import SEED, assert_groups_disjoint, group_folds, nested_cv


def test_group_folds_do_not_overlap():
    groups = ["a", "a", "b", "c", "d", "d"]
    splits = group_folds(groups, k=10**6, seed=42)
    assert_groups_disjoint(splits, groups)
    seen = []
    for _, test_idx in splits:
        seen.extend(test_idx.tolist())
    assert sorted(seen) == list(range(len(groups)))


def test_same_recipe_stays_together():
    groups = ["r1", "r1", "r2"]
    splits = group_folds(groups, k=2, seed=0)
    assert_groups_disjoint(splits, groups)
    for train_idx, test_idx in splits:
        train_groups = {groups[i] for i in train_idx}
        test_groups = {groups[i] for i in test_idx}
        assert not (train_groups & test_groups)


def test_noise_nested_r2_near_zero():
    rng = np.random.RandomState(SEED)
    count = 24
    features = rng.normal(size=(count, 3))
    target = rng.normal(size=count)
    groups = [f"g{i}" for i in range(count)]
    result = nested_cv("ridge", features, target, groups, seed=SEED)
    assert result["r2_test"] <= 0.2
    assert abs(result["baseline_r2"]) < 0.25
    assert result["cv_scheme"].startswith("nested")


def test_insample_can_look_good_while_nested_does_not():
    rng = np.random.RandomState(SEED)
    count = 24
    features = rng.normal(size=(count, 4))
    target = rng.normal(size=count)
    scaler = StandardScaler().fit(features)
    model = RandomForestRegressor(n_estimators=30, max_depth=6, random_state=SEED)
    model.fit(scaler.transform(features), target)
    insample = model.score(scaler.transform(features), target)
    result = nested_cv("ridge", features, target, [f"g{i}" for i in range(count)], seed=SEED)
    assert insample > 0.5
    assert result["r2_test"] < insample
