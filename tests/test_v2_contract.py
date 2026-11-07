"""Контракт pellet_ml и отказ грузить артефакт без пина или manifest."""
import json

import numpy as np
import pytest
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler

from app.services.ml_v2.lab_import import LabUnavailable
from app.services.ml_v2.predictor import V2Predictor, public_metrics, reset_predictor
from tests.lab_support import ensure_lab, lab_root, small_artifact

ROOT_RECIPES = None


@pytest.fixture(scope="module")
def artifact(tmp_path_factory):
    ensure_lab()
    return small_artifact(tmp_path_factory)


def test_contract_keys_and_interval():
    pellet_ml = ensure_lab()
    payload = pellet_ml.contract.make_contract(value=1.0, std=0.2, in_domain=True, model_name="ridge")
    for key in pellet_ml.constants.CONTRACT_FIELDS:
        assert key in payload
    assert payload["lo90"] < payload["value"] < payload["hi90"]


def test_unknown_additive_is_out_of_domain():
    pellet_ml = ensure_lab()
    train = [
        pellet_ml.parse_composition("100% Опилки"),
        pellet_ml.parse_composition("90% Опилки, 10% картон"),
        pellet_ml.parse_composition("70% Опилки, 30% картон"),
    ]
    domain = pellet_ml.domain.build_domain(train)
    info = pellet_ml.domain.assess({"Опилки": 80.0, "Неизвестная добавка": 20.0}, domain)
    assert info["in_domain"] is False
    assert info["new_components"]


def test_version_mismatch_rejected():
    pellet_ml = ensure_lab()
    assert pellet_ml.contract.versions_compatible({"scikit-learn": "1.8.0", "numpy": np.__version__})
    assert not pellet_ml.contract.versions_compatible({"scikit-learn": "0.24.2", "numpy": np.__version__})


def test_interval_from_bayes_has_std():
    ensure_lab()
    rng = np.random.RandomState(0)
    features = rng.normal(size=(30, 2))
    target = features[:, 0] * 2 + rng.normal(scale=0.1, size=30)
    scaler = StandardScaler().fit(features)
    model = BayesianRidge().fit(scaler.transform(features), target)
    pred, std = model.predict(scaler.transform(features[:1]), return_std=True)
    assert np.isfinite(pred[0])
    assert std[0] > 0


def test_loaded_artifact_contract_and_honest_metrics(artifact, monkeypatch):
    monkeypatch.setenv("PELLET_MODEL_DIR", str(artifact))
    reset_predictor()
    predictor = V2Predictor(artifact)
    assert predictor.load()
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["api_version"].split(".")[0] == "1"
    assert manifest["lab_commit"]
    assert manifest["metrics_are"] == "nested_group_cv_not_insample"
    oil = predictor.predict_contract("95% Опилки, 5% рапсовое масло", "q")
    assert oil["informative"] is True
    assert oil["in_domain"] is True
    assert oil["lo90"] <= oil["value"] <= oil["hi90"]
    outside = predictor.predict_contract({"Опилки": 80.0, "Древесная мука": 20.0}, "q")
    assert outside["in_domain"] is False
    for prop in ("density", "kf"):
        contract = predictor.predict_contract({"Опилки": 100.0}, prop)
        assert contract["informative"] is False
        assert contract["model_name"] == "baseline_mean"
    with pytest.raises(ValueError, match="неизвестный компонент"):
        predictor.predict_contract({"Опилки": 80.0, "банан": 20.0}, "q")
    shown = public_metrics(manifest["metrics"]["q"])
    assert "r2_train" not in shown
    assert shown["r2_score"] == pytest.approx(shown["r2_test_nested"])
    assert shown["r2_score"] > 0.9
    assert shown["r2_score"] < 0.99
    assert (lab_root() / "data" / "recipes.csv").is_file()


def test_manifest_without_lab_commit_is_rejected(artifact, tmp_path):
    broken = tmp_path / "broken"
    broken.mkdir()
    payload = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    payload.pop("lab_commit", None)
    (broken / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    (broken / "pellet_models.joblib").write_bytes((artifact / "pellet_models.joblib").read_bytes())
    assert V2Predictor(broken).load() is False


def test_product_package_does_not_embed_the_lab_core():
    root = lab_root().parents[0]
    del root
    package = Path_ml_v2()
    text = "\n".join(path.read_text(encoding="utf-8") for path in package.glob("*.py"))
    assert "lab_teacher_trainer" not in text
    assert "rnf_base_enrichment" not in text
    assert "ML_USE_IQR" not in text
    assert "train_and_save" not in text or "pellet_ml" in text


def Path_ml_v2():
    from pathlib import Path

    return Path(__file__).resolve().parents[1] / "app" / "services" / "ml_v2"


def test_missing_directory_does_not_raise_on_load(tmp_path):
    reset_predictor()
    with pytest.raises(LabUnavailable):
        V2Predictor(tmp_path).predict_contract({"Опилки": 100}, "q")
