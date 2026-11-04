"""Контракт прогноза v2 и отказ грузить артефакт с чужими версиями."""
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler

from app.services.ml_v2.applicability import assess, build_domain
from app.services.ml_v2.components_registry import parse_composition
from app.services.ml_v2.model_contract import CONTRACT_KEYS, make_contract, versions_compatible
from app.services.ml_v2.predictor import MODEL_DIR, get_predictor, public_metrics, reset_predictor

ROOT = Path(__file__).resolve().parents[1]


def test_contract_keys_and_interval():
    payload = make_contract(value=1.0, std=0.2, in_domain=True, model_name="bayes_ridge")
    for key in CONTRACT_KEYS:
        assert key in payload
    assert payload["lo90"] < payload["value"] < payload["hi90"]
    assert payload["lo90"] == pytest.approx(payload["value"] - 1.6448536269514722 * 0.2)


def test_unknown_additive_is_out_of_domain():
    train = [
        parse_composition("100% Опилки"),
        parse_composition("90% Опилки, 10% картон"),
        parse_composition("70% Опилки, 30% картон"),
    ]
    domain = build_domain(train)
    fresh = {"Опилки": 80.0, "Неизвестная добавка": 20.0}
    info = assess(fresh, domain)
    assert info["in_domain"] is False
    assert info["new_components"]


def test_version_mismatch_rejected():
    assert versions_compatible({"scikit-learn": "1.8.0", "numpy": np.__version__})
    assert not versions_compatible({"scikit-learn": "0.24.2", "numpy": np.__version__})


def test_interval_from_bayes_has_std():
    rng = np.random.RandomState(0)
    features = rng.normal(size=(30, 2))
    target = features[:, 0] * 2 + rng.normal(scale=0.1, size=30)
    scaler = StandardScaler().fit(features)
    model = BayesianRidge().fit(scaler.transform(features), target)
    pred, std = model.predict(scaler.transform(features[:1]), return_std=True)
    assert np.isfinite(pred[0])
    assert std[0] > 0


def test_loaded_artifact_contract_and_honest_metrics():
    reset_predictor()
    predictor = get_predictor()
    manifest = json.loads((MODEL_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["n_real"] == 48
    assert manifest["sha256"]
    oil = predictor.predict_contract({"Опилки": 95.0, "Рапсовое масло": 5.0}, "q")
    for key in CONTRACT_KEYS:
        assert key in oil
    assert oil["informative"] is True
    assert oil["in_domain"] is True
    assert oil["lo90"] < oil["value"] < oil["hi90"]

    outside = predictor.predict_contract({"Опилки": 80.0, "Древесная мука": 20.0}, "q")
    assert outside["in_domain"] is False

    for prop in ("density", "kf", "ad"):
        contract = predictor.predict_contract({"Опилки": 100.0}, prop)
        assert contract["informative"] is False
        assert contract["model_name"] == "baseline_mean"
        assert contract["value"] == pytest.approx(manifest["metrics"][prop]["train_mean"])
        assert contract["std"] == pytest.approx(manifest["metrics"][prop]["train_std"])

    with pytest.raises(ValueError, match="неизвестный компонент"):
        predictor.predict_contract({"Опилки": 80.0, "банан": 20.0}, "q")

    shown = public_metrics(manifest["metrics"]["q"])
    assert "r2_train" not in shown
    assert shown["r2_score"] == pytest.approx(manifest["metrics"]["q"]["r2_test"])
    assert shown["r2_score"] == pytest.approx(0.9506198546945188, abs=1e-6)
    assert shown["r2_score"] < 0.99
    assert manifest["metrics"]["q"]["model"] == "bayes_ridge"


def test_manifest_without_file_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr("app.services.ml_v2.predictor.MODEL_DIR", tmp_path)
    reset_predictor()
    from app.services.ml_v2.predictor import V2Predictor

    assert V2Predictor(tmp_path).load() is False
    reset_predictor()


def test_package_does_not_carry_forbidden_pieces():
    root = ROOT / "app" / "services" / "ml_v2"
    text = "\n".join(path.read_text(encoding="utf-8") for path in root.rglob("*.py"))
    assert "lab_teacher_trainer" not in text
    assert "rnf_base_enrichment" not in text
    assert "ML_USE_IQR" not in text
    assert "ML_WRITE_PREDICTIONS_TO_TRAINING" not in text
