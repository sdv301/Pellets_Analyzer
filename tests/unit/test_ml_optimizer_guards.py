import pytest

from app.services.ml_optimizer import MLCompositionOptimizer, PelletPropertyPredictor


@pytest.mark.unit
def test_prepare_composition_for_prediction_rejects_zero_total():
    predictor = PelletPropertyPredictor()
    predictor.feature_names = ["Опилки", "Солома"]

    with pytest.raises(ValueError):
        predictor.prepare_composition_for_prediction({"Опилки": 0.0, "Солома": 0.0})


@pytest.mark.unit
def test_optimizer_rejects_incompatible_constraints():
    class DummyPredictor:
        models = {"q": object()}
        feature_names = ["Опилки", "Солома"]

        def predict(self, composition, target_property):
            return 10.0

        target_properties_mapping = {"q": "Теплота сгорания"}

    optimizer = MLCompositionOptimizer(DummyPredictor())
    result = optimizer.optimize_composition(
        target_property="q",
        maximize=True,
        constraints={"Опилки": (70.0, 80.0), "Солома": (40.0, 60.0)},
    )
    assert result["success"] is False
    assert "минимальная сумма" in result["error"]
