import pytest


@pytest.mark.integration
def test_api_predict_composition_requires_payload(authed_client):
    response = authed_client.post("/api/predict_composition", json={})
    payload = response.get_json()

    assert payload["success"] is False
    assert "состав" in payload["error"].lower()


@pytest.mark.integration
def test_ml_predict_requires_target_property(authed_client):
    response = authed_client.post("/ml_predict", json={"composition": {"Опилки": 100}})
    payload = response.get_json()

    assert payload["success"] is False
    assert "состав или свойство" in payload["error"].lower()
