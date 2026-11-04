import pytest


@pytest.mark.integration
def test_calculate_economics_rejects_missing_components(authed_client):
    response = authed_client.post("/api/calculate_economics", json={})
    payload = response.get_json()

    assert response.status_code == 400
    assert payload["success"] is False
    assert "состав" in payload["error"].lower()


@pytest.mark.integration
def test_calculate_economics_handles_zero_efficiency(authed_client, seeded_components):
    response = authed_client.post(
        "/api/calculate_economics",
        json={
            "q_boiler": 80,
            "t_hours": 100,
            "distance": 10,
            "capacity_factor": 0.8,
            "efficiency": 0,
            "components": {"Опилки": 100},
        },
    )
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["success"] is False
    assert "кпд" in payload["error"].lower()


@pytest.mark.integration
def test_compare_economics_rejects_invalid_global_params(authed_client):
    response = authed_client.post(
        "/api/compare_economics",
        json={
            "global_params": {"q_boiler": "abc"},
            "scenarios": [{"name": "S1", "components": {"Опилки": 100}}],
        },
    )
    payload = response.get_json()
    assert payload["success"] is False
    assert "параметр" in payload["error"].lower()
