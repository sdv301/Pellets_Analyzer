import pytest


@pytest.mark.smoke
def test_public_pages_are_available(client):
    root_response = client.get("/")
    landing_response = client.get("/landing")

    assert root_response.status_code == 200
    assert landing_response.status_code == 200


@pytest.mark.smoke
def test_dashboard_redirects_when_unauthorized(client):
    response = client.get("/dashboard")
    assert response.status_code in (301, 302)
    assert "/login" in response.headers.get("Location", "")


@pytest.mark.smoke
def test_core_protected_pages_for_authorized_user(authed_client):
    economics_page = authed_client.get("/economics")
    ml_page = authed_client.get("/ml_dashboard")

    assert economics_page.status_code == 200
    assert ml_page.status_code == 200
