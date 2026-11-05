"""
tests/smoke/test_routes.py — Smoke-тесты всех зарегистрированных маршрутов (Фаза B2).

Покрывает сетку маршрутов из docs/FEATURES.md:
1. Доступность публичных страниц (GET 200).
2. Защита закрытых страниц: редирект на /login при обращении без сессии (302).
3. Доступность страниц для авторизованного пользователя (GET 200).
4. Ограничение доступа к панели администратора (пользователь 302/403, админ 200).
5. Минимальные POST-запросы к ключевым эндпоинтам.
"""

import pytest
import json


class TestPublicRoutes:
    """Проверка доступности публичных страниц."""

    @pytest.mark.smoke
    @pytest.mark.parametrize("url", [
        "/",
        "/landing",
        "/login",
        "/register",
        "/forgot-password",
        "/resend-verification",
    ])
    def test_public_pages_get_200(self, client, url):
        response = client.get(url)
        assert response.status_code == 200


class TestUnauthorizedRedirects:
    """Проверка перенаправления неавторизованных пользователей на страницу входа."""

    @pytest.mark.smoke
    @pytest.mark.parametrize("url", [
        "/dashboard",
        "/tables",
        "/ai_analysis",
        "/ai_ml_system_status",
        "/economics",
        "/compare",
        "/create_graph",
        "/ml_dashboard",
        "/ml_system_status",
        "/ml_train_status",
        "/ai_ml_recommendations",
        "/ai_ml_history",
        "/profile",
        "/ml/saved_models",
        "/admin/dashboard",
        "/admin/database",
        "/api/notifications",
    ])
    def test_protected_routes_redirect_to_login(self, client, url):
        response = client.get(url)
        assert response.status_code in (301, 302)
        assert "/login" in response.headers.get("Location", "")


class TestAuthenticatedGetRoutes:
    """Проверка доступности закрытых страниц для авторизованного пользователя."""

    @pytest.mark.smoke
    @pytest.mark.parametrize("url", [
        "/dashboard",
        "/tables",
        "/ai_analysis",
        "/ai_ml_system_status",
        "/economics",
        "/compare",
        "/create_graph",
        "/ml_dashboard",
        "/ml_system_status",
        "/ml_train_status",
        "/ai_ml_recommendations",
        "/ai_ml_history",
        "/profile",
        "/ml/saved_models",
        "/api/get_components_economics",
        "/api/download_economics",
        "/api/notifications",
    ])
    def test_user_get_pages_200(self, authenticated_client, url):
        response = authenticated_client.get(url)
        assert response.status_code == 200


class TestAdminAccessControl:
    """Проверка разграничения прав доступа к панели администратора."""

    @pytest.mark.smoke
    @pytest.mark.parametrize("url", [
        "/admin/dashboard",
        "/admin/database",
    ])
    def test_regular_user_cannot_access_admin(self, authenticated_client, url):
        response = authenticated_client.get(url)
        assert response.status_code in (302, 403)
        if response.status_code == 302:
            assert "/login" in response.headers.get("Location", "") or "/dashboard" in response.headers.get("Location", "")

    @pytest.mark.smoke
    @pytest.mark.parametrize("url", [
        "/admin/dashboard",
        "/admin/database",
    ])
    def test_admin_can_access_admin_pages(self, admin_client, url):
        response = admin_client.get(url)
        assert response.status_code == 200


class TestBasicPostRoutes:
    """Smoke-проверка обработки базовых POST-запросов."""

    @pytest.mark.smoke
    def test_search_and_clear_search(self, authenticated_client):
        res = authenticated_client.post('/search', data={'search_term': 'солома', 'search_column': 'composition'})
        assert res.status_code in (200, 302)

        res_clear = authenticated_client.post('/clear_search', data={})
        assert res_clear.status_code in (200, 302)

    @pytest.mark.smoke
    def test_compare_post_empty_or_valid(self, authenticated_client):
        # Минимальный запрос сравнения
        res = authenticated_client.post('/compare', data={'comp1': 'A', 'comp2': 'B'})
        assert res.status_code == 200
        data = res.get_json()
        assert data is not None

    @pytest.mark.smoke
    def test_economics_compare_post(self, authenticated_client):
        payload = json.dumps({'compositions': [{'name': 'Состав 1', 'components': [{'name': 'Опилки', 'ratio': 100}]}]})
        res = authenticated_client.post('/api/compare_economics', data=payload, content_type='application/json')
        assert res.status_code in (200, 400)

    @pytest.mark.smoke
    def test_create_graph_post(self, authenticated_client):
        res = authenticated_client.post('/create_graph', data={
            'x_param': 'war',
            'y_param': 'q',
            'graph_type': 'scatter',
            'backend': 'plotly'
        })
        assert res.status_code == 200

    @pytest.mark.smoke
    def test_ml_predict_post(self, authenticated_client):
        res = authenticated_client.post('/ml_predict', data=json.dumps({'composition': {'Опилки': 100}}), content_type='application/json')
        assert res.status_code in (200, 400)

    @pytest.mark.smoke
    def test_admin_clear_cache(self, admin_client):
        res = admin_client.post('/admin/clear_cache')
        assert res.status_code in (200, 302)

    @pytest.mark.smoke
    def test_notifications_payload_structure(self, authenticated_client):
        res = authenticated_client.get('/api/notifications')
        assert res.status_code == 200
        data = res.get_json()
        assert data.get('success') is True
        assert 'notifications' in data
        assert isinstance(data['notifications'], list)
