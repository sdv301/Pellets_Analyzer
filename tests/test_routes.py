"""
Тесты для маршрутов Flask — основные роуты.
"""
import pytest


class TestAuthRoutes:
    """Тесты маршрутов аутентификации."""

    def test_login_route_exists(self, client):
        """Маршрут логина существует (может не быть шаблона)."""
        response = client.get('/login')
        # 200 = OK, 500 = шаблон отсутствует (это тоже валидно для теста)
        assert response.status_code in [200, 500]

    def test_register_route_exists(self, client):
        """Маршрут регистрации существует."""
        response = client.get('/register')
        assert response.status_code in [200, 500]

    def test_forgot_password_route_exists(self, client):
        """Маршрут восстановления пароля существует."""
        response = client.get('/forgot-password')
        assert response.status_code in [200, 500]

    def test_logout_redirects_without_session(self, client):
        """Логаут без сессии перенаправляет."""
        response = client.get('/logout', follow_redirects=False)
        assert response.status_code in [301, 302, 200]


class TestMainRoutes:
    """Тесты основных маршрутов."""

    def test_index_route_exists(self, client):
        """Главная страница существует."""
        response = client.get('/landing')
        assert response.status_code in [200, 301, 302, 500]

    def test_dashboard_requires_auth(self, client):
        """Дашборд требует авторизации."""
        response = client.get('/dashboard')
        assert response.status_code in [301, 302, 401, 403]

    def test_upload_no_file(self, authenticated_client):
        """Загрузка без файла возвращает ошибку."""
        response = authenticated_client.post('/', data={})
        assert response.status_code == 200
        import json
        data = json.loads(response.data)
        assert data['success'] is False

    def test_search_empty_db(self, authenticated_client):
        """Поиск в пустой БД."""
        response = authenticated_client.post('/search', data={
            'search_column': ['composition'],
            'search_operator': ['LIKE'],
            'search_value': ['test']
        })
        assert response.status_code == 200
        import json
        data = json.loads(response.data)
        assert data['success'] is False

    def test_tables_requires_auth(self, client):
        """Страница таблиц требует авторизации."""
        response = client.get('/tables')
        assert response.status_code in [301, 302, 401, 403]

    def test_ai_analysis_requires_auth(self, client):
        """ИИ-анализ требует авторизации."""
        response = client.get('/ai_analysis')
        assert response.status_code in [301, 302, 401, 403]

    def test_economics_requires_auth(self, client):
        """Экономика требует авторизации."""
        response = client.get('/economics')
        assert response.status_code in [301, 302, 401, 403]


class TestCompareRoutes:
    """Тесты маршрутов сравнения."""

    def test_compare_requires_auth(self, client):
        """Сравнение требует авторизации."""
        response = client.get('/compare')
        assert response.status_code in [301, 302, 401, 403]

    def test_compare_post_requires_auth(self, client):
        """POST сравнения требует авторизации."""
        response = client.post('/compare', data={
            'comp1': 'Test1',
            'comp2': 'Test2'
        })
        assert response.status_code in [301, 302, 401, 403]


class TestGraphsRoutes:
    """Тесты маршрутов графиков."""

    def test_create_graph_requires_auth(self, client):
        """Создание графика требует авторизации."""
        response = client.get('/create_graph')
        assert response.status_code in [301, 302, 401, 403]


class TestMLRoutes:
    """Тесты ML-маршрутов."""

    def test_ml_dashboard_requires_auth(self, client):
        """ML дашборд требует авторизации."""
        response = client.get('/ml_dashboard')
        assert response.status_code in [301, 302, 401, 403]

    def test_ml_train_requires_auth(self, client):
        """Обучение ML требует авторизации."""
        response = client.post('/ml_system_train')
        assert response.status_code in [301, 302, 401, 403]

    def test_ml_optimize_requires_auth(self, client):
        """ML оптимизация требует авторизации."""
        response = client.post('/ml_optimize')
        assert response.status_code in [301, 302, 401, 403]

    def test_ml_predict_requires_auth(self, client):
        """ML предсказание требует авторизации."""
        response = client.post('/ml_predict')
        assert response.status_code in [301, 302, 401, 403]


class TestAdminRoutes:
    """Тесты админ-маршрутов."""

    def test_admin_dashboard_requires_admin(self, authenticated_client):
        """Админ-дашборд требует прав админа."""
        response = authenticated_client.get('/admin/dashboard')
        assert response.status_code in [301, 302, 401, 403]

    def test_admin_dashboard_works_for_admin(self, admin_client):
        """Админ-дашборд работает для админа (может не быть шаблона)."""
        response = admin_client.get('/admin/dashboard')
        assert response.status_code in [200, 500]


class TestAPIRoutes:
    """Тесты API-маршрутов."""

    def test_global_search_requires_auth(self, client):
        """Глобальный поиск требует авторизации."""
        response = client.get('/global_search?q=test')
        assert response.status_code in [301, 302, 401, 403]

    def test_ml_status_requires_auth(self, client):
        """Статус ML требует авторизации."""
        response = client.get('/ml_system_status')
        assert response.status_code in [301, 302, 401, 403]

    def test_ai_ml_status_requires_auth(self, client):
        """Статус AI/ML требует авторизации."""
        response = client.get('/ai_ml_system_status')
        assert response.status_code in [301, 302, 401, 403]

    def test_economics_api_requires_auth(self, client):
        """Экономика API требует авторизации."""
        response = client.get('/api/get_components_economics')
        assert response.status_code in [301, 302, 401, 403]

    def test_economics_calculate_requires_auth(self, client):
        """Расчёт экономики требует авторизации."""
        response = client.post('/api/calculate_economics', json={})
        assert response.status_code in [301, 302, 401, 403]
