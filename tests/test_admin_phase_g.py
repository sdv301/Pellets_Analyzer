# tests/test_admin_phase_g.py — Тесты обновленной структуры админ-панели (Фаза G: G1–G6)
import io
import json
import sqlite3
import pytest

from app.services.ml_v2 import switch


class TestAdminPhaseG:
    """Тестирование новых возможностей админ-панели G1–G6."""

    @pytest.fixture(autouse=True)
    def clean_switch_state(self):
        switch.set_runtime_override(None, None)
        yield
        switch.set_runtime_override(None, None)

    def test_admin_dashboard_renders_all_tabs(self, admin_client):
        """Проверка рендеринга всех вкладок админ-панели (G1–G6)."""
        res = admin_client.get('/admin/dashboard')
        assert res.status_code == 200
        html = res.data.decode('utf-8')
        assert 'Пользователи и роли (G1)' in html
        assert 'ML Модель и Тень (G2)' in html
        assert 'Лаборатория Tolko_model (G3)' in html
        assert 'Рецепты и данные (G4)' in html
        assert 'Журнал аудита (G5)' in html
        assert 'Настройки системы (G6)' in html
        assert 'lab-v0.2.0' in html

    def test_non_admin_cannot_access_dashboard(self, authenticated_client):
        """Не-администратор не имеет доступа к админ-панели."""
        res = authenticated_client.get('/admin/dashboard')
        assert res.status_code in (302, 403)

    def test_admin_generate_reset_link_success(self, admin_client, db_path):
        """G1: Генерация одноразовой ссылки сброса пароля без показа пароля."""
        # Находим тестового пользователя
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT id, username FROM users WHERE role_id != 2 LIMIT 1")
        row = cur.fetchone()
        conn.close()

        target_user_id = row[0] if row else 1

        res = admin_client.post(f'/admin/generate_reset_link/{target_user_id}')
        assert res.status_code == 200
        data = res.get_json()
        assert data['success'] is True
        assert 'reset_url' in data
        assert '/reset-password/' in data['reset_url']
        assert data['expires_in'] == '1 час'

    def test_admin_model_status_endpoint(self, admin_client):
        """G2, G3: Проверка эндпоинта статуса модели и лаборатории."""
        res = admin_client.get('/admin/model_status')
        assert res.status_code == 200
        data = res.get_json()
        assert data['success'] is True
        assert 'data' in data
        status = data['data']
        assert status['submodule_tag'] == 'lab-v0.2.0'
        assert status['api_version'] in ('1.0.0', '1.1.0')
        assert 'update_command' in status
        assert 'safety_notice' in status

    def test_admin_switch_model_and_rollback(self, admin_client):
        """G2: Переключение модели на v2, на тень и откат на legacy."""
        # 1. Переключение на тень (legacy + shadow=True)
        res = admin_client.post('/admin/switch_model', json={
            'version': 'legacy',
            'shadow': True
        })
        assert res.status_code == 200
        assert res.get_json()['shadow'] is True
        assert switch.shadow_enabled() is True
        assert switch.model_version() == 'legacy'

        # 2. Недопустимая версия возвращает 400
        res_bad = admin_client.post('/admin/switch_model', json={
            'version': 'invalid_ver_99',
            'shadow': False
        })
        assert res_bad.status_code == 400

        # 3. Откат на чистый legacy
        res_back = admin_client.post('/admin/switch_model', json={
            'version': 'legacy',
            'shadow': False
        })
        assert res_back.status_code == 200
        assert switch.shadow_enabled() is False
        assert switch.model_version() == 'legacy'

    def test_admin_shadow_logs_query_and_clear(self, admin_client, db_path):
        """G2: Добавление, чтение и очистка записей теневого лога."""
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute('''
            INSERT INTO ml_shadow_logs (target_property, legacy_value, v2_value, delta, in_domain, informative)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', ('density', 1050.0, 1055.5, 5.5, 1, 1))
        conn.commit()
        conn.close()

        # Запрос записей
        res = admin_client.get('/admin/shadow_logs?property=density')
        assert res.status_code == 200
        data = res.get_json()
        assert data['success'] is True
        assert data['count'] >= 1
        assert data['logs'][0]['target_property'] == 'density'

        # Очистка записей
        res_clear = admin_client.post('/admin/clear_shadow_logs')
        assert res_clear.status_code == 200
        assert res_clear.get_json()['deleted'] >= 1

        # Проверка после очистки
        res_after = admin_client.get('/admin/shadow_logs')
        assert res_after.get_json()['count'] == 0

    def test_admin_recipes_listing_and_filters(self, admin_client, db_path):
        """G4: Просмотр списка рецептур с пагинацией и фильтрами."""
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute('''
            INSERT OR REPLACE INTO measured_parameters (composition, density, source, is_questionable)
            VALUES (?, ?, ?, ?)
        ''', ('70% Опилки, 30% Солома', 1020.0, 'laboratory', 0))
        cur.execute('''
            INSERT OR REPLACE INTO measured_parameters (composition, density, source, is_questionable)
            VALUES (?, ?, ?, ?)
        ''', ('50% Опилки, 50% Торф', 980.0, 'literature', 1))
        conn.commit()
        conn.close()

        # Все рецепты
        res = admin_client.get('/admin/recipes?page=1&per_page=10')
        assert res.status_code == 200
        data = res.get_json()
        assert data['success'] is True
        assert data['total'] >= 2

        # Фильтр по источнику
        res_source = admin_client.get('/admin/recipes?source=literature')
        data_source = res_source.get_json()
        assert all(r['source'] == 'literature' for r in data_source['recipes'])

        # Фильтр по статусу сомнительности
        res_quest = admin_client.get('/admin/recipes?questionable=1')
        data_quest = res_quest.get_json()
        assert all(r['is_questionable'] == 1 for r in data_quest['recipes'])

    def test_admin_toggle_recipe_questionable(self, admin_client, db_path):
        """G4: Переключение статуса сомнительности опыта."""
        comp = '70% Опилки, 30% Солома'
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute('''
            INSERT OR REPLACE INTO measured_parameters (composition, density, is_questionable)
            VALUES (?, ?, 0)
        ''', (comp, 1020.0))
        conn.commit()
        conn.close()

        # Первое переключение: 0 -> 1
        res1 = admin_client.post('/admin/recipes/toggle_questionable', json={'composition': comp})
        assert res1.status_code == 200
        assert res1.get_json()['is_questionable'] == 1

        # Второе переключение: 1 -> 0
        res2 = admin_client.post('/admin/recipes/toggle_questionable', json={'composition': comp})
        assert res2.status_code == 200
        assert res2.get_json()['is_questionable'] == 0

    def test_admin_update_recipe_source(self, admin_client, db_path):
        """G4: Обновление источника рецептуры."""
        comp = '70% Опилки, 30% Солома'
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute('''
            INSERT OR REPLACE INTO measured_parameters (composition, density, source)
            VALUES (?, ?, 'laboratory')
        ''', (comp, 1020.0))
        conn.commit()
        conn.close()

        res = admin_client.post('/admin/recipes/update_source', json={
            'composition': comp,
            'source': 'literature'
        })
        assert res.status_code == 200
        assert res.get_json()['source'] == 'literature'

        # Недопустимый источник
        res_bad = admin_client.post('/admin/recipes/update_source', json={
            'composition': comp,
            'source': 'unknown_source'
        })
        assert res_bad.status_code == 400

    def test_admin_recipes_import_preview_and_confirm(self, admin_client, db_path):
        """G4: Предпросмотр и подтверждение импорта рецептур из CSV."""
        csv_content = (
            "composition,density,kf,kt,mass_loss,tign,tb,co,co2\n"
            "60% Опилки 40% Солома,1015.0,92.0,88.0,2.1,310.0,820.0,0.5,11.5\n"
            "80% Опилки 20% Картон,1080.0,96.0,91.0,1.8,325.0,840.0,0.4,12.2\n"
        )
        data = {
            'file': (io.BytesIO(csv_content.encode('utf-8')), 'test_recipes.csv')
        }

        # 1. Предпросмотр
        res_preview = admin_client.post(
            '/admin/recipes/import_preview',
            data=data,
            content_type='multipart/form-data'
        )
        assert res_preview.status_code == 200
        preview = res_preview.get_json()
        assert preview['success'] is True
        assert preview['total_rows'] == 2
        assert 'composition' in preview['columns']
        assert len(preview['preview_rows']) == 2

        # 2. Подтверждение импорта
        res_confirm = admin_client.post('/admin/recipes/import_confirm', json={
            'records': preview['preview_rows'],
            'source': 'laboratory'
        })
        assert res_confirm.status_code == 200
        confirm = res_confirm.get_json()
        assert confirm['success'] is True
        assert confirm['inserted_count'] == 2
        assert confirm['source'] == 'laboratory'

    def test_admin_update_settings(self, admin_client):
        """G6: Обновление системных параметров."""
        res = admin_client.post('/admin/settings/update', json={
            'max_upload_size_mb': 25,
            'session_lifetime_days': 14
        })
        assert res.status_code == 200
        data = res.get_json()
        assert data['success'] is True
        assert 'успешно' in data['message']
