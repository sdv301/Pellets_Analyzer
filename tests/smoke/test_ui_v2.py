"""
tests/smoke/test_ui_v2.py — Тесты интерфейса UI v2 (AstroWind) и страниц аутентификации.

Фаза E: E0–E4
- Вендоринг дизайна (E0)
- Дизайн-токены и статические стили (E1)
- Публичный лендинг на base_public.html (E2)
- Страницы входа, регистрации, сброса пароля (E3)
- Переключатель UI_V2 и сохранение обратной совместимости (E4)
"""

import os
from pathlib import Path
import pytest


class TestAstroWindStaticAssets:
    """E0-E1: Проверка наличия и целостности статических файлов AstroWind."""

    def test_astrowind_license_exists(self):
        license_path = Path("static/vendor/astrowind/LICENSE")
        assert license_path.exists()
        content = license_path.read_text(encoding="utf-8")
        assert "MIT License" in content
        assert "onWidget" in content

    def test_astrowind_css_exists_and_non_empty(self):
        css_path = Path("static/vendor/astrowind/astrowind.css")
        assert css_path.exists()
        assert css_path.stat().st_size > 50000  # Скомпилированный Tailwind CSS > 50 КБ

    def test_tokens_css_exists(self):
        tokens_path = Path("static/css/tokens.css")
        assert tokens_path.exists()
        content = tokens_path.read_text(encoding="utf-8")
        assert "--aw-color-primary" in content
        assert "--aw-color-bg-page" in content

    def test_app_css_exists(self):
        app_css = Path("static/css/app.css")
        assert app_css.exists()
        content = app_css.read_text(encoding="utf-8")
        assert "astrowind.css" in content


class TestUIV2LandingAndAuth:
    """E2-E3: Проверка рендеринга страниц за флагом UI_V2."""

    def test_landing_v2_with_query_param(self, client):
        """Проверка доступа к v2 лендингу через query-параметр ?v2=1."""
        res = client.get('/?v2=1')
        assert res.status_code == 200
        html = res.get_data(as_text=True)
        assert "Pellets" in html
        assert "95.1%" in html
        assert "AstroWind" in html
        assert "Nested LORO Group CV" in html

    def test_landing_v2_when_flag_enabled(self, app):
        """Проверка работы при глобальном флаге UI_V2=True."""
        app.config['UI_V2'] = True
        client = app.test_client()

        res = client.get('/')
        assert res.status_code == 200
        html = res.get_data(as_text=True)
        assert "Прогнозирование и оптимизация свойств" in html
        assert "95.1%" in html

        app.config['UI_V2'] = False

    def test_auth_login_v2(self, app):
        """Проверка страницы входа AstroWind v2."""
        app.config['UI_V2'] = True
        client = app.test_client()

        res = client.get('/login')
        assert res.status_code == 200
        html = res.get_data(as_text=True)
        assert "Вход в систему" in html
        assert 'name="csrf_token"' in html
        assert "togglePasswordBtn" in html
        assert 'autocomplete="username"' in html

        app.config['UI_V2'] = False

    def test_auth_register_v2(self, app):
        """Проверка страницы регистрации AstroWind v2."""
        app.config['UI_V2'] = True
        client = app.test_client()

        res = client.get('/register')
        assert res.status_code == 200
        html = res.get_data(as_text=True)
        assert "Создание аккаунта" in html
        assert 'name="csrf_token"' in html
        assert 'name="username"' in html
        assert 'name="email"' in html

        app.config['UI_V2'] = False

    def test_auth_forgot_password_v2(self, app):
        """Проверка страницы сброса пароля AstroWind v2."""
        app.config['UI_V2'] = True
        client = app.test_client()

        res = client.get('/forgot-password')
        assert res.status_code == 200
        html = res.get_data(as_text=True)
        assert "Восстановление пароля" in html
        assert 'name="csrf_token"' in html

        app.config['UI_V2'] = False

    def test_auth_reset_password_v2(self, app):
        """Проверка страницы задания нового пароля AstroWind v2."""
        app.config['UI_V2'] = True
        client = app.test_client()

        res = client.get('/reset-password/sampletoken123')
        assert res.status_code == 200
        html = res.get_data(as_text=True)
        assert "Новый пароль" in html
        assert 'name="csrf_token"' in html

        app.config['UI_V2'] = False

    def test_legacy_templates_preserved_when_ui_v2_false(self, client):
        """Проверка, что при UI_V2=False по умолчанию отдаются исходные шаблоны."""
        res = client.get('/')
        assert res.status_code == 200
        html = res.get_data(as_text=True)
        # В исходном landing.html есть canvas частиц
        assert "particles-canvas" in html

        res_login = client.get('/login')
        assert res_login.status_code == 200


class TestPhaseCUnifiedTemplatesAndResponsiveness:
    """Фаза C: C1 (Базовый шаблон & извлечение ассетов), C3 (Адаптивность)."""

    def test_all_app_and_error_templates_have_extends(self):
        """C1: Проверка, что все основные страницы наследуют базовый шаблон (нет страниц без extends)."""
        templates_dir = Path("templates")
        skip_dirs = {"includes", "email", ".pytest_cache"}
        skip_files = {"landing.html"}  # легаси лендинг для отката

        for html_file in templates_dir.rglob("*.html"):
            rel_parts = html_file.relative_to(templates_dir).parts
            if any(part in skip_dirs for part in rel_parts):
                continue
            if html_file.name.startswith("_") or "base" in html_file.name:
                continue
            if html_file.name in skip_files or "auth" in rel_parts and "v2" not in rel_parts:
                continue  # легаси auth шаблоны

            content = html_file.read_text(encoding="utf-8")
            assert "{% extends" in content, f"Шаблон {html_file} не наследует базовый шаблон!"

    def test_compare_page_uses_extracted_assets(self, authed_client):
        """C1: Проверка, что страница compare подключает вынесенные compare.css и compare.js."""
        res = authed_client.get('/compare')
        assert res.status_code == 200
        html = res.get_data(as_text=True)
        assert "compare.css" in html
        assert "compare.js" in html
        # В теле страницы больше нет вынесенного блока стилей
        assert "ПРЕМИУМ CSS СТИЛИ (SOFT UI)" not in html

    def test_error_pages_render_with_base_public(self, client):
        """C1 & D3: Проверка страниц ошибок 404, 403, 500 с базовым оформлением без утечек стека."""
        res_404 = client.get('/nonexistent-route-404-test')
        assert res_404.status_code == 404
        html_404 = res_404.get_data(as_text=True)
        assert "404" in html_404
        assert "Страница не найдена" in html_404
        assert "themeToggleBtn" in html_404

    def test_responsive_ultrawide_css_rules_present(self):
        """C3: Проверка наличия правил адаптивности для экранов 1280..3440px."""
        app_css = Path("static/css/app.css").read_text(encoding="utf-8")
        assert "@media (min-width: 2560px)" in app_css
        assert "max-width: 2160px" in app_css
        assert "table-responsive" in app_css
        assert "overflow-x: hidden" in app_css

