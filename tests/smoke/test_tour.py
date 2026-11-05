"""
tests/smoke/test_tour.py — Smoke-тесты интерактивного онбординг-тура (driver.js).
"""
from pathlib import Path
import pytest


class TestTourAssets:
    """Проверка наличия и доступности статических файлов онбординг-тура."""

    def test_driverjs_vendor_assets_exist(self):
        js_path = Path("static/vendor/driverjs/driver.js.iife.js")
        css_path = Path("static/vendor/driverjs/driver.css")
        assert js_path.exists(), "driver.js.iife.js должен существовать в static/vendor/driverjs/"
        assert css_path.exists(), "driver.css должен существовать в static/vendor/driverjs/"
        assert js_path.stat().st_size > 5000
        assert css_path.stat().st_size > 1000

    def test_app_tour_script_exists(self):
        tour_js = Path("static/js/app_tour.js")
        assert tour_js.exists()
        content = tour_js.read_text(encoding="utf-8")
        assert "driver.js.driver" in content
        assert "pellet_tour_completed" in content
        assert "startTourBtn" in content

    def test_tour_button_in_navigation_template(self):
        nav_html = Path("templates/includes/navigation.html").read_text(encoding="utf-8")
        assert 'id="startTourBtn"' in nav_html

    @pytest.mark.smoke
    def test_dashboard_includes_tour_scripts(self, authenticated_client):
        res = authenticated_client.get('/dashboard')
        assert res.status_code == 200
        html = res.get_data(as_text=True)
        assert "driver.js.iife.js" in html
        assert "app_tour.js" in html
        assert "driver.css" in html
