"""
Тесты для модуля security.py — безопасность.
"""
import pytest
from app.utils.security import (
    sanitize_input, sanitize_dict, validate_sql_input,
    allowed_file, validate_file_size,
    RateLimiter, BruteForceProtector, IPBlocklist
)


class TestSanitizeInput:
    """Тесты санитизации входных данных."""

    def test_removes_script_tags(self):
        """Удаляет <script> теги."""
        malicious = '<script>alert("xss")</script>Hello'
        result = sanitize_input(malicious)
        assert '<script>' not in result
        assert 'alert' not in result

    def test_removes_javascript_protocol(self):
        """Удаляет javascript: протокол."""
        malicious = '<a href="javascript:alert(1)">Click</a>'
        result = sanitize_input(malicious)
        assert 'javascript:' not in result.lower()

    def test_removes_event_handlers(self):
        """Удаляет обработчики событий."""
        malicious = '<img src=x onerror=alert(1)>'
        result = sanitize_input(malicious)
        assert 'onerror' not in result.lower()

    def test_escapes_html_entities(self):
        """Экранирует HTML-сущности."""
        text = '<div>Test & "quotes" \'single\'</div>'
        result = sanitize_input(text)
        assert '&lt;' in result
        assert '&gt;' in result
        assert '&quot;' in result

    def test_non_string_passthrough(self):
        """Не-строковые значения возвращаются как есть."""
        assert sanitize_input(123) == 123
        assert sanitize_input(None) is None

    def test_removes_iframe(self):
        """Удаляет iframe теги."""
        malicious = '<iframe src="evil.com"></iframe>'
        result = sanitize_input(malicious)
        assert '<iframe' not in result


class TestSanitizeDict:
    """Тесты рекурсивной санитизации словарей."""

    def test_sanitizes_string_values(self):
        """Санитизирует строковые значения."""
        data = {'name': '<script>alert(1)</script>'}
        result = sanitize_dict(data)
        assert '<script>' not in result['name']

    def test_sanitizes_nested_dicts(self):
        """Санитизирует вложенные словари."""
        data = {'user': {'bio': '<script>evil()</script>'}}
        result = sanitize_dict(data)
        assert '<script>' not in result['user']['bio']

    def test_sanitizes_lists(self):
        """Санитизирует списки."""
        data = {'comments': ['<script>bad()</script>', 'Good comment']}
        result = sanitize_dict(data)
        assert '<script>' not in result['comments'][0]

    def test_preserves_numbers(self):
        """Сохраняет числовые значения."""
        data = {'age': 25, 'score': 95.5}
        result = sanitize_dict(data)
        assert result['age'] == 25
        assert result['score'] == 95.5


class TestValidateSQLInput:
    """Тесты проверки SQL-инъекций."""

    def test_safe_input(self):
        """Безопасный ввод проходит."""
        assert validate_sql_input('Hello World') is True
        assert validate_sql_input('123') is True

    def test_dangerous_keywords(self):
        """Опасные SQL-ключевые слова блокируются."""
        assert validate_sql_input('DROP TABLE users') is False
        assert validate_sql_input('DELETE FROM users') is False
        assert validate_sql_input("OR 1=1") is False
        assert validate_sql_input('UNION SELECT * FROM users') is False

    def test_non_string_input(self):
        """Не-строковый ввод проходит."""
        assert validate_sql_input(123) is True


class TestFileValidation:
    """Тесты валидации файлов."""

    def test_allowed_extensions(self):
        """Допустимые расширения файлов."""
        assert allowed_file('data.xlsx') is True
        assert allowed_file('data.xls') is True
        assert allowed_file('data.csv') is True
        assert allowed_file('data.json') is True

    def test_disallowed_extensions(self):
        """Недопустимые расширения файлов."""
        assert allowed_file('script.exe') is False
        assert allowed_file('malware.php') is False
        assert allowed_file('hack.sh') is False

    def test_case_insensitive(self):
        """Проверка без учёта регистра."""
        assert allowed_file('DATA.XLSX') is True
        assert allowed_file('Data.Csv') is True

    def test_file_size_validation(self):
        """Проверка размера файла."""
        import io
        # Маленький файл
        small_file = io.BytesIO(b'test data')
        assert validate_file_size(small_file) is True


class TestRateLimiter:
    """Тесты ограничения частоты запросов."""

    def test_allows_normal_requests(self):
        """Обычные запросы проходят."""
        limiter = RateLimiter()
        limiter.max_requests = 5
        limiter.window = 60

        for i in range(5):
            assert limiter.is_allowed('test_key') is True

    def test_blocks_excessive_requests(self):
        """Чрезмерные запросы блокируются."""
        limiter = RateLimiter()
        limiter.max_requests = 3
        limiter.window = 60

        for i in range(3):
            limiter.is_allowed('test_key')

        assert limiter.is_allowed('test_key') is False


class TestBruteForceProtector:
    """Тесты защиты от brute force."""

    def test_allows_normal_attempts(self):
        """Обычные попытки проходят."""
        protector = BruteForceProtector(max_attempts=3, lockout_time=60)

        assert protector.is_locked('user1') is False
        protector.record_attempt('user1')
        assert protector.is_locked('user1') is False

    def test_locks_after_max_attempts(self):
        """Блокировка после максимального числа попыток."""
        protector = BruteForceProtector(max_attempts=3, lockout_time=60)

        for i in range(3):
            protector.record_attempt('user1')

        assert protector.is_locked('user1') is True

    def test_clear_attempts(self):
        """Очистка попыток после успешного входа."""
        protector = BruteForceProtector(max_attempts=3, lockout_time=60)

        protector.record_attempt('user1')
        protector.record_attempt('user1')
        protector.clear_attempts('user1')

        assert protector.is_locked('user1') is False
        assert protector.get_remaining_attempts('user1') == 3

    def test_remaining_attempts(self):
        """Количество оставшихся попыток."""
        protector = BruteForceProtector(max_attempts=5, lockout_time=60)

        protector.record_attempt('user1')
        protector.record_attempt('user1')

        assert protector.get_remaining_attempts('user1') == 3


class TestIPBlocklist:
    """Тесты блокировки IP."""

    def test_block_and_unblock(self):
        """Блокировка и разблокировка IP."""
        blocklist = IPBlocklist()

        assert blocklist.is_blocked('192.168.1.1') is False

        blocklist.block('192.168.1.1')
        assert blocklist.is_blocked('192.168.1.1') is True

        blocklist.unblock('192.168.1.1')
        assert blocklist.is_blocked('192.168.1.1') is False

    def test_unblock_nonexistent(self):
        """Разблокировка несуществующего IP не вызывает ошибок."""
        blocklist = IPBlocklist()
        blocklist.unblock('10.0.0.1')  # Не должно вызвать ошибку


class TestPathTraversalSecurity:
    """D4: Тесты защиты от Path Traversal."""

    def test_safe_path_valid(self, tmp_path):
        from app.utils.security import is_safe_path
        base = tmp_path / "uploads"
        base.mkdir()
        target = base / "valid_file.csv"
        assert is_safe_path(str(base), str(target)) is True

    def test_safe_path_rejects_traversal(self, tmp_path):
        from app.utils.security import is_safe_path
        base = tmp_path / "uploads"
        base.mkdir()
        target = base / ".." / "secret.txt"
        assert is_safe_path(str(base), str(target)) is False

    def test_safe_path_handles_none_and_empty(self):
        from app.utils.security import is_safe_path
        assert is_safe_path("", "test") is False
        assert is_safe_path("base", "") is False
        assert is_safe_path(None, "test") is False


class TestStoredXSSInCompare:
    """D1: Тест предотвращения XSS в таблице сравнения."""

    def test_compare_escapes_script_in_composition(self, authenticated_client, db_path):
        import sqlite3
        from app.database.database import init_db
        init_db(db_path)
        conn = sqlite3.connect(db_path)
        xss_name = '<script>alert("xss")</script> 100% Опилки'
        comp2 = '100% Солома'
        conn.execute("""
            INSERT OR REPLACE INTO measured_parameters 
            (composition, war, ad, vd, q, cd, hd, nd, sd, od, density, tign, tb, mass_loss, kt, h, co2, co, so2, nox)
            VALUES (?, 10.0, 5.0, 75.0, 17.5, 45.0, 5.5, 0.5, 0.1, 40.0, 1100.0, 250.0, 300.0, 80.0, 1.2, 0.5, 12.0, 0.05, 0.01, 0.02)
        """, (xss_name,))
        conn.execute("""
            INSERT OR REPLACE INTO measured_parameters 
            (composition, war, ad, vd, q, cd, hd, nd, sd, od, density, tign, tb, mass_loss, kt, h, co2, co, so2, nox)
            VALUES (?, 12.0, 4.0, 70.0, 16.5, 42.0, 5.0, 0.4, 0.1, 38.0, 1050.0, 260.0, 310.0, 78.0, 1.1, 0.4, 11.0, 0.04, 0.01, 0.02)
        """, (comp2,))
        conn.commit()
        conn.close()

        resp = authenticated_client.post('/compare', data={
            'comp1': xss_name,
            'comp2': comp2,
            'paramGroup': 'all'
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['success'] is True
        comparison_html = data.get('comparison', '')
        assert '<script>alert("xss")</script>' not in comparison_html
        assert '&lt;script&gt;alert(&quot;xss&quot;)&lt;/script&gt;' in comparison_html or '&lt;script&gt;' in comparison_html


class TestAdminSecurity:
    """D2: Тесты пароля администратора."""

    def test_no_hardcoded_admin_password_on_init(self, tmp_path, monkeypatch):
        from app.auth.auth import init_auth_tables
        import sqlite3
        test_db = str(tmp_path / "auth_test.db")
        monkeypatch.delenv('ADMIN_PASSWORD', raising=False)
        init_auth_tables(test_db)

        conn = sqlite3.connect(test_db)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE username = 'admin'")
        admin = cursor.fetchone()
        conn.close()
        # Без ADMIN_PASSWORD в env админ не должен создаваться со статическим паролем
        assert admin is None

    def test_admin_password_not_overwritten_on_restart(self, tmp_path, monkeypatch):
        from app.auth.auth import init_auth_tables, create_admin_user
        import sqlite3
        test_db = str(tmp_path / "auth_restart.db")
        init_auth_tables(test_db)

        # Создаём админа с первым паролем
        create_admin_user(test_db, 'admin', 'admin@example.com', 'initial_pass_123')
        conn = sqlite3.connect(test_db)
        cursor = conn.cursor()
        cursor.execute("SELECT password_hash FROM users WHERE username = 'admin'")
        first_hash = cursor.fetchone()[0]
        conn.close()

        # Эмулируем перезапуск с другим ADMIN_PASSWORD
        monkeypatch.setenv('ADMIN_PASSWORD', 'different_password')
        init_auth_tables(test_db)

        conn = sqlite3.connect(test_db)
        cursor = conn.cursor()
        cursor.execute("SELECT password_hash FROM users WHERE username = 'admin'")
        second_hash = cursor.fetchone()[0]
        conn.close()

        # Существующий хеш не должен быть перезаписан
        assert first_hash == second_hash

    def test_create_admin_cli_command(self, app, runner):
        res = runner.invoke(args=['create-admin', '--username', 'superadmin', '--email', 'super@example.com', '--password', 'SecretPass123!'])
        assert res.exit_code == 0
        assert 'superadmin' in res.output


class TestConfigAndDebugSecurity:
    """D3: Тесты конфигурации безопасности и скрытия трейсбеков."""

    def test_secret_key_required_outside_testing(self, monkeypatch):
        from app import create_app
        monkeypatch.delenv('SECRET_KEY', raising=False)
        with pytest.raises(RuntimeError, match="SECRET_KEY"):
            create_app({'TESTING': False})

    def test_traceback_hidden_when_debug_false(self):
        from app import create_app
        test_app = create_app({'TESTING': True, 'DEBUG': False, 'SECRET_KEY': 'k', 'PROPAGATE_EXCEPTIONS': False})

        @test_app.route('/trigger-500-test')
        def trigger_500():
            raise ValueError("Secret database credentials leaked!")

        client = test_app.test_client()
        res = client.get('/trigger-500-test')
        assert res.status_code == 500
        html = res.get_data(as_text=True)
        assert "Secret database credentials leaked!" not in html
        assert "Traceback" not in html


class TestFileUploadSecurityRoutes:
    """D4: Тесты валидации загрузок и защиты от Path Traversal в роутах."""

    def test_upload_rejects_path_traversal_filename(self, authenticated_client):
        import io
        data = {'file': (io.BytesIO(b"dummy,content\n1,2"), '../evil.csv')}
        res = authenticated_client.post('/', data=data, content_type='multipart/form-data')
        # secure_filename превращает '../evil.csv' в 'evil.csv' (безопасно) или отклоняет
        assert res.status_code in (200, 400)
        json_data = res.get_json()
        if res.status_code == 400:
            assert json_data['success'] is False

    def test_upload_rejects_disallowed_extension(self, authenticated_client):
        import io
        data = {'file': (io.BytesIO(b"print('hack')"), 'exploit.py')}
        res = authenticated_client.post('/', data=data, content_type='multipart/form-data')
        assert res.status_code == 400
        assert res.get_json()['success'] is False

    def test_load_file_rejects_traversal(self, authenticated_client):
        res = authenticated_client.post('/load_file', data={'selected_file': '../../etc/passwd'})
        assert res.status_code == 400
        assert res.get_json()['success'] is False

    def test_upload_economics_rejects_disallowed_extension(self, authenticated_client):
        import io
        data = {'file': (io.BytesIO(b"malicious script"), 'payload.sh')}
        res = authenticated_client.post('/api/upload_economics', data=data, content_type='multipart/form-data')
        assert res.status_code == 400
        assert res.get_json()['success'] is False


class TestSecurityHeadersAndCSP:
    """D5: Тесты заголовков безопасности и Content Security Policy."""

    def test_security_headers_present(self, client):
        res = client.get('/login')
        assert res.status_code == 200
        assert res.headers.get('X-Frame-Options') == 'SAMEORIGIN'
        assert res.headers.get('X-Content-Type-Options') == 'nosniff'
        assert res.headers.get('X-XSS-Protection') == '1; mode=block'
        assert res.headers.get('Referrer-Policy') == 'strict-origin-when-cross-origin'
        assert 'Content-Security-Policy' in res.headers

    def test_csp_does_not_contain_unsafe_eval(self, client):
        res = client.get('/login')
        csp = res.headers.get('Content-Security-Policy', '')
        assert "script-src" in csp
        assert "'unsafe-eval'" not in csp


class TestBruteForceProtection:
    """D5: Тесты защиты от перебора паролей (10 попыток -> 429)."""

    def test_ten_failed_logins_return_429(self, client):
        from app.utils.security import brute_force
        brute_force.reset()

        # Первые 9 неудачных попыток возвращают 200 (страница с ошибкой)
        for i in range(9):
            res = client.post('/login', data={
                'email_or_username': 'brutetestuser',
                'password': f'wrongpass{i}'
            })
            assert res.status_code == 200

        # 10-я попытка блокирует и возвращает 429
        res10 = client.post('/login', data={
            'email_or_username': 'brutetestuser',
            'password': 'wrongpass10'
        })
        assert res10.status_code == 429

        # Последующие попытки сразу получают 429
        res11 = client.post('/login', data={
            'email_or_username': 'brutetestuser',
            'password': 'wrongpass11'
        })
        assert res11.status_code == 429

        brute_force.reset()


class TestCSRFProtection:
    """D5: Тесты CSRF-защиты при WTF_CSRF_ENABLED=True."""

    def test_post_without_csrf_returns_400(self, app):
        from app.utils.security import generate_csrf_token
        app.config['WTF_CSRF_ENABLED'] = True
        client = app.test_client()

        # Устанавливаем сессию с токеном
        with client.session_transaction() as sess:
            sess['csrf_token'] = 'valid-token-12345'

        # POST без токена -> 400
        res = client.post('/login', data={
            'email_or_username': 'admin',
            'password': 'password'
        })
        assert res.status_code == 400

        # POST с неверным токеном -> 400
        res_wrong = client.post('/login', data={
            'csrf_token': 'wrong-token',
            'email_or_username': 'admin',
            'password': 'password'
        })
        assert res_wrong.status_code == 400

        # POST с верным токеном в форме не даёт 400 CSRF
        res_valid = client.post('/login', data={
            'csrf_token': 'valid-token-12345',
            'email_or_username': 'admin',
            'password': 'wrongpassword'
        })
        assert res_valid.status_code != 400

        # POST с верным токеном в заголовке X-CSRFToken не даёт 400 CSRF
        res_header = client.post(
            '/login',
            data={'email_or_username': 'admin', 'password': 'wrongpassword'},
            headers={'X-CSRFToken': 'valid-token-12345'}
        )
        assert res_header.status_code != 400

        app.config['WTF_CSRF_ENABLED'] = False


class TestSessionCookieHardening:
    """D6: Тесты конфигурации и флагов cookie сессии."""

    def test_session_cookie_flags(self, app):
        assert app.config['SESSION_COOKIE_HTTPONLY'] is True
        assert app.config['SESSION_COOKIE_SAMESITE'] == 'Lax'
        assert app.config['PERMANENT_SESSION_LIFETIME'] == 7 * 24 * 3600
