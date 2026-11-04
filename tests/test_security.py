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
