# ============================================================
# Pellets Analyzer — Security Middleware для Flask
# ============================================================
# Подключается в main.py для защиты от атак
# ============================================================

from functools import wraps
from flask import request, jsonify, abort, session
import time
import re
import hashlib
import os
import secrets
from collections import defaultdict


# ============================================================
# 1. RATE LIMITING (Ограничение частоты запросов)
# ============================================================
class RateLimiter:
    """Простой rate limiter без внешних зависимостей.
    Для продакшена рекомендуется Flask-Limiter."""
    
    def __init__(self):
        self.requests = defaultdict(list)
        self.max_requests = 100  # Максимум запросов
        self.window = 60  # Окно в секундах
    
    def is_allowed(self, key, max_requests=None, window=None):
        limit = max_requests if max_requests is not None else self.max_requests
        win = window if window is not None else self.window
        now = time.time()
        # Очистка старых запросов
        self.requests[key] = [t for t in self.requests[key] if now - t < win]
        if len(self.requests[key]) >= limit:
            return False
        self.requests[key].append(now)
        return True


rate_limiter = RateLimiter()


def rate_limit(max_requests=100, window=60):
    """Декоратор для ограничения частоты запросов."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            client_ip = request.remote_addr
            key = f"{client_ip}:{request.endpoint}"
            
            # Простая реализация через сессию
            if 'rate_limit' not in session:
                session['rate_limit'] = {}
            
            now = time.time()
            if key not in session['rate_limit']:
                session['rate_limit'][key] = []
            
            # Очистка
            session['rate_limit'][key] = [t for t in session['rate_limit'][key] if now - t < window]
            
            if len(session['rate_limit'][key]) >= max_requests:
                return jsonify({
                    'error': 'Слишком много запросов. Попробуйте позже.',
                    'retry_after': window
                }), 429
            
            session['rate_limit'][key].append(now)
            return f(*args, **kwargs)
        return decorated_function
    return decorator


# ============================================================
# 2. XSS ЗАЩИТА (Санитизация входных данных)
# ============================================================
def sanitize_input(text):
    """Очистка входных данных от XSS."""
    if not isinstance(text, str):
        return text
    
    # Удаление опасных HTML-тегов
    dangerous_patterns = [
        r'<script[^>]*>.*?</script>',
        r'javascript\s*:',
        r'on\w+\s*=',
        r'<iframe[^>]*>.*?</iframe>',
        r'<object[^>]*>.*?</object>',
        r'<embed[^>]*>.*?</embed>',
        r'<form[^>]*>.*?</form>',
        r'<input[^>]*>',
        r'<svg[^>]*>.*?</svg>',
        r'<img[^>]*onerror',
    ]
    
    for pattern in dangerous_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
    
    # Экранирование HTML-сущностей
    text = text.replace('&', '&amp;')
    text = text.replace('<', '&lt;')
    text = text.replace('>', '&gt;')
    text = text.replace('"', '&quot;')
    text = text.replace("'", '&#x27;')
    
    return text


def sanitize_dict(data):
    """Рекурсивная санитизация словаря."""
    if isinstance(data, dict):
        return {k: sanitize_dict(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_dict(item) for item in data]
    elif isinstance(data, str):
        return sanitize_input(data)
    return data


# ============================================================
# 3. SQL INJECTION ЗАЩИТА
# ============================================================
def validate_sql_input(text):
    """Проверка входных данных на SQL-инъекции."""
    if not isinstance(text, str):
        return True
    
    dangerous_keywords = [
        'DROP TABLE', 'DROP DATABASE', 'ALTER TABLE', 'ALTER DATABASE',
        'CREATE TABLE', 'CREATE DATABASE', 'DELETE FROM', 'TRUNCATE',
        'INSERT INTO', 'UPDATE SET', 'EXEC(', 'EXECUTE(',
        'xp_', 'sp_', 'UNION SELECT', 'OR 1=1', "OR '1'='1'",
        '; --', '; /*', '*/', 'WAITFOR DELAY', 'BENCHMARK('
    ]
    
    text_upper = text.upper()
    for keyword in dangerous_keywords:
        if keyword.upper() in text_upper:
            return False
    return True


# ============================================================
# 4. CSRF ЗАЩИТА
# ============================================================
def generate_csrf_token():
    """Генерация CSRF-токена."""
    if 'csrf_token' not in session:
        session['csrf_token'] = secrets.token_hex(32)
    return session['csrf_token']


def validate_csrf_token():
    """Валидация CSRF-токена."""
    token = (
        request.form.get('csrf_token')
        or request.headers.get('X-CSRFToken')
        or request.headers.get('X-CSRF-Token')
    )
    if not token and request.is_json:
        json_data = request.get_json(silent=True)
        if isinstance(json_data, dict):
            token = json_data.get('csrf_token')

    expected = session.get('csrf_token')
    if not token or not expected or not secrets.compare_digest(str(token), str(expected)):
        abort(400, description='CSRF token missing or invalid')


def csrf_exempt(f):
    """Декоратор для исключения роута из проверки CSRF."""
    f._csrf_exempt = True
    return f


def check_csrf():
    """Проверка CSRF-токена перед обработкой запроса."""
    from flask import current_app
    if not current_app.config.get('WTF_CSRF_ENABLED', not current_app.config.get('TESTING', False)):
        return
    if request.method not in ['POST', 'PUT', 'DELETE', 'PATCH']:
        return
    if request.endpoint:
        view = current_app.view_functions.get(request.endpoint)
        if getattr(view, '_csrf_exempt', False):
            return
    validate_csrf_token()


def csrf_protect(f):
    """Декоратор для явной защиты от CSRF."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if request.method in ['POST', 'PUT', 'DELETE', 'PATCH']:
            validate_csrf_token()
        return f(*args, **kwargs)
    return decorated_function


# ============================================================
# 5. SECURITY HEADERS (Заголовки безопасности)
# ============================================================
def add_security_headers(response):
    """Добавление заголовков безопасности к каждому ответу."""
    # Защита от кликджекинга
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    
    # Защита от MIME-sniffing
    response.headers['X-Content-Type-Options'] = 'nosniff'
    
    # XSS Protection
    response.headers['X-XSS-Protection'] = '1; mode=block'
    
    # Политика реферера
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    
    # Content Security Policy (без 'unsafe-eval' - D5)
    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://cdn.plot.ly https://cdnjs.cloudflare.com https://buttons.github.io https://unpkg.com; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdnjs.cloudflare.com https://demos.creative-tim.com https://unpkg.com; "
        "img-src 'self' data: blob: https://cdn.plot.ly https://cdnjs.cloudflare.com; "
        "font-src 'self' data: https://fonts.gstatic.com https://cdnjs.cloudflare.com https://demos.creative-tim.com; "
        "connect-src 'self' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com https://unpkg.com https://cdn.plot.ly; "
        "frame-ancestors 'self';"
    )
    
    # Permissions Policy
    response.headers['Permissions-Policy'] = 'camera=(), microphone=(), geolocation=()'
    
    # HSTS (только для HTTPS)
    # response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    
    # Удаление заголовка сервера
    response.headers.pop('Server', None)
    
    return response


# ============================================================
# 6. ЗАЩИТА ОТ BRUTE FORCE
# ============================================================
class BruteForceProtector:
    """Защита от перебора паролей."""
    
    def __init__(self, max_attempts=10, lockout_time=900):
        self.max_attempts = max_attempts  # Максимум 10 попыток (D5)
        self.lockout_time = lockout_time  # Время блокировки (15 мин)
        self.attempts = defaultdict(list)
        self.locked = {}
    
    def is_locked(self, identifier):
        """Проверка, заблокирован ли пользователь."""
        if identifier in self.locked:
            if time.time() - self.locked[identifier] < self.lockout_time:
                return True
            else:
                del self.locked[identifier]
                self.attempts[identifier] = []
        return False
    
    def record_attempt(self, identifier):
        """Запись попытки входа."""
        now = time.time()
        self.attempts[identifier] = [t for t in self.attempts[identifier] if now - t < self.lockout_time]
        self.attempts[identifier].append(now)
        
        if len(self.attempts[identifier]) >= self.max_attempts:
            self.locked[identifier] = now
            return True
        return False
    
    def clear_attempts(self, identifier):
        """Очистка попыток после успешного входа."""
        if identifier in self.attempts:
            del self.attempts[identifier]
        if identifier in self.locked:
            del self.locked[identifier]

    def reset(self):
        """Полная очистка состояния (для тестов)."""
        self.attempts.clear()
        self.locked.clear()
    
    def get_remaining_attempts(self, identifier):
        """Количество оставшихся попыток."""
        now = time.time()
        valid_attempts = [t for t in self.attempts.get(identifier, []) if now - t < self.lockout_time]
        return max(0, self.max_attempts - len(valid_attempts))


brute_force = BruteForceProtector(max_attempts=10, lockout_time=900)


# ============================================================
# 7. ВАЛИДАЦИЯ ЗАГРУЖАЕМЫХ ФАЙЛОВ
# ============================================================
ALLOWED_EXTENSIONS = {'.xlsx', '.xls', '.csv', '.json'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB


def allowed_file(filename):
    """Проверка допустимого расширения файла."""
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def validate_file_size(file):
    """Проверка размера файла."""
    file.seek(0, 2)  # Переход в конец
    size = file.tell()
    file.seek(0)  # Возврат в начало
    return size <= MAX_FILE_SIZE


# ============================================================
# 8. IP БЛОКИРОВКА
# ============================================================
class IPBlocklist:
    """Простой блокатор IP-адресов."""
    
    def __init__(self):
        self.blocked_ips = set()
    
    def block(self, ip):
        self.blocked_ips.add(ip)
    
    def unblock(self, ip):
        self.blocked_ips.discard(ip)
    
    def is_blocked(self, ip):
        return ip in self.blocked_ips


ip_blocklist = IPBlocklist()


def check_ip_blocklist(f):
    """Декоратор для проверки IP-адреса."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if ip_blocklist.is_blocked(request.remote_addr):
            abort(403, description='Доступ запрещён')
        return f(*args, **kwargs)
    return decorated_function


# ============================================================
# 9. ЛОГИРОВАНИЕ БЕЗОПАСНОСТИ
# ============================================================
import logging

security_logger = logging.getLogger('security')
security_logger.setLevel(logging.WARNING)

# Файл для логов безопасности
security_handler = logging.FileHandler('security.log', encoding='utf-8')
security_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s'
))
security_logger.addHandler(security_handler)


def log_security_event(event_type, details, ip=None):
    """Логирование событий безопасности."""
    security_logger.warning(
        f"[{event_type}] IP: {ip or request.remote_addr} | {details}"
    )

def is_safe_path(base_dir, path, follow_symlinks=True):
    """Проверяет, что путь path действительно находится внутри base_dir (защита от path traversal)."""
    if not path or not base_dir:
        return False
    try:
        resolved_base = os.path.realpath(base_dir) if follow_symlinks else os.path.abspath(base_dir)
        resolved_path = os.path.realpath(path) if follow_symlinks else os.path.abspath(path)
        return os.path.commonpath([resolved_base, resolved_path]) == resolved_base
    except (ValueError, Exception):
        return False

