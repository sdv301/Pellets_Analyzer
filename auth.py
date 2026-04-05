# auth.py - Модуль аутентификации и управления пользователями
"""
Поддерживает:
- Регистрацию через email с подтверждением
- Вход/выход через Flask-Login
- Роли: user, admin
- Хеширование паролей через bcrypt
- Логи активности пользователей
- Сброс пароля через email
"""

import sqlite3
import os
import secrets
import logging
from datetime import datetime, timedelta
from functools import wraps
from typing import Optional, Dict, Any, List

import bcrypt
from flask import session, redirect, url_for, flash, request, jsonify
from flask_mail import Mail, Message

logger = logging.getLogger(__name__)

# ============================================================
# КОНФИГУРАЦИЯ EMAIL
# ============================================================

MAIL_CONFIG = {
    'MAIL_SERVER': os.environ.get('MAIL_SERVER', 'smtp.gmail.com'),
    'MAIL_PORT': int(os.environ.get('MAIL_PORT', '587')),
    'MAIL_USE_TLS': os.environ.get('MAIL_USE_TLS', 'True').lower() == 'true',
    'MAIL_USERNAME': os.environ.get('MAIL_USERNAME', ''),
    'MAIL_PASSWORD': os.environ.get('MAIL_PASSWORD', ''),
    'MAIL_DEFAULT_SENDER': os.environ.get('MAIL_DEFAULT_SENDER', 'noreply@pellets-analyzer.com'),
}

mail = Mail()


def init_mail(app):
    """Инициализация Flask-Mail."""
    app.config.update(MAIL_CONFIG)
    mail.init_app(app)
    return mail


# ============================================================
# ИНИЦИАЛИЗАЦИЯ ТАБЛИЦ ПОЛЬЗОВАТЕЛЕЙ
# ============================================================

def init_auth_tables(db_path: str = 'pellets_data.db'):
    """Создаёт таблицы для аутентификации, ролей, сессий и логов."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Таблица ролей
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS roles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL,
        description TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # Таблица пользователей
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role_id INTEGER DEFAULT 1,
        is_active INTEGER DEFAULT 1,
        is_verified INTEGER DEFAULT 0,
        verification_token TEXT,
        reset_token TEXT,
        reset_token_expires TIMESTAMP,
        avatar_url TEXT,
        full_name TEXT,
        company TEXT,
        phone TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_login TIMESTAMP,
        login_count INTEGER DEFAULT 0,
        FOREIGN KEY (role_id) REFERENCES roles(id)
    )
    ''')

    # Таблица сессий
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS user_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        session_token TEXT UNIQUE NOT NULL,
        ip_address TEXT,
        user_agent TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        expires_at TIMESTAMP,
        is_active INTEGER DEFAULT 1,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    )
    ''')

    # Таблица логов активности
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS activity_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        action TEXT NOT NULL,
        details TEXT,
        ip_address TEXT,
        user_agent TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
    )
    ''')

    # Таблица загрузок файлов (история)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS user_uploads (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        filename TEXT NOT NULL,
        file_size INTEGER,
        file_type TEXT,
        records_count INTEGER,
        status TEXT DEFAULT 'processed',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    )
    ''')

    # Таблица ML моделей пользователей
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS user_ml_models (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        model_name TEXT NOT NULL,
        target_property TEXT NOT NULL,
        algorithm TEXT,
        r2_score REAL,
        mae REAL,
        cv_r2 REAL,
        training_data_size INTEGER,
        model_path TEXT,
        is_active INTEGER DEFAULT 1,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    )
    ''')

    # Вставка ролей по умолчанию
    default_roles = [
        ('user', 'Обычный пользователь'),
        ('admin', 'Администратор системы'),
        ('moderator', 'Модератор'),
    ]
    for role_name, role_desc in default_roles:
        cursor.execute(
            'INSERT OR IGNORE INTO roles (name, description) VALUES (?, ?)',
            (role_name, role_desc)
        )

    # Создание/обновление администратора по умолчанию
    admin_username = os.environ.get('ADMIN_USERNAME', 'admin')
    admin_email = os.environ.get('ADMIN_EMAIL', 'sdv300@bk.ru')
    admin_password = os.environ.get('ADMIN_PASSWORD', 'admin2180')

    cursor.execute('SELECT id, password_hash FROM users WHERE email = ? OR username = ?', (admin_email, admin_username))
    existing_admin = cursor.fetchone()

    cursor.execute('SELECT id FROM roles WHERE name = ?', ('admin',))
    admin_role = cursor.fetchone()
    role_id = admin_role[0] if admin_role else 2

    if existing_admin:
        # Обновляем пароль и email существующего админа
        password_hash = hash_password(admin_password)
        cursor.execute('''
            UPDATE users SET email = ?, password_hash = ?, role_id = ?, is_active = 1, is_verified = 1, full_name = 'Администратор'
            WHERE id = ?
        ''', (admin_email, password_hash, role_id, existing_admin[0]))
        logger.info(f"Обновлён администратор: {admin_email}")
    else:
        # Создаём нового админа
        password_hash = hash_password(admin_password)
        cursor.execute('''
            INSERT INTO users (username, email, password_hash, role_id, is_active, is_verified, full_name)
            VALUES (?, ?, ?, ?, 1, 1, ?)
        ''', (admin_username, admin_email, password_hash, role_id, 'Администратор'))
        logger.info(f"Создан администратор: {admin_email}")

    # Индексы для оптимизации
    indexes = [
        'CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)',
        'CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)',
        'CREATE INDEX IF NOT EXISTS idx_users_role ON users(role_id)',
        'CREATE INDEX IF NOT EXISTS idx_sessions_user ON user_sessions(user_id)',
        'CREATE INDEX IF NOT EXISTS idx_sessions_token ON user_sessions(session_token)',
        'CREATE INDEX IF NOT EXISTS idx_logs_user ON activity_logs(user_id)',
        'CREATE INDEX IF NOT EXISTS idx_logs_action ON activity_logs(action)',
        'CREATE INDEX IF NOT EXISTS idx_logs_created ON activity_logs(created_at)',
        'CREATE INDEX IF NOT EXISTS idx_uploads_user ON user_uploads(user_id)',
        'CREATE INDEX IF NOT EXISTS idx_models_user ON user_ml_models(user_id)',
    ]
    for idx in indexes:
        cursor.execute(idx)

    conn.commit()
    conn.close()
    logger.info("Таблицы аутентификации инициализированы")


# ============================================================
# ХЕШИРОВАНИЕ ПАРОЛЕЙ
# ============================================================

def hash_password(password: str) -> str:
    """Хеширует пароль через bcrypt."""
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')


def check_password(password: str, password_hash: str) -> bool:
    """Проверяет пароль против хеша."""
    try:
        return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
    except Exception:
        return False


# ============================================================
# УПРАВЛЕНИЕ ПОЛЬЗОВАТЕЛЯМИ
# ============================================================

def create_user(
    db_path: str,
    username: str,
    email: str,
    password: str,
    full_name: str = '',
    role: str = 'user'
) -> Dict[str, Any]:
    """
    Создаёт нового пользователя.
    Returns: {'success': bool, 'user_id': int, 'error': str, 'verification_token': str}
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        # Проверка существования
        cursor.execute('SELECT id FROM users WHERE email = ? OR username = ?', (email, username))
        if cursor.fetchone():
            return {'success': False, 'error': 'Email или имя пользователя уже заняты'}

        # Получаем role_id
        cursor.execute('SELECT id FROM roles WHERE name = ?', (role,))
        role_row = cursor.fetchone()
        role_id = role_row[0] if role_row else 1

        password_hash = hash_password(password)
        verification_token = secrets.token_urlsafe(32)

        cursor.execute('''
            INSERT INTO users (username, email, password_hash, role_id, verification_token, full_name)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (username, email, password_hash, role_id, verification_token, full_name))

        user_id = cursor.lastrowid
        conn.commit()

        log_activity(db_path, user_id, 'register', f'Регистрация пользователя: {email}')

        return {
            'success': True,
            'user_id': user_id,
            'verification_token': verification_token,
        }
    except Exception as e:
        conn.rollback()
        logger.error(f"Ошибка создания пользователя: {e}")
        return {'success': False, 'error': str(e)}
    finally:
        conn.close()


def verify_user(db_path: str, token: str) -> Dict[str, Any]:
    """Подтверждает email пользователя по токену."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT id FROM users WHERE verification_token = ?', (token,))
        row = cursor.fetchone()
        if not row:
            return {'success': False, 'error': 'Неверный токен подтверждения'}

        cursor.execute('''
            UPDATE users SET is_verified = 1, verification_token = NULL WHERE id = ?
        ''', (row[0],))
        conn.commit()

        log_activity(db_path, row[0], 'verify_email', 'Email подтверждён')
        return {'success': True, 'user_id': row[0]}
    except Exception as e:
        conn.rollback()
        return {'success': False, 'error': str(e)}
    finally:
        conn.close()


def authenticate_user(db_path: str, email_or_username: str, password: str) -> Dict[str, Any]:
    """
    Аутентифицирует пользователя.
    Returns: {'success': bool, 'user': dict, 'error': str}
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        cursor.execute('''
            SELECT u.*, r.name as role_name
            FROM users u
            LEFT JOIN roles r ON u.role_id = r.id
            WHERE u.email = ? OR u.username = ?
        ''', (email_or_username, email_or_username))
        user = cursor.fetchone()

        if not user:
            return {'success': False, 'error': 'Пользователь не найден'}

        if not check_password(password, user['password_hash']):
            return {'success': False, 'error': 'Неверный пароль'}

        if not user['is_active']:
            return {'success': False, 'error': 'Аккаунт деактивирован'}

        if not user['is_verified']:
            return {'success': False, 'error': 'Email не подтверждён. Проверьте почту.'}

        # Обновляем last_login и login_count
        cursor.execute('''
            UPDATE users SET last_login = CURRENT_TIMESTAMP, login_count = login_count + 1 WHERE id = ?
        ''', (user['id'],))
        conn.commit()

        log_activity(db_path, user['id'], 'login', f'Вход в систему: {user["email"]}')

        user_dict = dict(user)
        return {'success': True, 'user': user_dict}
    except Exception as e:
        logger.error(f"Ошибка аутентификации: {e}")
        return {'success': False, 'error': str(e)}
    finally:
        conn.close()


def get_user_by_id(db_path: str, user_id: int) -> Optional[Dict[str, Any]]:
    """Получает пользователя по ID."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        cursor.execute('''
            SELECT u.*, r.name as role_name, r.description as role_description
            FROM users u
            LEFT JOIN roles r ON u.role_id = r.id
            WHERE u.id = ?
        ''', (user_id,))
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_user_by_email(db_path: str, email: str) -> Optional[Dict[str, Any]]:
    """Получает пользователя по email."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        cursor.execute('''
            SELECT u.*, r.name as role_name
            FROM users u
            LEFT JOIN roles r ON u.role_id = r.id
            WHERE u.email = ?
        ''', (email,))
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def update_user_profile(db_path: str, user_id: int, **kwargs) -> Dict[str, Any]:
    """Обновляет профиль пользователя."""
    allowed_fields = {'full_name', 'company', 'phone', 'avatar_url'}
    updates = {k: v for k, v in kwargs.items() if k in allowed_fields}
    if not updates:
        return {'success': False, 'error': 'Нет допустимых полей для обновления'}

    set_clause = ', '.join(f'{k} = ?' for k in updates)
    values = list(updates.values()) + [user_id]

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(f'UPDATE users SET {set_clause} WHERE id = ?', values)
        conn.commit()
        log_activity(db_path, user_id, 'update_profile', f'Обновление профиля: {list(updates.keys())}')
        return {'success': True}
    except Exception as e:
        conn.rollback()
        return {'success': False, 'error': str(e)}
    finally:
        conn.close()


def change_password(db_path: str, user_id: int, old_password: str, new_password: str) -> Dict[str, Any]:
    """Изменяет пароль пользователя."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT password_hash FROM users WHERE id = ?', (user_id,))
        row = cursor.fetchone()
        if not row:
            return {'success': False, 'error': 'Пользователь не найден'}

        if not check_password(old_password, row['password_hash']):
            return {'success': False, 'error': 'Неверный текущий пароль'}

        new_hash = hash_password(new_password)
        cursor.execute('UPDATE users SET password_hash = ? WHERE id = ?', (new_hash, user_id))
        conn.commit()
        log_activity(db_path, user_id, 'change_password', 'Пароль изменён')
        return {'success': True}
    except Exception as e:
        conn.rollback()
        return {'success': False, 'error': str(e)}
    finally:
        conn.close()


def request_password_reset(db_path: str, email: str) -> Dict[str, Any]:
    """Создаёт токен сброса пароля."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT id FROM users WHERE email = ?', (email,))
        row = cursor.fetchone()
        if not row:
            # Не раскрываем, существует ли email
            return {'success': True, 'message': 'Если email существует, мы отправили ссылку для сброса'}

        reset_token = secrets.token_urlsafe(32)
        expires = datetime.utcnow() + timedelta(hours=1)
        cursor.execute('''
            UPDATE users SET reset_token = ?, reset_token_expires = ? WHERE id = ?
        ''', (reset_token, expires, row[0]))
        conn.commit()

        log_activity(db_path, row[0], 'reset_request', 'Запрос сброса пароля')
        return {'success': True, 'reset_token': reset_token, 'user_id': row[0]}
    except Exception as e:
        conn.rollback()
        return {'success': False, 'error': str(e)}
    finally:
        conn.close()


def reset_password(db_path: str, token: str, new_password: str) -> Dict[str, Any]:
    """Сбрасывает пароль по токену."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute('''
            SELECT id FROM users WHERE reset_token = ? AND reset_token_expires > CURRENT_TIMESTAMP
        ''', (token,))
        row = cursor.fetchone()
        if not row:
            return {'success': False, 'error': 'Неверный или просроченный токен'}

        new_hash = hash_password(new_password)
        cursor.execute('''
            UPDATE users SET password_hash = ?, reset_token = NULL, reset_token_expires = NULL WHERE id = ?
        ''', (new_hash, row[0]))
        conn.commit()

        log_activity(db_path, row[0], 'reset_password', 'Пароль сброшен')
        return {'success': True}
    except Exception as e:
        conn.rollback()
        return {'success': False, 'error': str(e)}
    finally:
        conn.close()


# ============================================================
# ЛОГИРОВАНИЕ АКТИВНОСТИ
# ============================================================

def log_activity(db_path: str, user_id: Optional[int], action: str, details: str = ''):
    """Записывает действие в лог активности."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        ip_address = request.remote_addr if request else ''
        user_agent = request.headers.get('User-Agent', '') if request else ''
        cursor.execute('''
            INSERT INTO activity_logs (user_id, action, details, ip_address, user_agent)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, action, details, ip_address, user_agent))
        conn.commit()
    except Exception as e:
        logger.error(f"Ошибка логирования: {e}")
    finally:
        conn.close()


def get_activity_logs(
    db_path: str,
    user_id: Optional[int] = None,
    action: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
) -> List[Dict[str, Any]]:
    """Получает логи активности."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        query = '''
            SELECT al.*, u.username, u.email
            FROM activity_logs al
            LEFT JOIN users u ON al.user_id = u.id
            WHERE 1=1
        '''
        params = []
        if user_id:
            query += ' AND al.user_id = ?'
            params.append(user_id)
        if action:
            query += ' AND al.action = ?'
            params.append(action)
        query += ' ORDER BY al.created_at DESC LIMIT ? OFFSET ?'
        params.extend([limit, offset])

        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def get_user_stats(db_path: str) -> Dict[str, Any]:
    """Получает статистику пользователей."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        stats = {}
        cursor.execute('SELECT COUNT(*) FROM users')
        stats['total_users'] = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM users WHERE is_active = 1')
        stats['active_users'] = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM users WHERE is_verified = 1')
        stats['verified_users'] = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM users WHERE role_id = (SELECT id FROM roles WHERE name = "admin")')
        stats['admin_users'] = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM activity_logs WHERE DATE(created_at) = DATE("now")')
        stats['today_actions'] = cursor.fetchone()[0]

        cursor.execute('''
            SELECT u.username, u.email, u.last_login, u.login_count
            FROM users u
            ORDER BY u.last_login DESC
            LIMIT 10
        ''')
        stats['recent_users'] = [
            {'username': r[0], 'email': r[1], 'last_login': r[2], 'login_count': r[3]}
            for r in cursor.fetchall()
        ]

        return stats
    finally:
        conn.close()


# ============================================================
# УПРАВЛЕНИЕ ЗАГРУЗКАМИ
# ============================================================

def log_upload(db_path: str, user_id: int, filename: str, file_size: int = 0,
               file_type: str = '', records_count: int = 0, status: str = 'processed'):
    """Записывает информацию о загрузке файла."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO user_uploads (user_id, filename, file_size, file_type, records_count, status)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (user_id, filename, file_size, file_type, records_count, status))
        conn.commit()
    except Exception as e:
        logger.error(f"Ошибка логирования загрузки: {e}")
    finally:
        conn.close()


def get_user_uploads(db_path: str, user_id: int, limit: int = 50) -> List[Dict[str, Any]]:
    """Получает историю загрузок пользователя."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        cursor.execute('''
            SELECT * FROM user_uploads
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
        ''', (user_id, limit))
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


# ============================================================
# УПРАВЛЕНИЕ ML МОДЕЛЯМИ
# ============================================================

def save_ml_model(db_path: str, user_id: int, model_name: str, target_property: str,
                  algorithm: str = '', r2_score: float = 0, mae: float = 0,
                  cv_r2: float = 0, training_data_size: int = 0, model_path: str = ''):
    """Сохраняет информацию о ML модели пользователя."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO user_ml_models (user_id, model_name, target_property, algorithm,
                                        r2_score, mae, cv_r2, training_data_size, model_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, model_name, target_property, algorithm, r2_score, mae, cv_r2,
              training_data_size, model_path))
        conn.commit()
        return cursor.lastrowid
    except Exception as e:
        logger.error(f"Ошибка сохранения ML модели: {e}")
        return None
    finally:
        conn.close()


def get_user_ml_models(db_path: str, user_id: int) -> List[Dict[str, Any]]:
    """Получает ML модели пользователя."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        cursor.execute('''
            SELECT * FROM user_ml_models
            WHERE user_id = ? AND is_active = 1
            ORDER BY created_at DESC
        ''', (user_id,))
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


# ============================================================
# ДЕКОРАТОРЫ ЗАЩИТЫ РОУТОВ
# ============================================================

def login_required(f):
    """Декоратор: требует авторизации."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Пожалуйста, войдите в систему.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


def admin_required(f):
    """Декоратор: требует роли администратора."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Пожалуйста, войдите в систему.', 'warning')
            return redirect(url_for('login'))

        if session.get('role_name') != 'admin':
            flash('Доступ запрещён. Требуются права администратора.', 'danger')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated_function


def moderator_required(f):
    """Декоратор: требует роли модератора или выше."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Пожалуйста, войдите в систему.', 'warning')
            return redirect(url_for('login'))

        role = session.get('role_name', '')
        if role not in ('admin', 'moderator'):
            flash('Доступ запрещён.', 'danger')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated_function


# ============================================================
# ОТПРАВКА EMAIL
# ============================================================

def send_verification_email(email: str, token: str, base_url: str):
    """Отправляет email с подтверждением регистрации."""
    verify_url = f"{base_url}/verify/{token}"
    try:
        msg = Message(
            subject='Подтверждение регистрации — Pellets Analyzer',
            recipients=[email],
            html=f'''
            <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
                <h2 style="color: #2d60ff;">Добро пожаловать в Pellets Analyzer!</h2>
                <p>Для завершения регистрации перейдите по ссылке:</p>
                <p><a href="{verify_url}" style="background: #2d60ff; color: white; padding: 12px 24px;
                   text-decoration: none; border-radius: 8px; display: inline-block;">Подтвердить email</a></p>
                <p>Или скопируйте ссылку: <code>{verify_url}</code></p>
                <hr>
                <p style="color: #888; font-size: 12px;">Если вы не регистрировались, проигнорируйте это письмо.</p>
            </div>
            '''
        )
        mail.send(msg)
        logger.info(f"Email подтверждения отправлен: {email}")
        return True
    except Exception as e:
        logger.error(f"Ошибка отправки email: {e}")
        return False


def send_reset_email(email: str, token: str, base_url: str):
    """Отправляет email для сброса пароля."""
    reset_url = f"{base_url}/reset-password/{token}"
    try:
        msg = Message(
            subject='Сброс пароля — Pellets Analyzer',
            recipients=[email],
            html=f'''
            <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
                <h2 style="color: #dc3545;">Сброс пароля</h2>
                <p>Вы запросили сброс пароля. Перейдите по ссылке:</p>
                <p><a href="{reset_url}" style="background: #dc3545; color: white; padding: 12px 24px;
                   text-decoration: none; border-radius: 8px; display: inline-block;">Сбросить пароль</a></p>
                <p>Ссылка действительна 1 час.</p>
                <hr>
                <p style="color: #888; font-size: 12px;">Если вы не запрашивали сброс, проигнорируйте это письмо.</p>
            </div>
            '''
        )
        mail.send(msg)
        logger.info(f"Email сброса пароля отправлен: {email}")
        return True
    except Exception as e:
        logger.error(f"Ошибка отправки email сброса: {e}")
        return False
