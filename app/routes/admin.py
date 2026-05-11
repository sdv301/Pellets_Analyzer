# app/routes/admin.py — Админ-панель
import sqlite3
from flask import Blueprint, render_template, request, jsonify, session
from app.auth.auth import login_required, admin_required, get_activity_logs, get_user_stats, log_activity, hash_password
from app.database.database import query_db

admin_bp = Blueprint('admin', __name__)

_db_path = 'pellets_data.db'
_ALLOWED_ROLE_NAMES = {'user', 'moderator', 'admin'}

def set_db_path(path):
    global _db_path
    _db_path = path


def _count_admins(cursor):
    cursor.execute('''
        SELECT COUNT(*)
        FROM users u
        JOIN roles r ON r.id = u.role_id
        WHERE r.name = 'admin'
    ''')
    row = cursor.fetchone()
    return int(row[0]) if row else 0


def _get_role_id(cursor, role_name):
    cursor.execute('SELECT id FROM roles WHERE name = ?', (role_name,))
    row = cursor.fetchone()
    return row[0] if row else None


def _delete_user_dependencies(cursor, user_id):
    """Удаляет связанные записи пользователя перед удалением аккаунта."""
    cursor.execute('DELETE FROM user_sessions WHERE user_id = ?', (user_id,))
    cursor.execute('DELETE FROM user_uploads WHERE user_id = ?', (user_id,))
    cursor.execute('DELETE FROM user_ml_models WHERE user_id = ?', (user_id,))
    cursor.execute('UPDATE activity_logs SET user_id = NULL WHERE user_id = ?', (user_id,))


@admin_bp.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    stats = get_user_stats(_db_path)
    activity_logs = get_activity_logs(_db_path, limit=100)

    conn = sqlite3.connect(_db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        cursor.execute('''
            SELECT u.*, r.name as role_name
            FROM users u
            LEFT JOIN roles r ON u.role_id = r.id
            ORDER BY u.created_at DESC
        ''')
        all_users = [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()

    return render_template('Admin/admin_dashboard.html',
                          stats=stats,
                          all_users=all_users,
                          activity_logs=activity_logs)


@admin_bp.route('/admin/database')
@admin_required
def admin_database():
    page = max(int(request.args.get('page', 1)), 1)
    per_page = min(max(int(request.args.get('per_page', 25)), 5), 200)
    selected_table = request.args.get('table', '').strip()

    conn = sqlite3.connect(_db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name NOT LIKE 'sqlite_%'
            ORDER BY name
        """)
        table_names = [row['name'] for row in cursor.fetchall()]

        if selected_table not in table_names and table_names:
            selected_table = table_names[0]

        columns = []
        rows = []
        total_rows = 0

        if selected_table:
            cursor.execute(f'SELECT COUNT(*) AS cnt FROM "{selected_table}"')
            total_rows = int(cursor.fetchone()['cnt'])

            offset = (page - 1) * per_page
            cursor.execute(f'SELECT * FROM "{selected_table}" LIMIT ? OFFSET ?', (per_page, offset))
            rows = [dict(row) for row in cursor.fetchall()]
            columns = list(rows[0].keys()) if rows else []
            if not columns:
                cursor.execute(f'PRAGMA table_info("{selected_table}")')
                columns = [row['name'] for row in cursor.fetchall()]

        total_pages = max(1, (total_rows + per_page - 1) // per_page) if selected_table else 1
        if page > total_pages:
            page = total_pages
    finally:
        conn.close()

    return render_template(
        'Admin/database_browser.html',
        table_names=table_names,
        selected_table=selected_table,
        columns=columns,
        rows=rows,
        page=page,
        per_page=per_page,
        total_rows=total_rows,
        total_pages=total_pages,
    )


@admin_bp.route('/admin/toggle_user/<int:user_id>', methods=['POST'])
@admin_required
def admin_toggle_user(user_id):
    if user_id == session.get('user_id'):
        return jsonify({'success': False, 'error': 'Нельзя деактивировать себя'})

    conn = sqlite3.connect(_db_path)
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT is_active FROM users WHERE id = ?', (user_id,))
        row = cursor.fetchone()
        if row:
            new_status = 0 if row[0] else 1
            cursor.execute('UPDATE users SET is_active = ? WHERE id = ?', (new_status, user_id))
            conn.commit()
            log_activity(_db_path, session['user_id'], 'toggle_user',
                        f'Пользователь {user_id} {"деактивирован" if new_status == 0 else "активирован"}')
            return jsonify({'success': True})
        return jsonify({'success': False, 'error': 'Пользователь не найден'})
    except Exception as e:
        conn.rollback()
        return jsonify({'success': False, 'error': str(e)})
    finally:
        conn.close()


@admin_bp.route('/admin/change_role/<int:user_id>', methods=['POST'])
@admin_required
def admin_change_role(user_id):
    if user_id == session.get('user_id'):
        return jsonify({'success': False, 'error': 'Нельзя менять роль самому себе'})

    payload = request.get_json(silent=True) or {}
    new_role = str(payload.get('role', '')).strip().lower()
    if new_role not in _ALLOWED_ROLE_NAMES:
        return jsonify({'success': False, 'error': 'Недопустимая роль'})

    conn = sqlite3.connect(_db_path)
    cursor = conn.cursor()
    try:
        cursor.execute('''
            SELECT u.id, u.username, r.name
            FROM users u
            LEFT JOIN roles r ON r.id = u.role_id
            WHERE u.id = ?
        ''', (user_id,))
        user_row = cursor.fetchone()
        if not user_row:
            return jsonify({'success': False, 'error': 'Пользователь не найден'})

        current_role = user_row[2]
        if current_role == new_role:
            return jsonify({'success': True, 'message': 'Роль не изменилась'})

        if current_role == 'admin' and new_role != 'admin' and _count_admins(cursor) <= 1:
            return jsonify({'success': False, 'error': 'Нельзя понизить последнего администратора'})

        new_role_id = _get_role_id(cursor, new_role)
        if new_role_id is None:
            return jsonify({'success': False, 'error': 'Роль отсутствует в базе'})

        cursor.execute('UPDATE users SET role_id = ? WHERE id = ?', (new_role_id, user_id))
        conn.commit()

        log_activity(
            _db_path,
            session['user_id'],
            'admin_change_role',
            f'Роль пользователя {user_row[1]} изменена: {current_role} -> {new_role}',
        )
        return jsonify({'success': True, 'message': 'Роль пользователя обновлена'})
    except Exception as e:
        conn.rollback()
        return jsonify({'success': False, 'error': str(e)})
    finally:
        conn.close()


@admin_bp.route('/admin/delete_user/<int:user_id>', methods=['POST'])
@admin_required
def admin_delete_user(user_id):
    if user_id == session.get('user_id'):
        return jsonify({'success': False, 'error': 'Нельзя удалить собственный аккаунт'})

    conn = sqlite3.connect(_db_path)
    cursor = conn.cursor()
    try:
        cursor.execute('''
            SELECT u.id, u.username, u.email, r.name
            FROM users u
            LEFT JOIN roles r ON r.id = u.role_id
            WHERE u.id = ?
        ''', (user_id,))
        user_row = cursor.fetchone()
        if not user_row:
            return jsonify({'success': False, 'error': 'Пользователь не найден'})

        if user_row[3] == 'admin' and _count_admins(cursor) <= 1:
            return jsonify({'success': False, 'error': 'Нельзя удалить последнего администратора'})

        # Безопасная очистка зависимостей (даже если FK CASCADE не активирован)
        _delete_user_dependencies(cursor, user_id)
        cursor.execute('DELETE FROM users WHERE id = ?', (user_id,))
        conn.commit()

        log_activity(
            _db_path,
            session['user_id'],
            'admin_delete_user',
            f'Удален пользователь {user_row[1]} ({user_row[2]})',
        )
        return jsonify({'success': True, 'message': 'Пользователь удален'})
    except Exception as e:
        conn.rollback()
        return jsonify({'success': False, 'error': str(e)})
    finally:
        conn.close()


@admin_bp.route('/admin/bulk_users_action', methods=['POST'])
@admin_required
def admin_bulk_users_action():
    payload = request.get_json(silent=True) or {}
    action = str(payload.get('action', '')).strip()
    user_ids = payload.get('user_ids', [])

    if not isinstance(user_ids, list) or not user_ids:
        return jsonify({'success': False, 'error': 'Список пользователей пуст'})

    try:
        user_ids = [int(uid) for uid in user_ids]
    except Exception:
        return jsonify({'success': False, 'error': 'Некорректные идентификаторы пользователей'})

    # Самого себя нельзя изменять в bulk-операциях
    current_user_id = int(session.get('user_id', 0))
    user_ids = [uid for uid in user_ids if uid != current_user_id]
    if not user_ids:
        return jsonify({'success': False, 'error': 'Недоступно для выбранных пользователей'})

    conn = sqlite3.connect(_db_path)
    cursor = conn.cursor()
    try:
        placeholders = ','.join(['?'] * len(user_ids))
        cursor.execute(f'''
            SELECT u.id, u.username, u.email, r.name
            FROM users u
            LEFT JOIN roles r ON r.id = u.role_id
            WHERE u.id IN ({placeholders})
        ''', tuple(user_ids))
        selected_users = cursor.fetchall()

        if not selected_users:
            return jsonify({'success': False, 'error': 'Пользователи не найдены'})

        selected_admin_ids = [row[0] for row in selected_users if row[3] == 'admin']

        if action == 'change_role':
            new_role = str(payload.get('role', '')).strip().lower()
            if new_role not in _ALLOWED_ROLE_NAMES:
                return jsonify({'success': False, 'error': 'Недопустимая роль'})

            if new_role != 'admin' and selected_admin_ids:
                admins_total = _count_admins(cursor)
                if admins_total - len(selected_admin_ids) <= 0:
                    return jsonify({'success': False, 'error': 'Нельзя понизить последнего администратора'})

            role_id = _get_role_id(cursor, new_role)
            if role_id is None:
                return jsonify({'success': False, 'error': 'Роль отсутствует в базе'})

            cursor.execute(
                f'UPDATE users SET role_id = ? WHERE id IN ({placeholders})',
                (role_id, *user_ids),
            )
            conn.commit()

            log_activity(
                _db_path,
                current_user_id,
                'admin_bulk_change_role',
                f'Массовая смена роли на {new_role}: {len(user_ids)} пользователей',
            )
            return jsonify({'success': True, 'message': f'Роль "{new_role}" назначена для {len(user_ids)} пользователей'})

        if action == 'delete':
            admins_total = _count_admins(cursor)
            if selected_admin_ids and admins_total - len(selected_admin_ids) <= 0:
                return jsonify({'success': False, 'error': 'Нельзя удалить последнего администратора'})

            deleted_count = 0
            for uid in user_ids:
                _delete_user_dependencies(cursor, uid)
                cursor.execute('DELETE FROM users WHERE id = ?', (uid,))
                deleted_count += cursor.rowcount
            conn.commit()

            log_activity(
                _db_path,
                current_user_id,
                'admin_bulk_delete_user',
                f'Массовое удаление пользователей: {deleted_count}',
            )
            return jsonify({'success': True, 'message': f'Удалено пользователей: {deleted_count}'})

        return jsonify({'success': False, 'error': 'Неизвестное действие'})
    except Exception as e:
        conn.rollback()
        return jsonify({'success': False, 'error': str(e)})
    finally:
        conn.close()


@admin_bp.route('/admin/clear_cache', methods=['POST'])
@admin_required
def admin_clear_cache():
    try:
        conn = sqlite3.connect(_db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM user_sessions')
        conn.commit()
        conn.close()
        log_activity(_db_path, session['user_id'], 'clear_cache', 'Очистка сессий')
        return jsonify({'success': True, 'message': 'Сессии очищены'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@admin_bp.route('/ml/saved_models')
@login_required
def user_ml_saved_models():
    from app.auth.auth import get_user_ml_models
    user_id = session.get('user_id')
    models = get_user_ml_models(_db_path, user_id)
    return render_template('Admin/ml_saved_models.html',
                          segment='ML Модели',
                          saved_models=__import__('pandas').DataFrame(models) if models else __import__('pandas').DataFrame())


@admin_bp.route('/admin/reset_user_password/<int:user_id>', methods=['POST'])
@admin_required
def admin_reset_user_password(user_id):
    """Админ сбрасывает пароль пользователя."""
    if user_id == session.get('user_id'):
        return jsonify({'success': False, 'error': 'Нельзя менять пароль себе'})

    data = request.get_json() or {}
    new_password = data.get('new_password', '').strip()

    if not new_password or len(new_password) < 6:
        return jsonify({'success': False, 'error': 'Пароль должен быть минимум 6 символов'})

    conn = sqlite3.connect(_db_path)
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT username, email FROM users WHERE id = ?', (user_id,))
        row = cursor.fetchone()
        if not row:
            return jsonify({'success': False, 'error': 'Пользователь не найден'})

        new_hash = hash_password(new_password)
        cursor.execute('UPDATE users SET password_hash = ? WHERE id = ?', (new_hash, user_id))
        conn.commit()
        log_activity(_db_path, session['user_id'], 'admin_reset_password',
                    f'Админ сбросил пароль для {row[0]} ({row[1]})')
        return jsonify({'success': True, 'message': 'Пароль изменён'})
    except Exception as e:
        conn.rollback()
        return jsonify({'success': False, 'error': str(e)})
    finally:
        conn.close()
