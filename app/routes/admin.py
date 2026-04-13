# app/routes/admin.py — Админ-панель
import sqlite3
from flask import Blueprint, render_template, request, jsonify, session
from app.auth.auth import login_required, admin_required, get_activity_logs, get_user_stats, log_activity, hash_password
from app.models.database import query_db

admin_bp = Blueprint('admin', __name__)

_db_path = 'pellets_data.db'

def set_db_path(path):
    global _db_path
    _db_path = path


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
