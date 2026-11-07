# app/routes/admin.py — Админ-панель (Фаза G: G1–G6)
import json
import logging
import os
import sqlite3
from datetime import datetime

from flask import Blueprint, jsonify, render_template, request, session, url_for
from werkzeug.utils import secure_filename

from app.auth.auth import (
    admin_required,
    get_activity_logs,
    get_user_stats,
    hash_password,
    is_smtp_configured,
    log_activity,
    login_required,
    request_password_reset,
)
from app.database.database import (
    clear_shadow_logs,
    get_recipes_paged,
    get_shadow_logs,
    get_system_settings,
    query_db,
    set_system_setting,
    toggle_recipe_questionable,
    update_recipe_source,
)
from app.services.ml_v2 import switch

logger = logging.getLogger(__name__)

admin_bp = Blueprint('admin', __name__)

_db_path = 'pellets_data.db'
_ALLOWED_ROLE_NAMES = {'user', 'moderator', 'admin'}


def _get_model_status() -> dict:
    """Собирает статус ML-модели и подмодуля lab (G2, G3)."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    manifest_path = os.path.join(base_dir, 'models', 'v2', 'manifest.json')
    artifact_path = os.path.join(base_dir, 'models', 'v2', 'pellet_models.joblib')

    manifest_data = {}
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest_data = json.load(f)
        except Exception as exc:
            logger.warning("Не удалось прочитать manifest.json: %s", exc)

    artifact_exists = os.path.exists(artifact_path)
    artifact_size_kb = round(os.path.getsize(artifact_path) / 1024, 1) if artifact_exists else 0
    mtime = datetime.fromtimestamp(os.path.getmtime(artifact_path)).strftime('%Y-%m-%d %H:%M') if artifact_exists else 'N/A'

    return {
        'model_version': switch.model_version(),
        'shadow_enabled': switch.shadow_enabled(),
        'manifest': manifest_data,
        'artifact_exists': artifact_exists,
        'artifact_size_kb': artifact_size_kb,
        'artifact_mtime': mtime,
        'submodule_tag': 'lab-v0.2.0',
        'api_version': manifest_data.get('api_version', '1.0.0'),
        'metrics': manifest_data.get('metrics', {}),
        'update_command': 'powershell .\\scripts\\update_lab.ps1',
        'safety_notice': 'Сборка и обновление моделей подмодуля lab требуют оффлайн-валидации (21 тест) и запрещены из веб-интерфейса из соображений безопасности.',
    }


def _get_settings_dict() -> dict:
    """Возвращает системные настройки приложения (G6)."""
    settings = get_system_settings(_db_path)
    return {
        'max_upload_size_mb': int(settings.get('max_upload_size_mb', 10)),
        'session_lifetime_days': int(settings.get('session_lifetime_days', 7)),
        'smtp_configured': is_smtp_configured(),
        'mail_server': os.environ.get('MAIL_SERVER', 'smtp.yandex.ru' if is_smtp_configured() else 'Не настроен'),
        'csrf_protection': True,
        'brute_force_protection': True,
        'model_version': settings.get('MODEL_VERSION', switch.model_version()),
        'shadow_enabled': settings.get('ML_SHADOW_V2', '1' if switch.shadow_enabled() else '0') == '1',
    }


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

    model_status = _get_model_status()
    shadow_logs = get_shadow_logs(_db_path, limit=50)
    recipes_sample, recipes_total = get_recipes_paged(_db_path, page=1, per_page=15)
    system_settings = _get_settings_dict()

    return render_template(
        'Admin/admin_dashboard.html',
        stats=stats,
        all_users=all_users,
        activity_logs=activity_logs,
        model_status=model_status,
        shadow_logs=shadow_logs,
        recipes_sample=recipes_sample,
        recipes_total=recipes_total,
        system_settings=system_settings,
    )


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

        if selected_table and selected_table in table_names:
            cursor.execute(f'SELECT COUNT(*) AS cnt FROM "{selected_table}"')  # nosec: B608
            total_rows = int(cursor.fetchone()['cnt'])

            offset = (page - 1) * per_page
            cursor.execute(f'SELECT * FROM "{selected_table}" LIMIT ? OFFSET ?', (per_page, offset))  # nosec: B608
            rows = [dict(row) for row in cursor.fetchall()]
            columns = list(rows[0].keys()) if rows else []
            if not columns:
                cursor.execute(f'PRAGMA table_info("{selected_table}")')  # nosec: B608
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
        ''', tuple(user_ids))  # nosec: B608
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
                f'UPDATE users SET role_id = ? WHERE id IN ({placeholders})',  # nosec: B608
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


# ============================================================
# МАРШРУТЫ ФАЗЫ G: G1, G2, G4, G6
# ============================================================

@admin_bp.route('/admin/generate_reset_link/<int:user_id>', methods=['POST'])
@admin_required
def admin_generate_reset_link(user_id):
    """Генерирует одноразовую ссылку сброса пароля без раскрытия паролей (G1)."""
    conn = sqlite3.connect(_db_path)
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT username, email FROM users WHERE id = ?', (user_id,))
        row = cursor.fetchone()
        if not row:
            return jsonify({'success': False, 'error': 'Пользователь не найден'}), 404
        username, email = row[0], row[1]
    finally:
        conn.close()

    if not email:
        return jsonify({'success': False, 'error': 'У пользователя не указан email'}), 400

    res = request_password_reset(_db_path, email)
    if not res.get('success') or 'reset_token' not in res:
        return jsonify({'success': False, 'error': res.get('error', 'Не удалось создать токен сброса')}), 500

    token = res['reset_token']
    try:
        reset_url = url_for('auth.reset_password_route', token=token, _external=True)
    except Exception:
        reset_url = f"/reset-password/{token}"

    log_activity(
        _db_path,
        session.get('user_id'),
        'admin_generate_reset_link',
        f'Сгенерирована ссылка сброса пароля для {username} ({email})'
    )

    return jsonify({
        'success': True,
        'reset_url': reset_url,
        'username': username,
        'email': email,
        'expires_in': '1 час'
    })


@admin_bp.route('/admin/model_status')
@admin_required
def admin_model_status():
    """Возвращает текущий статус модели и лаборатории в JSON (G2, G3)."""
    return jsonify({'success': True, 'data': _get_model_status()})


@admin_bp.route('/admin/switch_model', methods=['POST'])
@admin_required
def admin_switch_model():
    """Переключение версии модели (legacy/v2) и теневого режима (G2)."""
    payload = request.get_json(silent=True) or {}
    version = str(payload.get('version', '')).strip().lower()
    shadow = bool(payload.get('shadow', False))

    if version not in ('legacy', 'v2'):
        return jsonify({'success': False, 'error': 'Недопустимая версия модели. Разрешены: legacy, v2'}), 400

    if version == 'v2':
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        artifact_path = os.path.join(base_dir, 'models', 'v2', 'pellet_models.joblib')
        if not os.path.exists(artifact_path):
            return jsonify({
                'success': False,
                'error': 'Артефакт models/v2/pellet_models.joblib не найден. Соберите модель перед переключением.'
            }), 400

    username = session.get('username', 'admin')
    user_id = session.get('user_id')

    set_system_setting(_db_path, 'MODEL_VERSION', version, username)
    set_system_setting(_db_path, 'ML_SHADOW_V2', '1' if shadow else '0', username)
    switch.set_runtime_override(version, shadow)

    log_activity(
        _db_path,
        user_id,
        'switch_model_version',
        f'Модель переключена: версия={version}, тень={shadow}'
    )

    return jsonify({
        'success': True,
        'message': f'Модель успешно переключена на {version} (тень: {"включена" if shadow else "выключена"})',
        'version': version,
        'shadow': shadow
    })


@admin_bp.route('/admin/shadow_logs')
@admin_required
def admin_shadow_logs():
    """Возвращает записи теневого лога с фильтром по свойству (G2)."""
    prop = request.args.get('property', '').strip()
    limit = min(max(int(request.args.get('limit', 50)), 1), 500)
    logs = get_shadow_logs(_db_path, limit=limit, property_name=prop if prop else None)
    return jsonify({'success': True, 'logs': logs, 'count': len(logs)})


@admin_bp.route('/admin/clear_shadow_logs', methods=['POST'])
@admin_required
def admin_clear_shadow_logs():
    """Очищает теневой лог ML v2 (G2)."""
    deleted = clear_shadow_logs(_db_path)
    log_activity(_db_path, session.get('user_id'), 'clear_shadow_logs', f'Очищен теневой лог ({deleted} записей)')
    return jsonify({'success': True, 'deleted': deleted})


@admin_bp.route('/admin/recipes')
@admin_required
def admin_recipes():
    """Возвращает список рецептур с пагинацией и фильтрами (G4)."""
    page = max(int(request.args.get('page', 1)), 1)
    per_page = min(max(int(request.args.get('per_page', 25)), 5), 100)
    source = request.args.get('source', '').strip()
    questionable_raw = request.args.get('questionable')
    questionable = int(questionable_raw) if questionable_raw in ('0', '1') else None
    search = request.args.get('search', '').strip()

    recipes, total = get_recipes_paged(
        _db_path,
        page=page,
        per_page=per_page,
        source_filter=source,
        questionable_filter=questionable,
        search=search
    )
    pages = max(1, (total + per_page - 1) // per_page)

    return jsonify({
        'success': True,
        'recipes': recipes,
        'total': total,
        'page': page,
        'pages': pages
    })


@admin_bp.route('/admin/recipes/toggle_questionable', methods=['POST'])
@admin_required
def admin_toggle_recipe_questionable():
    """Помечает опыт как сомнительный или подтверждённый (G4)."""
    payload = request.get_json(silent=True) or {}
    composition = str(payload.get('composition', '')).strip()
    if not composition:
        return jsonify({'success': False, 'error': 'Не указана рецептура'}), 400

    new_val = toggle_recipe_questionable(_db_path, composition)
    if new_val is None:
        return jsonify({'success': False, 'error': 'Рецепт не найден в базе'}), 404

    status_str = 'сомнительный' if new_val == 1 else 'подтверждённый'
    log_activity(
        _db_path,
        session.get('user_id'),
        'recipe_toggle_questionable',
        f'Рецепт «{composition[:40]}...»: статус изменён на {status_str}'
    )

    return jsonify({
        'success': True,
        'composition': composition,
        'is_questionable': new_val,
        'status_text': status_str
    })


@admin_bp.route('/admin/recipes/update_source', methods=['POST'])
@admin_required
def admin_update_recipe_source():
    """Обновляет источник рецепта (laboratory, literature, ml_prediction) (G4)."""
    payload = request.get_json(silent=True) or {}
    composition = str(payload.get('composition', '')).strip()
    source = str(payload.get('source', '')).strip().lower()

    if not composition:
        return jsonify({'success': False, 'error': 'Не указана рецептура'}), 400

    if source not in ('laboratory', 'literature', 'ml_prediction'):
        return jsonify({'success': False, 'error': 'Недопустимый источник. Разрешены: laboratory, literature, ml_prediction'}), 400

    success = update_recipe_source(_db_path, composition, source)
    if not success:
        return jsonify({'success': False, 'error': 'Не удалось обновить источник рецепта'}), 404

    log_activity(
        _db_path,
        session.get('user_id'),
        'recipe_update_source',
        f'Рецепт «{composition[:40]}...»: источник установлен в {source}'
    )

    return jsonify({
        'success': True,
        'composition': composition,
        'source': source
    })


@admin_bp.route('/admin/recipes/import_preview', methods=['POST'])
@admin_required
def admin_recipes_import_preview():
    """Предпросмотр импорта данных перед сохранением (G4)."""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'Файл не прикреплён'}), 400

    file = request.files['file']
    if not file or not file.filename:
        return jsonify({'success': False, 'error': 'Имя файла пустое'}), 400

    filename = secure_filename(file.filename)
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ('.csv', '.xlsx', '.xls'):
        return jsonify({'success': False, 'error': 'Разрешены только форматы .csv, .xlsx, .xls'}), 400

    try:
        import pandas as pd
        if ext == '.csv':
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)

        if df.empty:
            return jsonify({'success': False, 'error': 'Таблица пуста'}), 400

        df.columns = [str(c).strip().lower() for c in df.columns]
        if 'composition' not in df.columns:
            return jsonify({
                'success': False,
                'error': 'Отсутствует обязательная колонка "composition" (состав рецептуры)'
            }), 400

        total_rows = len(df)
        columns = list(df.columns)
        preview_rows = df.head(5).fillna('').to_dict(orient='records')

        conn = sqlite3.connect(_db_path)
        cur = conn.cursor()
        cur.execute("SELECT composition FROM measured_parameters")
        existing_compositions = {row[0] for row in cur.fetchall()}
        conn.close()

        duplicates_count = sum(1 for c in df['composition'] if str(c).strip() in existing_compositions)

        return jsonify({
            'success': True,
            'filename': filename,
            'total_rows': total_rows,
            'columns': columns,
            'preview_rows': preview_rows,
            'duplicates_count': duplicates_count
        })

    except Exception as e:
        logger.error(f"Ошибка предпросмотра импорта: {e}")
        return jsonify({'success': False, 'error': f'Ошибка чтения файла: {str(e)}'}), 400


@admin_bp.route('/admin/recipes/import_confirm', methods=['POST'])
@admin_required
def admin_recipes_import_confirm():
    """Подтверждение импорта данных с флагом источника (G4)."""
    payload = request.get_json(silent=True) or {}
    records = payload.get('records', [])
    source = str(payload.get('source', 'laboratory')).strip().lower()

    if source not in ('laboratory', 'literature', 'ml_prediction'):
        return jsonify({'success': False, 'error': 'Недопустимый источник'}), 400

    if not isinstance(records, list) or not records:
        return jsonify({'success': False, 'error': 'Список записей пуст'}), 400

    try:
        conn = sqlite3.connect(_db_path)
        cur = conn.cursor()

        inserted_count = 0
        for rec in records:
            comp = str(rec.get('composition', '')).strip()
            if not comp:
                continue

            density = rec.get('density')
            kf = rec.get('kf')
            kt = rec.get('kt')
            mass_loss = rec.get('mass_loss')
            tign = rec.get('tign')
            tb = rec.get('tb')
            co = rec.get('co')
            co2 = rec.get('co2')
            is_quest = 1 if rec.get('is_questionable') else 0

            cur.execute('''
                INSERT INTO measured_parameters (
                    composition, density, kf, kt, mass_loss, tign, tb, co, co2, source, is_questionable
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(composition) DO UPDATE SET
                    density = COALESCE(excluded.density, measured_parameters.density),
                    kf = COALESCE(excluded.kf, measured_parameters.kf),
                    kt = COALESCE(excluded.kt, measured_parameters.kt),
                    mass_loss = COALESCE(excluded.mass_loss, measured_parameters.mass_loss),
                    tign = COALESCE(excluded.tign, measured_parameters.tign),
                    tb = COALESCE(excluded.tb, measured_parameters.tb),
                    co = COALESCE(excluded.co, measured_parameters.co),
                    co2 = COALESCE(excluded.co2, measured_parameters.co2),
                    source = excluded.source,
                    is_questionable = excluded.is_questionable
            ''', (comp, density, kf, kt, mass_loss, tign, tb, co, co2, source, is_quest))
            inserted_count += 1

        conn.commit()
        conn.close()

        log_activity(
            _db_path,
            session.get('user_id'),
            'recipe_import',
            f'Импортировано {inserted_count} рецептов (источник: {source})'
        )

        return jsonify({'success': True, 'inserted_count': inserted_count, 'source': source})

    except Exception as e:
        logger.error(f"Ошибка сохранения импорта: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@admin_bp.route('/admin/settings/update', methods=['POST'])
@admin_required
def admin_update_settings():
    """Сохранение системных настроек (G6)."""
    payload = request.get_json(silent=True) or request.form.to_dict()
    username = session.get('username', 'admin')

    max_upload_size_mb = payload.get('max_upload_size_mb')
    if max_upload_size_mb is not None:
        try:
            val = min(max(int(max_upload_size_mb), 1), 100)
            set_system_setting(_db_path, 'max_upload_size_mb', str(val), username)
        except ValueError:
            pass

    session_lifetime_days = payload.get('session_lifetime_days')
    if session_lifetime_days is not None:
        try:
            val = min(max(int(session_lifetime_days), 1), 30)
            set_system_setting(_db_path, 'session_lifetime_days', str(val), username)
        except ValueError:
            pass

    log_activity(
        _db_path,
        session.get('user_id'),
        'update_settings',
        f'Обновлены системные настройки пользователем {username}'
    )

    return jsonify({'success': True, 'message': 'Настройки успешно сохранены'})

