# app/auth/routes.py — Роуты аутентификации
from flask import Blueprint, render_template, request, session, flash, redirect, url_for
from app.auth.auth import (
    create_user, verify_user, authenticate_user,
    get_user_by_id, update_user_profile, change_password as auth_change_password,
    request_password_reset, reset_password,
    get_activity_logs, get_user_stats, get_user_uploads, get_user_ml_models,
    log_activity, log_upload,
    login_required, admin_required,
    send_verification_email, send_reset_email, resend_verification_email,
)

auth_bp = Blueprint('auth', __name__)

# Путь к БД — будет установлен при инициализации
_db_path = 'pellets_data.db'


def set_db_path(path):
    global _db_path
    _db_path = path


def _get_auth_template(template_name):
    """Выбирает шаблон: AstroWind v2 при UI_V2=1 (или ?v2=1), иначе классический."""
    from flask import current_app
    if current_app.config.get('UI_V2', False) or request.args.get('v2') == '1':
        return f'auth/v2/{template_name}'
    return f'auth/{template_name}'


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """Вход в систему."""
    from app.utils.security import brute_force
    if request.method == 'POST':
        client_ip = request.remote_addr or '127.0.0.1'
        email_or_username = request.form.get('email_or_username', '').strip()
        password = request.form.get('password', '')
        remember = request.form.get('remember') == 'on'

        identifier = f"{client_ip}:{email_or_username.lower()}"
        if brute_force.is_locked(identifier) or brute_force.is_locked(client_ip):
            flash('Слишком много неудачных попыток входа. Пожалуйста, подождите 15 минут.', 'danger')
            return render_template(_get_auth_template('login.html')), 429

        result = authenticate_user(_db_path, email_or_username, password)

        if result['success']:
            brute_force.clear_attempts(identifier)
            brute_force.clear_attempts(client_ip)
            user = result['user']
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['email'] = user['email']
            session['role_name'] = user['role_name']
            session['full_name'] = user.get('full_name', '')
            session.permanent = remember

            log_activity(_db_path, user['id'], 'login', f'Вход: {user["email"]}')
            flash(f'Добро пожаловать, {user["username"]}!', 'success')
            return redirect(url_for('main.dashboard'))
        else:
            locked_ident = brute_force.record_attempt(identifier)
            locked_ip = brute_force.record_attempt(client_ip)
            if locked_ident or locked_ip:
                flash('Слишком много неудачных попыток входа. Пожалуйста, подождите 15 минут.', 'danger')
                return render_template(_get_auth_template('login.html')), 429
            flash(result['error'], 'danger')

    return render_template(_get_auth_template('login.html'))


@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    """Регистрация нового пользователя."""
    from app.utils.security import rate_limiter
    if request.method == 'POST':
        client_ip = request.remote_addr or '127.0.0.1'
        if not rate_limiter.is_allowed(f"register:{client_ip}", max_requests=10, window=60):
            flash('Слишком много попыток регистрации. Попробуйте позже.', 'danger')
            return render_template(_get_auth_template('register.html')), 429

        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        full_name = request.form.get('full_name', '').strip()
        company = request.form.get('company', '').strip()

        if password != confirm_password:
            flash('Пароли не совпадают!', 'danger')
            return render_template(_get_auth_template('register.html'))

        if len(password) < 6:
            flash('Пароль должен быть минимум 6 символов!', 'danger')
            return render_template(_get_auth_template('register.html'))

        result = create_user(_db_path, username, email, password, full_name=full_name or username)

        if result['success']:
            from app.auth.auth import is_smtp_configured
            if not is_smtp_configured():
                # SMTP не настроен — авто-верификация пользователя
                import sqlite3
                conn = sqlite3.connect(_db_path)
                conn.execute('UPDATE users SET is_verified = 1, verification_token = NULL WHERE id = ?', (result['user_id'],))
                conn.commit()
                conn.close()
                flash('Регистрация успешна! Войдите в систему.', 'success')
            else:
                base_url = request.url_root.rstrip('/')
                email_sent = send_verification_email(email, result['verification_token'], base_url, username=username)

                if email_sent:
                    flash('Регистрация успешна! Проверьте email для подтверждения.', 'success')
                else:
                    flash('Регистрация успешна! Но возникла проблема с отправкой email. Вы можете <a href="' + url_for('auth.resend_verification') + '" class="alert-link">запросить письмо повторно</a>.', 'warning')
            return redirect(url_for('auth.login'))
        else:
            flash(result['error'], 'danger')

    return render_template(_get_auth_template('register.html'))


@auth_bp.route('/verify/<token>')
def verify_email(token):
    """Подтверждение email."""
    result = verify_user(_db_path, token)
    if result['success']:
        return render_template('auth/email_verified.html')
    else:
        flash(result['error'], 'danger')
        return redirect(url_for('auth.login'))


@auth_bp.route('/resend-verification', methods=['GET', 'POST'])
def resend_verification():
    """Повторная отправка email подтверждения."""
    from app.utils.security import rate_limiter
    if request.method == 'POST':
        client_ip = request.remote_addr or '127.0.0.1'
        if not rate_limiter.is_allowed(f"resend_verif:{client_ip}", max_requests=10, window=60):
            flash('Слишком много запросов. Попробуйте позже.', 'danger')
            return render_template('auth/resend_verification.html'), 429

        email = request.form.get('email', '').strip()
        if not email:
            flash('Введите email!', 'danger')
            return render_template('auth/resend_verification.html')

        base_url = request.url_root.rstrip('/')
        result = resend_verification_email(email, _db_path, base_url)

        if result['success']:
            flash(result['message'], 'success')
            return redirect(url_for('auth.login'))
        else:
            flash(result['error'], 'danger')

    return render_template('auth/resend_verification.html')


@auth_bp.route('/logout')
def logout():
    """Выход из системы."""
    user_id = session.get('user_id')
    if user_id:
        log_activity(_db_path, user_id, 'logout', 'Выход из системы')
    session.clear()
    flash('Вы вышли из системы.', 'info')
    return redirect(url_for('auth.login'))


@auth_bp.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    """Запрос сброса пароля."""
    from app.utils.security import rate_limiter
    if request.method == 'POST':
        client_ip = request.remote_addr or '127.0.0.1'
        if not rate_limiter.is_allowed(f"forgot:{client_ip}", max_requests=10, window=60):
            flash('Слишком много запросов на сброс пароля. Попробуйте позже.', 'danger')
            return render_template(_get_auth_template('forgot_password.html')), 429

        email = request.form.get('email', '').strip()
        result = request_password_reset(_db_path, email)

        if result['success'] and 'reset_token' in result:
            base_url = request.url_root.rstrip('/')
            send_reset_email(email, result['reset_token'], base_url)

        flash(result.get('message', 'Если email существует, мы отправили ссылку для сброса.'), 'info')
        return redirect(url_for('auth.login'))

    return render_template(_get_auth_template('forgot_password.html'))


@auth_bp.route('/reset-password/<token>', methods=['GET', 'POST'])
def reset_password_route(token):
    """Сброс пароля по токену."""
    if request.method == 'POST':
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')

        if password != confirm_password:
            flash('Пароли не совпадают!', 'danger')
            return render_template(_get_auth_template('reset_password.html'), token=token)

        result = reset_password(_db_path, token, password)
        if result['success']:
            flash('Пароль успешно изменён! Войдите в систему.', 'success')
            return redirect(url_for('auth.login'))
        else:
            flash(result['error'], 'danger')

    return render_template(_get_auth_template('reset_password.html'), token=token)


@auth_bp.route('/profile')
@login_required
def profile():
    """Страница профиля пользователя."""
    user_id = session.get('user_id')
    user = get_user_by_id(_db_path, user_id)
    if not user:
        session.clear()
        return redirect(url_for('auth.login'))

    uploads = get_user_uploads(_db_path, user_id)
    ml_models = get_user_ml_models(_db_path, user_id)

    admin_stats = None
    activity_logs = []
    if user['role_name'] == 'admin':
        admin_stats = get_user_stats(_db_path)
        activity_logs = get_activity_logs(_db_path, limit=50)

    return render_template('Admin/profile.html',
                          user=user,
                          uploads=uploads,
                          ml_models=ml_models,
                          admin_stats=admin_stats,
                          activity_logs=activity_logs)


@auth_bp.route('/profile/update', methods=['POST'])
@login_required
def profile_update():
    """Обновление профиля."""
    user_id = session.get('user_id')
    full_name = request.form.get('full_name', '').strip()
    company = request.form.get('company', '').strip()
    phone = request.form.get('phone', '').strip()

    result = update_user_profile(_db_path, user_id,
                                 full_name=full_name or None,
                                 company=company or None,
                                 phone=phone or None)
    if result['success']:
        flash('Профиль обновлён!', 'success')
    else:
        flash(result['error'], 'danger')
    return redirect(url_for('auth.profile'))


@auth_bp.route('/profile/change-password', methods=['POST'])
@login_required
def change_password():
    """Смена пароля."""
    user_id = session.get('user_id')
    old_password = request.form.get('old_password', '')
    new_password = request.form.get('new_password', '')
    confirm_password = request.form.get('confirm_password', '')

    if new_password != confirm_password:
        flash('Новые пароли не совпадают!', 'danger')
        return redirect(url_for('auth.profile'))

    result = auth_change_password(_db_path, user_id, old_password, new_password)
    if result['success']:
        flash('Пароль успешно изменён!', 'success')
    else:
        flash(result['error'], 'danger')
    return redirect(url_for('auth.profile'))
