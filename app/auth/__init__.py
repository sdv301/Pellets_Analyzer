# app/auth/__init__.py
from app.auth.auth import (
    init_auth_tables, init_mail, mail,
    create_user, verify_user, authenticate_user,
    get_user_by_id, get_user_by_email, update_user_profile, change_password,
    request_password_reset, reset_password,
    get_activity_logs, get_user_stats, get_user_uploads, get_user_ml_models,
    log_activity, log_upload,
    login_required, admin_required, moderator_required,
    send_verification_email, send_reset_email, resend_verification_email,
)

__all__ = [
    'init_auth_tables', 'init_mail', 'mail',
    'create_user', 'verify_user', 'authenticate_user',
    'get_user_by_id', 'get_user_by_email', 'update_user_profile', 'change_password',
    'request_password_reset', 'reset_password',
    'get_activity_logs', 'get_user_stats', 'get_user_uploads', 'get_user_ml_models',
    'log_activity', 'log_upload',
    'login_required', 'admin_required', 'moderator_required',
    'send_verification_email', 'send_reset_email', 'resend_verification_email',
]
