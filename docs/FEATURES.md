# Инвентаризация функционала и маршрутов (FEATURES.md)

Документ сформирован в рамках **Фазы B (шаг B1)** плана развития Pellets_Analyzer.
Содержит реестр всех зарегистрированных маршрутов приложения, методов, требуемых уровней доступа, шаблонов и статуса работоспособности.

---

## Сводная статистика

- **Всего маршрутов (правил url_map):** 56
- **Модулей (Blueprints):** 7 (`auth`, `main`, `compare`, `economics`, `graphs`, `ml`, `admin`)
- **Базовый уровень безопасности:** Все закрытые роуты защищены сессионной авторизацией и проверкой ролей.

---

## Модуль `auth` (10 маршрутов)

| URL | Метод | Эндпоинт | Доступ | Шаблон | Описание | Статус |
|---|---|---|---|---|---|:---:|
| `/profile/change-password` | POST | `auth.change_password` | Авторизован (login_required) | - | Смена пароля. | ✅ |
| `/forgot-password` | GET, POST | `auth.forgot_password` | Публичный | `auth/forgot_password.html` | Запрос сброса пароля. | ✅ |
| `/login` | GET, POST | `auth.login` | Публичный | `auth/login.html` | Вход в систему. | ✅ |
| `/logout` | GET | `auth.logout` | Публичный | - | Выход из системы. | ✅ |
| `/profile` | GET | `auth.profile` | Авторизован (login_required) | `Admin/profile.html` | Страница профиля пользователя. | ✅ |
| `/profile/update` | POST | `auth.profile_update` | Авторизован (login_required) | - | Обновление профиля. | ✅ |
| `/register` | GET, POST | `auth.register` | Публичный | `auth/register.html` | Регистрация нового пользователя. | ✅ |
| `/resend-verification` | GET, POST | `auth.resend_verification` | Публичный | `auth/resend_verification.html` | Повторная отправка email подтверждения. | ✅ |
| `/reset-password/<token>` | GET, POST | `auth.reset_password_route` | Публичный | `auth/reset_password.html` | Сброс пароля по токену. | ✅ |
| `/verify/<token>` | GET | `auth.verify_email` | Публичный | `auth/email_verified.html` | Подтверждение email. | ✅ |

## Модуль `main` (15 маршрутов)

| URL | Метод | Эндпоинт | Доступ | Шаблон | Описание | Статус |
|---|---|---|---|---|---|:---:|
| `/add_data` | POST | `main.add_data` | Авторизован (login_required) | - | - | ✅ |
| `/ai_analysis` | GET | `main.ai_analysis` | Авторизован (login_required) | `ai_analysis.html` | - | ✅ |
| `/ai_ml_system_status` | GET | `main.ai_ml_system_status` | Авторизован (login_required) | - | - | ✅ |
| `/clear_search` | POST | `main.clear_search` | Авторизован (login_required) | - | - | ✅ |
| `/dashboard` | GET | `main.dashboard` | Авторизован (login_required) | `index.html` | - | ✅ |
| `/economics` | GET | `main.economics` | Авторизован (login_required) | `economics.html` | - | ✅ |
| `/global_search` | GET | `main.global_search` | Авторизован (login_required) | - | - | ✅ |
| `/<path:path>.map` | GET | `main.ignore_map_files` | Публичный | - | - | ✅ |
| `/` | GET | `main.index` | Публичный | `landing.html` | Лендинг-визитка — доступна без авторизации. | ✅ |
| `/landing` | GET | `main.index` | Публичный | `landing.html` | Лендинг-визитка — доступна без авторизации. | ✅ |
| `/load_file` | POST | `main.load_file` | Авторизован (login_required) | - | - | ✅ |
| `/ai_ml_analysis` | POST | `main.perform_ai_ml_analysis` | Авторизован (login_required) | - | - | ✅ |
| `/search` | POST | `main.search` | Авторизован (login_required) | - | - | ✅ |
| `/tables` | GET | `main.tables` | Авторизован (login_required) | `Admin/tables.html` | - | ✅ |
| `/` | POST | `main.upload_file` | Авторизован (login_required) | - | - | ✅ |

## Модуль `compare` (2 маршрутов)

| URL | Метод | Эндпоинт | Доступ | Шаблон | Описание | Статус |
|---|---|---|---|---|---|:---:|
| `/compare` | GET | `compare.compare` | Авторизован (login_required) | `compare.html` | - | ✅ |
| `/compare` | POST | `compare.compare_data` | Авторизован (login_required) | - | - | ✅ |

## Модуль `economics` (7 маршрутов)

| URL | Метод | Эндпоинт | Доступ | Шаблон | Описание | Статус |
|---|---|---|---|---|---|:---:|
| `/api/calculate_economics` | POST | `economics.calculate_economics` | Авторизован (login_required) | - | - | ✅ |
| `/api/compare_economics` | POST | `economics.compare_economics` | Авторизован (login_required) | - | - | ✅ |
| `/api/download_economics` | GET | `economics.download_economics` | Авторизован (login_required) | - | - | ✅ |
| `/api/export_economics_compare_excel` | POST | `economics.export_economics_compare_excel` | Авторизован (login_required) | - | - | ✅ |
| `/api/export_economics_compare_pdf` | POST | `economics.export_economics_compare_pdf` | Авторизован (login_required) | - | - | ✅ |
| `/api/get_components_economics` | GET | `economics.get_components_economics` | Авторизован (login_required) | - | - | ✅ |
| `/api/upload_economics` | POST | `economics.upload_economics` | Авторизован (login_required) | - | - | ✅ |

## Модуль `graphs` (1 маршрутов)

| URL | Метод | Эндпоинт | Доступ | Шаблон | Описание | Статус |
|---|---|---|---|---|---|:---:|
| `/create_graph` | GET, POST | `graphs.create_graph` | Авторизован (login_required) | `create_graph.html` | - | ✅ |

## Модуль `ml` (12 маршрутов)

| URL | Метод | Эндпоинт | Доступ | Шаблон | Описание | Статус |
|---|---|---|---|---|---|:---:|
| `/ai_ml_history` | GET | `ml.ai_ml_history` | Авторизован (login_required) | - | - | ✅ |
| `/ai_ml_recommendations` | GET | `ml.ai_ml_recommendations` | Авторизован (login_required) | - | - | ✅ |
| `/api/predict_composition` | POST | `ml.api_predict_composition` | Авторизован (login_required) | - | - | ✅ |
| `/ml_augment_database` | POST | `ml.ml_augment_database` | Авторизован (login_required) | - | - | ✅ |
| `/ml_auto_retrain` | POST | `ml.ml_auto_retrain` | Авторизован (login_required) | - | - | ✅ |
| `/ml_dashboard` | GET | `ml.ml_dashboard` | Авторизован (login_required) | `ml_dashboard.html` | - | ✅ |
| `/ml_load_excel` | POST | `ml.ml_load_excel` | Авторизован (login_required) | - | Загружает данные из Excel файла в БД. | ✅ |
| `/ml_optimize` | POST | `ml.ml_optimize` | Авторизован (login_required) | - | - | ✅ |
| `/ml_predict` | POST | `ml.ml_predict` | Авторизован (login_required) | - | - | ✅ |
| `/ml_system_status` | GET | `ml.ml_system_status` | Авторизован (login_required) | - | - | ✅ |
| `/ml_system_train` | POST | `ml.ml_system_train` | Авторизован (login_required) | - | - | ✅ |
| `/ml_train_status` | GET | `ml.ml_train_status` | Авторизован (login_required) | - | Проверка статуса обучения. | ✅ |

## Модуль `admin` (9 маршрутов)

| URL | Метод | Эндпоинт | Доступ | Шаблон | Описание | Статус |
|---|---|---|---|---|---|:---:|
| `/admin/bulk_users_action` | POST | `admin.admin_bulk_users_action` | Администратор (admin_required) | - | - | ✅ |
| `/admin/change_role/<int:user_id>` | POST | `admin.admin_change_role` | Администратор (admin_required) | - | - | ✅ |
| `/admin/clear_cache` | POST | `admin.admin_clear_cache` | Администратор (admin_required) | - | - | ✅ |
| `/admin/dashboard` | GET | `admin.admin_dashboard` | Администратор (admin_required) | `Admin/admin_dashboard.html` | - | ✅ |
| `/admin/database` | GET | `admin.admin_database` | Администратор (admin_required) | `Admin/database_browser.html` | - | ✅ |
| `/admin/delete_user/<int:user_id>` | POST | `admin.admin_delete_user` | Администратор (admin_required) | - | - | ✅ |
| `/admin/reset_user_password/<int:user_id>` | POST | `admin.admin_reset_user_password` | Администратор (admin_required) | - | Админ сбрасывает пароль пользователя. | ✅ |
| `/admin/toggle_user/<int:user_id>` | POST | `admin.admin_toggle_user` | Администратор (admin_required) | - | - | ✅ |
| `/ml/saved_models` | GET | `admin.user_ml_saved_models` | Авторизован (login_required) | `Admin/ml_saved_models.html` | - | ✅ |
