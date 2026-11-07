# app/routes/ml.py — ML Dashboard, обучение, оптимизация
import json
import os
import sqlite3
import threading
import time
import logging
from collections import deque
from typing import Optional
from flask import Blueprint, current_app, has_app_context, render_template, request, jsonify, session
from app.auth.auth import login_required
from app.database.database import query_db
from app.services.ml_v2.switch import model_version, predict_one

ml_bp = Blueprint('ml', __name__)
logger = logging.getLogger(__name__)

_db_path = os.path.join('data', 'pellets_data.db')

# Хранилище статуса обучения (синхронизируется с таблицей system_settings в SQLite)
_training_status = {
    'is_training': False,
    'progress': 0,
    'message': '',
    'result': None,
    'error': None,
    'started_at': None,
    'finished_at': None,
}
_ml_warmup_started = False
_ml_warmup_completed = False
_ml_warmup_error = None
_ml_warmup_lock = threading.Lock()
_endpoint_timings = {
    '/ml_optimize': deque(maxlen=200),
    '/ml_system_train': deque(maxlen=200),
}


def _get_db_path() -> str:
    if has_app_context():
        return current_app.config.get('DATABASE_PATH') or current_app.config.get('DATABASE') or _db_path
    return _db_path


def set_db_path(path):
    global _db_path
    _db_path = path
    if has_app_context():
        current_app.config['DATABASE_PATH'] = path


def _update_training_status(db_path: Optional[str] = None, **kwargs):
    """Обновляет статус обучения в памяти и сохраняет в SQLite system_settings."""
    global _training_status
    _training_status.update(kwargs)
    path = db_path or _get_db_path()
    if path:
        try:
            from app.database.database import set_system_setting
            set_system_setting(path, "ml_training_status", json.dumps(_training_status, default=str), updated_by="ml_worker")
        except Exception as exc:
            logger.debug("Could not persist ml_training_status to SQLite: %s", exc)


def _read_training_status(db_path: Optional[str] = None) -> dict:
    """Считывает статус обучения из SQLite system_settings с фолбэком на локальный кэш."""
    global _training_status
    path = db_path or _get_db_path()
    if path:
        try:
            from app.database.database import get_system_setting
            raw = get_system_setting(path, "ml_training_status")
            if raw:
                data = json.loads(raw)
                if isinstance(data, dict):
                    _training_status.update(data)
        except Exception as exc:
            logger.debug("Could not read ml_training_status from SQLite: %s", exc)
    return _training_status


def _get_ai_ml_analyzer():
    """Получает синглтон AIMLAnalyzer из app.config вместо создания на каждый запрос."""
    db_path = _get_db_path()
    if has_app_context() and current_app.config.get('AI_ML_ANALYZER'):
        analyzer = current_app.config['AI_ML_ANALYZER']
        if getattr(analyzer, 'db_path', None) == db_path:
            return analyzer
    from app.services.ai_ml_analyzer import AIMLAnalyzer
    analyzer = AIMLAnalyzer(db_path)
    if has_app_context():
        current_app.config['AI_ML_ANALYZER'] = analyzer
    return analyzer


def _percentile(values, q):
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = int(round((q / 100.0) * (len(sorted_values) - 1)))
    return float(sorted_values[idx])


def _record_endpoint_timing(endpoint, elapsed_sec):
    timings = None
    if has_app_context():
        ext_timings = current_app.extensions.setdefault('endpoint_timings', {})
        if endpoint not in ext_timings:
            ext_timings[endpoint] = deque(maxlen=200)
        timings = ext_timings[endpoint]
    else:
        timings = _endpoint_timings.get(endpoint)
    if timings is None:
        return
    elapsed_ms = elapsed_sec * 1000.0
    timings.append(elapsed_ms)
    p50 = _percentile(list(timings), 50)
    p95 = _percentile(list(timings), 95)
    logger.info("%s latency_ms=%.2f p50=%.2f p95=%.2f n=%s", endpoint, elapsed_ms, p50, p95, len(timings))


def _resolve_ml_system():
    from app.services.ml_optimizer import get_ml_system
    db_path = _get_db_path()
    started_at = time.perf_counter()
    ml_system = get_ml_system()
    # Переопределяем путь к БД (без полной перезагрузки)
    if ml_system.db_path != db_path:
        ml_system.db_path = db_path
        ml_system.reload_data()
    if has_app_context():
        current_app.extensions['ml_system'] = ml_system
    elapsed_ms = (time.perf_counter() - started_at) * 1000.0
    logger.info("_get_ml_system resolve_ms=%.2f db=%s", elapsed_ms, db_path)
    return ml_system


def _warmup_ml_system_worker():
    global _ml_warmup_completed, _ml_warmup_error
    try:
        _resolve_ml_system()
        _ml_warmup_completed = True
        _ml_warmup_error = None
        logger.info("ML warmup completed successfully")
    except Exception as exc:
        _ml_warmup_error = str(exc)
        logger.warning("ML warmup failed: %s", exc)


def _ensure_ml_warmup():
    """Синхронная или ленивая инициализация без запуска фоновых потоков в воркере."""
    if model_version() == "v2":
        return
    global _ml_warmup_started, _ml_warmup_completed
    with _ml_warmup_lock:
        if has_app_context():
            warmup = current_app.extensions.setdefault('ml_warmup', {'started': False, 'completed': False, 'error': None})
            if warmup['started']:
                return
            warmup['started'] = True
        else:
            if _ml_warmup_started:
                return
            _ml_warmup_started = True
    # Выполняется ленивая инициализация без фонового потока
    try:
        _warmup_ml_system_worker()
    except Exception as exc:
        logger.warning("ML warmup failed: %s", exc)


def _get_ml_system():
    return _resolve_ml_system()


def _legacy_predict(ml_system, composition, target_property):
    return ml_system.predictor.predict(composition, target_property)


def _contract_fields(contract):
    if not contract:
        return {}
    return {
        "std": contract.get("std"),
        "lo90": contract.get("lo90"),
        "hi90": contract.get("hi90"),
        "in_domain": contract.get("in_domain"),
        "informative": contract.get("informative"),
        "note": contract.get("note") or "",
        "model_name": contract.get("model_name"),
        "model_version": "v2",
    }


def _v2_training_refused_payload():
    """v2 не переобучается на БД продукта и не пишет туда прогнозы."""
    from app.services.ml_v2.predictor import v2_system_status

    status = v2_system_status()
    message = (
        "Режим v2 использует зафиксированный артефакт. "
        "Переобучение на данных продукта не выполняется, прогнозы в БД не пишутся. "
        "Метрики — nested CV из manifest."
    )
    _update_training_status(
        is_training=False,
        progress=100,
        message=message,
        error=None,
        result={
            "success": True,
            "message": message,
            "model_version": "v2",
            "metrics": status["model_metrics"],
        },
    )
    return message


def _train_async_task(target_properties, algorithm, input_features, user_id=None):
    """Фоновая задача обучения."""
    db_path = _get_db_path()
    started_at = time.perf_counter()
    try:
        _update_training_status(
            db_path=db_path,
            is_training=True,
            progress=10,
            message='Загрузка данных...',
            result=None,
            error=None,
            started_at=time.time(),
        )

        ml_system = _get_ml_system()

        _update_training_status(
            db_path=db_path,
            progress=30,
            message='Подготовка данных...',
        )
        time.sleep(0.5)  # UI update

        _update_training_status(
            db_path=db_path,
            progress=50,
            message='Обучение моделей...',
        )

        result = ml_system.train_models(target_properties, algorithm, selected_features=input_features)

        _update_training_status(
            db_path=db_path,
            progress=90,
            message='Сохранение результатов...',
        )

        if result.get('success'):
            # Сохраняем модели в user_ml_models для отображения
            if user_id:
                from app.auth.auth import save_ml_model
                status = ml_system.get_ml_system_status()
                for prop, metrics in status.get('model_metrics', {}).items():
                    training_metrics = metrics.get('training_metrics', {})
                    composition_text = ', '.join(status.get('available_components', [])[:5])
                    save_ml_model(
                        db_path, user_id,
                        model_name=f'Model_{prop}',
                        target_property=prop,
                        algorithm=metrics.get('algorithm_used', algorithm),
                        r2_score=training_metrics.get('r2_score', 0),
                        mae=training_metrics.get('mae', 0),
                        cv_r2=training_metrics.get('cv_r2', 0),
                        training_data_size=status.get('training_data_size', 0),
                        composition_text=composition_text
                    )

            _update_training_status(
                db_path=db_path,
                result=result,
                progress=100,
                message='Обучение завершено!',
            )
        else:
            err = result.get('error', 'Обучение не удалось')
            _update_training_status(
                db_path=db_path,
                error=err,
                message=err,
            )

    except Exception as e:
        _update_training_status(
            db_path=db_path,
            error=str(e),
            message=f'Ошибка: {str(e)}',
        )
    finally:
        _update_training_status(
            db_path=db_path,
            is_training=False,
            finished_at=time.time(),
        )
        logger.info("ml_train_async completed in %.2fms", (time.perf_counter() - started_at) * 1000.0)


@ml_bp.route('/ml_dashboard')
@login_required
def ml_dashboard():
    # В режиме v2 legacy-прогрев не нужен: он мог бы обучить старую модель и записать её на диск.
    if model_version() != "v2":
        _ensure_ml_warmup()
    return render_template(
        'ml_dashboard.html',
        segment='ML Анализ'
    )


def _require_admin_role():
    """Проверяет роль администратора для защищённых ML операций."""
    if session.get('role_name') != 'admin':
        return jsonify({
            'success': False,
            'error': 'Доступ запрещён. Операция доступна только администраторам.'
        }), 403
    return None


@ml_bp.route('/ml_system_train', methods=['POST'])
@login_required
def ml_system_train():
    started_at = time.perf_counter()
    admin_gate = _require_admin_role()
    if admin_gate:
        return admin_gate

    status = _read_training_status()
    if status.get('is_training'):
        return jsonify({'success': False, 'error': 'Обучение уже запущено. Дождитесь завершения.'})

    if model_version() == "v2":
        message = _v2_training_refused_payload()
        return jsonify({"success": True, "message": message, "task_id": "ml_train", "model_version": "v2"})

    # Legacy: запуск обучения в веб-процессе отключен согласно политике безопасности
    _record_endpoint_timing('/ml_system_train', time.perf_counter() - started_at)
    return jsonify({
        'success': False,
        'error': 'Обучение моделей в веб-воркере отключено согласно политике безопасности (ML_TRAINING_POLICY). '
                 'Запустите переобучение через CLI: python scripts/train_legacy.py',
        'cli_command': 'python scripts/train_legacy.py',
    })


@ml_bp.route('/ml_train_status')
@login_required
def ml_train_status():
    """Проверка статуса обучения."""
    status = _read_training_status()
    return jsonify({
        'is_training': status.get('is_training', False),
        'progress': status.get('progress', 0),
        'message': status.get('message', ''),
        'result': status.get('result'),
        'error': status.get('error'),
    })


@ml_bp.route('/ml_augment_database', methods=['POST'])
@login_required
def ml_augment_database():
    admin_gate = _require_admin_role()
    if admin_gate:
        return admin_gate

    try:
        ml_system = _get_ml_system()
        try:
            variations_count = int(request.form.get('variations_count', 3))
            confidence_interval = float(request.form.get('confidence_interval', 5.0))
        except ValueError:
            return jsonify({'success': False, 'error': 'Некорректные параметры для аугментации'})

        result = ml_system.augment_database(variations_count=variations_count, confidence_interval=confidence_interval)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': f'Ошибка аугментации: {str(e)}'})


@ml_bp.route('/ml_optimize', methods=['POST'])
@login_required
def ml_optimize():
    started_at = time.perf_counter()

    try:
        target_property = request.form.get('target_property')
        maximize = request.form.get('maximize', 'true').lower() == 'true'

        if not target_property:
            return jsonify({'success': False, 'error': 'Не указано целевое свойство'})

        constraints = {}
        for key in request.form:
            if key.startswith('min_'):
                comp = key.replace('min_', '')
                min_val_str = request.form.get(key, '0')
                max_val_str = request.form.get(f'max_{comp}', '100')
                try:
                    min_val = float(min_val_str) if min_val_str.strip() else 0.0
                    max_val = float(max_val_str) if max_val_str.strip() else 100.0
                except ValueError:
                    return jsonify({'success': False, 'error': f'Некорректное значение для компонента {comp}'})
                constraints[comp] = (min_val, max_val)

        if model_version() == "v2":
            from app.services.ml_v2.optimizer import optimize_composition_v2
            result = optimize_composition_v2(
                target_property=target_property,
                maximize=maximize,
                constraints=constraints,
                db_path=_get_db_path(),
            )
            return jsonify(result)

        ml_system = _get_ml_system()
        result = ml_system.optimize_composition(target_property=target_property, maximize=maximize, constraints=constraints)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': f'Ошибка оптимизации: {str(e)}'})
    finally:
        _record_endpoint_timing('/ml_optimize', time.perf_counter() - started_at)


@ml_bp.route('/ml_predict', methods=['POST'])
@login_required
def ml_predict():
    try:
        data = request.get_json(silent=True) or {}
        if not isinstance(data, dict):
            return jsonify({'success': False, 'error': 'Некорректный формат запроса'})
        composition = data.get('composition', {})
        target_property = data.get('target_property')

        if not composition or not target_property:
            return jsonify({'success': False, 'error': 'Не указаны состав или свойство'})

        if model_version() == "v2":
            result = predict_one(composition, target_property, legacy_predict=lambda *_args: None)
        else:
            ml_system = _get_ml_system()
            result = predict_one(
                composition,
                target_property,
                legacy_predict=lambda comp, prop: _legacy_predict(ml_system, comp, prop),
            )
        prediction = result["prediction"]
        if prediction is None:
            return jsonify({'success': False, 'error': 'Не удалось сделать предсказание', 'model_version': result['model_version']})
        body = {
            'success': True,
            'prediction': float(prediction),
            'property': target_property,
            'model_version': result['model_version'],
        }
        if result.get('contract'):
            body['contract'] = result['contract']
        return jsonify(body)
    except Exception as e:
        return jsonify({'success': False, 'error': f'Ошибка предсказания: {str(e)}'})


@ml_bp.route('/ml_system_status')
@login_required
def ml_system_status():
    if model_version() == "v2":
        try:
            from app.services.ml_v2.predictor import v2_system_status
            return jsonify({'success': True, 'system_status': v2_system_status()})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e), 'model_version': 'v2'})
    try:
        ml_system = _get_ml_system()
        # Не перезагружаем данные каждый раз — используем кэш
        status = ml_system.get_ml_system_status()
        return jsonify({'success': True, 'system_status': status})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@ml_bp.route('/api/predict_composition', methods=['POST'])
@login_required
def api_predict_composition():
    try:
        data = request.get_json(silent=True) or {}
        if not isinstance(data, dict):
            return jsonify({'success': False, 'error': 'Некорректный формат запроса'})
        composition = data.get('composition', {})
        if not isinstance(composition, dict) or not composition:
            return jsonify({'success': False, 'error': 'Не указан состав для предсказания'})
        if model_version() == "v2":
            from app.services.ml_v2.lab_import import require_lab
            from app.services.ml_v2.predictor import get_predictor

            predictor = get_predictor()
            pellet_ml = require_lab()
            contracts = predictor.predict_all(composition)
            results = {}
            for prop, contract in contracts.items():
                results[prop] = {
                    'value': round(float(contract['value']), 3),
                    'display_name': pellet_ml.properties.display_name(prop),
                    **_contract_fields(contract),
                }
            return jsonify({'success': True, 'model_version': 'v2', 'results': results})

        ml_system = _get_ml_system()

        if not ml_system.predictor.is_trained:
            return jsonify({'success': False, 'error': 'Модели не обучены'})

        results = {}
        target_props = ml_system.predictor.main_target_properties
        for prop in target_props:
            outcome = predict_one(
                composition,
                prop,
                legacy_predict=lambda comp, target: _legacy_predict(ml_system, comp, target),
            )
            pred = outcome['prediction']
            if pred is not None:
                display_name = ml_system.predictor.target_properties_mapping.get(prop, prop)
                results[prop] = {'value': round(float(pred), 3), 'display_name': display_name}

        return jsonify({'success': True, 'model_version': 'legacy', 'results': results})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@ml_bp.route('/ml_auto_retrain', methods=['POST'])
@login_required
def ml_auto_retrain():
    admin_gate = _require_admin_role()
    if admin_gate:
        return admin_gate

    if model_version() == "v2":
        return jsonify({
            'success': False,
            'model_version': 'v2',
            'error': 'v2 не переобучается на данных продукта и не записывает прогнозы в БД.',
        })
    return jsonify({
        'success': False,
        'model_version': 'legacy',
        'error': 'Автоматическое переобучение в веб-воркере отключено согласно политике безопасности (ML_TRAINING_POLICY). '
                 'Используйте CLI: python scripts/train_legacy.py',
        'cli_command': 'python scripts/train_legacy.py',
    })


@ml_bp.route('/ml_load_excel', methods=['POST'])
@login_required
def ml_load_excel():
    """Загружает данные из Excel файла в БД."""
    admin_gate = _require_admin_role()
    if admin_gate:
        return admin_gate

    try:
        ml_system = _get_ml_system()
        result = ml_system.load_excel_data()
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': f'Ошибка загрузки: {str(e)}'})


@ml_bp.route('/ai_ml_recommendations')
@login_required
def ai_ml_recommendations():
    try:
        analyzer = _get_ai_ml_analyzer()
        recommendations = analyzer.get_system_recommendations()
        return jsonify({'success': True, 'recommendations': recommendations['recommendations'], 'total_count': recommendations['total_count']})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@ml_bp.route('/ai_ml_history')
@login_required
def ai_ml_history():
    try:
        analyzer = _get_ai_ml_analyzer()
        history = analyzer.get_analysis_history(limit=5)
        return jsonify({'success': True, 'history': history})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@ml_bp.route('/api/ml/save_recipe', methods=['POST'])
@login_required
def api_save_recipe():
    """Сохраняет рецепт в серверную базу данных (для операторов и лаборантов)."""
    try:
        data = request.get_json(silent=True) or {}
        composition = data.get('composition', {})
        if not composition:
            return jsonify({'success': False, 'error': 'Состав не указан'}), 400

        user_id = session.get('user_id')
        recipe_name = (data.get('recipe_name') or '').strip() or f"Рецепт {time.strftime('%Y-%m-%d %H:%M')}"
        predictions = data.get('predictions', {})
        lab_actuals = data.get('lab_actuals', {})
        residuals = data.get('residuals', {})
        notes = data.get('notes', '')

        db_path = _get_db_path()
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute('''
            INSERT INTO saved_recipes (user_id, recipe_name, composition_json, predictions_json, lab_actuals_json, residuals_json, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_id,
            recipe_name,
            json.dumps(composition, ensure_ascii=False),
            json.dumps(predictions, ensure_ascii=False),
            json.dumps(lab_actuals, ensure_ascii=False),
            json.dumps(residuals, ensure_ascii=False),
            notes
        ))
        conn.commit()
        recipe_id = cur.lastrowid
        conn.close()

        return jsonify({
            'success': True,
            'message': 'Рецепт успешно сохранён в серверной базе данных',
            'recipe_id': recipe_id
        })
    except Exception as e:
        logger.warning(f"Ошибка сохранения рецепта: {e}")
        return jsonify({'success': False, 'error': f'Ошибка сохранения: {str(e)}'}), 500


@ml_bp.route('/api/ml/saved_recipes', methods=['GET'])
@login_required
def api_get_saved_recipes():
    """Возвращает историю сохранённых рецептур."""
    try:
        user_id = session.get('user_id')
        role_name = session.get('role_name', 'operator')
        db_path = _get_db_path()
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        if role_name == 'admin':
            cur.execute('''
                SELECT id, user_id, recipe_name, composition_json, predictions_json, lab_actuals_json, residuals_json, notes, created_at
                FROM saved_recipes
                ORDER BY created_at DESC
                LIMIT 50
            ''')
        else:
            cur.execute('''
                SELECT id, user_id, recipe_name, composition_json, predictions_json, lab_actuals_json, residuals_json, notes, created_at
                FROM saved_recipes
                WHERE user_id = ? OR user_id IS NULL
                ORDER BY created_at DESC
                LIMIT 50
            ''', (user_id,))

        rows = []
        for r in cur.fetchall():
            rows.append({
                'id': r['id'],
                'recipe_name': r['recipe_name'],
                'composition': json.loads(r['composition_json'] or '{}'),
                'predictions': json.loads(r['predictions_json'] or '{}'),
                'lab_actuals': json.loads(r['lab_actuals_json'] or '{}'),
                'residuals': json.loads(r['residuals_json'] or '{}'),
                'created_at': r['created_at'],
                'notes': r['notes']
            })
        conn.close()
        return jsonify({'success': True, 'recipes': rows})
    except Exception as e:
        logger.warning(f"Ошибка загрузки рецептов: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
