# app/routes/ml.py — ML Dashboard, обучение, оптимизация
import threading
import time
from flask import Blueprint, render_template, request, jsonify
from app.auth.auth import login_required
from app.models.database import query_db

ml_bp = Blueprint('ml', __name__)

_db_path = 'pellets_data.db'

# Глобальное хранилище статуса обучения
_training_status = {
    'is_training': False,
    'progress': 0,
    'message': '',
    'result': None,
    'error': None,
    'started_at': None,
    'finished_at': None,
}

def set_db_path(path):
    global _db_path
    _db_path = path


def _get_ml_system():
    from app.services.ml_optimizer import get_ml_system
    ml_system = get_ml_system()
    ml_system.reload_data()
    return ml_system


def _train_async_task(target_properties, algorithm, input_features):
    """Фоновая задача обучения."""
    global _training_status
    try:
        _training_status['is_training'] = True
        _training_status['progress'] = 10
        _training_status['message'] = 'Загрузка данных...'
        _training_status['result'] = None
        _training_status['error'] = None

        ml_system = _get_ml_system()

        _training_status['progress'] = 30
        _training_status['message'] = 'Подготовка данных...'
        time.sleep(0.5)  # UI update

        _training_status['progress'] = 50
        _training_status['message'] = 'Обучение моделей...'

        result = ml_system.train_models(target_properties, algorithm, selected_features=input_features)

        _training_status['progress'] = 90
        _training_status['message'] = 'Сохранение результатов...'

        if result.get('success'):
            _training_status['result'] = result
            _training_status['progress'] = 100
            _training_status['message'] = 'Обучение завершено!'
        else:
            _training_status['error'] = result.get('error', 'Обучение не удалось')
            _training_status['message'] = _training_status['error']

    except Exception as e:
        _training_status['error'] = str(e)
        _training_status['message'] = f'Ошибка: {str(e)}'
    finally:
        _training_status['is_training'] = False
        _training_status['finished_at'] = time.time()


@ml_bp.route('/admin/ml/dashboard')
@login_required
def ml_dashboard():
    from app.routes.main import get_uploaded_files
    uploaded_files = get_uploaded_files()
    measured_data = query_db(_db_path, "measured_parameters")

    ml_system = _get_ml_system()
    status = ml_system.get_ml_system_status()

    history_list = []
    try:
        from app.models.database import get_ml_optimizations
        optimizations_history = get_ml_optimizations(_db_path, limit=10)
        if hasattr(optimizations_history, 'empty') and not optimizations_history.empty:
            history_list = optimizations_history.to_dict('records')
            for item in history_list:
                if 'optimal_composition' in item and isinstance(item['optimal_composition'], dict):
                    item['optimal_composition_text'] = ", ".join([f"{v}% {k}" for k, v in item['optimal_composition'].items()])
                else:
                    item['optimal_composition_text'] = "Нет данных"
    except Exception:
        pass

    return render_template(
        'ml_dashboard.html',
        segment='ML Анализ',
        uploaded_files=uploaded_files,
        compositions=measured_data['composition'].tolist() if not measured_data.empty and 'composition' in measured_data.columns else [],
        ml_status=status,
        optimization_history=history_list
    )


@ml_bp.route('/ml_system_train', methods=['POST'])
@login_required
def ml_system_train():
    global _training_status
    if _training_status['is_training']:
        return jsonify({'success': False, 'error': 'Обучение уже запущено. Дождитесь завершения.'})

    try:
        target_properties = request.form.getlist('target_properties[]')
        if not target_properties:
            from app.services.ml_optimizer import PelletPropertyPredictor
            target_properties = list(PelletPropertyPredictor().target_properties_mapping.keys())

        algorithm = request.form.get('algorithm', 'random_forest').lower()
        input_features = request.form.getlist('input_features[]')
        if not input_features:
            input_features = None

        # Запуск обучения в фоне
        thread = threading.Thread(
            target=_train_async_task,
            args=(target_properties, algorithm, input_features)
        )
        thread.daemon = True
        thread.start()

        return jsonify({
            'success': True,
            'message': 'Обучение запущено в фоновом режиме. Отслеживайте прогресс.',
            'task_id': 'ml_train'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': f'Ошибка запуска обучения: {str(e)}'})


@ml_bp.route('/ml_train_status')
@login_required
def ml_train_status():
    """Проверка статуса обучения."""
    global _training_status
    return jsonify({
        'is_training': _training_status['is_training'],
        'progress': _training_status['progress'],
        'message': _training_status['message'],
        'result': _training_status['result'],
        'error': _training_status['error'],
    })


@ml_bp.route('/ml_augment_database', methods=['POST'])
@login_required
def ml_augment_database():
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
    try:
        ml_system = _get_ml_system()
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

        result = ml_system.optimize_composition(target_property=target_property, maximize=maximize, constraints=constraints)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': f'Ошибка оптимизации: {str(e)}'})


@ml_bp.route('/ml_predict', methods=['POST'])
@login_required
def ml_predict():
    try:
        ml_system = _get_ml_system()
        data = request.get_json()
        composition = data.get('composition', {})
        target_property = data.get('target_property')

        if not composition or not target_property:
            return jsonify({'success': False, 'error': 'Не указаны состав или свойство'})

        prediction = ml_system.predictor.predict(composition, target_property)
        if prediction is not None:
            return jsonify({'success': True, 'prediction': prediction, 'property': target_property})
        else:
            return jsonify({'success': False, 'error': 'Не удалось сделать предсказание'})
    except Exception as e:
        return jsonify({'success': False, 'error': f'Ошибка предсказания: {str(e)}'})


@ml_bp.route('/ml_system_status')
@login_required
def ml_system_status():
    try:
        ml_system = _get_ml_system()
        ml_system.reload_data()
        status = ml_system.get_ml_system_status()
        return jsonify({'success': True, 'system_status': status})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@ml_bp.route('/api/predict_composition', methods=['POST'])
@login_required
def api_predict_composition():
    try:
        data = request.json
        composition = data.get('composition', {})
        ml_system = _get_ml_system()

        if not ml_system.predictor.is_trained:
            return jsonify({'success': False, 'error': 'Модели не обучены'})

        results = {}
        target_props = ml_system.predictor.main_target_properties
        for prop in target_props:
            pred = ml_system.predictor.predict(composition, prop)
            if pred is not None:
                display_name = ml_system.predictor.target_properties_mapping.get(prop, prop)
                results[prop] = {'value': round(float(pred), 3), 'display_name': display_name}

        return jsonify({'success': True, 'results': results})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@ml_bp.route('/ml_auto_retrain', methods=['POST'])
@login_required
def ml_auto_retrain():
    try:
        ml_system = _get_ml_system()
        result = ml_system.retrain_on_new_data()
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@ml_bp.route('/ml_load_excel', methods=['POST'])
@login_required
def ml_load_excel():
    """Загружает данные из Excel файла в БД."""
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
        from app.services.ai_ml_analyzer import AIMLAnalyzer
        analyzer = AIMLAnalyzer(_db_path)
        recommendations = analyzer.get_system_recommendations()
        return jsonify({'success': True, 'recommendations': recommendations['recommendations'], 'total_count': recommendations['total_count']})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@ml_bp.route('/ai_ml_history')
@login_required
def ai_ml_history():
    try:
        from app.services.ai_ml_analyzer import AIMLAnalyzer
        analyzer = AIMLAnalyzer(_db_path)
        history = analyzer.get_analysis_history(limit=5)
        return jsonify({'success': True, 'history': history})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
