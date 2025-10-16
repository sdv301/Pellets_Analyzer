# main.py - основное приложение со всеми роутами
from flask import Flask, render_template, request, jsonify, session, flash
from flask_session import Session
import pandas as pd
import os
from data_processor import process_data_source
from database import query_db, insert_data, init_db
from ai_ml_integration import AIMLAnalyzer
from gui import *
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'Uploads'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = os.path.join(os.path.dirname(__file__), 'sessions')
app.config['SECRET_KEY'] = 'your-secret-key'
Session(app)
ai_ml_analyzer = AIMLAnalyzer()
db_path = 'pellets_data.db'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)
init_db(db_path)  # Инициализация базы данных

@app.route('/')
def index():
    uploaded_files = os.listdir(app.config['UPLOAD_FOLDER']) if os.path.exists(app.config['UPLOAD_FOLDER']) else []
    show_data = session.get('show_data', False)
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 10))
    
    tables = []
    total_rows = []
    
    # ВСЕГДА загружаем данные из базы для отображения
    measured_data = query_db(db_path, "measured_parameters")
    components_data = query_db(db_path, "components")
    
    if not measured_data.empty:
        total_measured = len(measured_data)
        total_rows.append(total_measured)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        df_pag = measured_data.iloc[start_idx:end_idx] if total_measured > 0 else pd.DataFrame()
        tables.append({
            'name': 'Измеренные параметры',
            'data': df_pag.to_html(classes='table table-striped table-sm', index=False) if not df_pag.empty else 'Таблица пуста.'
        })

    if not components_data.empty:
        total_components = len(components_data)
        total_rows.append(total_components)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        df_pag = components_data.iloc[start_idx:end_idx] if total_components > 0 else pd.DataFrame()
        tables.append({
            'name': 'Компоненты',
            'data': df_pag.to_html(classes='table table-striped table-sm', index=False) if not df_pag.empty else 'Таблица пуста.'
        })
    
    # Если есть данные в базе, показываем их
    if not measured_data.empty or not components_data.empty:
        show_data = True
        session['show_data'] = True

    return render_template(
        'index.html',
        segment='Главная',
        uploaded_files=uploaded_files,
        tables=tables,
        show_data=show_data,
        page=page,
        per_page=per_page,
        total_rows=total_rows
    )

@app.route('/', methods=['POST'])
def upload_file():
    uploaded_files = os.listdir(app.config['UPLOAD_FOLDER']) if os.path.exists(app.config['UPLOAD_FOLDER']) else []
    if 'file' not in request.files:
        flash('Файл не предоставлен.', 'danger')
        return jsonify({
            'success': False,
            'message': 'Файл не предоставлен.',
            'uploaded_files': uploaded_files
        })
    file = request.files['file']
    if file.filename == '':
        flash('Файл не выбран.', 'danger')
        return jsonify({
            'success': False,
            'message': 'Файл не выбран.',
            'uploaded_files': uploaded_files
        })
    if file and (file.filename.endswith('.csv') or file.filename.endswith('.xlsx')):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        try:
            messages, components_sheet_name, sheet_data = process_data_source(file_path, db_path)
            
            # Загружаем данные из базы для отображения
            measured_data = query_db(db_path, "measured_parameters")
            components_data = query_db(db_path, "components")
            
            if not sheet_data:
                flash('Файл не содержит данных для отображения.', 'warning')
            else:
                flash(f'Загружено {len(sheet_data)} листов из файла {file.filename}.', 'success')
            
            # Сохраняем в сессию
            session['sheet_data'] = [
                {'name': s['name'], 'data': s['data'].to_json(orient='records', force_ascii=False)} for s in sheet_data
            ]
            session['components_sheet_name'] = components_sheet_name
            session['show_data'] = True
            
            # Сохраняем данные из базы в сессию для немедленного отображения
            session['measured_data'] = measured_data.to_json(orient='records', force_ascii=False)
            session['components_data'] = components_data.to_json(orient='records', force_ascii=False)
            
            flash(f'Данные сохранены в сессию: {len(sheet_data)} листов.', 'info')
            
            # Возвращаем данные для обновления фронтенда
            return jsonify({
                'success': True,
                'message': 'Файл успешно загружен.',
                'uploaded_files': uploaded_files,
                'messages': messages,
                'measured_data': measured_data.head(20).to_html(classes='table table-striped table-sm', index=False) if not measured_data.empty else '',
                'components_data': components_data.head(20).to_html(classes='table table-striped table-sm', index=False) if not components_data.empty else '',
                'total_measured': len(measured_data),
                'total_components': len(components_data),
                'refresh_page': True
            })
        except Exception as e:
            flash(f'Ошибка обработки файла {file.filename}: {str(e)}', 'danger')
            return jsonify({
                'success': False,
                'message': f'Ошибка обработки файла: {str(e)}',
                'uploaded_files': uploaded_files
            })
    flash('Недопустимый формат файла.', 'danger')
    return jsonify({
        'success': False,
        'message': 'Недопустимый формат файла.',
        'uploaded_files': uploaded_files
    })

@app.route('/load_file', methods=['POST'])
def load_file():
    selected_file = request.form.get('selected_file')
    uploaded_files = os.listdir(app.config['UPLOAD_FOLDER']) if os.path.exists(app.config['UPLOAD_FOLDER']) else []
    if not selected_file:
        return jsonify({
            'success': False,
            'message': 'Файл не выбран.',
            'uploaded_files': uploaded_files
        })
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], selected_file)
    if not os.path.exists(file_path):
        return jsonify({
            'success': False,
            'message': f'Файл {selected_file} не найден.',
            'uploaded_files': uploaded_files
        })
    try:
        process_messages, components_sheet_name = process_data_source(file_path, db_path)
        for msg in process_messages:
            category = "danger" if "Error" in msg or "Warning" in msg else "success"
            flash(msg, category)
        
        measured_data = query_db(db_path, "measured_parameters")
        components_data = query_db(db_path, "components")
        graph, _, _ = generate_graph(measured_data)  # Обновленный вызов
        if measured_data.empty and components_data.empty:
            return jsonify({
                'success': False,
                'message': 'Данные обработаны, но таблицы пусты. Проверьте формат файла или названия столбцов.',
                'uploaded_files': uploaded_files
            })
        session['show_data'] = True
        session['data_loaded'] = True
        session['measured_data'] = measured_data.to_json()
        session['components_data'] = components_data.to_json()
        session['graph'] = graph
        session['components_sheet_name'] = components_sheet_name
        return jsonify({
            'success': True,
            'message': f'Данные из файла {selected_file} успешно загружены.',
            'measured_data': measured_data.head(10).to_html(classes='table table-striped table-sm', index=False),
            'components_data': components_data.head(10).to_html(classes='table table-striped table-sm', index=False),
            'graph': graph,
            'uploaded_files': uploaded_files,
            'components_sheet_name': components_sheet_name,
            'total_measured': len(measured_data),
            'total_components': len(components_data)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Ошибка обработки файла {selected_file}: {str(e)}',
            'uploaded_files': uploaded_files
        })

@app.route('/search', methods=['POST'])
def search():
    try:
        search_column = request.form.get('search_column')
        search_operator = request.form.get('search_operator', '=')
        search_value = request.form.get('search_value', '').strip()
        
        print(f"=== ПОИСК: column={search_column}, operator={search_operator}, value={search_value}")
        
        # Базовая проверка
        if not search_column or not search_value:
            return jsonify({
                'success': False,
                'message': 'Заполните все поля поиска'
            })
        
        # Получаем все данные
        all_measured_data = query_db(db_path, "measured_parameters")
        
        if all_measured_data.empty:
            return jsonify({
                'success': False,
                'message': 'Нет данных для поиска. Сначала загрузите файл.'
            })
        
        print(f"=== ВСЕХ ДАННЫХ: {len(all_measured_data)} строк")
        
        # Простая и надежная фильтрация
        filtered_data = simple_filter(all_measured_data, search_column, search_operator, search_value)
        
        print(f"=== НАЙДЕНО: {len(filtered_data)} строк")
        
        if filtered_data.empty:
            return jsonify({
                'success': True,
                'message': 'По вашему запросу ничего не найдено',
                'measured_data': '<div class="alert alert-info">По вашему запросу ничего не найдено</div>',
                'total_measured': 0
            })
        
        # Сохраняем ТОЛЬКО результаты поиска
        session['search_results'] = filtered_data.to_json(orient='records', force_ascii=False)
        session['search_performed'] = True
        session['show_data'] = True
        
        return jsonify({
            'success': True,
            'message': f'Найдено {len(filtered_data)} записей',
            'refresh_page': True  # Перезагружаем страницу чтобы показать результаты
        })
        
    except Exception as e:
        print(f"=== ОШИБКА ПОИСКА: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Ошибка при поиске: {str(e)}'
        })

@app.route('/add_data', methods=['POST'])
def add_data():
    try:
        data = {
            'composition': request.form.get('composition'),
            'density': float(request.form.get('density', '')) if request.form.get('density', '') else None,
            'kf': float(request.form.get('kf', '')) if request.form.get('kf', '') else None,
            'kt': float(request.form.get('kt', '')) if request.form.get('kt', '') else None,
            'h': float(request.form.get('h', '')) if request.form.get('h', '') else None,
            'mass_loss': float(request.form.get('mass_loss', '')) if request.form.get('mass_loss', '') else None,
            'tign': float(request.form.get('tign', '')) if request.form.get('tign', '') else None,
            'tb': float(request.form.get('tb', '')) if request.form.get('tb', '') else None,
            'tau_d1': float(request.form.get('tau_d1', '')) if request.form.get('tau_d1', '') else None,
            'tau_d2': float(request.form.get('tau_d2', '')) if request.form.get('tau_d2', '') else None,
            'tau_b': float(request.form.get('tau_b', '')) if request.form.get('tau_b', '') else None,
            'co2': float(request.form.get('co2', '')) if request.form.get('co2', '') else None,
            'co': float(request.form.get('co', '')) if request.form.get('co', '') else None,
            'so2': float(request.form.get('so2', '')) if request.form.get('so2', '') else None,
            'nox': float(request.form.get('nox', '')) if request.form.get('nox', '') else None,
            'q': float(request.form.get('q', '')) if request.form.get('q', '') else None,
            'ad': float(request.form.get('ad', '')) if request.form.get('ad', '') else None
        }
        df = pd.DataFrame([data])
        insert_data(db_path, "measured_parameters", df)
        measured_data = query_db(db_path, "measured_parameters")
        components_data = query_db(db_path, "components")
        graph, _, _ = generate_graph(measured_data)  # Обновленный вызов
        session['show_data'] = True
        session['data_loaded'] = True
        session['measured_data'] = measured_data.to_json()
        session['components_data'] = components_data.to_json()
        session['graph'] = graph
        return jsonify({
            'success': True,
            'message': 'Данные успешно добавлены.',
            'measured_data': measured_data.head(10).to_html(classes='table table-striped table-sm', index=False),
            'components_data': components_data.head(10).to_html(classes='table table-striped table-sm', index=False),
            'graph': graph,
            'uploaded_files': os.listdir(app.config['UPLOAD_FOLDER']) if os.path.exists(app.config['UPLOAD_FOLDER']) else [],
            'components_sheet_name': session.get('components_sheet_name', 'Таблица компонентов'),
            'total_measured': len(measured_data),
            'total_components': len(components_data)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Ошибка при добавлении данных: {str(e)}',
            'uploaded_files': os.listdir(app.config['UPLOAD_FOLDER']) if os.path.exists(app.config['UPLOAD_FOLDER']) else []
        })

@app.route('/tables')
def tables():
    uploaded_files = os.listdir(app.config['UPLOAD_FOLDER']) if os.path.exists(app.config['UPLOAD_FOLDER']) else []
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 10))
    
    search_performed = session.get('search_performed', False)
    tables = []
    total_rows = []
    
    # Обрабатываем результаты поиска
    if search_performed:
        try:
            measured_data_json = session.get('search_results', '[]')
            measured_data = pd.read_json(measured_data_json) if measured_data_json != '[]' else pd.DataFrame()
            
            if not measured_data.empty:
                total_measured = len(measured_data)
                total_rows.append(total_measured)
                start_idx = (page - 1) * per_page
                end_idx = start_idx + per_page
                df_pag = measured_data.iloc[start_idx:end_idx] if total_measured > 0 else pd.DataFrame()
                
                tables.append({
                    'name': 'Результаты поиска',
                    'data': df_pag.to_html(classes='table table-striped table-sm', index=False) if not df_pag.empty else '<div class="alert alert-info">Ничего не найдено</div>'
                })
        except Exception as e:
            print(f"Ошибка загрузки результатов поиска: {e}")
    
    # Если нет результатов поиска, показываем все данные
    if not tables:
        measured_data = query_db(db_path, "measured_parameters")
        components_data = query_db(db_path, "components")
        
        if not measured_data.empty:
            total_measured = len(measured_data)
            total_rows.append(total_measured)
            start_idx = (page - 1) * per_page
            end_idx = start_idx + per_page
            df_pag = measured_data.iloc[start_idx:end_idx] if total_measured > 0 else pd.DataFrame()
            tables.append({
                'name': 'Измеренные параметры',
                'data': df_pag.to_html(classes='table table-striped table-sm', index=False) if not df_pag.empty else 'Таблица пуста.'
            })

        if not components_data.empty:
            total_components = len(components_data)
            total_rows.append(total_components)
            start_idx = (page - 1) * per_page
            end_idx = start_idx + per_page
            df_pag = components_data.iloc[start_idx:end_idx] if total_components > 0 else pd.DataFrame()
            tables.append({
                'name': 'Компоненты',
                'data': df_pag.to_html(classes='table table-striped table-sm', index=False) if not df_pag.empty else 'Таблица пуста.'
            })
    
    show_data = len(tables) > 0

    return render_template(
        'tables.html',
        segment='Таблицы',
        uploaded_files=uploaded_files,
        tables=tables,
        show_data=show_data,
        page=page,
        per_page=per_page,
        total_rows=total_rows,
        search_performed=search_performed
    )

@app.route('/clear_search', methods=['POST'])
def clear_search():
    session.pop('search_results', None)
    session.pop('search_performed', None)
    return jsonify({
        'success': True,
        'message': 'Поиск очищен, показаны все данные'
    })

@app.route('/compare')
def compare():
    uploaded_files = os.listdir(app.config['UPLOAD_FOLDER']) if os.path.exists(app.config['UPLOAD_FOLDER']) else []
    measured_data = query_db(db_path, "measured_parameters")
    compositions = measured_data['composition'].tolist() if not measured_data.empty else []
    return render_template('compare.html', segment='Сравнительная таблица',uploaded_files=uploaded_files, compositions=compositions)

@app.route('/compare', methods=['POST'])
def compare_data():
    try:
        # Получаем все выбранные составы
        compositions = []
        i = 1
        while True:
            comp = request.form.get(f'comp{i}')
            if comp:
                compositions.append(comp)
                i += 1
            else:
                break
        
        print(f"=== СРАВНЕНИЕ: выбрано {len(compositions)} составов: {compositions}")
        
        if len(compositions) < 2:
            return jsonify({
                'success': False,
                'message': 'Выберите хотя бы два состава для сравнения.'
            })
        
        # Получаем выбранные критерии
        selected_criteria = request.form.getlist('criteria[]')
        print(f"=== ВЫБРАННЫЕ КРИТЕРИИ: {selected_criteria}")
        
        # Получаем данные из базы
        measured_data = query_db(db_path, "measured_parameters")
        
        if measured_data.empty:
            return jsonify({
                'success': False,
                'message': 'В базе данных нет измеренных параметров.'
            })
        
        # Фильтруем данные по выбранным составам
        comparison_data = pd.DataFrame()
        found_compositions = []
        
        for comp in compositions:
            comp_data = measured_data[measured_data['composition'] == comp]
            if not comp_data.empty:
                comparison_data = pd.concat([comparison_data, comp_data])
                found_compositions.append(comp)
        
        if comparison_data.empty:
            return jsonify({
                'success': False,
                'message': 'Выбранные составы не найдены в базе данных.'
            })
        
        # Если не все составы найдены, сообщаем об этом
        if len(found_compositions) != len(compositions):
            missing = set(compositions) - set(found_compositions)
            flash(f'Некоторые составы не найдены: {", ".join(missing)}', 'warning')
        
        # Получаем настройки фильтрации
        show_all = request.form.get('show_all') == 'on'
        show_diff = request.form.get('show_diff') == 'on'
        param_group = request.form.get('paramGroup', 'all')
        
        # Применяем фильтрацию параметров с учетом выбранных критериев
        filtered_data = filter_parameters_with_criteria(
            comparison_data, 
            param_group, 
            show_diff and not show_all,
            selected_criteria
        )
        
        # Если после фильтрации данных нет
        if filtered_data.empty:
            return jsonify({
                'success': True,
                'message': 'После применения фильтров нет данных для отображения.',
                'comparison': '<div class="alert alert-info">Нет данных, соответствующих выбранным фильтрам</div>',
                'compositions': found_compositions,
                'stats': {'compositions_count': 0, 'parameters_count': 0, 'total_rows': 0}
            })
        
        return jsonify({
            'success': True,
            'message': f'Сравнение {len(found_compositions)} составов выполнено успешно.',
            'comparison': filtered_data.to_html(
                classes='table table-striped table-sm comparison-table', 
                index=False, 
                escape=False,
                na_rep='N/A'
            ),
            'compositions': found_compositions,
            'stats': get_comparison_stats(filtered_data)
        })
        
    except Exception as e:
        print(f"=== ОШИБКА СРАВНЕНИЯ: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'message': f'Ошибка при сравнении: {str(e)}'
        })

def filter_parameters_with_criteria(data, param_group, show_diff_only=False, selected_criteria=None):
    """Фильтрует параметры по группам и выбранным критериям"""
    
    # Сохраняем оригинальные названия колонок для внутренней обработки
    original_data = data.copy()
    
    # Русские названия колонок для отображения
    COLUMN_NAMES = {
        'composition': 'Состав',
        'density': 'Плотность, кг/м³',
        'q': 'Теплота сгорания, МДж/кг',
        'ad': 'Зольность, %',
        'kf': 'Ударопрочность, %',
        'kt': 'Устойчивость к нагрузкам, %',
        'h': 'Гигроскопичность, %',
        'mass_loss': 'Потеря массы, %',
        'tign': 'Температура зажигания, °C',
        'tb': 'Температура выгорания, °C',
        'tau_d1': 'Задержка газофазного зажигания, с',
        'tau_d2': 'Задержка гетерогенного зажигания, с',
        'tau_b': 'Время горения, с',
        'co2': 'Концентрация CO₂, %',
        'co': 'Концентрация CO, %',
        'so2': 'Концентрация SO₂, ppm',
        'nox': 'Концентрация NOx, ppm',
        'war': 'Влажность, %',
        'vd': 'Летучие вещества, %',
        'cd': 'Содержание углерода, %',
        'hd': 'Содержание водорода, %',
        'nd': 'Содержание азота, %',
        'sd': 'Содержание серы, %',
        'od': 'Содержание кислорода, %'
    }
    
    # Группы параметров
    param_groups = {
        'thermal': ['q', 'tign', 'tb', 'tau_b', 'tau_d1', 'tau_d2'],
        'mechanical': ['density', 'kf', 'kt', 'h', 'mass_loss'],
        'chemical': ['ad', 'cd', 'hd', 'nd', 'sd', 'od', 'vd', 'war'],
        'emissions': ['co2', 'co', 'so2', 'nox'],
        'combustion': ['mass_loss', 'tau_b', 'tau_d1', 'tau_d2', 'tign', 'tb']
    }
    
    # Определяем какие колонки оставить
    if selected_criteria:
        # Используем выбранные критерии
        selected_params = ['composition'] + selected_criteria
    elif param_group != 'all' and param_group in param_groups:
        # Используем группу параметров
        selected_params = ['composition'] + param_groups[param_group]
    else:
        # Все параметры
        selected_params = original_data.columns.tolist()
    
    # Оставляем только существующие колонки
    existing_params = [col for col in selected_params if col in original_data.columns]
    
    if not existing_params:
        return pd.DataFrame()
        
    filtered_data = original_data[existing_params].copy()
    
    # Переименовываем колонки для отображения (только если есть что переименовывать)
    rename_dict = {col: COLUMN_NAMES.get(col, col) for col in filtered_data.columns if col in COLUMN_NAMES}
    filtered_data = filtered_data.rename(columns=rename_dict)
    
    # Дополнительная логика для показа только значимых различий
    if show_diff_only and len(original_data['composition'].unique()) >= 2:
        try:
            numeric_cols = [col for col in existing_params if col != 'composition' and col in original_data.select_dtypes(include=[np.number]).columns]
            significant_cols = ['composition']
            
            for col in numeric_cols:
                if col != 'composition':
                    composition_stats = original_data.groupby('composition')[col].mean()
                    if len(composition_stats) >= 2:
                        max_val = composition_stats.max()
                        min_val = composition_stats.min()
                        if max_val > 0:
                            difference_pct = ((max_val - min_val) / max_val) * 100
                            if difference_pct >= 10:
                                significant_cols.append(col)
            
            if len(significant_cols) > 1:
                # Фильтруем по значимым колонкам, но сохраняем оригинальные названия для фильтрации
                temp_data = original_data[significant_cols].copy()
                # Переименовываем после фильтрации
                temp_rename_dict = {col: COLUMN_NAMES.get(col, col) for col in temp_data.columns if col in COLUMN_NAMES}
                filtered_data = temp_data.rename(columns=temp_rename_dict)
                
        except Exception as e:
            print(f"Ошибка при фильтрации значимых различий: {e}")
    
    return filtered_data

@app.route('/ai_analysis')
def ai_analysis():
    """Страница интеллектуального ML анализа"""
    uploaded_files = os.listdir(app.config['UPLOAD_FOLDER']) if os.path.exists(app.config['UPLOAD_FOLDER']) else []
    return render_template('ai_analysis.html', segment='ИИ-анализ', uploaded_files=uploaded_files)

@app.route('/ai_ml_system_status')
def ai_ml_system_status():
    """Возвращает статус системы"""
    try:
        data_summary = ai_ml_analyzer.get_data_summary()
        ml_status = ai_ml_analyzer.get_ml_models_status()
        
        return jsonify({
            'success': True,
            'data_summary': data_summary,
            'ml_status': ml_status
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        })

@app.route('/ai_ml_analysis', methods=['POST'])
def perform_ai_ml_analysis():
    """Выполняет интеллектуальный ML анализ"""
    try:
        data = request.get_json()
        user_query = data.get('query', '')
        
        if not user_query:
            return jsonify({'success': False, 'message': 'Пустой запрос'})
        
        # Здесь будет интеграция с ИИ API
        # Пока используем заглушку с локальной логикой
        analysis_result = ai_ml_analyzer.analyze_with_ai(user_query)
        
        return jsonify({
            'success': True,
            'analysis': analysis_result.get('analysis', 'Анализ выполнен'),
            'recommendations': analysis_result.get('recommendations', ''),
            'optimal_composition': analysis_result.get('optimal_composition', {})
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Ошибка анализа: {str(e)}'
        })

@app.route('/create_graph', methods=['GET', 'POST'])
def create_graph():
    uploaded_files = os.listdir(app.config['UPLOAD_FOLDER']) if os.path.exists(app.config['UPLOAD_FOLDER']) else []
    measured_data = query_db(db_path, "measured_parameters")
    parameters = measured_data.columns.tolist() if not measured_data.empty else []
    
    # Получаем выбранный тип визуализации
    selected_viz_type = request.form.get('viz_type', 'matplotlib') if request.method == 'POST' else 'matplotlib'
    
    # Выбираем соответствующий список графиков
    if selected_viz_type == 'plotly':
        graphs = PLOTLY_GRAPHS
    elif selected_viz_type == 'seaborn':
        graphs = SEABORN_GRAPHS
    else:  # matplotlib
        graphs = MATPLOTLIB_GRAPHS

    if request.method == 'POST':
        try:
            viz_type = request.form.get('viz_type', 'matplotlib')
            graph_type = request.form.get('graph_type', 'scatter')
            x_param = request.form.get('x_param', 'ad')
            y_param = request.form.get('y_param', 'q')
            z_param = request.form.get('z_param', '')
            color_param = request.form.get('color_param', '')
            size_param = request.form.get('size_param', '')
            animation_param = request.form.get('animation_param', '')
            theme = request.form.get('theme', 'default')
            title = request.form.get('title', '')
            width = int(request.form.get('width', 800))
            height = int(request.form.get('height', 600))
            show_grid = request.form.get('show_grid') == 'on'
            
            selected_compositions_json = request.form.get('selected_compositions', '[]')
            try:
                selected_compositions = json.loads(selected_compositions_json) if selected_compositions_json else []
            except json.JSONDecodeError:
                selected_compositions = []

            # ДЕБАГ ИНФОРМАЦИЯ
            print(f"=== CREATE_GRAPH DEBUG ===")
            print(f"Selected compositions JSON: {selected_compositions_json}")
            print(f"Selected compositions: {selected_compositions} (type: {type(selected_compositions)}, len: {len(selected_compositions)})")
            
    
            stats = get_data_statistics(measured_data)

            if measured_data.empty:
                return jsonify({
                    'success': False,
                    'message': 'Нет данных для построения графика. Сначала загрузите файл.',
                    'stats': stats
                })
            
            # ИСПРАВЛЕННАЯ ПРОВЕРКА: Если нет выбранных составов, используем ВСЕ доступные
            if selected_compositions is not None and len(selected_compositions) == 0:
                print(f"=== ПУСТОЙ СПИСОК СОСТАВОВ, ИСПОЛЬЗУЕМ ВСЕ ДОСТУПНЫЕ")
                # Если составы не выбраны, используем все доступные
                if 'composition' in measured_data.columns:
                    selected_compositions = measured_data['composition'].unique().tolist()
                else:
                    selected_compositions = None
            
            # Выбор типа визуализации с поддержкой фильтрации составов
            graph = None
            graph_message = ""
            graph_output_type = "matplotlib"
            available_compositions = []
            
            if viz_type == 'plotly':
                graph, graph_message, available_compositions = generate_plotly_graph(
                    measured_data, x_param, y_param, graph_type,
                    z_param, color_param, size_param, animation_param,
                    theme, title, width, height, show_grid, selected_compositions
                )
                graph_output_type = 'plotly'
                
            elif viz_type == 'seaborn':
                graph, graph_message, available_compositions = generate_seaborn_plot(
                    measured_data, x_param, y_param, graph_type, theme, color_param, selected_compositions
                )
                graph_output_type = 'matplotlib'
                
            else:  # matplotlib
                graph, graph_message, available_compositions = generate_graph(
                    measured_data, x_param, y_param, graph_type,
                    z_param, color_param, size_param, animation_param,
                    theme, title, width, height, show_grid, selected_compositions
                )
                graph_output_type = 'matplotlib'
            
            if not graph:
                return jsonify({
                    'success': False if graph_message else True,
                    'message': graph_message or 'Нет данных для отображения с выбранными составами',
                    'stats': stats,
                    'available_compositions': available_compositions
                })
                
            return jsonify({
                'success': True,
                'message': graph_message,
                'graph': graph,
                'graph_type': graph_output_type,
                'stats': stats,
                'available_compositions': available_compositions
            })
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({
                'success': False,
                'message': f'Ошибка при создании графика: {str(e)}',
                'stats': get_data_statistics(measured_data)
            })

    # GET запрос - ИСПРАВЛЕННАЯ СТРОКА
    graph, _, _ = generate_graph(measured_data)  # Теперь распаковываем 3 значения
    stats = get_data_statistics(measured_data)
    
    return render_template(
        'create_graph.html',
        segment='Создание графика',
        uploaded_files=uploaded_files,
        parameters=parameters,
        graphs=graphs,
        viz_types=VIZ_TYPES,
        selected_viz_type=selected_viz_type,
        graph=graph,
        components_sheet_name=session.get('components_sheet_name', 'Таблица компонентов'),
        stats=stats
    )

@app.route('/<path:path>.map')
def ignore_map_files(path):
    return '', 204

@app.route('/ml_dashboard')
def ml_dashboard():
    """Страница ML анализа и оптимизации"""
    uploaded_files = os.listdir(app.config['UPLOAD_FOLDER']) if os.path.exists(app.config['UPLOAD_FOLDER']) else []
    measured_data = query_db(db_path, "measured_parameters")
    
    # Lazy import: Импортируем внутри роута, чтобы избежать цикла
    from ml_optimizer import get_ml_system
    ml_system = get_ml_system()  # Получаем глобальный экземпляр
    
    # Получаем статус ML моделей
    ml_status = {}
    try:
        ml_status = ml_system.get_ml_system_status()
    except Exception as e:
        print(f"Ошибка получения статуса ML: {e}")
        print(f"Training data size: {len(ml_system.training_data) if ml_system.training_data is not None else 'None'}")

    return render_template(
        'ml_dashboard.html', 
        segment='ML Анализ',
        uploaded_files=uploaded_files,
        compositions=measured_data['composition'].tolist() if not measured_data.empty else [],
        ml_status=ml_status
        
    )

@app.route('/ml_system_train', methods=['POST'])
def ml_system_train():
    """Обучает всю ML систему"""
    try:
        from ml_optimizer import get_ml_system
        ml_system = get_ml_system()
        
        # Получаем target_properties из формы
        target_properties = request.form.getlist('target_properties[]')
        if not target_properties:
            from ml_optimizer import PelletPropertyPredictor
            target_properties = list(PelletPropertyPredictor().target_properties_mapping.keys())
        
        print(f"🚀 Запуск обучения ML системы для свойств: {target_properties}")
        
        # Получаем алгоритм из запроса
        algorithm = request.form.get('algorithm', 'random_forest').lower()
        
        # Обучаем модель
        success = ml_system.train_models(target_properties, algorithm)
        
        if success:
            status = ml_system.get_ml_system_status()
            
            # ФОРМИРУЕМ ОТВЕТ ДЛЯ ФРОНТЕНДА
            response_data = {
                'success': True,
                'message': 'ML система успешно обучена!',
                'status': status,
                'trained_count': len(status.get('trained_models', [])),
                'metrics': {}
            }
            
            # Добавляем метрики для каждой обученной модели
            for prop in status.get('trained_models', []):
                metrics = status['model_metrics'].get(prop, {})
                training_metrics = metrics.get('training_metrics', {})
                response_data['metrics'][prop] = {
                    'r2_score': training_metrics.get('r2_score', 0),
                    'mae': training_metrics.get('mae', 0),
                    'cv_r2': training_metrics.get('cv_r2', 0)
                }
            
            print(f"✅ Обучение завершено. Метрики: {response_data['metrics']}")
            return jsonify(response_data)
        else:
            return jsonify({
                'success': False,
                'error': 'Не удалось обучить ML систему. Проверьте данные.'
            })
            
    except Exception as e:
        print(f"❌ Ошибка обучения системы: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Ошибка обучения системы: {str(e)}'
        })

@app.route('/ml_optimize', methods=['POST'])
def ml_optimize():
    try:
        from ml_optimizer import get_ml_system
        ml_system = get_ml_system()
        
        target_property = request.form.get('target_property')
        maximize = request.form.get('maximize', 'true').lower() == 'true'
        
        if not target_property:
            return jsonify({'success': False, 'error': 'Не указано целевое свойство'})
        
        # Получаем ограничения из формы
        constraints = {}
        for key in request.form:
            if key.startswith('min_'):
                comp = key.replace('min_', '')
                min_val_str = request.form.get(key, '0')  # По умолчанию 0, если пусто
                max_val_str = request.form.get(f'max_{comp}', '100')  # По умолчанию 100, если пусто
                
                # Обрабатываем пустые строки
                try:
                    min_val = float(min_val_str) if min_val_str.strip() else 0.0
                    max_val = float(max_val_str) if max_val_str.strip() else 100.0
                except ValueError:
                    return jsonify({
                        'success': False,
                        'error': f'Некорректное значение для компонента {comp}: min={min_val_str}, max={max_val_str}'
                    })
                
                constraints[comp] = (min_val, max_val)
        
        print(f"🎯 Запуск оптимизации для {target_property} (maximize: {maximize})")
        
        result = ml_system.optimize_composition(
            target_property=target_property,
            maximize=maximize,
            constraints=constraints
        )
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Ошибка оптимизации: {str(e)}'
        })

@app.route('/ml_predict', methods=['POST'])
def ml_predict():
    """Предсказывает свойства для заданного состава"""
    try:
        from ml_optimizer import get_ml_system
        ml_system = get_ml_system()
        data = request.get_json()
        composition = data.get('composition', {})
        target_property = data.get('target_property')
        
        if not composition or not target_property:
            return jsonify({'success': False, 'error': 'Не указаны состав или свойство'})
        
        prediction = ml_system.predictor.predict(composition, target_property)
        
        if prediction is not None:
            return jsonify({
                'success': True,
                'prediction': prediction,
                'property': target_property
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Не удалось сделать предсказание'
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Ошибка предсказания: {str(e)}'
        })

@app.route('/ml_system_status')
def ml_system_status():
    """Возвращает полный статус ML системы"""
    try:
        # Lazy import внутри роута
        from ml_optimizer import get_ml_system
        ml_system = get_ml_system()
        status = ml_system.get_ml_system_status()
        return jsonify({
            'success': True,
            'system_status': status  # Для совместимости с JS
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/ml_history')
def ml_history():
    """Страница истории ML оптимизаций"""
    uploaded_files = os.listdir(app.config['UPLOAD_FOLDER']) if os.path.exists(app.config['UPLOAD_FOLDER']) else []
    
    # Получаем историю оптимизаций
    from database import get_ml_optimizations
    optimizations_history = get_ml_optimizations(db_path, limit=20)
    
    return render_template(
        'ml_history.html', 
        segment='История ML',
        uploaded_files=uploaded_files,
        optimizations_history=optimizations_history
    )

@app.route('/ml_saved_models')
def ml_saved_models():
    """Страница сохраненных ML моделей"""
    uploaded_files = os.listdir(app.config['UPLOAD_FOLDER']) if os.path.exists(app.config['UPLOAD_FOLDER']) else []
    
    # Получаем сохраненные модели
    from database import get_active_ml_models
    saved_models = get_active_ml_models(db_path)
    
    return render_template(
        'ml_saved_models.html', 
        segment='Сохраненные ML модели',
        uploaded_files=uploaded_files,
        saved_models=saved_models
    )

@app.route('/ml_verify_optimization', methods=['POST'])
def ml_verify_optimization():
    """Добавляет реальные измерения для ML оптимизации"""
    try:
        optimization_id = request.form.get('optimization_id')
        actual_properties = {
            'density': float(request.form.get('density', 0)) if request.form.get('density') else None,
            'q': float(request.form.get('q', 0)) if request.form.get('q') else None,
            'ad': float(request.form.get('ad', 0)) if request.form.get('ad') else None,
            # Добавьте другие свойства
        }
        
        from database import update_ml_optimization_with_actual
        success = update_ml_optimization_with_actual(db_path, optimization_id, actual_properties)
        
        if success:
            # Переобучаем модели на верифицированных данных
            from ml_optimizer import get_ml_system
            ml_system = get_ml_system()
            retrain_result = ml_system.retrain_on_new_data()
            
            return jsonify({
                'success': True,
                'message': 'Реальные измерения добавлены и модели переобучены',
                'retrain_result': retrain_result
            })
        else:
            return jsonify({'success': False, 'error': 'Ошибка добавления измерений'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/ml_auto_retrain', methods=['POST'])
def ml_auto_retrain():
    """Автоматическое переобучение на новых данных"""
    try:
        from ml_optimizer import get_ml_system
        ml_system = get_ml_system()
        
        result = ml_system.retrain_on_new_data()
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/ai_ml_recommendations')
def ai_ml_recommendations():
    """Возвращает рекомендации по улучшению системы"""
    try:
        recommendations = ai_ml_analyzer.get_system_recommendations()
        return jsonify({
            'success': True,
            'recommendations': recommendations['recommendations'],
            'total_count': recommendations['total_count']
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/ai_ml_history')
def ai_ml_history():
    """Возвращает историю анализов"""
    try:
        history = ai_ml_analyzer.get_analysis_history(limit=5)
        return jsonify({
            'success': True,
            'history': history
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


if __name__ == '__main__':
    app.run(debug=True)