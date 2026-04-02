# main.py - основное приложение со всеми роутами
from flask import Flask, render_template, request, jsonify, session, flash
from flask_session import Session
import pandas as pd
import os
import numpy as np
import secrets
from data_processor import process_data_source
from database import query_db, insert_data, init_db
from ai_ml_integration import AIMLAnalyzer
from gui import *
import json
import logging
import traceback

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'Uploads'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = os.path.join(os.path.dirname(__file__), 'sessions')
# БЕЗОПАСНОСТЬ: Секретный ключ из переменной окружения или генерируется случайно
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', secrets.token_hex(24))

# --- ОТКЛЮЧЕНИЕ КЭШИРОВАНИЯ (Чтобы страницы всегда обновлялись) ---
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app_errors.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

@app.errorhandler(Exception)
def handle_exception(e):
    error_info = traceback.format_exc()
    app.logger.error(f"КРИТИЧЕСКАЯ ОШИБКА:\n{error_info}")
    return f"""
        <h2 style='color:red;'>Сервер упал (Ошибка 500)</h2>
        <p>Но теперь мы поймали ошибку! Она только что была записана в файл.</p>
        <p><b>Что делать:</b> Зайдите в папку с проектом, найдите файл <code>app_errors.log</code>, откройте его через Блокнот и скопируйте текст оттуда.</p>
    """, 500
# -----------------------------

# Helper-функции для уменьшения дублирования кода
def get_uploaded_files():
    """Возвращает список загруженных файлов"""
    upload_folder = app.config.get('UPLOAD_FOLDER', 'Uploads')
    return os.listdir(upload_folder) if os.path.exists(upload_folder) else []

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response
# -------------------------------------------------------------------

Session(app)
ai_ml_analyzer = AIMLAnalyzer()
db_path = 'pellets_data.db'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)
init_db(db_path)  # Инициализация базы данных

@app.route('/')
def index():
    uploaded_files = get_uploaded_files()
    show_data = session.get('show_data', False)
    
    # Статистика и данные для графиков (BankDash)
    total_measured_count = 0
    total_components_count = 0
    chart_labels = []
    chart_data = []

    try:
        measured_data = query_db(db_path, "measured_parameters")
        components_data = query_db(db_path, "components")
        
        if not measured_data.empty:
            total_measured_count = len(measured_data)
        
        if not components_data.empty:
            total_components_count = len(components_data)
            
            # Подготовка данных для графика: Топ-6 компонентов по частоте
            if 'composition' in components_data.columns:
                # Извлекаем названия компонентов (убираем проценты и лишние пробелы для простоты графиков)
                # В реальной базе лучше использовать парсер, здесь берем первые слова как базу
                raw_compositions = components_data['composition'].astype(str).tolist()
                parsed_components = []
                for comp in raw_compositions:
                    # Простая эвристика: берем первое слово (например "Сосна" из "Сосна 80%") 
                    # или оставляем как есть, если это одно слово
                    base_comp = comp.split(' ')[0].split('_')[0].strip(',.-')
                    if base_comp:
                        parsed_components.append(base_comp)
                
                from collections import Counter
                comp_counts = Counter(parsed_components)
                # Берем топ-7 самых частых
                top_comps = comp_counts.most_common(7)
                chart_labels = [item[0] for item in top_comps]
                chart_data = [item[1] for item in top_comps]
                
        if total_measured_count > 0 or total_components_count > 0:
            show_data = True
            session['show_data'] = True
    except Exception as e:
        print(f"Ошибка при получении статистики для дашборда: {e}")

    return render_template(
        'index.html',
        segment='Главная',
        uploaded_files=uploaded_files,
        show_data=show_data,
        total_measured_count=total_measured_count,
        total_components_count=total_components_count,
        chart_labels=chart_labels,
        chart_data=chart_data
    )

@app.route('/', methods=['POST'])
def upload_file():
    uploaded_files = get_uploaded_files()
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
    uploaded_files = get_uploaded_files()
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
        process_messages, components_sheet_name, sheet_data = process_data_source(file_path, db_path)
        for msg in process_messages:
            category = "danger" if "Error" in msg or "Warning" in msg else "success"
            flash(msg, category)
        
        measured_data = query_db(db_path, "measured_parameters")
        components_data = query_db(db_path, "components")
        graph, message, compositions = generate_graph(measured_data)  # Обновленный вызов
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
        # Получаем данные из формы
        search_columns = request.form.getlist('search_column')
        search_operators = request.form.getlist('search_operator')
        search_values = request.form.getlist('search_value')
        search_values_max = request.form.getlist('search_value_max')
        
        all_measured_data = query_db(db_path, "measured_parameters")
        if all_measured_data.empty:
            return jsonify({'success': False, 'message': 'База данных пуста.'})
            
        # Начинаем со всех данных и последовательно применяем фильтры (логика И)
        filtered_data = all_measured_data.copy()
        
        applied_filters = 0
        for i in range(len(search_columns)):
            col = search_columns[i]
            if not col or col not in all_measured_data.columns:
                continue
                
            op = search_operators[i] if i < len(search_operators) else '='
            val = search_values[i].strip() if i < len(search_values) else ''
            val_max = search_values_max[i].strip() if i < len(search_values_max) else ''
            
            if not val and op != 'BETWEEN':
                continue
            
            applied_filters += 1
            
            try:
                # Специальная обработка для состава (строковый поиск)
                if col == 'composition':
                    if op == 'LIKE' or op == '=':
                        mask = filtered_data[col].astype(str).str.contains(val, case=False, na=False)
                    elif op == '!=':
                        mask = ~filtered_data[col].astype(str).str.contains(val, case=False, na=False)
                    else:
                        mask = pd.Series([False] * len(filtered_data))
                    filtered_data = filtered_data[mask]
                    continue

                # Числовой поиск - безопасная конвертация
                try:
                    f_val = float(val) if val else None
                    f_val_max = float(val_max) if val_max else None
                except (ValueError, TypeError):
                    print(f"Skipping invalid numeric search: {val} to {val_max}")
                    continue

                if op == 'BETWEEN':
                    if f_val is not None and f_val_max is not None:
                        mask = (filtered_data[col] >= f_val) & (filtered_data[col] <= f_val_max)
                    elif f_val is not None:
                        mask = filtered_data[col] >= f_val
                    elif f_val_max is not None:
                        mask = filtered_data[col] <= f_val_max
                    else:
                        mask = pd.Series([False] * len(filtered_data))
                else:
                    if f_val is None: # Если не BETWEEN и значение пустое - пропускаем
                        continue
                        
                    if op == '=':
                        mask = filtered_data[col] == f_val
                    elif op == '!=':
                        mask = filtered_data[col] != f_val
                    elif op == '>':
                        mask = filtered_data[col] > f_val
                    elif op == '>=':
                        mask = filtered_data[col] >= f_val
                    elif op == '<':
                        mask = filtered_data[col] < f_val
                    elif op == '<=':
                        mask = filtered_data[col] <= f_val
                    elif op == 'LIKE':
                        mask = filtered_data[col].astype(str).str.contains(str(val), na=False)
                    else:
                        mask = pd.Series([False] * len(filtered_data))
                
                # Применяем фильтр к текущим результатам (логика И)
                filtered_data = filtered_data[mask]

            except Exception as e:
                print(f"Search criterion error for col {col}: {e}")
                continue
                
        if applied_filters == 0:
            return jsonify({'success': False, 'message': 'Критерии поиска не заданы.'})

        if filtered_data.empty:
            return jsonify({'success': True, 'message': 'Ничего не найдено', 'count': 0, 'refresh_page': True})
            
        session['search_results'] = filtered_data.to_json(orient='records', force_ascii=False)
        session['search_performed'] = True
        return jsonify({
            'success': True, 
            'message': f'Найдено записей: {len(filtered_data)}',
            'count': len(filtered_data),
            'refresh_page': True
        })
        
    except Exception as e:
        import traceback
        print(f"Search API Error: {traceback.format_exc()}")
        return jsonify({'success': False, 'message': f'Ошибка поиска: {str(e)}'})

@app.route('/global_search', methods=['GET'])
def global_search():
    query = request.args.get('q', '').strip()
    if not query or len(query) < 2:
        return jsonify({'success': False, 'results': []})
        
    try:
        measured_data = query_db(db_path, "measured_parameters")
        if measured_data.empty or 'composition' not in measured_data.columns:
            return jsonify({'success': True, 'results': []})
            
        # Поиск по вхождению (регистронезависимый)
        mask = measured_data['composition'].astype(str).str.contains(query, case=False, na=False)
        results_df = measured_data[mask].head(5)
        
        results = []
        for _, row in results_df.iterrows():
            results.append({
                'composition': row['composition'],
                'q': round(row['q'], 2) if pd.notna(row.get('q')) else None,
                'ad': round(row['ad'], 2) if pd.notna(row.get('ad')) else None
            })
            
        return jsonify({'success': True, 'results': results})
    except Exception as e:
        print(f"Global search error: {e}")
        return jsonify({'success': False, 'message': str(e)})

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
            'ad': float(request.form.get('ad', '')) if request.form.get('ad', '') else None,
            'war': float(request.form.get('war', '')) if request.form.get('war', '') else None,
            'vd': float(request.form.get('vd', '')) if request.form.get('vd', '') else None,
            'cd': float(request.form.get('cd', '')) if request.form.get('cd', '') else None,
            'hd': float(request.form.get('hd', '')) if request.form.get('hd', '') else None,
            'nd': float(request.form.get('nd', '')) if request.form.get('nd', '') else None,
            'sd': float(request.form.get('sd', '')) if request.form.get('sd', '') else None,
            'od': float(request.form.get('od', '')) if request.form.get('od', '') else None
        }
        df = pd.DataFrame([data])
        insert_data(db_path, "measured_parameters", df)
        measured_data = query_db(db_path, "measured_parameters")
        components_data = query_db(db_path, "components")
        graph, message, compositions = generate_graph(measured_data)  # Обновленный вызов
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
            'uploaded_files': get_uploaded_files(),
            'components_sheet_name': session.get('components_sheet_name', 'Таблица компонентов'),
            'total_measured': len(measured_data),
            'total_components': len(components_data)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Ошибка при добавлении данных: {str(e)}',
            'uploaded_files': get_uploaded_files()
        })

@app.route('/tables')
def tables():
    uploaded_files = get_uploaded_files()
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 10))
    
    search_performed = session.get('search_performed', False)
    quick_search = request.args.get('search', '').strip()
    
    # Обработка перехода из глобального поиска
    if quick_search:
        try:
            measured_data = query_db(db_path, "measured_parameters")
            if not measured_data.empty and 'composition' in measured_data.columns:
                mask = measured_data['composition'].astype(str) == quick_search
                filtered_data = measured_data[mask]
                session['search_results'] = filtered_data.to_json(orient='records', force_ascii=False)
                session['search_performed'] = True
                search_performed = True
        except Exception as e:
            print(f"Ошибка быстрого поиска: {e}")

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
    uploaded_files = get_uploaded_files()
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
        
        if len(compositions) < 2:
            return jsonify({
                'success': False,
                'message': 'Выберите хотя бы два состава для сравнения.'
            })
        
        # Получаем выбранные критерии
        selected_criteria = request.form.getlist('criteria[]')
        
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
    uploaded_files = get_uploaded_files()
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

@app.route('/economics')
def economics():
    """Страница экономической оценки"""
    uploaded_files = get_uploaded_files()
    return render_template('economics.html', segment='Экономическая часть', uploaded_files=uploaded_files)

@app.route('/create_graph', methods=['GET', 'POST'])
def create_graph():
    uploaded_files = get_uploaded_files()
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
            
            # ДЕБАГ
            selected_compositions_json = request.form.get('selected_compositions', '[]')
            try:
                selected_compositions = json.loads(selected_compositions_json) if selected_compositions_json else []
            except json.JSONDecodeError:
                selected_compositions = []
            
    
            stats = get_data_statistics(measured_data)

            if measured_data.empty:
                return jsonify({
                    'success': False,
                    'message': 'Нет данных для построения графика. Сначала загрузите файл.',
                    'stats': stats
                })
            
            # ИСПРАВЛЕННАЯ ПРОВЕРКА: Если нет выбранных составов, используем ВСЕ доступные
            if selected_compositions is not None and len(selected_compositions) == 0:
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
    graph, message, compositions = generate_graph(measured_data)  # Теперь распаковываем 3 значения
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

@app.route('/profile')
def profile():
    """Страница профиля и настроек пользователя"""
    return render_template('profile.html', segment='Профиль')

@app.route('/<path:path>.map')
def ignore_map_files(path):
    return '', 204

@app.route('/ml_dashboard')
def ml_dashboard():
    try:
        uploaded_files = get_uploaded_files()
        measured_data = query_db(db_path, "measured_parameters")
        
        from ml_optimizer import get_ml_system
        ml_system = get_ml_system()
        ml_system.reload_data()
        
        status = ml_system.get_ml_system_status()
        
        history_list = []
        try:
            from database import get_ml_optimizations
            optimizations_history = get_ml_optimizations(db_path, limit=10)
            if hasattr(optimizations_history, 'empty') and not optimizations_history.empty:
                history_list = optimizations_history.to_dict('records')
                for item in history_list:
                    if 'optimal_composition' in item and isinstance(item['optimal_composition'], dict):
                        item['optimal_composition_text'] = ", ".join([f"{v}% {k}" for k, v in item['optimal_composition'].items()])
                    else:
                        item['optimal_composition_text'] = "Нет данных"
        except Exception as db_err:
            print(f"Ошибка БД (история): {db_err}")

        return render_template(
            'ml_dashboard.html', 
            segment='ML Анализ',
            uploaded_files=uploaded_files,
            compositions = measured_data['composition'].tolist() if not measured_data.empty and 'composition' in measured_data.columns else [],
            ml_status=status,
            optimization_history=history_list # Исправлено имя переменной под ваш HTML
        )
    except Exception as e:
        import traceback
        err = traceback.format_exc()
        # ХИТРОСТЬ: Возвращаем 200 ОК, чтобы браузер точно напечатал ошибку на экране!
        return f"<h2 style='color:red;'>Внимание, найдена ошибка в коде Python:</h2><pre style='background:#f4f4f4; padding:20px; font-size:16px; border-left: 5px solid red;'>{err}</pre>", 200

@app.route('/ml_system_train', methods=['POST'])
def ml_system_train():
    """Обучает всю ML систему"""
    try:
        from ml_optimizer import get_ml_system
        ml_system = get_ml_system()
        ml_system.reload_data()  # Перед тренировкой обязательно обновляем данные
        
        # Получаем target_properties из формы
        target_properties = request.form.getlist('target_properties[]')
        if not target_properties:
            from ml_optimizer import PelletPropertyPredictor
            target_properties = list(PelletPropertyPredictor().target_properties_mapping.keys())
        
        print(f"🚀 Запуск обучения ML системы для свойств: {target_properties}")
        
        # Получаем алгоритм из запроса
        algorithm = request.form.get('algorithm', 'random_forest').lower()
        
        # Получаем выбранные входные компоненты (фичи) для обучения
        input_features = request.form.getlist('input_features[]')
        if not input_features:
            input_features = None # По-умолчанию использовать все, если ничего не пришло
            
        print(f"🔧 Выбранные входные компоненты: {input_features if input_features else 'Все (по умолчанию)'}")
        
        # Обучаем модель
        result = ml_system.train_models(target_properties, algorithm, selected_features=input_features)
        
        if result.get('success'):
            status = ml_system.get_ml_system_status()
            
            # ФОРМИРУЕМ ОТВЕТ ДЛЯ ФРОНТЕНДА
            response_data = {
                'success': True,
                'message': 'ML система успешно обучена!',
                'status': status,
                'trained_count': result.get('trained_count', len(status.get('trained_models', []))),
                'skipped': result.get('skipped', []),
                'metrics': result.get('metrics', {})
            }
            
            print(f"✅ Обучение завершено. Метрики: {response_data['metrics']}")
            return jsonify(response_data)
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Не удалось обучить ML систему. Проверьте данные.'),
                'skipped': result.get('skipped', [])
            })
            
    except Exception as e:
        print(f"❌ Ошибка обучения системы: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Ошибка обучения системы: {str(e)}'
        })

@app.route('/ml_augment_database', methods=['POST'])
def ml_augment_database():
    """Масштабирует базу данных и переобучает систему"""
    try:
        from ml_optimizer import get_ml_system
        ml_system = get_ml_system()
        
        # Получаем параметры из формы
        try:
            variations_count = int(request.form.get('variations_count', 3))
            confidence_interval = float(request.form.get('confidence_interval', 5.0))
        except ValueError:
            return jsonify({
                'success': False,
                'error': 'Некорректные параметры для аугментации (ожидаются числа)'
            })
            
        print(f"🚀 Запуск аугментации: вариаций={variations_count}, интервал={confidence_interval}%")
        
        result = ml_system.augment_database(
            variations_count=variations_count,
            confidence_interval=confidence_interval
        )
        
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Ошибка аугментации: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Ошибка аугментации: {str(e)}'
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
        ml_system.reload_data() # Обновляем чтобы подтянуть новые фичи
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

@app.route('/api/predict_composition', methods=['POST'])
def api_predict_composition():
    """API для мгновенного предсказания (песочница)"""
    try:
        data = request.json
        composition = data.get('composition', {})
        
        from ml_optimizer import get_ml_system
        ml_system = get_ml_system()
        
        if not ml_system.predictor.is_trained:
            return jsonify({'success': False, 'error': 'Модели не обучены'})
            
        results = {}
        target_props = ml_system.predictor.main_target_properties
        
        for prop in target_props:
            pred = ml_system.predictor.predict(composition, prop)
            if pred is not None:
                display_name = ml_system.predictor.target_properties_mapping.get(prop, prop)
                results[prop] = {
                    'value': round(float(pred), 3),
                    'display_name': display_name
                }
                
        return jsonify({
            'success': True,
            'results': results
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/ml_history')
def ml_history():
    """Страница истории ML оптимизаций"""
    uploaded_files = get_uploaded_files()
    
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
    uploaded_files = get_uploaded_files()
    
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
        success = update_ml_optimization_with_actual(db_path, int(optimization_id), actual_properties)
        
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

# Эталонные данные из нового листа "Исходники"
# Плотность (Ro) скрыта в коде для расчетов экономики
COMPONENTS_DATA = {
    "Опилки": {"War": 10.8, "Ad": 0.18, "Qas": 19.79, "Hd": 6.82, "Qai": 18.18, "Cc": 1.0, "Cu": 0.2, "Cg": 1.2, "Ro": 1048.0},
    "Солома": {"War": 9.83, "Ad": 4.33, "Qas": 16.75, "Hd": 6.54, "Qai": 15.26, "Cc": 1.4, "Cu": 0.9, "Cg": 2.8, "Ro": 977.0},
    "Картон": {"War": 5.02, "Ad": 3.16, "Qas": 16.73, "Hd": 6.50, "Qai": 15.28, "Cc": 2.0, "Cu": 1.1, "Cg": 2.4, "Ro": 1040.0},
    "Подсолнечный жмых": {"War": 6.16, "Ad": 4.09, "Qas": 20.79, "Hd": 7.36, "Qai": 19.17, "Cc": 15.0, "Cu": 0.6, "Cg": 2.0, "Ro": 969.0},
    "Рисовая шелуха": {"War": 6.34, "Ad": 10.85, "Qas": 15.95, "Hd": 5.30, "Qai": 14.81, "Cc": 2.3, "Cu": 1.2, "Cg": 3.3, "Ro": 1105.0},
    "Угольный шлам": {"War": 2.20, "Ad": 62.58, "Qas": 8.00, "Hd": 1.87, "Qai": 7.79, "Cc": 3.0, "Cu": 0.9, "Cg": 2.5, "Ro": 1000.0},
    "Торф": {"War": 9.90, "Ad": 20.70, "Qas": 11.80, "Hd": 2.93, "Qai": 11.09, "Cc": 4.5, "Cu": 1.5, "Cg": 3.5, "Ro": 1051.0},
    "Бурый уголь": {"War": 10.76, "Ad": 4.50, "Qas": 22.91, "Hd": 5.54, "Qai": 21.60, "Cc": 6.0, "Cu": 1.2, "Cg": 2.9, "Ro": 1123.0},
    "СМС": {"War": 13.71, "Ad": 24.87, "Qas": 10.14, "Hd": 4.29, "Qai": 9.19, "Cc": 115.0, "Cu": 3.8, "Cg": 5.5, "Ro": 1000.0},
    "Пластик": {"War": 2.00, "Ad": 0.20, "Qas": 22.80, "Hd": 13.50, "Qai": 19.83, "Cc": 22.5, "Cu": 2.3, "Cg": 5.0, "Ro": 1008.0},
    "Листья": {"War": 7.48, "Ad": 13.14, "Qas": 17.05, "Hd": 6.92, "Qai": 15.63, "Cc": 0.0, "Cu": 1.7, "Cg": 4.0, "Ro": 1099.0},
    "Отработанное моторное масло": {"War": 0.0, "Ad": 1.68, "Qas": 45.24, "Hd": 13.20, "Qai": 42.37, "Cc": 11.5, "Cu": 0.0, "Cg": 0.0, "Ro": 1000.0},
    "Рапсовое масло": {"War": 0.0, "Ad": 0.0, "Qas": 40.89, "Hd": 10.20, "Qai": 38.63, "Cc": 90.0, "Cu": 0.0, "Cg": 0.0, "Ro": 1000.0}
}

@app.route('/api/get_components_economics')
def get_components_economics():
    """Возвращает список компонентов с их экономическими параметрами"""
    try:
        df = query_db(db_path, "components")
        if df.empty:
            # Возвращаем встроенные данные как фоллбек
            fallback = []
            for name, d in COMPONENTS_DATA.items():
                fallback.append({
                    'component': name,
                    'ro': d['Ro'],
                    'cost_raw': d['Cc'],
                    'cost_crush': d['Cu'],
                    'cost_granule': d['Cg']
                })
            return jsonify({'success': True, 'components': fallback})
        
        result = []
        for _, row in df.iterrows():
            name = str(row['component'])
            name_clean = name.strip() # Очищаем от случайных пробелов из Excel!
            
            if name_clean in COMPONENTS_DATA:
                # Если компонент базовый, ЖЕСТКО берем правильные экономические цифры
                d = COMPONENTS_DATA[name_clean]
                ro = d['Ro']
                c_raw = d['Cc']
                c_crush = d['Cu']
                c_gran = d['Cg']
            else:
                # Если это новый/кастомный компонент, берем из БД
                ro = row.get('ro')
                c_raw = row.get('cost_raw')
                c_crush = row.get('cost_crush')
                c_gran = row.get('cost_granule')
                
                ro = float(ro) if not pd.isna(ro) else 1000.0
                c_raw = float(c_raw) if not pd.isna(c_raw) else 0.0
                c_crush = float(c_crush) if not pd.isna(c_crush) else 0.0
                c_gran = float(c_gran) if not pd.isna(c_gran) else 0.0

            result.append({
                'component': name, # Оставляем оригинальное имя для связи с интерфейсом
                'ro': float(ro),
                'cost_raw': float(c_raw),
                'cost_crush': float(c_crush),
                'cost_granule': float(c_gran)
            })
            
        return jsonify({'success': True, 'components': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

from flask import send_file
import io

@app.route('/api/download_economics', methods=['GET'])
def download_economics():
    """Генерирует Excel-файл с новым стандартом колонок"""
    try:
        excel_data = []
        
        # Берем только эталонные данные для шаблона
        for name, d in COMPONENTS_DATA.items():
            excel_data.append({
                'Компоненты': name,
                'War, %': d['War'],
                'Ad, %': d['Ad'],
                'Qas,V, МДж/кг': d['Qas'],
                'Hd, %': d['Hd'],
                'Qai,V, МДж/кг': d['Qai'],
                'Cc, руб/кг': d['Cc'],
                'Cи, руб/кг': d['Cu'],
                'Cг, руб/кг': d['Cg']
            })
        
        out_df = pd.DataFrame(excel_data)
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            out_df.to_excel(writer, index=False, sheet_name='Цены и Характеристики')
            worksheet = writer.sheets['Цены и Характеристики']
            for i, col in enumerate(out_df.columns):
                worksheet.column_dimensions[chr(65 + i)].width = 18
                
        output.seek(0)
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name='Шаблон_Цен_Экономика.xlsx'
        )
    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': f'Ошибка при формировании файла: {str(e)}'})

@app.route('/api/upload_economics', methods=['POST'])
def upload_economics():
    """Загружает новый формат Excel и обновляет базу данных"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'Файл не найден'})
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'Файл не выбран'})

    try:
        df = pd.read_excel(file)
        
        # Проверяем новые колонки
        required_cols = ['Компоненты', 'Qai,V, МДж/кг', 'Cc, руб/кг', 'Cи, руб/кг', 'Cг, руб/кг']
        for col in required_cols:
            if col not in df.columns:
                return jsonify({'success': False, 'error': f'Неверный формат. Не найдена колонка: {col}'})

        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM components")
        
        for _, row in df.iterrows():
            name = str(row['Компоненты']).strip()
            
            # Если компонент базовый, берем его скрытую плотность, иначе 1000
            ro = COMPONENTS_DATA.get(name, {}).get('Ro', 1000.0)
            
            cursor.execute("""
                INSERT INTO components (component, ro, q, cost_raw, cost_crush, cost_granule) 
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                name,
                float(ro), # Скрытая плотность
                float(row['Qai,V, МДж/кг']),
                float(row['Cc, руб/кг']),
                float(row['Cи, руб/кг']),
                float(row['Cг, руб/кг'])
            ))
            
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': 'Цены успешно обновлены!'})
    except Exception as e:
        return jsonify({'success': False, 'error': f'Ошибка при чтении файла: {str(e)}'})

@app.route('/api/calculate_economics', methods=['POST'])
def calculate_economics():
    """Рассчитывает экономические показатели по формулам из методики (Пример расчета по эксельке.docx)"""
    try:
        import math
        data = request.json
        
        # 1. Входные параметры от пользователя (из формы интерфейса)
        q_boiler = float(data.get('q_boiler', 80))    # Мощность котла, кВт
        t_hours = float(data.get('t_hours', 720))     # Время работы, ч
        distance = float(data.get('distance', 258))   # Расстояние, км
        capacity_factor = float(data.get('capacity_factor', 0.8))  # Коэфф. использования мощности (a)
        efficiency = float(data.get('efficiency', 0.9))  # КПД котла (b)
        components = data.get('components', {})  # {"Опилки": 67.5, "Солома": 27.5, ...}
        
        # --- Скрытые константы из Excel/Методики ---
        # Коэффициенты из формулы объема
        COEFF_A = 0.8        # a - коэффициент использования мощности
        COEFF_C = 3.6        # c - коэффициент пересчета мощности в МДж/кг (1 кВт*ч = 3.6 МДж)
        COEFF_B = 0.9        # b - КПД котла
        
        # Экономика
        RENT_COST_PER_M2 = 380.0        # С_аренда - стоимость аренды 1 м², руб/м²
        STORAGE_LOAD_KG_PER_M2 = 1100.0 # Норматив нагрузки на склад, кг/м²
        TRUCK_BASE_RATE = 6000.0        # БС - базовая ставка за машину, руб
        TRUCK_COST_PER_KM = 85.0        # С_км - стоимость 1 км пути, руб/км
        TRUCK_CAPACITY_KG = 20000.0     # m_г - грузоподъемность одной машины, кг
        # -------------------------------------------

        # Загружаем актуальные данные из БД
        db_components = query_db(db_path, "components")
        db_map = {str(row['component']): row for _, row in db_components.iterrows()} if not db_components.empty else {}

        # 2. Расчет средневзвешенных характеристик смеси
        mix_ro = 0.0   # Плотность смеси (ρ), кг/м³
        mix_qai = 0.0  # Низшая теплота сгорания смеси (Qi,v_a), МДж/кг
        mix_cc = 0.0   # Стоимость сырья смеси (С_сырье), руб/кг
        mix_cu = 0.0   # Стоимость измельчения смеси (С_измельчение), руб/кг
        mix_cg = 0.0   # Стоимость гранулирования смеси (С_пеллетирование), руб/кг
        
        for comp_name, percentage in components.items():
            fraction = float(percentage) / 100.0
            comp_name_clean = str(comp_name).strip()
            
            # ПРИОРИТЕТ: Встроенные точные данные -> База данных
            if comp_name_clean in COMPONENTS_DATA:
                c_data = COMPONENTS_DATA[comp_name_clean]
                ro = c_data['Ro']
                qai = c_data['Qai']  # Низшая теплота сгорания
                cc = c_data['Cc']
                cu = c_data['Cu']
                cg = c_data['Cg']
            else:
                db_row = db_map.get(comp_name, {})
                ro = float(db_row.get('ro', 1000.0)) if not pd.isna(db_row.get('ro')) else 1000.0
                qai = float(db_row.get('q', 0.0)) if not pd.isna(db_row.get('q')) else 0.0
                cc = float(db_row.get('cost_raw', 0.0)) if not pd.isna(db_row.get('cost_raw')) else 0.0
                cu = float(db_row.get('cost_crush', 0.0)) if not pd.isna(db_row.get('cost_crush')) else 0.0
                cg = float(db_row.get('cost_granule', 0.0)) if not pd.isna(db_row.get('cost_granule')) else 0.0

            # Взвешенное усреднение по доле компонента
            mix_ro += fraction * float(ro)
            mix_qai += fraction * float(qai)
            mix_cc += fraction * float(cc)
            mix_cu += fraction * float(cu)
            mix_cg += fraction * float(cg)
        
        if mix_ro <= 0 or mix_qai <= 0:
            return jsonify({'success': False, 'error': 'Некорректный состав или отсутствуют данные о теплоте сгорания/плотности'})

        # 3. Расчет объема и массы топлива ПО МЕТОДИКЕ
                # Формула: Vнеобх = (Q × T × a × c) / (Qi,v_a × b × ρ)
        # где: Q - мощность котла, кВт
        #      T - время работы, ч
        #      a - коэфф. использования мощности (0.8)
        #      c - коэфф. пересчета в МДж (3.6)
        #      Qi,v_a - теплота сгорания смеси, МДж/кг
        #      b - КПД котла (0.9)
        #      ρ - плотность смеси, кг/м³
        
        numerator = q_boiler * t_hours * COEFF_A * COEFF_C
        denominator = mix_qai * COEFF_B * mix_ro
        
        v_neobh_m3 = numerator / denominator  # Необходимый объем, м³
        
        # Формула массы: mнеобх = Vнеобх × ρ
        mass_kg = v_neobh_m3 * mix_ro
        mass_kg = round(mass_kg)  # Округляем до целых кг
        
        # Пересчитываем объем для точности
        v_neobh_m3 = mass_kg / mix_ro

        # 4. Затраты на хранение (SC) по методике
        # Формула: SC = С_аренда × (mнеобх / 1100)
        # где 1100 - норматив нагрузки на 1 м²
        storage_area_m2 = math.ceil(mass_kg / STORAGE_LOAD_KG_PER_M2)
        storage_cost = RENT_COST_PER_M2 * storage_area_m2

        # 5. Затраты на производство (РС) по методике
        # Формула: РС = (С_сырье + С_измельчение + С_пеллетирование) × mнеобх
        production_cost_per_kg = mix_cc + mix_cu + mix_cg
        production_cost = production_cost_per_kg * mass_kg

        # 6. Затраты на транспортировку (ТС) по методике
        # Формула: С_рейса = БС × С_км × S × 2
        #          К = mнеобх / m_г (округление до целых)
        #          ТС = С_рейса × К
        trip_cost = TRUCK_BASE_RATE + (TRUCK_COST_PER_KM * distance * 2)
        trucks_needed = math.ceil(mass_kg / TRUCK_CAPACITY_KG)
        transport_cost = trip_cost * trucks_needed

        # 7. Итоговая стоимость по методике
        # Формула: С = РС + ТС + SC
        total_cost = production_cost + transport_cost + storage_cost
        
        return jsonify({
            'success': True,
            'mix_metrics': {
                'density': round(mix_ro, 0),
                'heat_capacity': round(mix_qai, 2),
                'cost_raw': round(mix_cc, 2),
                'cost_crush': round(mix_cu, 2),
                'cost_granule': round(mix_cg, 2)
            },
            'storage': {
                'volume_m3': round(v_neobh_m3, 2),
                'mass_kg': int(mass_kg),
                'area_m2': int(storage_area_m2),
                'cost': round(storage_cost, 2)
            },
            'production': {
                'cost_per_kg': round(production_cost_per_kg, 2),
                'cost': round(production_cost, 2)
            },
            'transport': {
                'trip_cost': round(trip_cost, 2),
                'trucks': int(trucks_needed),
                'cost': round(transport_cost, 2)
            },
            'total_cost': round(total_cost, 2)
        })
        
    except Exception as e:
        import traceback
        print(f"Error calculating economics: {traceback.format_exc()}")
        return jsonify({'success': False, 'error': f'Ошибка расчета: {str(e)}'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)