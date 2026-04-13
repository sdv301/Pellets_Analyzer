# app/routes/main.py — Основные роуты (dashboard, upload, search, tables)
import os
import json
import pandas as pd
from flask import Blueprint, render_template, request, jsonify, session, flash, redirect, url_for
from app.auth.auth import login_required
from app.models.database import query_db
from app.services.data_processor import process_data_source
from app.services.gui import generate_graph, get_data_statistics

main_bp = Blueprint('main', __name__)

# Путь к БД — будет установлен при инициализации
_db_path = 'pellets_data.db'


def set_db_path(path):
    global _db_path
    _db_path = path


def get_uploaded_files():
    """Возвращает список загруженных файлов"""
    upload_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'Uploads')
    return os.listdir(upload_folder) if os.path.exists(upload_folder) else []


@main_bp.route('/')
@main_bp.route('/landing')
def index():
    """Лендинг-визитка — доступна без авторизации."""
    return render_template('landing.html')


@main_bp.route('/dashboard')
@login_required
def dashboard():
    uploaded_files = get_uploaded_files()
    show_data = session.get('show_data', False)

    total_measured_count = 0
    total_components_count = 0
    chart_labels = []
    chart_data = []

    try:
        measured_data = query_db(_db_path, "measured_parameters")
        components_data = query_db(_db_path, "components")

        if not measured_data.empty:
            total_measured_count = len(measured_data)

        if not components_data.empty:
            total_components_count = len(components_data)

            if 'composition' in components_data.columns:
                raw_compositions = components_data['composition'].astype(str).tolist()
                parsed_components = []
                for comp in raw_compositions:
                    base_comp = comp.split(' ')[0].split('_')[0].strip(',.-')
                    if base_comp:
                        parsed_components.append(base_comp)

                from collections import Counter
                comp_counts = Counter(parsed_components)
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


@main_bp.route('/', methods=['POST'])
@login_required
def upload_file():
    uploaded_files = get_uploaded_files()
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'Файл не предоставлен.', 'uploaded_files': uploaded_files})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'Файл не выбран.', 'uploaded_files': uploaded_files})

    upload_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'Uploads')
    if file and (file.filename.endswith('.csv') or file.filename.endswith('.xlsx')):
        file_path = os.path.join(upload_folder, file.filename)
        file.save(file_path)
        try:
            messages, components_sheet_name, sheet_data = process_data_source(file_path, _db_path)

            measured_data = query_db(_db_path, "measured_parameters")
            components_data = query_db(_db_path, "components")

            if not sheet_data:
                flash('Файл не содержит данных для отображения.', 'warning')
            else:
                flash(f'Загружено {len(sheet_data)} листов из файла {file.filename}.', 'success')

            session['sheet_data'] = [
                {'name': s['name'], 'data': s['data'].to_json(orient='records', force_ascii=False)} for s in sheet_data
            ]
            session['components_sheet_name'] = components_sheet_name
            session['show_data'] = True
            session['measured_data'] = measured_data.to_json(orient='records', force_ascii=False)
            session['components_data'] = components_data.to_json(orient='records', force_ascii=False)

            flash(f'Данные сохранены в сессию: {len(sheet_data)} листов.', 'info')

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
            return jsonify({'success': False, 'message': f'Ошибка обработки файла: {str(e)}', 'uploaded_files': uploaded_files})

    flash('Недопустимый формат файла.', 'danger')
    return jsonify({'success': False, 'message': 'Недопустимый формат файла.', 'uploaded_files': uploaded_files})


@main_bp.route('/load_file', methods=['POST'])
@login_required
def load_file():
    selected_file = request.form.get('selected_file')
    uploaded_files = get_uploaded_files()
    if not selected_file:
        return jsonify({'success': False, 'message': 'Файл не выбран.', 'uploaded_files': uploaded_files})

    upload_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'Uploads')
    file_path = os.path.join(upload_folder, selected_file)
    if not os.path.exists(file_path):
        return jsonify({'success': False, 'message': f'Файл {selected_file} не найден.', 'uploaded_files': uploaded_files})

    try:
        process_messages, components_sheet_name, sheet_data = process_data_source(file_path, _db_path)
        for msg in process_messages:
            category = "danger" if "Error" in msg or "Warning" in msg else "success"
            flash(msg, category)

        measured_data = query_db(_db_path, "measured_parameters")
        components_data = query_db(_db_path, "components")
        graph, message, compositions = generate_graph(measured_data)

        if measured_data.empty and components_data.empty:
            return jsonify({'success': False, 'message': 'Данные обработаны, но таблицы пусты. Проверьте формат файла или названия столбцов.', 'uploaded_files': uploaded_files})

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
        return jsonify({'success': False, 'message': f'Ошибка обработки файла {selected_file}: {str(e)}', 'uploaded_files': uploaded_files})


@main_bp.route('/search', methods=['POST'])
@login_required
def search():
    try:
        search_columns = request.form.getlist('search_column')
        search_operators = request.form.getlist('search_operator')
        search_values = request.form.getlist('search_value')
        search_values_max = request.form.getlist('search_value_max')

        all_measured_data = query_db(_db_path, "measured_parameters")
        if all_measured_data.empty:
            return jsonify({'success': False, 'message': 'База данных пуста.'})

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
                if col == 'composition':
                    if op == 'LIKE' or op == '=':
                        mask = filtered_data[col].astype(str).str.contains(val, case=False, na=False)
                    elif op == '!=':
                        mask = ~filtered_data[col].astype(str).str.contains(val, case=False, na=False)
                    else:
                        mask = pd.Series([False] * len(filtered_data))
                    filtered_data = filtered_data[mask]
                    continue

                try:
                    f_val = float(val) if val else None
                    f_val_max = float(val_max) if val_max else None
                except (ValueError, TypeError):
                    continue

                if op == 'BETWEEN':
                    if f_val is not None and f_val_max is not None:
                        if f_val > f_val_max:
                            continue
                        mask = (filtered_data[col] >= f_val) & (filtered_data[col] <= f_val_max)
                    else:
                        continue
                elif op == 'APPROX':
                    if f_val is not None:
                        tolerance = f_val * 0.05
                        mask = (filtered_data[col] >= f_val - tolerance) & (filtered_data[col] <= f_val + tolerance)
                    else:
                        mask = pd.Series([False] * len(filtered_data))
                elif op == 'CONTAINS':
                    if f_val is not None:
                        mask = filtered_data[col].astype(str).str.contains(str(f_val), na=False, regex=False)
                    else:
                        mask = pd.Series([False] * len(filtered_data))
                else:
                    if f_val is None:
                        continue
                    if op == '=':
                        mask = filtered_data[col].round(2) == round(f_val, 2)
                    elif op == '!=':
                        mask = filtered_data[col].round(2) != round(f_val, 2)
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


@main_bp.route('/global_search', methods=['GET'])
@login_required
def global_search():
    query = request.args.get('q', '').strip()
    if not query or len(query) < 2:
        return jsonify({'success': False, 'results': []})

    try:
        measured_data = query_db(_db_path, "measured_parameters")
        if measured_data.empty or 'composition' not in measured_data.columns:
            return jsonify({'success': True, 'results': []})

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


@main_bp.route('/add_data', methods=['POST'])
@login_required
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
        from app.models.database import insert_data
        insert_data(_db_path, "measured_parameters", df)

        measured_data = query_db(_db_path, "measured_parameters")
        components_data = query_db(_db_path, "components")
        graph, message, compositions = generate_graph(measured_data)

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
        return jsonify({'success': False, 'message': f'Ошибка при добавлении данных: {str(e)}', 'uploaded_files': get_uploaded_files()})


@main_bp.route('/tables')
@login_required
def tables():
    uploaded_files = get_uploaded_files()
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 10))

    search_performed = session.get('search_performed', False)
    quick_search = request.args.get('search', '').strip()

    if quick_search:
        try:
            measured_data = query_db(_db_path, "measured_parameters")
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

    if not tables:
        measured_data = query_db(_db_path, "measured_parameters")
        components_data = query_db(_db_path, "components")

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
        'Admin/tables.html',
        segment='Таблицы',
        uploaded_files=uploaded_files,
        tables=tables,
        show_data=show_data,
        page=page,
        per_page=per_page,
        total_rows=total_rows,
        search_performed=search_performed
    )


@main_bp.route('/clear_search', methods=['POST'])
@login_required
def clear_search():
    session.pop('search_results', None)
    session.pop('search_performed', None)
    return jsonify({'success': True, 'message': 'Поиск очищен, показаны все данные'})


@main_bp.route('/ai_analysis')
@login_required
def ai_analysis():
    uploaded_files = get_uploaded_files()
    return render_template('ai_analysis.html', segment='ИИ-анализ', uploaded_files=uploaded_files)


@main_bp.route('/ai_ml_system_status')
@login_required
def ai_ml_system_status():
    try:
        from app.services.ai_ml_analyzer import AIMLAnalyzer
        analyzer = AIMLAnalyzer(_db_path)
        data_summary = analyzer.get_data_summary()
        ml_status = analyzer.get_ml_models_status()
        return jsonify({'success': True, 'data_summary': data_summary, 'ml_status': ml_status})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@main_bp.route('/ai_ml_analysis', methods=['POST'])
@login_required
def perform_ai_ml_analysis():
    try:
        from app.services.ai_ml_analyzer import AIMLAnalyzer
        analyzer = AIMLAnalyzer(_db_path)
        data = request.get_json()
        user_query = data.get('query', '')

        if not user_query:
            return jsonify({'success': False, 'message': 'Пустой запрос'})

        analysis_result = analyzer.analyze_with_ai(user_query)

        return jsonify({
            'success': True,
            'analysis': analysis_result.get('analysis', 'Анализ выполнен'),
            'recommendations': analysis_result.get('recommendations', ''),
            'optimal_composition': analysis_result.get('optimal_composition', {})
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Ошибка анализа: {str(e)}'})


@main_bp.route('/economics')
@login_required
def economics():
    uploaded_files = get_uploaded_files()
    return render_template('economics.html', segment='Экономическая часть', uploaded_files=uploaded_files)


@main_bp.route('/<path:path>.map')
def ignore_map_files(path):
    return '', 204
