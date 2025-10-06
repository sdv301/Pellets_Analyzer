from flask import Flask, render_template, request, jsonify, session, flash
from flask_session import Session
import pandas as pd
import os
from data_processor import process_data_source
from database import query_db, insert_data, init_db
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from io import StringIO
import numpy as np
from matplotlib.animation import FuncAnimation
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'Uploads'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = os.path.join(os.path.dirname(__file__), 'sessions')
app.config['SECRET_KEY'] = 'your-secret-key'
Session(app)
db_path = 'pellets_data.db'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)
init_db(db_path)  # Инициализация базы данных

# Словарь для перевода параметров
PARAM_NAMES = {
    'ad': 'Зольность (%)',
    'q': 'Теплота сгорания (МДж/кг)',
    'density': 'Плотность (кг/м³)',
    'kf': 'Коэффициент формы (%)',
    'kt': 'Коэффициент теплопроводности (%)',
    'h': 'Высота (%)',
    'mass_loss': 'Потеря массы (%)',
    'tign': 'Температура воспламенения (°C)',
    'tb': 'Температура горения (°C)',
    'tau_d1': 'Время задержки 1 (с)',
    'tau_d2': 'Время задержки 2 (с)',
    'tau_b': 'Время горения (с)',
    'co2': 'CO2 (%)',
    'co': 'CO (%)',
    'so2': 'SO2 (ppm)',
    'nox': 'NOx (ppm)',
    'war': 'Влажность (%)',
    'vd': 'Летучие вещества (%)',
    'cd': 'Углерод (%)',
    'hd': 'Водород (%)',
    'nd': 'Азот (%)',
    'sd': 'Сера (%)',
    'od': 'Кислород (%)'
}

# Список доступных графиков
GRAPHS = ['scatter', 'line', 'histogram', 'bar', 'box', 'heatmap', '3d_scatter', 'animated_scatter']

def apply_theme(theme):
    """Применяет тему к графику"""
    if theme == 'dark':
        plt.style.use('dark_background')
    elif theme == 'seaborn':
        plt.style.use('seaborn-v0_8')
    elif theme == 'ggplot':
        plt.style.use('ggplot')
    else:
        plt.style.use('default')

def generate_animated_graph(data, x_param, y_param, animation_param, theme, title):
    """Создает анимированный график"""
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        apply_theme(theme)
        
        # Получаем уникальные значения для анимации
        animation_values = sorted(data[animation_param].unique())
        
        def update(frame):
            ax.clear()
            current_value = animation_values[frame]
            frame_data = data[data[animation_param] == current_value]
            
            scatter = ax.scatter(frame_data[x_param], frame_data[y_param], 
                               alpha=0.7, s=50, c='blue')
            
            ax.set_xlabel(PARAM_NAMES.get(x_param, x_param.capitalize()))
            ax.set_ylabel(PARAM_NAMES.get(y_param, y_param.capitalize()))
            ax.set_title(f'{title or "Анимированный график"} - {animation_param}: {current_value}')
            ax.grid(True)
            
            return scatter,
        
        anim = FuncAnimation(fig, update, frames=len(animation_values), 
                           interval=500, blit=False, repeat=True)
        
        # Сохраняем анимацию
        buf = io.BytesIO()
        anim.save(buf, format='gif', writer='pillow', fps=2)
        buf.seek(0)
        graph = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        return graph, "Анимированный график создан успешно"
        
    except Exception as e:
        plt.close()
        return None, f"Ошибка при создании анимированного графика: {str(e)}"

def generate_graph(data, x_param='ad', y_param='q', graph_type='scatter', 
                  z_param=None, color_param=None, size_param=None, 
                  animation_param=None, theme='default', title=None,
                  width=800, height=600, show_grid=True):
    
    if data.empty or x_param not in data.columns or y_param not in data.columns:
        return None, "Нет данных для построения графика"
    
    try:
        # Применяем тему
        apply_theme(theme)
        
        plt.figure(figsize=(width/100, height/100))
        
        if graph_type == 'scatter':
            if color_param and color_param in data.columns:
                scatter = plt.scatter(data[x_param], data[y_param], 
                                    c=data[color_param], cmap='viridis', 
                                    alpha=0.7, s=50)
                plt.colorbar(scatter, label=PARAM_NAMES.get(color_param, color_param))
            else:
                plt.scatter(data[x_param], data[y_param], alpha=0.7, s=50)
                
        elif graph_type == 'line':
            plt.plot(data[x_param], data[y_param], linewidth=2)
            
        elif graph_type == 'histogram':
            plt.hist(data[x_param], bins=20, alpha=0.7, edgecolor='black')
            
        elif graph_type == 'bar':
            # Для bar chart группируем по x_param и усредняем y_param
            grouped = data.groupby(x_param)[y_param].mean()
            plt.bar(grouped.index.astype(str), grouped.values, alpha=0.7)
            plt.xticks(rotation=45)
            
        elif graph_type == 'box':
            data[[x_param, y_param]].boxplot()
            
        elif graph_type == 'heatmap':
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                corr_matrix = data[numeric_cols].corr()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
            else:
                return None, "Для тепловой карты нужно больше числовых параметров"
            
        elif graph_type == '3d_scatter' and z_param and z_param in data.columns:
            fig = plt.figure(figsize=(width/100, height/100))
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(data[x_param], data[y_param], data[z_param],
                               c=data[color_param] if color_param and color_param in data.columns else 'blue',
                               cmap='viridis' if color_param and color_param in data.columns else None,
                               alpha=0.7)
            ax.set_xlabel(PARAM_NAMES.get(x_param, x_param.capitalize()))
            ax.set_ylabel(PARAM_NAMES.get(y_param, y_param.capitalize()))
            ax.set_zlabel(PARAM_NAMES.get(z_param, z_param.capitalize()))
            if color_param and color_param in data.columns:
                fig.colorbar(scatter, ax=ax, label=PARAM_NAMES.get(color_param, color_param))
                
        elif graph_type == 'animated_scatter' and animation_param and animation_param in data.columns:
            return generate_animated_graph(data, x_param, y_param, animation_param, theme, title)
        else:
            # По умолчанию scatter plot
            plt.scatter(data[x_param], data[y_param], alpha=0.7, s=50)
        
        # Общие настройки для 2D графиков
        if not graph_type.startswith('3d') and graph_type != 'animated_scatter':
            plt.xlabel(PARAM_NAMES.get(x_param, x_param.capitalize()))
            plt.ylabel(PARAM_NAMES.get(y_param, y_param.capitalize()))
            plt.grid(show_grid)
            
        # Заголовок
        if title:
            plt.title(title)
        else:
            plt.title(f'{PARAM_NAMES.get(y_param, y_param.capitalize())} vs {PARAM_NAMES.get(x_param, x_param.capitalize())}')
        
        plt.tight_layout()
        
        # Сохраняем график
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        graph = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        return graph, "График создан успешно"
        
    except Exception as e:
        plt.close()
        return None, f"Ошибка при создании графика: {str(e)}"

def generate_graph_simple(data, x_param='ad', y_param='q', graph_type='scatter'):
    """Простая версия для обратной совместимости"""
    graph, _ = generate_graph(data, x_param, y_param, graph_type)
    return graph

def get_data_statistics(data):
    """Возвращает статистику по данным"""
    if data.empty:
        return {}
    
    stats = {}
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        # Преобразуем numpy типы в стандартные Python типы
        stats[col] = {
            'mean': float(data[col].mean()) if not pd.isna(data[col].mean()) else None,
            'std': float(data[col].std()) if not pd.isna(data[col].std()) else None,
            'min': float(data[col].min()) if not pd.isna(data[col].min()) else None,
            'max': float(data[col].max()) if not pd.isna(data[col].max()) else None,
            'count': int(data[col].count())
        }
    
    return stats

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

    # ВАЖНО: возвращаем render_template в конце функции
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
        graph = generate_graph_simple(measured_data)
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

def simple_filter(df, column, operator, value):
    """Простая и надежная фильтрация"""
    if column not in df.columns:
        print(f"=== СТОЛБЕЦ НЕ НАЙДЕН: {column}")
        return pd.DataFrame()
    
    try:
        # Для текстового поиска (составы)
        if column == 'composition':
            return df[df[column].astype(str).str.contains(value, case=False, na=False)]
        
        # Для числовых полей
        try:
            num_value = float(value)
            
            if operator == '=':
                return df[df[column] == num_value]
            elif operator == '>':
                return df[df[column] > num_value]
            elif operator == '>=':
                return df[df[column] >= num_value]
            elif operator == '<':
                return df[df[column] < num_value]
            elif operator == '<=':
                return df[df[column] <= num_value]
            elif operator == '!=':
                return df[df[column] != num_value]
            elif operator == 'LIKE':
                # Для числовых полей LIKE не имеет смысла, ищем точное совпадение
                return df[df[column] == num_value]
                
        except ValueError:
            # Если не удалось преобразовать в число, ищем как текст
            if operator == 'LIKE' or operator == '=':
                return df[df[column].astype(str).str.contains(value, case=False, na=False)]
            else:
                # Для других операторов с текстом возвращаем пустой результат
                return pd.DataFrame()
                
    except Exception as e:
        print(f"=== ОШИБКА ФИЛЬТРАЦИИ: {e}")
    
    return pd.DataFrame()

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
        graph = generate_graph_simple(measured_data)
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
        composition1 = request.form.get('comp1')
        composition2 = request.form.get('comp2')
        
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
        
        measured_data = query_db(db_path, "measured_parameters")
        
        # Фильтруем данные по выбранным составам
        comparison_data = pd.DataFrame()
        for comp in compositions:
            comp_data = measured_data[measured_data['composition'] == comp]
            if not comp_data.empty:
                comparison_data = pd.concat([comparison_data, comp_data])
        
        if comparison_data.empty:
            return jsonify({
                'success': False,
                'message': 'Выбранные составы не найдены в базе данных.'
            })
        
        # Получаем настройки фильтрации
        show_all = request.form.get('show_all') == 'on'
        show_diff = request.form.get('show_diff') == 'on'
        param_group = request.form.get('paramGroup', 'all')
        
        # Применяем фильтрацию параметров
        filtered_data = filter_parameters(comparison_data, param_group, show_diff)
        
        return jsonify({
            'success': True,
            'message': f'Сравнение {len(compositions)} составов выполнено успешно.',
            'comparison': filtered_data.to_html(classes='table table-striped table-sm comparison-table', index=False),
            'compositions': compositions,
            'stats': get_comparison_stats(comparison_data)
        })
        
    except Exception as e:
        print(f"=== ОШИБКА СРАВНЕНИЯ: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Ошибка при сравнении: {str(e)}'
        })

def filter_parameters(data, param_group, show_diff_only=False):
    """Фильтрует параметры по группам"""
    # Группы параметров
    param_groups = {
        'thermal': ['q', 'tign', 'tb', 'tau_b', 'tau_d1', 'tau_d2'],
        'mechanical': ['density', 'kf', 'kt', 'h', 'mass_loss'],
        'chemical': ['ad', 'cd', 'hd', 'nd', 'sd', 'od', 'vd'],
        'emissions': ['co2', 'co', 'so2', 'nox'],
        'combustion': ['mass_loss', 'tau_b', 'tau_d1', 'tau_d2', 'tign', 'tb']
    }
    
    if param_group != 'all' and param_group in param_groups:
        selected_params = ['composition'] + param_groups[param_group]
        # Оставляем только существующие колонки
        existing_params = [col for col in selected_params if col in data.columns]
        data = data[existing_params]
    
    return data

def get_comparison_stats(data):
    """Возвращает статистику для сравнения"""
    if data.empty:
        return {}
    
    numeric_data = data.select_dtypes(include=[np.number])
    stats = {
        'compositions_count': len(data['composition'].unique()),
        'parameters_count': len(numeric_data.columns),
        'total_rows': len(data)
    }
    
    return stats

@app.route('/ai_analysis')
def ai_analysis():
    uploaded_files = os.listdir(app.config['UPLOAD_FOLDER']) if os.path.exists(app.config['UPLOAD_FOLDER']) else []
    return render_template('ai_analysis.html',segment='ИИ-анализ', uploaded_files=uploaded_files)

@app.route('/ai_analysis', methods=['POST'])
def perform_ai_analysis():
    return jsonify({
        'success': True,
        'message': 'Анализ ИИ пока не реализован. Это заглушка.'
    })

@app.route('/create_graph', methods=['GET', 'POST'])
def create_graph():
    uploaded_files = os.listdir(app.config['UPLOAD_FOLDER']) if os.path.exists(app.config['UPLOAD_FOLDER']) else []
    measured_data = query_db(db_path, "measured_parameters")
    parameters = measured_data.columns.tolist() if not measured_data.empty else []
    graphs = ['scatter', 'line', 'histogram', 'bar', 'box', 'heatmap', '3d_scatter']
    messages = []

    if request.method == 'POST':
        try:
            graph_type = request.form.get('graph_type', 'scatter')
            x_param = request.form.get('x_param', 'ad')
            y_param = request.form.get('y_param', 'q')
            z_param = request.form.get('z_param', '')
            color_param = request.form.get('color_param', '')
            theme = request.form.get('theme', 'default')
            title = request.form.get('title', '')
            width = int(request.form.get('width', 800))
            height = int(request.form.get('height', 600))
            show_grid = request.form.get('show_grid') == 'on'
            selected_file = request.form.get('selected_file')
            
            stats = get_data_statistics(measured_data)

            if measured_data.empty:
                return jsonify({
                    'success': False,
                    'message': 'Нет данных для построения графика. Сначала загрузите файл.',
                    'stats': stats
                })
            
            # Создаем график из существующих данных
            graph, graph_message = generate_graph(
                measured_data, x_param, y_param, graph_type,
                z_param, color_param, None, None,
                theme, title, width, height, show_grid
            )
            
            if not graph:
                return jsonify({
                    'success': False,
                    'message': graph_message or 'Не удалось создать график',
                    'stats': stats
                })
                
            return jsonify({
                'success': True,
                'message': graph_message,
                'graph': graph,
                'stats': stats
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'message': f'Ошибка при создании графика: {str(e)}',
                'stats': get_data_statistics(measured_data)
            })

    # GET запрос
    graph, _ = generate_graph(measured_data)
    stats = get_data_statistics(measured_data)
    
    return render_template(
        'create_graph.html',
        segment='Создание графика',
        uploaded_files=uploaded_files,
        parameters=parameters,
        graphs=graphs,
        messages=messages,
        graph=graph,
        components_sheet_name=session.get('components_sheet_name', 'Таблица компонентов'),
        stats=stats
    )

@app.route('/<path:path>.map')
def ignore_map_files(path):
    return '', 204

if __name__ == '__main__':
    app.run(debug=True)