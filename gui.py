from flask import Flask, render_template, request, redirect, url_for, send_file, session, flash
import os
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from werkzeug.utils import secure_filename
from data_processor import process_data_source
from database import init_db, query_db, insert_data
from ai_integration import ask_ai
import base64
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'Uploads'
app.config['ALLOWED_EXTENSIONS'] = {'.csv', '.xlsx'}
app.config['SECRET_KEY'] = 'your_secret_key_here'

db_path = "pellets_data.db"
init_db(db_path)  # Создание таблиц при старте

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def allowed_file(filename):
    if not filename:
        print("Filename is empty or None")
        return False
    file_ext = os.path.splitext(filename)[1].lower()
    is_allowed = file_ext in app.config['ALLOWED_EXTENSIONS']
    print(f"Checking file: {filename}, extension: {file_ext}, allowed: {is_allowed}")
    return is_allowed

def generate_graph(df):
    if df.empty:
        print("Пропуск генерации графика: DataFrame пустой")
        return None
    if 'ad' not in df.columns or 'q' not in df.columns:
        print(f"Пропуск генерации графика: отсутствуют столбцы 'ad' или 'q'. Доступные столбцы: {list(df.columns)}")
        return None
    try:
        start_time = time.time()
        plt.figure(figsize=(6, 4), dpi=80)
        plt.scatter(df['ad'], df['q'], s=30)
        plt.title('Q vs Ad')
        plt.xlabel('Ad, %')
        plt.ylabel('Q, МДж/кг')
        plt.grid(True)
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()
        print(f"Генерация графика заняла {time.time() - start_time:.2f} секунд")
        return img_base64
    except Exception as e:
        print(f"Ошибка генерации графика: {str(e)}")
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    show_data = session.get('show_data', False)
    try:
        start_time = time.time()
        measured_data = query_db(db_path, "measured_parameters")
        components_data = query_db(db_path, "components")
        print(f"Запрос к базе (measured_parameters) занял {time.time() - start_time:.2f} секунд")
        print(f"Запрос к базе (components) занял {time.time() - start_time:.2f} секунд")
        print(f"Столбцы measured_data: {list(measured_data.columns)}")
        print(f"Столбцы components_data: {list(components_data.columns)}")
    except Exception as e:
        flash(f"Ошибка запроса к базе данных: {str(e)}", "danger")
        measured_data = pd.DataFrame()
        components_data = pd.DataFrame()

    filtered_data = measured_data.copy()

    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if not file or file.filename == '':
                flash("Файл не выбран.", "danger")
                print("Файл не выбран")
            elif file and allowed_file(file.filename):
                # Используем оригинальное имя файла с поддержкой UTF-8
                original_filename = file.filename  # Не используем secure_filename, если UTF-8 важен
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
                try:
                    start_time = time.time()
                    file.save(file_path)
                    print(f"Сохранен файл: {file_path} (имя: {original_filename})")
                    if os.path.exists(file_path):
                        process_messages = process_data_source(file_path, db_path)
                        print(f"Сообщения обработки: {process_messages}")
                        for msg in process_messages:
                            category = "danger" if "Error" in msg or "Warning" in msg else "success"
                            flash(msg, category)
                        
                        # Перезапрашиваем данные
                        try:
                            measured_data = query_db(db_path, "measured_parameters")
                            components_data = query_db(db_path, "components")
                            filtered_data = measured_data.copy()
                            print(f"Столбцы measured_data после обработки: {list(measured_data.columns)}")
                            print(f"Столбцы components_data после обработки: {list(components_data.columns)}")
                            if measured_data.empty and components_data.empty:
                                flash("Данные обработаны, но таблицы пусты. Проверьте формат файла или названия столбцов.", "danger")
                            else:
                                flash("Файл успешно обработан. Данные загружены.", "success")
                                show_data = True
                                session['show_data'] = True
                            print(f"Обработка файла заняла {time.time() - start_time:.2f} секунд")
                        except Exception as e:
                            flash(f"Ошибка запроса к базе данных после загрузки: {str(e)}", "danger")
                            print(f"Ошибка запроса к базе после обработки: {str(e)}")
                    else:
                        flash(f"Файл не сохранен: {file_path}", "danger")
                        print(f"Файл не сохранен: {file_path}")
                except Exception as e:
                    flash(f"Ошибка сохранения файла: {str(e)}", "danger")
                    print(f"Ошибка сохранения файла: {str(e)}")
            else:
                flash(f"Недопустимый формат файла: {file.filename}. Разрешены только .csv и .xlsx.", "danger")
                print(f"Недопустимый формат файла: {file.filename}")
        
        elif 'search' in request.form:
            start_time = time.time()
            rho = request.form.get('rho')
            kt = request.form.get('kt')
            ad = request.form.get('ad')
            q = request.form.get('q')
            num_components = request.form.get('num_components')
            
            if not measured_data.empty:
                try:
                    if rho:
                        filtered_data = filtered_data[filtered_data['density'] >= float(rho)]
                    if kt:
                        filtered_data = filtered_data[filtered_data['kt'] >= float(kt)]
                    if ad:
                        filtered_data = filtered_data[filtered_data['ad'] <= float(ad)]
                    if q:
                        filtered_data = filtered_data[filtered_data['q'] >= float(q)]
                    if num_components:
                        filtered_data = filtered_data[filtered_data['composition'].str.count('%') == int(num_components) + 1]
                    flash("Поиск выполнен успешно.", "success")
                except Exception as e:
                    flash(f"Ошибка при фильтрации данных: {str(e)}", "danger")
            else:
                flash("Нет данных для фильтрации.", "danger")
            print(f"Search processing took {time.time() - start_time:.2f} seconds")

    # Подготовка данных для отображения
    start_time = time.time()
    measured_data_html = filtered_data.to_html(classes='table table-striped', index=False) if not filtered_data.empty else "<p class= card-header pb-0 >Нет данных для отображения в таблице измеренных параметров.</p>"
    components_data_html = components_data.to_html(classes='table table-striped', index=False) if not components_data.empty else "<p class= card-header pb-0 >Нет данных для отображения в таблице измеренных параметров.</p>"
    print(f"HTML table generation took {time.time() - start_time:.2f} seconds")
    
    graph = generate_graph(filtered_data)
    
    uploaded_files = os.listdir(app.config['UPLOAD_FOLDER']) if os.path.exists(app.config['UPLOAD_FOLDER']) else []
    
    return render_template('index.html', 
                           segment='dashboard',
                           measured_data=measured_data_html,
                           components_data=components_data_html,
                           graph=graph,
                           uploaded_files=uploaded_files,
                           show_data=show_data)

@app.route('/create_graph', methods=['GET', 'POST'])
def create_graph():
    uploaded_files = os.listdir(app.config['UPLOAD_FOLDER']) if os.path.exists(app.config['UPLOAD_FOLDER']) else []
    graphs = ["Q vs Ad", "CO2 vs Density", "Kf vs Kt"]
    
    messages = []
    if request.method == 'POST':
        graph_type = request.form.get('graph_type')
        x_param = request.form.get('x_param')
        y_param = request.form.get('y_param')
        file = request.form.get('file')
        messages = [f"График {graph_type} создан по параметрам {x_param} vs {y_param} из файла {file}"]
    
    return render_template('create_graph.html', 
                           segment='create_graph',
                           graphs=graphs, 
                           uploaded_files=uploaded_files, 
                           messages=messages)

@app.route('/ai_analysis')
def ai_analysis():
    try:
        start_time = time.time()
        measured_data = query_db(db_path, "measured_parameters")
        print(f"Database query (ai_analysis) took {time.time() - start_time:.2f} seconds")
    except Exception as e:
        flash(f"Ошибка запроса к базе данных: {str(e)}", "danger")
        return render_template('ai_analysis.html', 
                              segment='ai_analysis', 
                              ai_response="Ошибка запроса к базе данных.")
    
    if measured_data.empty:
        flash("Нет данных для анализа.", "danger")
        return render_template('ai_analysis.html', 
                              segment='ai_analysis', 
                              ai_response="Нет данных для анализа.")
    
    ai_query = "Проанализируйте тренды теплоты сгорания (q) в следующих данных: " + measured_data[['composition', 'q']].head(10).to_string()
    try:
        start_time = time.time()
        ai_response = ask_ai(ai_query, use_openai=True)
        print(f"AI analysis took {time.time() - start_time:.2f} seconds")
        return render_template('ai_analysis.html', 
                              segment='ai_analysis', 
                              ai_response=ai_response)
    except Exception as e:
        flash(f"Ошибка анализа AI: {str(e)}", "danger")
        return render_template('ai_analysis.html', 
                              segment='ai_analysis', 
                              ai_response="Ошибка анализа AI.")

@app.route('/compare', methods=['GET', 'POST'])
def compare():
    uploaded_files = os.listdir(app.config['UPLOAD_FOLDER']) if os.path.exists(app.config['UPLOAD_FOLDER']) else []
    selected_file = request.form.get('file') if request.method == 'POST' else uploaded_files[0] if uploaded_files else None
    compositions = []
    comparison_html = None
    if selected_file:
        try:
            start_time = time.time()
            measured_data = query_db(db_path, "measured_parameters")
            print(f"Database query (compare) took {time.time() - start_time:.2f} seconds")
            compositions = measured_data['composition'].tolist()
        except Exception as e:
            flash(f"Ошибка запроса к базе данных: {str(e)}", "danger")
            measured_data = pd.DataFrame()
    
    if request.method == 'POST':
        comp1 = request.form.get('comp1')
        comp2 = request.form.get('comp2')
        show_all = request.form.get('show_all')
        show_diff = request.form.get('show_diff')
        if comp1 and comp2:
            try:
                df1 = measured_data[measured_data['composition'] == comp1]
                df2 = measured_data[measured_data['composition'] == comp2]
                if show_all:
                    comparison_html = pd.concat([df1, df2]).to_html(classes='table table-striped', index=False)
                elif show_diff:
                    diff = df1.ne(df2)
                    comparison_html = df1[diff].to_html(classes='table table-striped', index=False)
                else:
                    comparison_html = pd.concat([df1, df2]).to_html(classes='table table-striped', index=False)
                flash("Сравнение выполнено успешно.", "success")
            except Exception as e:
                flash(f"Ошибка при сравнении: {str(e)}", "danger")
    
    return render_template('compare.html', 
                          segment='compare',
                          comparison=comparison_html, 
                          uploaded_files=uploaded_files, 
                          compositions=compositions, 
                          file=selected_file)

@app.route('/export/<table>', methods=['GET'])
def export(table):
    try:
        start_time = time.time()
        df = query_db(db_path, table)
        print(f"Database query (export) took {time.time() - start_time:.2f} seconds")
    except Exception as e:
        flash(f"Ошибка запроса к базе данных: {str(e)}", "danger")
        return "Ошибка запроса к базе данных."
    if df.empty:
        flash("Нет данных для экспорта.", "danger")
        return "Нет данных для экспорта."
    
    csv_path = f"{table}.csv"
    df.to_csv(csv_path, index=False)
    return send_file(csv_path, as_attachment=True)

@app.route('/add_data', methods=['POST'])
def add_data():
    try:
        start_time = time.time()
        composition = request.form.get('composition')
        density = request.form.get('density')
        kf = request.form.get('kf')
        kt = request.form.get('kt')
        h = request.form.get('h')
        mass_loss = request.form.get('mass_loss')
        tign = request.form.get('tign')
        tb = request.form.get('tb')
        tau_d1 = request.form.get('tau_d1')
        tau_d2 = request.form.get('tau_d2')
        tau_b = request.form.get('tau_b')
        co2 = request.form.get('co2')
        co = request.form.get('co')
        so2 = request.form.get('so2')
        nox = request.form.get('nox')
        q = request.form.get('q')
        ad = request.form.get('ad')
        
        df = pd.DataFrame({
            'composition': [composition],
            'density': [float(density) if density else None],
            'kf': [float(kf) if kf else None],
            'kt': [float(kt) if kt else None],
            'h': [float(h) if h else None],
            'mass_loss': [float(mass_loss) if mass_loss else None],
            'tign': [float(tign) if tign else None],
            'tb': [float(tb) if tb else None],
            'tau_d1': [float(tau_d1) if tau_d1 else None],
            'tau_d2': [float(tau_d2) if tau_d2 else None],
            'tau_b': [float(tau_b) if tau_b else None],
            'co2': [float(co2) if co2 else None],
            'co': [float(co) if co else None],
            'so2': [float(so2) if so2 else None],
            'nox': [float(nox) if nox else None],
            'q': [float(q) if q else None],
            'ad': [float(ad) if ad else None]
        })
        insert_data(db_path, "measured_parameters", df)
        flash("Данные успешно добавлены.", "success")
        print(f"Data insertion took {time.time() - start_time:.2f} seconds")
    except Exception as e:
        flash(f"Ошибка добавления данных: {str(e)}", "danger")
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=False)