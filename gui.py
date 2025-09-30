from flask import Flask, render_template, request, redirect, url_for, send_file, session
import os
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from werkzeug.utils import secure_filename
from data_processor import process_data_source
from database import query_db, insert_data
from ai_integration import ask_ai
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'xlsx'}
app.config['SECRET_KEY'] = 'your_secret_key_here'  # Добавь для работы session

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def generate_graph(df):
    if df.empty:
        return None
    plt.figure(figsize=(8, 6))
    plt.scatter(df['ad'], df['q'])
    plt.title('Q vs Ad')
    plt.xlabel('Ad, %')
    plt.ylabel('Q, МДж/кг')
    plt.grid(True)
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()
    return img_base64

@app.route('/', methods=['GET', 'POST'])
def index():
    db_path = "pellets_data.db"
    messages = []
    show_data = 'show_data' in session or request.method == 'POST' and 'file' in request.files
    
    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                messages.append("No file selected.")
            elif file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                if not filename.endswith(('.csv', '.xlsx')):
                    filename += '.' + file.filename.rsplit('.', 1)[1].lower()
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                print(f"Saving file to: {file_path}")
                file.save(file_path)
                if os.path.exists(file_path):
                    print(f"File exists, processing: {file_path}")
                    messages.extend(process_data_source(file_path, db_path))
                    show_data = True
                else:
                    messages.append(f"File not saved: {file_path}")
            else:
                messages.append("Invalid file format. Please upload a CSV or Excel file.")
        
        elif 'search' in request.form:
            rho = request.form.get('rho')
            kt = request.form.get('kt')
            ad = request.form.get('ad')
            q = request.form.get('q')
            num_components = request.form.get('num_components')
            
            measured_data = query_db(db_path, "measured_parameters")
            if not measured_data.empty:
                if rho:
                    measured_data = measured_data[measured_data['density'] >= float(rho)]
                if kt:
                    measured_data = measured_data[measured_data['kt'] >= float(kt)]
                if ad:
                    measured_data = measured_data[measured_data['ad'] <= float(ad)]
                if q:
                    measured_data = measured_data[measured_data['q'] >= float(q)]
                if num_components:
                    measured_data = measured_data[measured_data['composition'].str.count('%') == int(num_components) + 1]
                messages.append("Search applied successfully.")
            else:
                messages.append("No data to filter.")

    measured_data = query_db(db_path, "measured_parameters")
    components_data = query_db(db_path, "components")
    
    measured_data_html = measured_data.to_html(classes='table table-striped', index=False) if not measured_data.empty else "<p>No data in measured_parameters</p>"
    components_data_html = components_data.to_html(classes='table table-striped', index=False) if not components_data.empty else "<p>No data in components</p>"
    
    graph = generate_graph(measured_data)
    
    uploaded_files = os.listdir(app.config['UPLOAD_FOLDER'])
    
    session['show_data'] = show_data
    return render_template('index.html', 
                           segment='dashboard',  # Добавлено
                           measured_data=measured_data_html,
                           components_data=components_data_html,
                           graph=graph,
                           messages=messages,
                           uploaded_files=uploaded_files,
                           show_data=show_data)

@app.route('/create_graph', methods=['GET', 'POST'])
def create_graph():
    db_path = "pellets_data.db"
    uploaded_files = os.listdir(app.config['UPLOAD_FOLDER'])
    graphs = ["Q vs Ad", "CO2 vs Density", "Kf vs Kt"]
    
    if request.method == 'POST':
        graph_type = request.form.get('graph_type')
        x_param = request.form.get('x_param')
        y_param = request.form.get('y_param')
        file = request.form.get('file')
        messages = [f"График {graph_type} создан по параметрам {x_param} vs {y_param} из файла {file}"]
        return render_template('create_graph.html', segment='create_graph',  # Добавлено
                               graphs=graphs, uploaded_files=uploaded_files, messages=messages)
    
    return render_template('create_graph.html', segment='create_graph',  # Добавлено
                           graphs=graphs, uploaded_files=uploaded_files)

@app.route('/ai_analysis')
def ai_analysis():
    db_path = "pellets_data.db"
    measured_data = query_db(db_path, "measured_parameters")
    if measured_data.empty:
        return render_template('ai_analysis.html', segment='ai_analysis',  # Добавлено
                              ai_response="No data for analysis.")
    
    ai_query = "Analyze the heat of combustion (q) trends in the following data: " + measured_data[['composition', 'q']].to_string()
    try:
        ai_response = ask_ai(ai_query, use_openai=True)
        return render_template('ai_analysis.html', segment='ai_analysis',  # Добавлено
                              ai_response=ai_response)
    except Exception as e:
        return render_template('ai_analysis.html', segment='ai_analysis',  # Добавлено
                              ai_response=f"AI analysis failed: {e}")

@app.route('/compare', methods=['GET', 'POST'])
def compare():
    db_path = "pellets_data.db"
    uploaded_files = os.listdir(app.config['UPLOAD_FOLDER'])
    file = request.form.get('file') if request.method == 'POST' else uploaded_files[0] if uploaded_files else None
    compositions = []
    if file:
        measured_data = query_db(db_path, "measured_parameters")
        compositions = measured_data['composition'].tolist()
    
    if request.method == 'POST':
        comp1 = request.form.get('comp1')
        comp2 = request.form.get('comp2')
        show_all = request.form.get('show_all')
        if comp1 and comp2:
            df1 = measured_data[measured_data['composition'] == comp1]
            df2 = measured_data[measured_data['composition'] == comp2]
            if show_all:
                comparison_html = pd.concat([df1, df2]).to_html(classes='table table-striped', index=False)
            else:
                diff = df1.ne(df2)
                comparison_html = df1[diff].to_html(classes='table table-striped', index=False)
            return render_template('compare.html', segment='compare',  # Добавлено
                                  comparison=comparison_html, uploaded_files=uploaded_files, compositions=compositions, file=file)
    
    return render_template('compare.html', segment='compare',  # Добавлено
                          uploaded_files=uploaded_files, compositions=compositions, file=file)

@app.route('/export/<table>', methods=['GET'])
def export(table):
    db_path = "pellets_data.db"
    df = query_db(db_path, table)
    if df.empty:
        return "No data to export."
    
    csv_path = f"{table}.csv"
    df.to_csv(csv_path, index=False)
    return send_file(csv_path, as_attachment=True)

@app.route('/add_data', methods=['POST'])
def add_data():
    db_path = "pellets_data.db"
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
        'density': [float(density)],
        'kf': [float(kf)],
        'kt': [float(kt)],
        'h': [float(h)],
        'mass_loss': [float(mass_loss)],
        'tign': [float(tign)],
        'tb': [float(tb)],
        'tau_d1': [float(tau_d1)],
        'tau_d2': [float(tau_d2)],
        'tau_b': [float(tau_b)],
        'co2': [float(co2)],
        'co': [float(co)],
        'so2': [float(so2)],
        'nox': [float(nox)],
        'q': [float(q)],
        'ad': [float(ad)]
    })
    insert_data(db_path, "measured_parameters", df)
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)