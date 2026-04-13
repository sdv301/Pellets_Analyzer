# app/routes/economics.py — Экономический расчёт
from flask import Blueprint, request, jsonify, send_file
import pandas as pd
import io
import math
from app.auth.auth import login_required
from app.models.database import query_db

economics_bp = Blueprint('economics', __name__)

_db_path = 'pellets_data.db'

def set_db_path(path):
    global _db_path
    _db_path = path

# Эталонные данные
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


@economics_bp.route('/api/get_components_economics')
@login_required
def get_components_economics():
    try:
        df = query_db(_db_path, "components")
        if df.empty:
            fallback = []
            for name, d in COMPONENTS_DATA.items():
                fallback.append({'component': name, 'ro': d['Ro'], 'cost_raw': d['Cc'], 'cost_crush': d['Cu'], 'cost_granule': d['Cg']})
            return jsonify({'success': True, 'components': fallback})

        result = []
        for _, row in df.iterrows():
            name = str(row['component']).strip()
            if name in COMPONENTS_DATA:
                d = COMPONENTS_DATA[name]
                ro, c_raw, c_crush, c_gran = d['Ro'], d['Cc'], d['Cu'], d['Cg']
            else:
                ro = float(row.get('ro', 1000.0)) if not pd.isna(row.get('ro')) else 1000.0
                c_raw = float(row.get('cost_raw', 0.0)) if not pd.isna(row.get('cost_raw')) else 0.0
                c_crush = float(row.get('cost_crush', 0.0)) if not pd.isna(row.get('cost_crush')) else 0.0
                c_gran = float(row.get('cost_granule', 0.0)) if not pd.isna(row.get('cost_granule')) else 0.0

            result.append({'component': str(row['component']), 'ro': float(ro), 'cost_raw': float(c_raw), 'cost_crush': float(c_crush), 'cost_granule': float(c_gran)})

        return jsonify({'success': True, 'components': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@economics_bp.route('/api/download_economics', methods=['GET'])
@login_required
def download_economics():
    try:
        excel_data = []
        for name, d in COMPONENTS_DATA.items():
            excel_data.append({
                'Компоненты': name, 'War, %': d['War'], 'Ad, %': d['Ad'],
                'Qas,V, МДж/кг': d['Qas'], 'Hd, %': d['Hd'], 'Qai,V, МДж/кг': d['Qai'],
                'Cc, руб/кг': d['Cc'], 'Cи, руб/кг': d['Cu'], 'Cг, руб/кг': d['Cg']
            })
        out_df = pd.DataFrame(excel_data)
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            out_df.to_excel(writer, index=False, sheet_name='Цены и Характеристики')
            worksheet = writer.sheets['Цены и Характеристики']
            for i, col in enumerate(out_df.columns):
                worksheet.column_dimensions[chr(65 + i)].width = 18
        output.seek(0)
        return send_file(output, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', as_attachment=True, download_name='Шаблон_Цен_Экономика.xlsx')
    except Exception as e:
        return jsonify({'success': False, 'error': f'Ошибка при формировании файла: {str(e)}'})


@economics_bp.route('/api/upload_economics', methods=['POST'])
@login_required
def upload_economics():
    import sqlite3
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'Файл не найден'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'Файл не выбран'})

    try:
        df = pd.read_excel(file)
    except Exception as e:
        return jsonify({'success': False, 'error': f'Не удалось прочитать Excel: {str(e)}'})

    required_cols = ['Компоненты', 'Qai,V, МДж/кг', 'Cc, руб/кг', 'Cи, руб/кг', 'Cг, руб/кг']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        return jsonify({'success': False, 'error': f'Неверный формат. Отсутствуют колонки: {", ".join(missing)}'})

    df['Компоненты'] = df['Компоненты'].astype(str).str.strip()
    df = df[df['Компоненты'].str.len() > 0]
    if df.empty:
        return jsonify({'success': False, 'error': 'Файл не содержит данных о компонентах'})

    numeric_cols = ['Qai,V, МДж/кг', 'Cc, руб/кг', 'Cи, руб/кг', 'Cг, руб/кг']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    conn = sqlite3.connect(_db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM components")
        for _, row in df.iterrows():
            name = str(row['Компоненты']).strip()
            ro = COMPONENTS_DATA.get(name, {}).get('Ro', 1000.0)
            cursor.execute("""
                INSERT INTO components (component, ro, q, cost_raw, cost_crush, cost_granule)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (name, float(ro), float(row['Qai,V, МДж/кг']), float(row['Cc, руб/кг']), float(row['Cи, руб/кг']), float(row['Cг, руб/кг']))
            )
        conn.commit()
        return jsonify({'success': True, 'message': f'Цены успешно обновлены! Загружено компонентов: {len(df)}'})
    except Exception as e:
        conn.rollback()
        return jsonify({'success': False, 'error': f'Ошибка записи в БД: {str(e)}'})
    finally:
        conn.close()


@economics_bp.route('/api/calculate_economics', methods=['POST'])
@login_required
def calculate_economics():
    try:
        data = request.json
        q_boiler = float(data.get('q_boiler', 80))
        t_hours = float(data.get('t_hours', 720))
        distance = float(data.get('distance', 258))
        capacity_factor = float(data.get('capacity_factor', 0.8))
        efficiency = float(data.get('efficiency', 0.9))
        components = data.get('components', {})

        COEFF_C = 3.6
        RENT_COST_PER_M2 = 380.0
        STORAGE_LOAD_KG_PER_M2 = 1100.0
        TRUCK_BASE_RATE = 6000.0
        TRUCK_COST_PER_KM = 85.0
        TRUCK_CAPACITY_KG = 20000.0

        db_components = query_db(_db_path, "components")
        db_map = {str(row['component']): row for _, row in db_components.iterrows()} if not db_components.empty else {}

        mix_ro = mix_qai = mix_cc = mix_cu = mix_cg = 0.0
        for comp_name, percentage in components.items():
            fraction = float(percentage) / 100.0
            comp_name_clean = str(comp_name).strip()
            if comp_name_clean in COMPONENTS_DATA:
                c_data = COMPONENTS_DATA[comp_name_clean]
                ro, qai, cc, cu, cg = c_data['Ro'], c_data['Qai'], c_data['Cc'], c_data['Cu'], c_data['Cg']
            else:
                db_row = db_map.get(comp_name, {})
                ro = float(db_row.get('ro', 1000.0)) if not pd.isna(db_row.get('ro')) else 1000.0
                qai = float(db_row.get('q', 0.0)) if not pd.isna(db_row.get('q')) else 0.0
                cc = float(db_row.get('cost_raw', 0.0)) if not pd.isna(db_row.get('cost_raw')) else 0.0
                cu = float(db_row.get('cost_crush', 0.0)) if not pd.isna(db_row.get('cost_crush')) else 0.0
                cg = float(db_row.get('cost_granule', 0.0)) if not pd.isna(db_row.get('cost_granule')) else 0.0

            mix_ro += fraction * float(ro)
            mix_qai += fraction * float(qai)
            mix_cc += fraction * float(cc)
            mix_cu += fraction * float(cu)
            mix_cg += fraction * float(cg)

        if mix_ro <= 0 or mix_qai <= 0:
            return jsonify({'success': False, 'error': 'Некорректный состав или отсутствуют данные о теплоте сгорания/плотности'})

        numerator = q_boiler * t_hours * capacity_factor * COEFF_C
        denominator = mix_qai * efficiency * mix_ro
        v_neobh_m3 = numerator / denominator
        mass_kg = round(v_neobh_m3 * mix_ro)
        v_neobh_m3 = mass_kg / mix_ro

        storage_area_m2 = math.ceil(mass_kg / STORAGE_LOAD_KG_PER_M2)
        storage_cost = RENT_COST_PER_M2 * storage_area_m2

        production_cost_per_kg = mix_cc + mix_cu + mix_cg
        production_cost = production_cost_per_kg * mass_kg

        trip_cost = TRUCK_BASE_RATE + (TRUCK_COST_PER_KM * distance * 2)
        trucks_needed = math.ceil(mass_kg / TRUCK_CAPACITY_KG)
        transport_cost = trip_cost * trucks_needed

        total_cost = production_cost + transport_cost + storage_cost

        return jsonify({
            'success': True,
            'mix_metrics': {'density': round(mix_ro, 0), 'heat_capacity': round(mix_qai, 2), 'cost_raw': round(mix_cc, 2), 'cost_crush': round(mix_cu, 2), 'cost_granule': round(mix_cg, 2)},
            'storage': {'volume_m3': round(v_neobh_m3, 2), 'mass_kg': int(mass_kg), 'area_m2': int(storage_area_m2), 'cost': round(storage_cost, 2)},
            'production': {'cost_per_kg': round(production_cost_per_kg, 2), 'cost': round(production_cost, 2)},
            'transport': {'trip_cost': round(trip_cost, 2), 'trucks': int(trucks_needed), 'cost': round(transport_cost, 2)},
            'total_cost': round(total_cost, 2)
        })
    except Exception as e:
        import traceback
        print(f"Error calculating economics: {traceback.format_exc()}")
        return jsonify({'success': False, 'error': f'Ошибка расчета: {str(e)}'})
