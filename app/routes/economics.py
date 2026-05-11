# app/routes/economics.py — Экономический расчёт
from flask import Blueprint, request, jsonify, send_file
import pandas as pd
import io
import math
import time
import logging
import threading
from collections import deque
from datetime import datetime
from fpdf import FPDF
from app.auth.auth import login_required
from app.database.database import query_db

economics_bp = Blueprint('economics', __name__)
logger = logging.getLogger(__name__)

_db_path = 'pellets_data.db'
_db_component_map_cache = None
_db_component_map_cache_db_path = None
_db_component_map_cache_lock = threading.Lock()
_endpoint_timings = {
    '/api/calculate_economics': deque(maxlen=200),
    '/api/compare_economics': deque(maxlen=200),
}


def _percentile(values, q):
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = int(round((q / 100.0) * (len(sorted_values) - 1)))
    return float(sorted_values[idx])


def _record_timing(endpoint, elapsed_sec):
    timings = _endpoint_timings.get(endpoint)
    if timings is None:
        return
    elapsed_ms = elapsed_sec * 1000.0
    timings.append(elapsed_ms)
    p50 = _percentile(list(timings), 50)
    p95 = _percentile(list(timings), 95)
    logger.info("%s latency_ms=%.2f p50=%.2f p95=%.2f n=%s", endpoint, elapsed_ms, p50, p95, len(timings))


def _invalidate_component_map_cache():
    global _db_component_map_cache, _db_component_map_cache_db_path
    with _db_component_map_cache_lock:
        _db_component_map_cache = None
        _db_component_map_cache_db_path = None

def set_db_path(path):
    global _db_path
    if _db_path != path:
        _invalidate_component_map_cache()
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


def _build_db_component_map(force_refresh=False):
    global _db_component_map_cache, _db_component_map_cache_db_path
    with _db_component_map_cache_lock:
        if (
            not force_refresh
            and _db_component_map_cache is not None
            and _db_component_map_cache_db_path == _db_path
        ):
            return _db_component_map_cache

    db_components = query_db(_db_path, "components")
    component_map = {str(row['component']).strip(): row for _, row in db_components.iterrows()} if not db_components.empty else {}
    with _db_component_map_cache_lock:
        _db_component_map_cache = component_map
        _db_component_map_cache_db_path = _db_path
    return component_map


def _resolve_component_metrics(comp_name: str, db_map: dict):
    comp_name_clean = str(comp_name).strip()
    if comp_name_clean in COMPONENTS_DATA:
        c_data = COMPONENTS_DATA[comp_name_clean]
        return float(c_data['Ro']), float(c_data['Qai']), float(c_data['Cc']), float(c_data['Cu']), float(c_data['Cg'])

    db_row = db_map.get(comp_name_clean, {})
    ro = float(db_row.get('ro', 1000.0)) if not pd.isna(db_row.get('ro')) else 1000.0
    qai = float(db_row.get('q', 0.0)) if not pd.isna(db_row.get('q')) else 0.0
    cc = float(db_row.get('cost_raw', 0.0)) if not pd.isna(db_row.get('cost_raw')) else 0.0
    cu = float(db_row.get('cost_crush', 0.0)) if not pd.isna(db_row.get('cost_crush')) else 0.0
    cg = float(db_row.get('cost_granule', 0.0)) if not pd.isna(db_row.get('cost_granule')) else 0.0
    return ro, qai, cc, cu, cg


def _compute_economics_result(q_boiler, t_hours, distance, capacity_factor, efficiency, components, db_map):
    COEFF_C = 3.6
    RENT_COST_PER_M2 = 380.0
    STORAGE_LOAD_KG_PER_M2 = 1100.0
    TRUCK_BASE_RATE = 6000.0
    TRUCK_COST_PER_KM = 85.0
    TRUCK_CAPACITY_KG = 20000.0

    mix_ro = mix_qai = mix_cc = mix_cu = mix_cg = 0.0
    for comp_name, percentage in components.items():
        fraction = float(percentage) / 100.0
        ro, qai, cc, cu, cg = _resolve_component_metrics(comp_name, db_map)
        mix_ro += fraction * ro
        mix_qai += fraction * qai
        mix_cc += fraction * cc
        mix_cu += fraction * cu
        mix_cg += fraction * cg

    if mix_ro <= 0 or mix_qai <= 0:
        raise ValueError('Некорректный состав или отсутствуют данные о теплоте сгорания/плотности')
    if efficiency <= 0 or efficiency > 1:
        raise ValueError('КПД должен быть в диапазоне (0, 1]')

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
    return {
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
        'production': {'cost_per_kg': round(production_cost_per_kg, 2), 'cost': round(production_cost, 2)},
        'transport': {'trip_cost': round(trip_cost, 2), 'trucks': int(trucks_needed), 'cost': round(transport_cost, 2)},
        'total_cost': round(total_cost, 2)
    }


def _extract_global_params(payload: dict):
    global_params = payload.get('global_params', payload)
    try:
        params = {
            'q_boiler': float(global_params.get('q_boiler', 80)),
            't_hours': float(global_params.get('t_hours', 720)),
            'distance': float(global_params.get('distance', 258)),
            'capacity_factor': float(global_params.get('capacity_factor', 0.8)),
            'efficiency': float(global_params.get('efficiency', 0.9)),
        }
    except (TypeError, ValueError):
        raise ValueError('Некорректные числовые параметры расчета')

    if params['q_boiler'] <= 0:
        raise ValueError('Мощность котла должна быть больше 0')
    if params['t_hours'] <= 0:
        raise ValueError('Время работы должно быть больше 0')
    if params['distance'] < 0:
        raise ValueError('Расстояние не может быть отрицательным')
    if params['capacity_factor'] <= 0 or params['capacity_factor'] > 1:
        raise ValueError('Коэффициент загрузки должен быть в диапазоне (0, 1]')
    if params['efficiency'] <= 0 or params['efficiency'] > 1:
        raise ValueError('КПД должен быть в диапазоне (0, 1]')

    return params


def _build_compare_rows(global_params: dict, scenarios: list):
    db_map = _build_db_component_map()
    rows = []
    for idx, scenario in enumerate(scenarios, start=1):
        scenario_name = str(scenario.get('name') or f'Вариант {idx}')
        components = scenario.get('components', {})
        if not components:
            rows.append({'name': scenario_name, 'success': False, 'error': 'Пустой состав'})
            continue
        try:
            result = _compute_economics_result(
                global_params['q_boiler'],
                global_params['t_hours'],
                global_params['distance'],
                global_params['capacity_factor'],
                global_params['efficiency'],
                components,
                db_map
            )
            rows.append({'name': scenario_name, 'success': True, 'components': components, 'result': result})
        except Exception as ex:
            rows.append({'name': scenario_name, 'success': False, 'components': components, 'error': str(ex)})
    return rows


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
        _invalidate_component_map_cache()
        return jsonify({'success': True, 'message': f'Цены успешно обновлены! Загружено компонентов: {len(df)}'})
    except Exception as e:
        conn.rollback()
        return jsonify({'success': False, 'error': f'Ошибка записи в БД: {str(e)}'})
    finally:
        conn.close()


@economics_bp.route('/api/calculate_economics', methods=['POST'])
@login_required
def calculate_economics():
    started_at = time.perf_counter()
    try:
        data = request.get_json(silent=True) or {}
        if not isinstance(data, dict):
            return jsonify({'success': False, 'error': 'Некорректный формат запроса'}), 400

        params = _extract_global_params(data)
        components = data.get('components', {})
        if not isinstance(components, dict) or not components:
            return jsonify({'success': False, 'error': 'Не указан состав компонентов'}), 400

        result = _compute_economics_result(
            params['q_boiler'],
            params['t_hours'],
            params['distance'],
            params['capacity_factor'],
            params['efficiency'],
            components,
            _build_db_component_map()
        )
        return jsonify({'success': True, **result})
    except Exception as e:
        import traceback
        print(f"Error calculating economics: {traceback.format_exc()}")
        return jsonify({'success': False, 'error': f'Ошибка расчета: {str(e)}'})
    finally:
        _record_timing('/api/calculate_economics', time.perf_counter() - started_at)


@economics_bp.route('/api/compare_economics', methods=['POST'])
@login_required
def compare_economics():
    started_at = time.perf_counter()
    try:
        data = request.get_json(silent=True) or {}
        scenarios = data.get('scenarios', [])
        if not isinstance(scenarios, list) or len(scenarios) == 0:
            return jsonify({'success': False, 'error': 'Добавьте хотя бы один сценарий для сравнения'})
        params = _extract_global_params(data)
        compare_rows = _build_compare_rows(params, scenarios)
        valid_rows = [row for row in compare_rows if row.get('success')]
        min_total = min((row['result']['total_cost'] for row in valid_rows), default=None)
        best_name = next((row['name'] for row in valid_rows if row['result']['total_cost'] == min_total), None)
        return jsonify({
            'success': True,
            'params': params,
            'scenarios': compare_rows,
            'summary': {'best_total_cost': min_total, 'best_scenario': best_name, 'count': len(compare_rows)}
        })
    except Exception as e:
        return jsonify({'success': False, 'error': f'Ошибка сравнения: {str(e)}'})
    finally:
        _record_timing('/api/compare_economics', time.perf_counter() - started_at)


def _build_compare_export_dataframe(params: dict, compare_rows: list):
    rows = []
    for row in compare_rows:
        if not row.get('success'):
            rows.append({
                'Сценарий': row.get('name'),
                'Статус': 'Ошибка',
                'Ошибка': row.get('error', '')
            })
            continue
        r = row['result']
        rows.append({
            'Сценарий': row['name'],
            'Статус': 'OK',
            'Итоговая стоимость, руб': r['total_cost'],
            'Масса топлива, кг': r['storage']['mass_kg'],
            'Объем, м3': r['storage']['volume_m3'],
            'Площадь хранения, м2': r['storage']['area_m2'],
            'Стоимость хранения, руб': r['storage']['cost'],
            'Стоимость производства, руб': r['production']['cost'],
            'Стоимость транспорта, руб': r['transport']['cost'],
            'Себестоимость, руб/кг': r['production']['cost_per_kg'],
            'Теплота сгорания, МДж/кг': r['mix_metrics']['heat_capacity'],
            'Плотность, кг/м3': r['mix_metrics']['density']
        })
    df = pd.DataFrame(rows)
    params_df = pd.DataFrame([{
        'q_boiler': params['q_boiler'],
        't_hours': params['t_hours'],
        'distance': params['distance'],
        'capacity_factor': params['capacity_factor'],
        'efficiency': params['efficiency'],
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M')
    }])
    return df, params_df


@economics_bp.route('/api/export_economics_compare_excel', methods=['POST'])
@login_required
def export_economics_compare_excel():
    try:
        data = request.json or {}
        scenarios = data.get('scenarios', [])
        if not isinstance(scenarios, list) or len(scenarios) == 0:
            return jsonify({'success': False, 'error': 'Нет сценариев для выгрузки'})
        params = _extract_global_params(data)
        compare_rows = _build_compare_rows(params, scenarios)
        results_df, params_df = _build_compare_export_dataframe(params, compare_rows)

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            results_df.to_excel(writer, index=False, sheet_name='Сравнение')
            params_df.to_excel(writer, index=False, sheet_name='Параметры')

            ws = writer.sheets['Сравнение']
            for i, col in enumerate(results_df.columns, start=1):
                max_len = max(14, len(str(col)) + 2)
                ws.column_dimensions[chr(64 + i)].width = max_len
        output.seek(0)
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=f'Сравнение_экономики_{datetime.now().strftime("%Y%m%d_%H%M")}.xlsx'
        )
    except Exception as e:
        return jsonify({'success': False, 'error': f'Ошибка Excel-выгрузки: {str(e)}'})


@economics_bp.route('/api/export_economics_compare_pdf', methods=['POST'])
@login_required
def export_economics_compare_pdf():
    try:
        data = request.json or {}
        scenarios = data.get('scenarios', [])
        if not isinstance(scenarios, list) or len(scenarios) == 0:
            return jsonify({'success': False, 'error': 'Нет сценариев для выгрузки'})

        params = _extract_global_params(data)
        compare_rows = _build_compare_rows(params, scenarios)

        pdf = FPDF(orientation='L', unit='mm', format='A4')
        pdf.add_page()
        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 10, 'Economics compare report', new_x='LMARGIN', new_y='NEXT')
        pdf.set_font('Helvetica', '', 10)
        pdf.cell(
            0, 7,
            (
                f"q_boiler={params['q_boiler']}, t_hours={params['t_hours']}, distance={params['distance']}, "
                f"capacity_factor={params['capacity_factor']}, efficiency={params['efficiency']}"
            ),
            new_x='LMARGIN', new_y='NEXT'
        )
        pdf.ln(2)

        headers = ['Scenario', 'Status', 'Total cost', 'Mass kg', 'Storage', 'Production', 'Transport']
        col_widths = [55, 20, 35, 25, 35, 35, 35]

        pdf.set_font('Helvetica', 'B', 10)
        for idx, h in enumerate(headers):
            pdf.cell(col_widths[idx], 8, h, border=1)
        pdf.ln()

        pdf.set_font('Helvetica', '', 9)
        for row in compare_rows:
            if row.get('success'):
                result = row['result']
                values = [
                    row['name'],
                    'OK',
                    f"{result['total_cost']:.2f}",
                    f"{result['storage']['mass_kg']}",
                    f"{result['storage']['cost']:.2f}",
                    f"{result['production']['cost']:.2f}",
                    f"{result['transport']['cost']:.2f}",
                ]
            else:
                values = [row.get('name', ''), 'Error', '-', '-', '-', '-', '-']
            for idx, value in enumerate(values):
                pdf.cell(col_widths[idx], 8, str(value)[:40], border=1)
            pdf.ln()

        pdf_data = bytes(pdf.output(dest='S'))
        output = io.BytesIO(pdf_data)
        output.seek(0)
        return send_file(
            output,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'economics_compare_{datetime.now().strftime("%Y%m%d_%H%M")}.pdf'
        )
    except Exception as e:
        return jsonify({'success': False, 'error': f'Ошибка PDF-выгрузки: {str(e)}'})
