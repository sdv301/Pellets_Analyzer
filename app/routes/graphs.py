# app/routes/graphs.py — Создание графиков
from flask import Blueprint, render_template, request, jsonify, session
import json
import logging
import time
from app.auth.auth import login_required
from app.database.database import query_db
from app.services.gui import (
    generate_graph, generate_plotly_graph, generate_seaborn_plot,
    get_data_statistics, MATPLOTLIB_GRAPHS, PLOTLY_GRAPHS, SEABORN_GRAPHS, VIZ_TYPES
)

graphs_bp = Blueprint('graphs', __name__)
logger = logging.getLogger(__name__)

_db_path = 'pellets_data.db'

def set_db_path(path):
    global _db_path
    _db_path = path


def _resolve_3d_z_param(data, x_param, y_param, z_param):
    if data.empty:
        return z_param, []

    numeric_columns = data.select_dtypes(include='number').columns.tolist()
    candidates = [col for col in numeric_columns if col not in {x_param, y_param}]

    if z_param and z_param in candidates:
        return z_param, candidates
    if candidates:
        return candidates[0], candidates
    return z_param, candidates


@graphs_bp.route('/create_graph', methods=['GET', 'POST'])
@login_required
def create_graph():
    from app.routes.main import get_uploaded_files
    started_at = time.perf_counter()
    uploaded_files = get_uploaded_files()
    measured_preview = query_db(_db_path, "measured_parameters", query="SELECT * FROM {} LIMIT 1")
    parameters = measured_preview.columns.tolist() if not measured_preview.empty else []

    selected_viz_type = request.form.get('viz_type', 'matplotlib') if request.method == 'POST' else 'matplotlib'

    if selected_viz_type == 'plotly':
        graphs = PLOTLY_GRAPHS
    elif selected_viz_type == 'seaborn':
        graphs = SEABORN_GRAPHS
    else:
        graphs = MATPLOTLIB_GRAPHS

    if request.method == 'POST':
        try:
            measured_data = query_db(_db_path, "measured_parameters")
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

            stats = get_data_statistics(measured_data)

            if measured_data.empty:
                return jsonify({'success': False, 'message': 'Нет данных для построения графика. Сначала загрузите файл.', 'stats': stats})

            if graph_type == '3d_scatter':
                z_param, z_candidates = _resolve_3d_z_param(measured_data, x_param, y_param, z_param)
                if not z_param or z_param in {x_param, y_param} or z_param not in measured_data.columns:
                    candidates_text = ', '.join(z_candidates) if z_candidates else 'нет'
                    return jsonify({
                        'success': False,
                        'message': (
                            'Для 3D Scatter выберите корректный параметр Z. '
                            f'Доступные параметры Z: {candidates_text}'
                        ),
                        'stats': stats
                    })

            if selected_compositions is not None and len(selected_compositions) == 0:
                if 'composition' in measured_data.columns:
                    selected_compositions = measured_data['composition'].unique().tolist()
                else:
                    selected_compositions = None

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
                graph_output_type = 'plotly_json'
            elif viz_type == 'seaborn':
                graph, graph_message, available_compositions = generate_seaborn_plot(
                    measured_data, x_param, y_param, graph_type, theme, color_param, selected_compositions
                )
                graph_output_type = 'matplotlib'
            else:
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

            response = jsonify({
                'success': True,
                'message': graph_message,
                'graph': graph,
                'graph_type': graph_output_type,
                'stats': stats,
                'available_compositions': available_compositions
            })
            logger.info("graphs_create_post elapsed_ms=%.2f rows=%s", (time.perf_counter() - started_at) * 1000.0, len(measured_data))
            return response
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({'success': False, 'message': f'Ошибка при создании графика: {str(e)}', 'stats': get_data_statistics(measured_data)})

    # GET request
    # Ленивая инициализация: не строим график и статистику на открытии вкладки.
    # Это существенно ускоряет переключение между вкладками при больших таблицах.
    graph = None
    stats = {}

    param_labels = {
        'composition': 'Составы', 'density': 'Плотность, кг/м³', 'kf': 'Ударопрочность, %',
        'kt': 'Устойчивость к колебательным нагрузкам, %', 'h': 'Гигроскопичность, %',
        'mass_loss': 'Потеря массы, %', 'tign': 'Температура зажигания, °С', 'tb': 'Температура выгорания, °С',
        'tau_d1': 'Задержка газофазного зажигания, с', 'tau_d2': 'Задержка гетерогенного зажигания, с',
        'tau_b': 'Время горения, с', 'co2': 'Концентрация CO₂, %', 'co': 'Концентрация CO, %',
        'so2': 'Концентрация SO₂, ppm', 'nox': 'Концентрация NOx, ppm', 'q': 'Теплота сгорания, МДж/кг',
        'ad': 'Содержание золы на сухую массу, %', 'war': 'Влажность на аналитическую массу, %',
        'vd': 'Содержание летучих на сухую массу, %', 'cd': 'Содержание углерода на сухую массу, %',
        'hd': 'Содержание водорода на сухую массу, %', 'nd': 'Содержание азота на сухую массу, %',
        'sd': 'Содержание серы на сухую массу, %', 'od': 'Содержание кислорода на сухую массу, %',
        'component': 'Компоненты',
    }

    response = render_template(
        'create_graph.html',
        segment='Создание графика',
        uploaded_files=uploaded_files,
        parameters=parameters,
        param_labels=param_labels,
        graphs=graphs,
        viz_types=VIZ_TYPES,
        selected_viz_type=selected_viz_type,
        graph=graph,
        components_sheet_name=session.get('components_sheet_name', 'Таблица компонентов'),
        stats=stats
    )
    logger.info("graphs_create_get elapsed_ms=%.2f", (time.perf_counter() - started_at) * 1000.0)
    return response
