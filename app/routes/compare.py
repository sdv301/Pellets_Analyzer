# app/routes/compare.py — Роуты сравнения составов
from flask import Blueprint, render_template, request, jsonify, session, flash
import pandas as pd
from app.auth.auth import login_required
from app.database.database import query_db

compare_bp = Blueprint('compare', __name__)

_db_path = 'pellets_data.db'

def set_db_path(path):
    global _db_path
    _db_path = path


@compare_bp.route('/compare')
@login_required
def compare():
    from app.routes.main import get_uploaded_files
    uploaded_files = get_uploaded_files()
    measured_data = query_db(_db_path, "measured_parameters")
    compositions = measured_data['composition'].tolist() if not measured_data.empty else []
    return render_template('compare.html', segment='Сравнительная таблица', uploaded_files=uploaded_files, compositions=compositions)


@compare_bp.route('/compare', methods=['POST'])
@login_required
def compare_data():
    try:
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
            return jsonify({'success': False, 'message': 'Выберите хотя бы два состава для сравнения.'})

        selected_criteria = request.form.getlist('criteria[]')
        measured_data = query_db(_db_path, "measured_parameters")

        if measured_data.empty:
            return jsonify({'success': False, 'message': 'В базе данных нет измеренных параметров.'})

        comparison_data = pd.DataFrame()
        found_compositions = []

        for comp in compositions:
            comp_data = measured_data[measured_data['composition'] == comp]
            if not comp_data.empty:
                comparison_data = pd.concat([comparison_data, comp_data])
                found_compositions.append(comp)

        if comparison_data.empty:
            return jsonify({'success': False, 'message': 'Выбранные составы не найдены в базе данных.'})

        if len(found_compositions) != len(compositions):
            missing = set(compositions) - set(found_compositions)
            flash(f'Некоторые составы не найдены: {", ".join(missing)}', 'warning')

        show_all = request.form.get('show_all') == 'on'
        show_diff = request.form.get('show_diff') == 'on'
        param_group = request.form.get('paramGroup', 'all')

        filtered_data = filter_parameters_with_criteria(comparison_data, param_group, show_diff and not show_all, selected_criteria)

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
            'comparison': filtered_data.to_html(classes='table table-striped table-sm comparison-table', index=False, escape=False, na_rep='N/A'),
            'compositions': found_compositions,
            'stats': get_comparison_stats(filtered_data)
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Ошибка при сравнении: {str(e)}'})


def filter_parameters_with_criteria(data, param_group, show_diff_only=False, selected_criteria=None):
    from app.services.gui import filter_parameters
    return filter_parameters(data, param_group, show_diff_only)


def get_comparison_stats(data):
    from app.services.gui import get_comparison_stats as _stats
    return _stats(data)
