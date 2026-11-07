"""Пакет модулей визуализации и построения графиков Pellets Analyzer.

Содержит:
- palette: Палитры, названия параметров, стили и темы
- registry: Реестр типов графиков (ChartRegistry)
- stats: Статистические и фильтрационные функции
- plotly_charts: Интерактивные графики Plotly
- seaborn_charts: Статистические графики Seaborn
- matplotlib_charts: Статические и анимированные графики Matplotlib
"""

from app.services.charts.palette import (
    COLOR_PALETTE,
    PARAM_NAMES,
    PARAM_UNITS,
    MATPLOTLIB_GRAPHS,
    PLOTLY_GRAPHS,
    SPECIAL_GRAPH_PARAMS,
    SEABORN_GRAPHS,
    VIZ_TYPES,
    get_param_display_name,
    get_param_description,
    get_plotly_theme,
    get_seaborn_style,
    apply_theme,
)
from app.services.charts.registry import (
    ChartRegistry,
    register_chart,
)
from app.services.charts.stats import (
    get_data_statistics,
    simple_filter,
    filter_parameters,
    get_comparison_stats,
    get_compact_composition_table,
    get_key_params_summary,
)
from app.services.charts.plotly_charts import (
    generate_plotly_graph,
    create_radar_chart,
    create_plotly_radar_chart,
    create_animated_scatter,
)
from app.services.charts.seaborn_charts import (
    generate_seaborn_plot,
)
from app.services.charts.matplotlib_charts import (
    generate_graph,
    generate_graph_simple,
    generate_animated_graph,
    handle_scatter_plot,
    handle_bar_chart,
    handle_heatmap,
    handle_3d_scatter,
)

__all__ = [
    # Palette & Config
    'COLOR_PALETTE',
    'PARAM_NAMES',
    'PARAM_UNITS',
    'MATPLOTLIB_GRAPHS',
    'PLOTLY_GRAPHS',
    'SPECIAL_GRAPH_PARAMS',
    'SEABORN_GRAPHS',
    'VIZ_TYPES',
    'get_param_display_name',
    'get_param_description',
    'get_plotly_theme',
    'get_seaborn_style',
    'apply_theme',
    # Registry
    'ChartRegistry',
    'register_chart',
    # Stats
    'get_data_statistics',
    'simple_filter',
    'filter_parameters',
    'get_comparison_stats',
    'get_compact_composition_table',
    'get_key_params_summary',
    # Plotly
    'generate_plotly_graph',
    'create_radar_chart',
    'create_plotly_radar_chart',
    'create_animated_scatter',
    # Seaborn
    'generate_seaborn_plot',
    # Matplotlib
    'generate_graph',
    'generate_graph_simple',
    'generate_animated_graph',
    'handle_scatter_plot',
    'handle_bar_chart',
    'handle_heatmap',
    'handle_3d_scatter',
]
