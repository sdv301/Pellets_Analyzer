"""Палитра цветов, названия параметров и управление темами для графиков."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns

# Единая категориальная палитра для графиков (AstroWind + colorblind-friendly / WCAG AA) (F2)
COLOR_PALETTE = [
    "#0161EF",  # Primary Blue
    "#0284C7",  # Sky Blue
    "#10B981",  # Emerald Green
    "#F59E0B",  # Amber Yellow
    "#8B5CF6",  # Violet Purple
    "#EC4899",  # Rose Pink
    "#14B8A6",  # Teal
    "#F97316",  # Orange
]

# Настройка Plotly шаблонов с единой палитрой
_astrowind_light = go.layout.Template(
    layout=go.Layout(
        colorway=COLOR_PALETTE,
        font=dict(family="Inter, -apple-system, sans-serif", color="#0F172A"),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
    )
)
_astrowind_dark = go.layout.Template(
    layout=go.Layout(
        colorway=COLOR_PALETTE,
        font=dict(family="Inter, -apple-system, sans-serif", color="#F8FAFC"),
        paper_bgcolor="#0B0F19",
        plot_bgcolor="#111827",
    )
)
pio.templates["astrowind"] = _astrowind_light
pio.templates["astrowind_dark"] = _astrowind_dark
pio.templates.default = "astrowind"

try:
    sns.set_palette(COLOR_PALETTE)
except Exception:
    pass

# Словарь для перевода параметров
PARAM_NAMES = {
    'ad': 'содержание золы на сухую массу, %',
    'q': 'теплота сгорания, МДж/кг', 
    'density': 'плотность, кг/м3',
    'kf': 'ударопрочность, %',
    'kt': 'устойчивость к колебательным нагрузкам, %',
    'h': 'гигроскопичность, %',
    'mass_loss': 'потеря массы, %',
    'tign': 'температура зажигания, °С',
    'tb': 'температура выгорания, °C',
    'tau_d1': 'задержка газофазного зажигания, C',
    'tau_d2': 'задержка гетерогенного зажигания, C', 
    'tau_b': 'Время горения, С',
    'co2': 'концентрации диоксида углерода, %',
    'co': 'концентрации монооксида углерода, %',
    'so2': 'концентрации оксидов серы, ppm',
    'nox': 'концентрации оксидов азота, ppm',
    'war': 'влажность на аналитическую массу, %',
    'vd': 'содержание летучих на сухую массу, %',
    'cd': 'содержание углерода на сухую массу, %',
    'hd': 'содержание водорода на сухую массу, %',
    'nd': 'содержание азота на сухую массу, %',
    'sd': 'содержание серы на сухую массу, %',
    'od': 'содержание кислорода на сухую массу, %'
}

PARAM_UNITS = {
    'ad': '%', 'q': 'МДж/кг', 'density': 'кг/м³', 'kf': '%', 'kt': '%', 'h': '%',
    'mass_loss': '%', 'tign': '°C', 'tb': '°C', 'tau_d1': 'с', 'tau_d2': 'с', 
    'tau_b': 'с', 'co2': '%', 'co': '%', 'so2': 'ppm', 'nox': 'ppm', 'war': '%',
    'vd': '%', 'cd': '%', 'hd': '%', 'nd': '%', 'sd': '%', 'od': '%'
}

MATPLOTLIB_GRAPHS = [
    ('scatter', 'Точечная диаграмма'),
    ('line', 'Линейный график'),
    ('bar', 'Столбчатая диаграмма'),
    ('histogram', 'Гистограмма'),
    ('box', 'Box Plot'),
    ('pie', 'Круговая диаграмма'),
    ('heatmap', 'Тепловая карта'),
    ('3d_scatter', '3D Scatter')
]

PLOTLY_GRAPHS = [
    ('scatter', 'Точечная диаграмма'),
    ('line', 'Линейный график'),
    ('bar', 'Столбчатая диаграмма'),
    ('histogram', 'Гистограмма'),
    ('box', 'Box Plot'),
    ('violin', 'Violin Plot'),
    ('pie', 'Круговая диаграмма'),
    ('heatmap', 'Тепловая карта'),
    ('radar', 'Радарная диаграмма'),
    ('3d_scatter', '3D Scatter')
]

SPECIAL_GRAPH_PARAMS = {
    'pie': {
        'x_description': 'Категория (текстовый параметр)',
        'y_description': 'Значение (числовой параметр)'
    },
    'sunburst': {
        'x_description': 'Вторичная категория',
        'y_description': 'Значение для отображения'
    },
    'treemap': {
        'x_description': 'Вторичная категория', 
        'y_description': 'Значение для отображения'
    }
}

SEABORN_GRAPHS = [
    ('scatter', 'Точечная диаграмма'),
    ('line', 'Линейный график'),
    ('bar', 'Столбчатая диаграмма'),
    ('histogram', 'Гистограмма'),
    ('box', 'Box Plot'),
    ('violin', 'Violin Plot'),
    ('heatmap', 'Тепловая карта')
]

VIZ_TYPES = [
    ('matplotlib', 'Matplotlib (статические)'),
    ('plotly', 'Plotly (интерактивные)'),
    ('seaborn', 'Seaborn (статистические)')
]


def get_param_display_name(param):
    """Возвращает отображаемое имя параметра с единицей измерения"""
    name = PARAM_NAMES.get(param, param.capitalize())
    unit = PARAM_UNITS.get(param, '')
    return f"{name} ({unit})" if unit else name


def get_param_description(graph_type, param_type):
    """Возвращает описание параметра для специальных графиков"""
    if graph_type in SPECIAL_GRAPH_PARAMS:
        return SPECIAL_GRAPH_PARAMS[graph_type].get(param_type, '')
    return ''


def get_plotly_theme(theme):
    """Возвращает тему Plotly с поддержкой единой палитры (F2)"""
    themes = {
        'default': 'astrowind',
        'light': 'astrowind',
        'dark': 'astrowind_dark',
        'plotly_white': 'astrowind',
        'plotly_dark': 'astrowind_dark',
        'seaborn': 'seaborn',
        'ggplot': 'ggplot2'
    }
    return themes.get(theme, 'astrowind')


def get_seaborn_style(theme):
    """Возвращает стиль Seaborn для темы"""
    styles = {
        'default': 'whitegrid',
        'light': 'whitegrid',
        'dark': 'darkgrid',
        'minimal': 'ticks',
        'classic': 'white'
    }
    return styles.get(theme, 'whitegrid')


def apply_theme(theme):
    """Применяет тему оформления Matplotlib/Seaborn"""
    if theme == 'dark':
        plt.style.use('dark_background')
        sns.set_style('darkgrid')
    elif theme == 'minimal':
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_style('ticks')
    elif theme == 'classic':
        plt.style.use('classic')
        sns.set_style('white')
    else:
        plt.style.use('default')
        sns.set_style('whitegrid')
