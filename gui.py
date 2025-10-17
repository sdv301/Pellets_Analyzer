# gui.py - только графические функции
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import seaborn as sns
import json
import base64
from io import StringIO
import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Настройка Plotly для тем
pio.templates.default = "plotly_white"

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

# Списки поддерживаемых графиков для каждого типа визуализации
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
    ('sunburst', 'Sunburst'),
    ('treemap', 'Treemap'),
    ('3d_scatter', '3D Scatter'),
    ('animated_scatter', 'Анимированный график')
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

# Типы визуализации
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

def get_plotly_theme(theme):
    """Возвращает тему Plotly"""
    themes = {
        'default': 'plotly_white',
        'dark': 'plotly_dark',
        'seaborn': 'seaborn',
        'ggplot': 'ggplot2'
    }
    return themes.get(theme, 'plotly_white')

def create_radar_chart(data, color_param, template, title):
    """Создает радарную диаграмму"""
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 3:
        return None
    
    # Берем первые 6 числовых колонок для радара
    radar_cols = numeric_cols[:6]
    
    if color_param and color_param in data.columns:
        categories = data[color_param].unique()
    else:
        categories = ['Все данные']
    
    fig = go.Figure()
    
    for category in categories:
        if color_param and color_param in data.columns:
            category_data = data[data[color_param] == category]
        else:
            category_data = data
            
        values = [category_data[col].mean() for col in radar_cols]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=[get_param_display_name(col) for col in radar_cols],
            fill='toself',
            name=str(category)
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, max([max(trace.r) for trace in fig.data]) * 1.1])
        ),
        title=title or "Радарная диаграмма параметров",
        template=template
    )
    
    return fig

def generate_plotly_graph(data, x_param='ad', y_param='q', graph_type='scatter',
                         z_param=None, color_param=None, size_param=None, 
                         animation_param=None, theme='default', title=None,
                         width=800, height=600, show_grid=True, selected_compositions=None):
    """Создает интерактивные графики с использованием Plotly, с исправленным box plot и улучшенной обработкой данных"""
    
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.io as pio
    import numpy as np
    import traceback
    
    # Проверка входных данных
    if data.empty or x_param not in data.columns:
        return None, "Нет данных или неверный параметр X для построения графика", []
    
    try:
        # ФИЛЬТРАЦИЯ ПО ВЫБРАННЫМ СОСТАВАМ
        filtered_data = data.copy()
        available_compositions = []
        
        if 'composition' in data.columns:
            available_compositions = data['composition'].unique().tolist()
            
            if selected_compositions is not None:
                if len(selected_compositions) == 0:
                    return None, "Нет выбранных составов для отображения", available_compositions
                filtered_data = data[data['composition'].isin(selected_compositions)]
                
            if filtered_data.empty:
                return None, "Нет данных для выбранных составов", available_compositions
        
        # Настройка темы
        template = get_plotly_theme(theme)
        
        # Определяем количество элементов для легенды
        legend_items = len(filtered_data['composition'].unique()) if 'composition' in filtered_data.columns else 0
        
        # Адаптивные настройки легенды
        legend_font_size = 10 if legend_items <= 10 else (9 if legend_items <= 15 else 8)
        legend_y_position = 0.98 if legend_items <= 10 else (0.97 if legend_items <= 15 else 0.95)
        right_margin = 150 if legend_items <= 10 else (160 if legend_items <= 15 else 180)
        
        fig = None
        
        # Общие настройки для hover-данных
        hover_data = {'composition': True} if 'composition' in filtered_data.columns else None
        
        # Обработка разных типов графиков
        try:
            if graph_type == 'scatter':
                if y_param not in filtered_data.columns or filtered_data[y_param].dropna().empty:
                    return None, f"Неверный параметр Y ({y_param}) или нет данных", available_compositions
                
                fig = px.scatter(
                    filtered_data,
                    x=x_param,
                    y=y_param,
                    color='composition' if 'composition' in filtered_data.columns else (color_param if color_param and color_param in filtered_data.columns else None),
                    size=size_param if size_param and size_param in filtered_data.columns else None,
                    hover_name='composition' if 'composition' in filtered_data.columns else None,
                    hover_data=hover_data,
                    title=title or f"{get_param_display_name(y_param)} vs {get_param_display_name(x_param)}",
                    template=template
                )

                fig.update_traces(
                    hovertemplate=(
                        "<b>%{hovertext}</b><br>" +
                        f"{get_param_display_name(x_param)}: %{{x:.2f}}<br>" +
                        f"{get_param_display_name(y_param)}: %{{y:.2f}}<br>" +
                        "<extra></extra>"
                    ) if 'composition' in filtered_data.columns else (
                        f"{get_param_display_name(x_param)}: %{{x:.2f}}<br>" +
                        f"{get_param_display_name(y_param)}: %{{y:.2f}}<br>" +
                        "<extra></extra>"
                    )
                )
                
            elif graph_type == 'line':
                if y_param not in filtered_data.columns or filtered_data[y_param].dropna().empty:
                    return None, f"Неверный параметр Y ({y_param}) или нет данных", available_compositions
                
                sorted_data = filtered_data.sort_values(by=x_param)
                fig = px.line(
                    sorted_data,
                    x=x_param,
                    y=y_param,
                    color='composition' if 'composition' in sorted_data.columns else (color_param if color_param and color_param in sorted_data.columns else None),
                    hover_name='composition' if 'composition' in sorted_data.columns else None,
                    title=title or f"Линейный график: {get_param_display_name(y_param)} vs {get_param_display_name(x_param)}",
                    template=template
                )
                fig.update_traces(
                    mode='lines+markers',
                    marker=dict(size=6, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')),
                    line=dict(width=2),
                    hovertemplate=(
                        "<b>%{hovertext}</b><br>" +
                        f"{get_param_display_name(x_param)}: %{{x:.2f}}<br>" +
                        f"{get_param_display_name(y_param)}: %{{y:.2f}}<br>" +
                        "<extra></extra>"
                    ) if 'composition' in sorted_data.columns else (
                        f"{get_param_display_name(x_param)}: %{{x:.2f}}<br>" +
                        f"{get_param_display_name(y_param)}: %{{y:.2f}}<br>" +
                        "<extra></extra>"
                    )
                )
                
            elif graph_type == 'bar':
                if y_param not in filtered_data.columns or filtered_data[y_param].dropna().empty:
                    return None, f"Неверный параметр Y ({y_param}) или нет данных", available_compositions
                
                if filtered_data[x_param].dtype == 'object' or len(filtered_data[x_param].unique()) <= 20:
                    grouped = filtered_data.groupby(['composition', x_param])[y_param].mean().reset_index() if 'composition' in filtered_data.columns else filtered_data.groupby(x_param)[y_param].mean().reset_index()
                    fig = px.bar(
                        grouped,
                        x=x_param,
                        y=y_param,
                        color='composition' if 'composition' in filtered_data.columns else (color_param if color_param and color_param in filtered_data.columns else None),
                        barmode='group',
                        hover_name='composition' if 'composition' in filtered_data.columns else None,
                        title=title or f"Среднее {get_param_display_name(y_param)} по {get_param_display_name(x_param)}",
                        template=template
                    )
                    fig.update_traces(
                        hovertemplate=(
                            "<b>%{hovertext}</b><br>" +
                            f"{get_param_display_name(x_param)}: %{{x}}<br>" +
                            f"{get_param_display_name(y_param)}: %{{y:.2f}}<br>" +
                            "<extra></extra>"
                        ) if 'composition' in filtered_data.columns else (
                            f"{get_param_display_name(x_param)}: %{{x}}<br>" +
                            f"{get_param_display_name(y_param)}: %{{y:.2f}}<br>" +
                            "<extra></extra>"
                        )
                    )
                else:
                    fig = px.histogram(
                        filtered_data,
                        x=x_param,
                        color='composition' if 'composition' in filtered_data.columns else (color_param if color_param and color_param in filtered_data.columns else None),
                        title=title or f"Распределение {get_param_display_name(x_param)}",
                        template=template
                    )
                    fig.update_traces(
                        hovertemplate=(
                            f"{get_param_display_name(x_param)}: %{{x:.2f}}<br>" +
                            "Количество: %{y}<br>" +
                            "<extra></extra>"
                        )
                    )
                
            elif graph_type == 'histogram':
                fig = px.histogram(
                    filtered_data,
                    x=x_param,
                    color='composition' if 'composition' in filtered_data.columns else (color_param if color_param and color_param in filtered_data.columns else None),
                    title=title or f"Гистограмма {get_param_display_name(x_param)}",
                    template=template
                )
                fig.update_traces(
                    hovertemplate=(
                        f"{get_param_display_name(x_param)}: %{{x:.2f}}<br>" +
                        "Количество: %{y}<br>" +
                        "<extra></extra>"
                    )
                )
                
            elif graph_type == 'box':
                # Проверка наличия y_param и данных для него
                if y_param not in filtered_data.columns or filtered_data[y_param].dropna().empty:
                    return None, f"Неверный параметр Y ({y_param}) или нет данных для построения box plot", available_compositions
                
                # Проверка, что y_param числовой
                if not pd.api.types.is_numeric_dtype(filtered_data[y_param]):
                    return None, f"Параметр Y ({y_param}) должен быть числовым для box plot", available_compositions
                
                # Если x_param числовой и имеет много уникальных значений, выполняем биннинг
                if pd.api.types.is_numeric_dtype(filtered_data[x_param]) and len(filtered_data[x_param].unique()) > 20:
                    data = filtered_data[x_param].dropna()
                    if data.empty:
                        return None, f"Нет данных для параметра X ({x_param}) после удаления NaN", available_compositions
                    
                    # Автоматическое определение количества бинов
                    data_min, data_max = data.min(), data.max()
                    bins = min(10, max(5, len(data) // 10))
                    bin_edges = np.linspace(data_min, data_max, bins + 1)
                    
                    # Создаем категории на основе диапазонов
                    bin_labels = [f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}" for i in range(len(bin_edges)-1)]
                    filtered_data = filtered_data.copy()
                    filtered_data['binned_x'] = pd.cut(filtered_data[x_param], bins=bin_edges, labels=bin_labels, include_lowest=True)
                    x_param_to_use = 'binned_x'
                else:
                    x_param_to_use = x_param
                
                # Проверка, что x_param_to_use (оригинальный или биннированный) имеет данные
                if filtered_data[x_param_to_use].dropna().empty:
                    return None, f"Параметр X ({x_param_to_use}) не содержит данных после обработки", available_compositions
                
                # Определяем параметр для цвета
                color = 'composition' if 'composition' in filtered_data.columns else (color_param if color_param and color_param in filtered_data.columns else None)
                
                # Создаем box plot
                fig = px.box(
                    filtered_data,
                    x=x_param_to_use,
                    y=y_param,
                    color=color,
                    points='all',  # Показываем все точки
                    notched=True,   # Выемка для медианы
                    hover_name='composition' if 'composition' in filtered_data.columns else None,
                    hover_data=hover_data,
                    title=title or f"Box Plot: {get_param_display_name(y_param)} по {get_param_display_name(x_param)}",
                    template=template
                )
                
                # Настраиваем hover-шаблон и внешний вид
                fig.update_traces(
                    boxmean=True,  # Показываем среднее значение
                    jitter=0.3,    # Разброс точек
                    pointpos=0,    # Позиция точек
                    hovertemplate=(
                        "<b>%{hovertext}</b><br>" +
                        f"{get_param_display_name(x_param)}: %{{x}}<br>" +
                        f"{get_param_display_name(y_param)}: %{{y:.2f}}<br>" +
                        "<extra></extra>"
                    ) if 'composition' in filtered_data.columns else (
                        f"{get_param_display_name(x_param)}: %{{x}}<br>" +
                        f"{get_param_display_name(y_param)}: %{{y:.2f}}<br>" +
                        "<extra></extra>"
                    )
                )

            elif graph_type == 'violin':
                if y_param not in filtered_data.columns or filtered_data[y_param].dropna().empty:
                    return None, f"Неверный параметр Y ({y_param}) или нет данных", available_compositions
                
                fig = px.violin(
                    filtered_data,
                    x=x_param,
                    y=y_param,
                    color='composition' if 'composition' in filtered_data.columns else (color_param if color_param and color_param in filtered_data.columns else None),
                    box=True,
                    points='all',
                    title=title or f"Violin Plot: {get_param_display_name(y_param)}",
                    template=template
                )
                fig.update_traces(
                    hovertemplate=(
                        "<b>%{hovertext}</b><br>" +
                        f"{get_param_display_name(x_param)}: %{{x}}<br>" +
                        f"{get_param_display_name(y_param)}: %{{y:.2f}}<br>" +
                        "<extra></extra>"
                    ) if 'composition' in filtered_data.columns else (
                        f"{get_param_display_name(x_param)}: %{{x}}<br>" +
                        f"{get_param_display_name(y_param)}: %{{y:.2f}}<br>" +
                        "<extra></extra>"
                    )
                )
                
            elif graph_type == 'sunburst':
                if 'composition' not in filtered_data.columns:
                    return None, "Для sunburst нужна колонка 'composition'", available_compositions
                if y_param not in filtered_data.columns or filtered_data[y_param].dropna().empty:
                    return None, f"Неверный параметр Y ({y_param}) или нет данных", available_compositions
                fig = px.sunburst(
                    filtered_data,
                    path=['composition', x_param],
                    values=y_param,
                    title=title or f"Sunburst: {get_param_display_name(y_param)} по составам и {get_param_display_name(x_param)}",
                    template=template
                )
                fig.update_traces(
                    hovertemplate=(
                        "<b>%{label}</b><br>" +
                        f"Значение: %{{value:.2f}}<br>" +
                        "<extra></extra>"
                    )
                )
                
            elif graph_type == 'treemap':
                if 'composition' not in filtered_data.columns:
                    return None, "Для treemap нужна колонка 'composition'", available_compositions
                if y_param not in filtered_data.columns or filtered_data[y_param].dropna().empty:
                    return None, f"Неверный параметр Y ({y_param}) или нет данных", available_compositions
                fig = px.treemap(
                    filtered_data,
                    path=['composition', x_param],
                    values=y_param,
                    title=title or f"Treemap: {get_param_display_name(y_param)} по составам и {get_param_display_name(x_param)}",
                    template=template
                )
                fig.update_traces(
                    hovertemplate=(
                        "<b>%{label}</b><br>" +
                        f"Значение: %{{value:.2f}}<br>" +
                        "<extra></extra>"
                    )
                )
                
            elif graph_type == 'pie':
                try:
                    if pd.api.types.is_numeric_dtype(filtered_data[x_param]):
                        data = filtered_data[x_param].dropna()
                        if data.empty:
                            return None, "Нет данных для построения круговой диаграммы", available_compositions
                        
                        data_min, data_max = data.min(), data.max()
                        bins = min(10, max(5, len(data) // 10))
                        bin_edges = np.linspace(data_min, data_max, bins + 1)
                        
                        category_counts = {}
                        for i in range(len(bin_edges) - 1):
                            lower, upper = bin_edges[i], bin_edges[i + 1]
                            count = ((data >= lower) & (data < upper)).sum() if i < len(bin_edges) - 2 else ((data >= lower) & (data <= upper)).sum()
                            category_counts[f"{lower:.2f}-{upper:.2f}"] = count
                        
                        pie_data = pd.DataFrame({
                            'category': list(category_counts.keys()),
                            'count': list(category_counts.values())
                        })
                    else:
                        pie_data = filtered_data[x_param].value_counts().reset_index()
                        pie_data.columns = ['category', 'count']
                    
                    pie_data = pie_data[pie_data['count'] > 0]
                    if len(pie_data) > 10:
                        pie_data = pie_data.sort_values('count', ascending=False)
                        main_categories = pie_data.head(9)
                        other_sum = pie_data['count'].iloc[9:].sum()
                        if other_sum > 0:
                            other_row = pd.DataFrame({'category': ['Другие'], 'count': [other_sum]})
                            pie_data = pd.concat([main_categories, other_row], ignore_index=True)
                    
                    fig = px.pie(
                        pie_data,
                        values='count',
                        names='category',
                        title=title or f"Распределение {get_param_display_name(x_param)}",
                        template=template
                    )
                    fig.update_traces(
                        hovertemplate=(
                            "<b>%{label}</b><br>" +
                            "Количество: %{value}<br>" +
                            "Доля: %{percent}<br>" +
                            "<extra></extra>"
                        ),
                        textposition='inside',
                        textinfo='percent+label'
                    )
                    
                except Exception as e:
                    traceback.print_exc()
                    return None, f"Ошибка при создании круговой диаграммы: {str(e)}", available_compositions
                
            elif graph_type == 'heatmap':
                numeric_cols = filtered_data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) < 2:
                    return None, "Для тепловой карты нужно как минимум 2 числовых параметра", available_compositions
                corr_matrix = filtered_data[numeric_cols].corr()
                fig = px.imshow(
                    corr_matrix,
                    x=[get_param_display_name(col) for col in numeric_cols],
                    y=[get_param_display_name(col) for col in numeric_cols],
                    color_continuous_scale='RdBu_r',
                    zmin=-1,
                    zmax=1,
                    text_auto='.2f',
                    title=title or "Тепловая карта корреляций",
                    template=template
                )
                fig.update_traces(
                    hovertemplate=(
                        "X: %{x}<br>" +
                        "Y: %{y}<br>" +
                        "Корреляция: %{z:.2f}<br>" +
                        "<extra></extra>"
                    )
                )
                
            elif graph_type == 'radar':
                try:
                    fig = create_plotly_radar_chart(filtered_data, color_param, template, title)
                    if fig is None:
                        return None, "Не удалось создать радарную диаграмму: недостаточно числовых параметров", available_compositions
                except Exception as e:
                    return None, f"Ошибка при создании радарной диаграммы: {str(e)}", available_compositions
                
            elif graph_type == '3d_scatter' and z_param and z_param in filtered_data.columns:
                if y_param not in filtered_data.columns or filtered_data[y_param].dropna().empty:
                    return None, f"Неверный параметр Y ({y_param}) или нет данных", available_compositions
                fig = px.scatter_3d(
                    filtered_data,
                    x=x_param,
                    y=y_param,
                    z=z_param,
                    color='composition' if 'composition' in filtered_data.columns else (color_param if color_param and color_param in filtered_data.columns else None),
                    size=size_param if size_param and size_param in filtered_data.columns else None,
                    hover_name='composition' if 'composition' in filtered_data.columns else None,
                    title=title or f"3D Scatter: {get_param_display_name(z_param)} vs {get_param_display_name(y_param)} vs {get_param_display_name(x_param)}",
                    template=template
                )
                fig.update_traces(
                    hovertemplate=(
                        "<b>%{hovertext}</b><br>" +
                        f"{get_param_display_name(x_param)}: %{{x:.2f}}<br>" +
                        f"{get_param_display_name(y_param)}: %{{y:.2f}}<br>" +
                        f"{get_param_display_name(z_param)}: %{{z:.2f}}<br>" +
                        "<extra></extra>"
                    ) if 'composition' in filtered_data.columns else (
                        f"{get_param_display_name(x_param)}: %{{x:.2f}}<br>" +
                        f"{get_param_display_name(y_param)}: %{{y:.2f}}<br>" +
                        f"{get_param_display_name(z_param)}: %{{z:.2f}}<br>" +
                        "<extra></extra>"
                    )
                )
                
            elif graph_type == 'animated_scatter' and animation_param and animation_param in filtered_data.columns:
                try:
                    fig = create_animated_scatter(
                        filtered_data,
                        x_param,
                        y_param,
                        animation_param,
                        template,
                        title,
                        color_param='composition' if 'composition' in filtered_data.columns else color_param,
                        size_param=size_param
                    )
                    if fig is None:
                        return None, "Не удалось создать анимированный график", available_compositions
                except Exception as e:
                    return None, f"Ошибка при создании анимированного графика: {str(e)}", available_compositions
                
            else:
                return None, f"Неподдерживаемый тип графика: {graph_type}", available_compositions
        
            # Общие настройки для всех графиков
            if fig:
                fig.update_layout(
                    width=width,
                    height=height,
                    template=template,
                    showlegend=(legend_items > 0),
                    legend=dict(
                        title=dict(
                            text='Составы' if 'composition' in filtered_data.columns else 'Категории',
                            font=dict(size=legend_font_size)
                        ),
                        orientation="v",
                        yanchor="top",
                        y=legend_y_position,
                        xanchor="left",
                        x=1.02,
                        bgcolor='rgba(255,255,255,0.9)',
                        bordercolor='lightgray',
                        borderwidth=1,
                        font=dict(size=legend_font_size),
                        itemsizing='constant',
                        itemwidth=30
                    ),
                    font=dict(size=12),
                    margin=dict(l=50, r=right_margin, t=50, b=50),
                    hovermode='closest',
                    hoverlabel=dict(
                        bgcolor="white",
                        font_size=12,
                        font_family="Arial"
                    )
                )
                
                if show_grid and graph_type not in ['pie', 'sunburst', 'treemap', 'heatmap', 'radar']:
                    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
                    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
                
                # Конвертация в HTML
                graph_html = pio.to_html(
                    fig,
                    include_plotlyjs='cdn',
                    full_html=False,
                    config={
                        'responsive': True,
                        'displayModeBar': True,
                        'displaylogo': False,
                        'modeBarButtonsToAdd': ['hoverClosestGl2d', 'hoverCompareGl2d']
                    }
                )
                
                return graph_html, f"{graph_type.capitalize()} график создан успешно", available_compositions
            
            return None, "Не удалось создать график", available_compositions
        
        except Exception as e:
            error_msg = f"Ошибка при создании графика {graph_type}: {str(e)}"
            traceback.print_exc()
            return None, error_msg, available_compositions
    
    except Exception as e:
        error_msg = f"Ошибка при обработке данных: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return None, error_msg, available_compositions


def create_animated_scatter(data, x_param, y_param, animation_param, template, title, color_param=None, size_param=None):
    """Создает анимированный scatter plot"""
    fig = px.scatter(data, x=x_param, y=y_param,
                   animation_frame=animation_param,
                   color=color_param if color_param and color_param in data.columns else None,
                   size=size_param if size_param and size_param in data.columns else None,
                   hover_name=data.index if hasattr(data, 'index') else None,
                   title=title or f"Анимированный график по {get_param_display_name(animation_param)}",
                   template=template)
    
    return fig

def generate_seaborn_plot(data, x_param='ad', y_param='q', plot_type='scatter', 
                         theme='default', color_param=None, selected_compositions=None):
    """Создает статические графики с использованием Seaborn"""
    
    # ФИЛЬТРАЦИЯ ПО СОСТАВАМ
    filtered_data = data.copy()
    available_compositions = []
    
    if 'composition' in data.columns:
        available_compositions = data['composition'].unique().tolist()
        
        if selected_compositions:
            filtered_data = data[data['composition'].isin(selected_compositions)]
            if filtered_data.empty:
                return None, "Нет данных для выбранных составов", available_compositions
    
    try:
        # Применяем тему Seaborn
        sns.set_theme(style=get_seaborn_style(theme))
        
        plt.figure(figsize=(12, 8))
        
        # АВТОМАТИЧЕСКИ ИСПОЛЬЗУЕМ СОСТАВ ДЛЯ ЦВЕТА ЕСЛИ НЕ УКАЗАН color_param
        use_composition_color = ('composition' in filtered_data.columns and 
                               not color_param and plot_type not in ['pie', 'heatmap'])
        
        if plot_type == 'scatter':
            if use_composition_color:
                sns.scatterplot(data=filtered_data, x=x_param, y=y_param, 
                               hue='composition', palette='viridis', s=50)
            elif color_param and color_param in filtered_data.columns:
                sns.scatterplot(data=filtered_data, x=x_param, y=y_param, 
                               hue=color_param, palette='viridis', s=50)
            else:
                sns.scatterplot(data=filtered_data, x=x_param, y=y_param, s=50)
            
        elif plot_type == 'line':
            # ДЛЯ ЛИНЕЙНОГО ГРАФИКА - важно сортировать по X
            sorted_data = filtered_data.sort_values(by=x_param)
            if use_composition_color:
                sns.lineplot(data=sorted_data, x=x_param, y=y_param, 
                            hue='composition', palette='viridis', marker='o')
            elif color_param and color_param in filtered_data.columns:
                sns.lineplot(data=sorted_data, x=x_param, y=y_param, 
                            hue=color_param, palette='viridis', marker='o')
            else:
                sns.lineplot(data=sorted_data, x=x_param, y=y_param, marker='o')
            
        elif plot_type == 'violin':
            if use_composition_color:
                sns.violinplot(data=filtered_data, x=x_param, y=y_param, 
                              hue='composition', palette='viridis')
            elif color_param and color_param in filtered_data.columns:
                sns.violinplot(data=filtered_data, x=x_param, y=y_param, 
                              hue=color_param, palette='viridis')
            else:
                sns.violinplot(data=filtered_data, x=x_param, y=y_param)
            
        elif plot_type == 'box':
            if use_composition_color:
                sns.boxplot(data=filtered_data, x=x_param, y=y_param, 
                           hue='composition', palette='viridis')
            elif color_param and color_param in filtered_data.columns:
                sns.boxplot(data=filtered_data, x=x_param, y=y_param, 
                           hue=color_param, palette='viridis')
            else:
                sns.boxplot(data=filtered_data, x=x_param, y=y_param)
            
        elif plot_type == 'histogram':
            if use_composition_color:
                sns.histplot(data=filtered_data, x=x_param, hue='composition', 
                            kde=True, palette='viridis', multiple="layer")
            elif color_param and color_param in filtered_data.columns:
                sns.histplot(data=filtered_data, x=x_param, hue=color_param, 
                            kde=True, palette='viridis', multiple="layer")
            else:
                sns.histplot(data=filtered_data, x=x_param, kde=True)
            
        elif plot_type == 'bar':
            if filtered_data[x_param].dtype == 'object' or len(filtered_data[x_param].unique()) <= 20:
                if use_composition_color:
                    sns.barplot(data=filtered_data, x=x_param, y=y_param, 
                               hue='composition', palette='viridis')
                elif color_param and color_param in filtered_data.columns:
                    sns.barplot(data=filtered_data, x=x_param, y=y_param, 
                               hue=color_param, palette='viridis')
                else:
                    sns.barplot(data=filtered_data, x=x_param, y=y_param, palette='viridis')
            else:
                if use_composition_color:
                    sns.histplot(data=filtered_data, x=x_param, hue='composition', 
                                kde=True, palette='viridis')
                elif color_param and color_param in filtered_data.columns:
                    sns.histplot(data=filtered_data, x=x_param, hue=color_param, 
                                kde=True, palette='viridis')
                else:
                    sns.histplot(data=filtered_data, x=x_param, kde=True)
                
        elif plot_type == 'heatmap':
            numeric_data = filtered_data.select_dtypes(include=[np.number])
            if len(numeric_data.columns) >= 2:
                corr_matrix = numeric_data.corr()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
            else:
                return None, "Для тепловой карты нужно как минимум 2 числовых параметра", available_compositions
        
        elif plot_type == 'pie':
            # Круговая диаграмма через matplotlib
            pie_data = filtered_data[x_param].value_counts()
            
            if len(pie_data) > 10:
                main_categories = pie_data.head(9)
                other_sum = pie_data.iloc[9:].sum()
                if other_sum > 0:
                    main_categories = pd.concat([main_categories, pd.Series([other_sum], index=['Другие'])])
                pie_data = main_categories
            
            colors = sns.color_palette('viridis', len(pie_data))
            plt.pie(pie_data.values, labels=pie_data.index, autopct='%1.1f%%', 
                   startangle=90, colors=colors)
            plt.axis('equal')
        
        # Устанавливаем заголовок
        if plot_type == 'pie':
            plt.title(f"Распределение {get_param_display_name(x_param)}")
        elif plot_type == 'heatmap':
            plt.title("Тепловая карта корреляций")
        elif plot_type == 'histogram':
            plt.title(f"Гистограмма {get_param_display_name(x_param)}")
        elif plot_type == 'line':
            plt.title(f"Линейный график: {get_param_display_name(y_param)} vs {get_param_display_name(x_param)}")
        else:
            plt.title(f"{get_param_display_name(y_param)} vs {get_param_display_name(x_param)}")
        
        # Добавляем подписи осей для не-круговых диаграмм
        if plot_type != 'pie':
            plt.xlabel(get_param_display_name(x_param))
            if plot_type != 'histogram':  # У гистограммы только X ось
                plt.ylabel(get_param_display_name(y_param))
        
        # Улучшаем легенду для составов
        if use_composition_color and plot_type not in ['pie', 'heatmap']:
            plt.legend(title='Составы', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        
        plt.tight_layout()
        
        # Сохраняем в base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        graph = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        return graph, "Seaborn график создан успешно", available_compositions
        
    except Exception as e:
        plt.close()
        import traceback
        traceback.print_exc()
        return None, f"Ошибка при создании Seaborn графика: {str(e)}", available_compositions

def create_plotly_radar_chart(data, color_param, template, title):
    """Создает радарную диаграмму для Plotly"""
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 3:
        return None
    
    # Берем первые 6 числовых колонок для радара
    radar_cols = numeric_cols[:6]
    
    if color_param and color_param in data.columns:
        categories = data[color_param].unique()[:5]  # Ограничиваем количество категорий
    else:
        categories = ['Все данные']
    
    fig = go.Figure()
    
    for category in categories:
        if color_param and color_param in data.columns:
            category_data = data[data[color_param] == category]
        else:
            category_data = data
            
        # Вычисляем средние значения для каждой колонки
        values = []
        valid_cols = []
        for col in radar_cols:
            mean_val = category_data[col].mean()
            if not pd.isna(mean_val):
                values.append(mean_val)
                valid_cols.append(col)
        
        if len(values) >= 3:  # Минимум 3 параметра для радара
            # Нормализуем значения от 0 до 100
            max_val = max(values)
            if max_val > 0:
                normalized_values = [v / max_val * 100 for v in values]
            else:
                normalized_values = values
                
            fig.add_trace(go.Scatterpolar(
                r=normalized_values,
                theta=[get_param_display_name(col) for col in valid_cols],
                fill='toself',
                name=str(category)
            ))
    
    if len(fig.data) == 0:
        return None
        
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        title=title or "Радарная диаграмма параметров",
        template=template,
        showlegend=True
    )
    
    return fig

def get_seaborn_style(theme):
    """Возвращает стиль Seaborn"""
    styles = {
        'default': 'whitegrid',
        'dark': 'darkgrid',
        'seaborn': 'whitegrid',
        'ggplot': 'whitegrid'
    }
    return styles.get(theme, 'whitegrid')

def apply_theme(theme):
    """Применяет тему к графику"""
    plt.style.use('default')  # Сбрасываем стиль
    if theme == 'dark':
        plt.style.use('dark_background')
    elif theme == 'seaborn':
        plt.style.use('seaborn-v0_8')
    elif theme == 'ggplot':
        plt.style.use('ggplot')
    else:
        plt.style.use('default')

def generate_animated_graph(data, x_param, y_param, animation_param, theme, title, selected_compositions=None):
    """Создает анимированный график"""
    
    # ФИЛЬТРАЦИЯ ПО СОСТАВАМ
    filtered_data = data.copy()
    available_compositions = []
    
    if 'composition' in data.columns:
        available_compositions = data['composition'].unique().tolist()
        
        if selected_compositions:
            filtered_data = data[data['composition'].isin(selected_compositions)]
            if filtered_data.empty:
                return None, "Нет данных для выбранных составов", available_compositions
    
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        apply_theme(theme)
        
        # Получаем уникальные значения для анимации
        animation_values = sorted(filtered_data[animation_param].unique())
        
        def update(frame):
            ax.clear()
            current_value = animation_values[frame]
            frame_data = filtered_data[filtered_data[animation_param] == current_value]
            
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
        
        return graph, "Анимированный график создан успешно", available_compositions
        
    except Exception as e:
        plt.close()
        return None, f"Ошибка при создании анимированного графика: {str(e)}", available_compositions

def generate_graph(data, x_param='ad', y_param='q', graph_type='scatter', 
                  z_param=None, color_param=None, size_param=None, 
                  animation_param=None, theme='default', title=None,
                  width=800, height=600, show_grid=True, selected_compositions=None):
    
    # ФИЛЬТРАЦИЯ ПО СОСТАВАМ
    filtered_data = data.copy()
    available_compositions = []
    
    if 'composition' in data.columns:
        available_compositions = data['composition'].unique().tolist()
        
        # ВАЖНОЕ ИСПРАВЛЕНИЕ: Если selected_compositions пустой массив, возвращаем None
        if selected_compositions is not None:
            if len(selected_compositions) == 0:
                # Пустой список составов - возвращаем None
                return None, "Нет выбранных составов для отображения", available_compositions
            elif len(selected_compositions) > 0:
                # Есть выбранные составы - фильтруем данные
                filtered_data = data[data['composition'].isin(selected_compositions)]
                
            if filtered_data.empty:
                return None, "Нет данных для выбранных составов", available_compositions
    
    if filtered_data.empty or x_param not in filtered_data.columns:
        return None, "Нет данных для построения графика", available_compositions
    
    try:
        # Применяем тему
        apply_theme(theme)
        
        # Для гистограмм и круговых диаграмм используем только X параметр
        if graph_type in ['histogram', 'pie']:
            fig, ax = plt.subplots(figsize=(width/100, height/100))
        else:
            fig, ax = plt.subplots(figsize=(width/100, height/100))
        
        # АВТОМАТИЧЕСКИ ИСПОЛЬЗУЕМ СОСТАВ ДЛЯ ЦВЕТА ЕСЛИ НЕ УКАЗАН color_param
        use_composition_color = ('composition' in filtered_data.columns and 
                               not color_param and graph_type not in ['pie', 'heatmap'])
        
        if graph_type == 'scatter':
            if use_composition_color:
                # Используем composition для цвета
                unique_compositions = filtered_data['composition'].unique()
                colors = plt.cm.viridis(np.linspace(0, 1, len(unique_compositions)))
                color_map = dict(zip(unique_compositions, colors))
                
                for composition in unique_compositions:
                    comp_data = filtered_data[filtered_data['composition'] == composition]
                    ax.scatter(comp_data[x_param], comp_data[y_param], 
                              c=[color_map[composition]], label=composition,
                              alpha=0.7, s=50)
            elif color_param and color_param in filtered_data.columns:
                scatter = ax.scatter(filtered_data[x_param], filtered_data[y_param], 
                                    c=filtered_data[color_param], cmap='viridis', 
                                    alpha=0.7, s=50)
                plt.colorbar(scatter, label=PARAM_NAMES.get(color_param, color_param))
            else:
                ax.scatter(filtered_data[x_param], filtered_data[y_param], alpha=0.7, s=50)
                
        elif graph_type == 'line':
            # СОРТИРУЕМ ДАННЫЕ ДЛЯ ЛИНЕЙНОГО ГРАФИКА
            sorted_data = filtered_data.sort_values(by=x_param)
            
            if use_composition_color:
                unique_compositions = sorted_data['composition'].unique()
                colors = plt.cm.viridis(np.linspace(0, 1, len(unique_compositions)))
                
                for i, composition in enumerate(unique_compositions):
                    comp_data = sorted_data[sorted_data['composition'] == composition]
                    ax.plot(comp_data[x_param], comp_data[y_param], 
                           color=colors[i], label=composition,
                           linewidth=2, marker='o')
            else:
                ax.plot(sorted_data[x_param], sorted_data[y_param], linewidth=2, marker='o')
            
        elif graph_type == 'histogram':
            # ИСПРАВЛЕННАЯ ГИСТОГРАММА
            if use_composition_color:
                unique_compositions = filtered_data['composition'].unique()
                colors = plt.cm.viridis(np.linspace(0, 1, len(unique_compositions)))
                
                for i, composition in enumerate(unique_compositions):
                    comp_data = filtered_data[filtered_data['composition'] == composition]
                    ax.hist(comp_data[x_param].dropna(), bins=15, alpha=0.7, 
                           color=colors[i], label=composition, edgecolor='black')
            else:
                ax.hist(filtered_data[x_param].dropna(), bins=20, alpha=0.7, edgecolor='black')
            
            ax.set_xlabel(get_param_display_name(x_param))
            ax.set_ylabel('Частота')
            
        elif graph_type == 'bar':
            if filtered_data[x_param].dtype == 'object' or len(filtered_data[x_param].unique()) <= 20:
                if use_composition_color:
                    # Группируем по composition и x_param
                    grouped = filtered_data.groupby(['composition', x_param])[y_param].mean().reset_index()
                    unique_x = grouped[x_param].unique()
                    unique_compositions = grouped['composition'].unique()
                    
                    bar_width = 0.8 / len(unique_compositions)
                    x_positions = np.arange(len(unique_x))
                    
                    for i, composition in enumerate(unique_compositions):
                        comp_data = grouped[grouped['composition'] == composition]
                        values = [comp_data[comp_data[x_param] == x_val][y_param].values[0] 
                                if len(comp_data[comp_data[x_param] == x_val]) > 0 else 0 
                                for x_val in unique_x]
                        
                        ax.bar(x_positions + i * bar_width, values, bar_width,
                              label=composition, alpha=0.7)
                    
                    ax.set_xticks(x_positions + bar_width * (len(unique_compositions) - 1) / 2)
                    ax.set_xticklabels([str(x) for x in unique_x], rotation=45)
                else:
                    grouped = filtered_data.groupby(x_param)[y_param].mean()
                    x_positions = range(len(grouped))
                    ax.bar(x_positions, grouped.values, alpha=0.7)
                    ax.set_xticks(x_positions)
                    ax.set_xticklabels([str(label) for label in grouped.index], rotation=45)
                
                ax.set_xlabel(get_param_display_name(x_param))
                ax.set_ylabel(get_param_display_name(y_param))
            else:
                if use_composition_color:
                    unique_compositions = filtered_data['composition'].unique()
                    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_compositions)))
                    
                    for i, composition in enumerate(unique_compositions):
                        comp_data = filtered_data[filtered_data['composition'] == composition]
                        ax.hist(comp_data[x_param].dropna(), bins=15, alpha=0.7,
                               color=colors[i], label=composition, edgecolor='black')
                else:
                    ax.hist(filtered_data[x_param].dropna(), bins=20, alpha=0.7, edgecolor='black')
                
                ax.set_xlabel(get_param_display_name(x_param))
                ax.set_ylabel('Частота')
                
        elif graph_type == 'box':
            if use_composition_color:
                data_to_plot = []
                labels = []
                unique_compositions = filtered_data['composition'].unique()[:10]  # Ограничиваем количество
                
                for composition in unique_compositions:
                    comp_data = filtered_data[filtered_data['composition'] == composition]
                    if not comp_data.empty:
                        data_to_plot.append(comp_data[y_param].dropna())
                        labels.append(composition)
                
                if data_to_plot:
                    box_plot = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
                    # Раскрашиваем боксы
                    colors = plt.cm.viridis(np.linspace(0, 1, len(data_to_plot)))
                    for patch, color in zip(box_plot['boxes'], colors):
                        patch.set_facecolor(color)
            else:
                data_to_plot = [filtered_data[filtered_data[x_param] == cat][y_param].dropna() 
                              for cat in filtered_data[x_param].unique()[:10]]
                ax.boxplot(data_to_plot)
                ax.set_xticklabels([str(cat) for cat in filtered_data[x_param].unique()[:10]], rotation=45)
            
            ax.set_xlabel(get_param_display_name(x_param) if not use_composition_color else 'Составы')
            ax.set_ylabel(get_param_display_name(y_param))
            
        elif graph_type == 'pie':
            pie_data = filtered_data[x_param].value_counts()
            
            if len(pie_data) > 10:
                main_categories = pie_data.head(9)
                other_sum = pie_data.iloc[9:].sum()
                if other_sum > 0:
                    main_categories = pd.concat([main_categories, pd.Series([other_sum], index=['Другие'])])
                pie_data = main_categories
            
            ax.pie(pie_data.values, labels=pie_data.index, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            
        elif graph_type == 'heatmap':
            numeric_cols = filtered_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                corr_matrix = filtered_data[numeric_cols].corr()
                im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto')
                ax.set_xticks(range(len(corr_matrix.columns)))
                ax.set_yticks(range(len(corr_matrix.columns)))
                ax.set_xticklabels(corr_matrix.columns, rotation=45)
                ax.set_yticklabels(corr_matrix.columns)
                plt.colorbar(im, ax=ax)
            else:
                return None, "Для тепловой карты нужно больше числовых параметров", available_compositions
            
        elif graph_type == '3d_scatter' and z_param and z_param in filtered_data.columns:
            fig = plt.figure(figsize=(width/100, height/100))
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(filtered_data[x_param], filtered_data[y_param], filtered_data[z_param],
                               c=filtered_data[color_param] if color_param and color_param in filtered_data.columns else 'blue',
                               cmap='viridis' if color_param and color_param in filtered_data.columns else None,
                               alpha=0.7)
            ax.set_xlabel(get_param_display_name(x_param))
            ax.set_ylabel(get_param_display_name(y_param))
            ax.set_zlabel(get_param_display_name(z_param))
            if color_param and color_param in filtered_data.columns:
                plt.colorbar(scatter, ax=ax, label=get_param_display_name(color_param))
                
        elif graph_type == 'animated_scatter' and animation_param and animation_param in filtered_data.columns:
            return generate_animated_graph(filtered_data, x_param, y_param, animation_param, theme, title)
        else:
            ax.scatter(filtered_data[x_param], filtered_data[y_param], alpha=0.7, s=50)
        
        # Общие настройки для 2D графиков (кроме круговых и 3D)
        if not graph_type.startswith('3d') and graph_type != 'animated_scatter' and graph_type != 'pie':
            if graph_type != 'histogram':  # Для гистограммы уже установили подписи
                ax.set_xlabel(get_param_display_name(x_param))
                ax.set_ylabel(get_param_display_name(y_param))
            ax.grid(show_grid)
            
        # Заголовок
        if title:
            ax.set_title(title)
        else:
            if graph_type == 'pie':
                ax.set_title(f'Распределение {get_param_display_name(x_param)}')
            elif graph_type == 'histogram':
                ax.set_title(f'Гистограмма {get_param_display_name(x_param)}')
            elif graph_type == 'line':
                ax.set_title(f'Линейный график: {get_param_display_name(y_param)} vs {get_param_display_name(x_param)}')
            else:
                ax.set_title(f'{get_param_display_name(y_param)} vs {get_param_display_name(x_param)}')
        
        #ДОБАВЛЯЕМ ЛЕГЕНДУ ДЛЯ СОСТАВОВ
        if use_composition_color and graph_type not in ['pie', 'heatmap', '3d_scatter']:
            ax.legend(title='Составы', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        
        # Сохраняем график
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        graph = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        return graph, "График создан успешно", available_compositions
        
    except Exception as e:
        plt.close()
        import traceback
        traceback.print_exc()
        return None, f"Ошибка при создании графика: {str(e)}", available_compositions

# Вспомогательные функции для generate_graph
def handle_scatter_plot(ax, data, x_param, y_param, color_param):
    """Обрабатывает scatter plot"""
    if color_param and color_param in data.columns:
        scatter = ax.scatter(data[x_param], data[y_param], 
                            c=data[color_param], cmap='viridis', 
                            alpha=0.7, s=50)
        plt.colorbar(scatter, label=PARAM_NAMES.get(color_param, color_param))
        return scatter
    else:
        return ax.scatter(data[x_param], data[y_param], alpha=0.7, s=50)

def handle_bar_chart(ax, data, x_param, y_param):
    """Обрабатывает bar chart"""
    try:
        # Пытаемся сгруппировать по x_param
        if data[x_param].dtype == 'object' or len(data[x_param].unique()) <= 10:
            grouped = data.groupby(x_param)[y_param].mean()
            x_positions = range(len(grouped))
            ax.bar(x_positions, grouped.values, alpha=0.7)
            ax.set_xticks(x_positions)
            ax.set_xticklabels([str(label) for label in grouped.index], rotation=45)
        else:
            # Если слишком много уникальных значений, используем гистограмму
            ax.hist(data[x_param].dropna(), bins=20, alpha=0.7, edgecolor='black')
    except Exception as e:
        raise ValueError(f"Ошибка при создании bar chart: {e}")

def handle_heatmap(data, fig, ax, theme):
    """Обрабатывает heatmap"""
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        return None, "Для тепловой карты нужно как минимум 2 числовых параметра"
    
    corr_matrix = data[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    graph = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    
    return graph, "Тепловая карта создана успешно"

def handle_3d_scatter(ax, data, x_param, y_param, z_param, color_param):
    """Обрабатывает 3D scatter plot"""
    if color_param and color_param in data.columns:
        scatter = ax.scatter(data[x_param], data[y_param], data[z_param],
                           c=data[color_param], cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, ax=ax, label=PARAM_NAMES.get(color_param, color_param))
    else:
        ax.scatter(data[x_param], data[y_param], data[z_param], alpha=0.7)
    
    ax.set_xlabel(PARAM_NAMES.get(x_param, x_param.capitalize()))
    ax.set_ylabel(PARAM_NAMES.get(y_param, y_param.capitalize()))
    ax.set_zlabel(PARAM_NAMES.get(z_param, z_param.capitalize()))

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

def filter_parameters(data, param_group, show_diff_only=False):
    """Фильтрует параметры по группам с улучшенной логикой"""
    
    # Группы параметров с русскими описаниями
    param_groups = {
        'thermal': ['q', 'tign', 'tb', 'tau_b', 'tau_d1', 'tau_d2'],
        'mechanical': ['density', 'kf', 'kt', 'h', 'mass_loss'],
        'chemical': ['ad', 'cd', 'hd', 'nd', 'sd', 'od', 'vd', 'war'],
        'emissions': ['co2', 'co', 'so2', 'nox'],
        'combustion': ['mass_loss', 'tau_b', 'tau_d1', 'tau_d2', 'tign', 'tb']
    }
    
    if param_group != 'all' and param_group in param_groups:
        selected_params = ['composition'] + param_groups[param_group]
        # Оставляем только существующие колонки
        existing_params = [col for col in selected_params if col in data.columns]
        
        if not existing_params:
            return pd.DataFrame()  # Возвращаем пустой DataFrame если нет подходящих колонок
            
        filtered_data = data[existing_params]
    else:
        filtered_data = data.copy()
    
    # Дополнительная логика для показа только значимых различий
    if show_diff_only and len(data['composition'].unique()) >= 2:
        try:
            numeric_cols = filtered_data.select_dtypes(include=[np.number]).columns
            significant_cols = ['composition']  # Всегда оставляем composition
            
            for col in numeric_cols:
                if col != 'composition':
                    # Вычисляем различия между составами
                    composition_stats = filtered_data.groupby('composition')[col].mean()
                    if len(composition_stats) >= 2:
                        max_val = composition_stats.max()
                        min_val = composition_stats.min()
                        if max_val > 0:  # Избегаем деления на ноль
                            difference_pct = ((max_val - min_val) / max_val) * 100
                            if difference_pct >= 10:  # Порог 10%
                                significant_cols.append(col)
            
            # Если нашли значимые колонки, фильтруем
            if len(significant_cols) > 1:
                filtered_data = filtered_data[significant_cols]
                
        except Exception as e:
            print(f"Ошибка при фильтрации значимых различий: {e}")
            # В случае ошибки возвращаем все данные
    
    return filtered_data

def get_comparison_stats(data):
    """Возвращает статистику для сравнения"""
    if data.empty:
        return {}
    
    # Определяем имя колонки с составами (может быть 'composition' или 'Состав')
    composition_column = 'composition' if 'composition' in data.columns else 'Состав'
    
    if composition_column not in data.columns:
        return {
            'compositions_count': 0,
            'parameters_count': len(data.select_dtypes(include=[np.number]).columns),
            'total_rows': len(data)
        }
    
    numeric_data = data.select_dtypes(include=[np.number])
    stats = {
        'compositions_count': len(data[composition_column].unique()),
        'parameters_count': len(numeric_data.columns),
        'total_rows': len(data)
    }

    return stats

def get_compact_composition_table(data, max_compositions=10):
    """Создает компактную таблицу составов для легенды"""
    if data.empty or 'composition' not in data.columns:
        return None
    
    compositions = data['composition'].unique()
    if len(compositions) > max_compositions:
        # Группируем редко встречающиеся составы
        main_compositions = compositions[:max_compositions-1]
        other_count = len(compositions) - len(main_compositions)
        compositions = list(main_compositions) + [f'Другие ({other_count} составов)']
    
    # Создаем компактное представление
    compact_data = []
    for comp in compositions:
        comp_data = data[data['composition'] == comp]
        if not comp_data.empty:
            compact_data.append({
                'composition': comp,
                'samples': len(comp_data),
                'key_params': get_key_params_summary(comp_data)
            })
    
    return pd.DataFrame(compact_data)

def get_key_params_summary(comp_data, max_params=3):
    """Возвращает ключевые параметры для состава"""
    numeric_cols = comp_data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return "Нет числовых данных"
    
    # Выбираем параметры с наибольшей вариативностью
    summary = []
    for col in numeric_cols[:max_params]:
        mean_val = comp_data[col].mean()
        if not pd.isna(mean_val):
            summary.append(f"{PARAM_NAMES.get(col, col)}: {mean_val:.2f}")
    
    return "; ".join(summary) if summary else "Нет данных"

def get_param_description(graph_type, param_type):
    """Возвращает описание параметра для специальных графиков"""
    if graph_type in SPECIAL_GRAPH_PARAMS:
        return SPECIAL_GRAPH_PARAMS[graph_type].get(param_type, '')
    return ''