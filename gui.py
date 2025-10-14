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
    'ad': 'Зольность',
    'q': 'Теплота сгорания', 
    'density': 'Плотность',
    'kf': 'Коэффициент формы',
    'kt': 'Коэффициент теплопроводности',
    'h': 'Высота',
    'mass_loss': 'Потеря массы',
    'tign': 'Температура воспламенения',
    'tb': 'Температура горения',
    'tau_d1': 'Время задержки 1',
    'tau_d2': 'Время задержки 2', 
    'tau_b': 'Время горения',
    'co2': 'CO2',
    'co': 'CO',
    'so2': 'SO2',
    'nox': 'NOx',
    'war': 'Влажность',
    'vd': 'Летучие вещества',
    'cd': 'Углерод',
    'hd': 'Водород',
    'nd': 'Азот',
    'sd': 'Сера',
    'od': 'Кислород'
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
                         width=800, height=600, show_grid=True):
    """Создает интерактивные графики с использованием Plotly"""
    
    if data.empty:
        return None, "Нет данных для построения графика"
    
    if x_param not in data.columns:
        return None, f"Параметр X '{x_param}' не найден в данных"
    
    try:
        # Настройка темы
        template = get_plotly_theme(theme)
        
        fig = None
        
        if graph_type == 'scatter':
            fig = px.scatter(data, x=x_param, y=y_param,
                           color=color_param if color_param and color_param in data.columns else None,
                           size=size_param if size_param and size_param in data.columns else None,
                           title=title or f"{get_param_display_name(y_param)} vs {get_param_display_name(x_param)}")
            
        elif graph_type == 'line':
            sorted_data = data.sort_values(by=x_param)
            fig = px.line(sorted_data, x=x_param, y=y_param,
                         color=color_param if color_param and color_param in data.columns else None,
                         title=title or f"Линейный график: {get_param_display_name(y_param)} vs {get_param_display_name(x_param)}")
            
        elif graph_type == 'bar':
            if data[x_param].dtype == 'object' or len(data[x_param].unique()) <= 20:
                grouped = data.groupby(x_param)[y_param].mean().reset_index()
                fig = px.bar(grouped, x=x_param, y=y_param,
                           color=color_param if color_param and color_param in data.columns else None,
                           title=title or f"Среднее {get_param_display_name(y_param)} по {get_param_display_name(x_param)}")
            else:
                fig = px.histogram(data, x=x_param,
                                 title=title or f"Распределение {get_param_display_name(x_param)}")
                
        elif graph_type == 'histogram':
            fig = px.histogram(data, x=x_param,
                             color=color_param if color_param and color_param in data.columns else None,
                             title=title or f"Гистограмма {get_param_display_name(x_param)}")
            
        elif graph_type == 'box':
            fig = px.box(data, x=x_param, y=y_param,
                        color=color_param if color_param and color_param in data.columns else None,
                        title=title or f"Box plot: {get_param_display_name(y_param)}")
            
        elif graph_type == 'violin':
            fig = px.violin(data, x=x_param, y=y_param,
                          color=color_param if color_param and color_param in data.columns else None,
                          title=title or f"Violin plot: {get_param_display_name(y_param)}")
            
        elif graph_type == 'pie':
            if data[x_param].dtype == 'object' or len(data[x_param].unique()) <= 20:
                pie_data = data[x_param].value_counts().reset_index()
                pie_data.columns = ['category', 'count']
                fig = px.pie(pie_data, values='count', names='category',
                            title=title or f"Распределение {get_param_display_name(x_param)}")
            else:
                return None, "Для круговой диаграммы нужны категориальные данные"
                
        elif graph_type == 'heatmap':
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                corr_matrix = data[numeric_cols].corr()
                fig = px.imshow(corr_matrix, 
                              color_continuous_scale='RdBu_r',
                              title=title or "Тепловая карта корреляций")
            else:
                return None, "Для тепловой карты нужно как минимум 2 числовых параметра"
                
        elif graph_type == 'radar':
            fig = create_plotly_radar_chart(data, color_param, template, title)
            if fig is None:
                return None, "Не удалось создать радарную диаграмму"
                
        elif graph_type == '3d_scatter' and z_param and z_param in data.columns:
            fig = px.scatter_3d(data, x=x_param, y=y_param, z=z_param,
                              color=color_param if color_param and color_param in data.columns else None,
                              title=title or "3D Scatter Plot")
            
        elif graph_type == 'animated_scatter' and animation_param and animation_param in data.columns:
            fig = px.scatter(data, x=x_param, y=y_param,
                           animation_frame=animation_param,
                           color=color_param if color_param and color_param in data.columns else None,
                           size=size_param if size_param and size_param in data.columns else None,
                           title=title or f"Анимированный график по {get_param_display_name(animation_param)}")
            
        else:
            fig = px.scatter(data, x=x_param, y=y_param,
                           title=title or f"{get_param_display_name(y_param)} vs {get_param_display_name(x_param)}")
        
        # Общие настройки
        if fig:
            fig.update_layout(
                width=width,
                height=height,
                template=template,
                showlegend=True,
                font=dict(size=12),
                margin=dict(l=50, r=50, t=50, b=50)
            )
            
            if show_grid:
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
            
            # Конвертируем в HTML
            graph_html = pio.to_html(
                fig,
                include_plotlyjs='cdn',
                full_html=False,
                config={
                    'responsive': True,
                    'displayModeBar': True,
                    'displaylogo': False
                }
            )
            
            return graph_html, "Plotly график создан успешно"
        else:
            return None, "Не удалось создать график"
            
    except Exception as e:
        import traceback
        error_msg = f"Ошибка при создании Plotly графика: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return None, error_msg

def create_correlation_heatmap(data, template, title):
    """Создает тепловую карту корреляций с Seaborn стилем"""
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) < 2:
        return None
    
    corr_matrix = data[numeric_cols].corr()
    
    # Создаем heatmap с Plotly
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=[get_param_display_name(col) for col in corr_matrix.columns],
        y=[get_param_display_name(col) for col in corr_matrix.index],
        colorscale='RdBu_r',
        zmid=0,
        hoverongaps=False,
        text=corr_matrix.round(2).values,
        texttemplate='%{text}',
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title=title or "Тепловая карта корреляций",
        template=template,
        xaxis_title="Параметры",
        yaxis_title="Параметры"
    )
    
    return fig

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

def generate_seaborn_plot(data, x_param='ad', y_param='q', plot_type='scatter', theme='default', color_param=None):
    """Создает статические графики с использованием Seaborn"""
    try:
        # Применяем тему Seaborn
        sns.set_theme(style=get_seaborn_style(theme))
        
        plt.figure(figsize=(12, 8))
        
        if plot_type == 'scatter':
            if color_param and color_param in data.columns:
                sns.scatterplot(data=data, x=x_param, y=y_param, hue=color_param, palette='viridis')
            else:
                sns.scatterplot(data=data, x=x_param, y=y_param)
            
        elif plot_type == 'line':
            # ДЛЯ ЛИНЕЙНОГО ГРАФИКА - важно сортировать по X
            sorted_data = data.sort_values(by=x_param)
            if color_param and color_param in data.columns:
                sns.lineplot(data=sorted_data, x=x_param, y=y_param, hue=color_param, palette='viridis')
            else:
                sns.lineplot(data=sorted_data, x=x_param, y=y_param)
            
        elif plot_type == 'violin':
            if color_param and color_param in data.columns:
                sns.violinplot(data=data, x=x_param, y=y_param, hue=color_param, palette='viridis')
            else:
                sns.violinplot(data=data, x=x_param, y=y_param)
            
        elif plot_type == 'box':
            if color_param and color_param in data.columns:
                sns.boxplot(data=data, x=x_param, y=y_param, hue=color_param, palette='viridis')
            else:
                sns.boxplot(data=data, x=x_param, y=y_param)
            
        elif plot_type == 'histogram':
            # ИСПРАВЛЕННАЯ ГИСТОГРАММА
            if color_param and color_param in data.columns:
                sns.histplot(data=data, x=x_param, hue=color_param, kde=True, palette='viridis', multiple="layer")
            else:
                sns.histplot(data=data, x=x_param, kde=True)
            
        elif plot_type == 'bar':
            # Для bar chart группируем и усредняем
            if data[x_param].dtype == 'object' or len(data[x_param].unique()) <= 20:
                if color_param and color_param in data.columns:
                    grouped = data.groupby([x_param, color_param])[y_param].mean().reset_index()
                    sns.barplot(data=grouped, x=x_param, y=y_param, hue=color_param, palette='viridis')
                else:
                    grouped = data.groupby(x_param)[y_param].mean().reset_index()
                    sns.barplot(data=grouped, x=x_param, y=y_param, palette='viridis')
            else:
                # Если много уникальных значений, используем гистограмму
                if color_param and color_param in data.columns:
                    sns.histplot(data=data, x=x_param, hue=color_param, kde=True, palette='viridis')
                else:
                    sns.histplot(data=data, x=x_param, kde=True)
                
        elif plot_type == 'heatmap':
            numeric_data = data.select_dtypes(include=[np.number])
            if len(numeric_data.columns) >= 2:
                corr_matrix = numeric_data.corr()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
            else:
                return None, "Для тепловой карты нужно как минимум 2 числовых параметра"
        
        elif plot_type == 'pie':
            # Круговая диаграмма через matplotlib
            pie_data = data[x_param].value_counts()
            
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
        
        plt.tight_layout()
        
        # Сохраняем в base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        graph = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        return graph, "Seaborn график создан успешно"
        
    except Exception as e:
        plt.close()
        import traceback
        traceback.print_exc()
        return None, f"Ошибка при создании Seaborn графика: {str(e)}"

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
    
    if data.empty or x_param not in data.columns:
        return None, "Нет данных для построения графика"
    
    try:
        # Применяем тему
        apply_theme(theme)
        
        # Для гистограмм и круговых диаграмм используем только X параметр
        if graph_type in ['histogram', 'pie']:
            fig, ax = plt.subplots(figsize=(width/100, height/100))
        else:
            fig, ax = plt.subplots(figsize=(width/100, height/100))
        
        if graph_type == 'scatter':
            if color_param and color_param in data.columns:
                scatter = ax.scatter(data[x_param], data[y_param], 
                                    c=data[color_param], cmap='viridis', 
                                    alpha=0.7, s=50)
                plt.colorbar(scatter, label=PARAM_NAMES.get(color_param, color_param))
            else:
                ax.scatter(data[x_param], data[y_param], alpha=0.7, s=50)
                
        elif graph_type == 'line':
            # СОРТИРУЕМ ДАННЫЕ ДЛЯ ЛИНЕЙНОГО ГРАФИКА
            sorted_data = data.sort_values(by=x_param)
            ax.plot(sorted_data[x_param], sorted_data[y_param], linewidth=2, marker='o')
            
        elif graph_type == 'histogram':
            # ИСПРАВЛЕННАЯ ГИСТОГРАММА
            ax.hist(data[x_param].dropna(), bins=20, alpha=0.7, edgecolor='black')
            ax.set_xlabel(get_param_display_name(x_param))
            ax.set_ylabel('Частота')
            
        elif graph_type == 'bar':
            if data[x_param].dtype == 'object' or len(data[x_param].unique()) <= 20:
                grouped = data.groupby(x_param)[y_param].mean()
                x_positions = range(len(grouped))
                ax.bar(x_positions, grouped.values, alpha=0.7)
                ax.set_xticks(x_positions)
                ax.set_xticklabels([str(label) for label in grouped.index], rotation=45)
                ax.set_xlabel(get_param_display_name(x_param))
                ax.set_ylabel(get_param_display_name(y_param))
            else:
                ax.hist(data[x_param].dropna(), bins=20, alpha=0.7, edgecolor='black')
                ax.set_xlabel(get_param_display_name(x_param))
                ax.set_ylabel('Частота')
                
        elif graph_type == 'box':
            data_to_plot = [data[data[x_param] == cat][y_param].dropna() 
                          for cat in data[x_param].unique()[:10]]  # Ограничиваем количество категорий
            ax.boxplot(data_to_plot)
            ax.set_xticklabels([str(cat) for cat in data[x_param].unique()[:10]], rotation=45)
            ax.set_xlabel(get_param_display_name(x_param))
            ax.set_ylabel(get_param_display_name(y_param))
            
        elif graph_type == 'pie':
            pie_data = data[x_param].value_counts()
            
            if len(pie_data) > 10:
                main_categories = pie_data.head(9)
                other_sum = pie_data.iloc[9:].sum()
                if other_sum > 0:
                    main_categories = pd.concat([main_categories, pd.Series([other_sum], index=['Другие'])])
                pie_data = main_categories
            
            ax.pie(pie_data.values, labels=pie_data.index, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            
        elif graph_type == 'heatmap':
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                corr_matrix = data[numeric_cols].corr()
                im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto')
                ax.set_xticks(range(len(corr_matrix.columns)))
                ax.set_yticks(range(len(corr_matrix.columns)))
                ax.set_xticklabels(corr_matrix.columns, rotation=45)
                ax.set_yticklabels(corr_matrix.columns)
                plt.colorbar(im, ax=ax)
            else:
                return None, "Для тепловой карты нужно больше числовых параметров"
            
        elif graph_type == '3d_scatter' and z_param and z_param in data.columns:
            fig = plt.figure(figsize=(width/100, height/100))
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(data[x_param], data[y_param], data[z_param],
                               c=data[color_param] if color_param and color_param in data.columns else 'blue',
                               cmap='viridis' if color_param and color_param in data.columns else None,
                               alpha=0.7)
            ax.set_xlabel(get_param_display_name(x_param))
            ax.set_ylabel(get_param_display_name(y_param))
            ax.set_zlabel(get_param_display_name(z_param))
            if color_param and color_param in data.columns:
                plt.colorbar(scatter, ax=ax, label=get_param_display_name(color_param))
                
        elif graph_type == 'animated_scatter' and animation_param and animation_param in data.columns:
            return generate_animated_graph(data, x_param, y_param, animation_param, theme, title)
        else:
            ax.scatter(data[x_param], data[y_param], alpha=0.7, s=50)
        
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
        import traceback
        traceback.print_exc()
        return None, f"Ошибка при создании графика: {str(e)}"

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
    
    numeric_data = data.select_dtypes(include=[np.number])
    stats = {
        'compositions_count': len(data['composition'].unique()),
        'parameters_count': len(numeric_data.columns),
        'total_rows': len(data)
    }
    
    return stats