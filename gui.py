from flask import Flask, render_template, request, jsonify, session, flash
from flask_session import Session
import pandas as pd
import os
from data_processor import process_data_source
from database import query_db, insert_data, init_db
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
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'Uploads'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = os.path.join(os.path.dirname(__file__), 'sessions')
app.config['SECRET_KEY'] = 'your-secret-key'
Session(app)
db_path = 'pellets_data.db'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)
init_db(db_path)  # Инициализация базы данных


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

# Список доступных графиков
GRAPHS = ['scatter', 'line', 'histogram', 'bar', 'box', 'heatmap', 'violin', 
          'pie', 'sunburst', '3d_scatter', 'animated_scatter', 'radar', 'treemap']

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

def create_animated_scatter(data, x_param, y_param, animation_param, template, title):
    """Создает анимированный scatter plot"""
    fig = px.scatter(data, x=x_param, y=y_param,
                   animation_frame=animation_param,
                   color=color_param if color_param in data.columns else None,
                   size=size_param if size_param in data.columns else None,
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

@app.route('/')
def index():
    uploaded_files = os.listdir(app.config['UPLOAD_FOLDER']) if os.path.exists(app.config['UPLOAD_FOLDER']) else []
    show_data = session.get('show_data', False)
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 10))
    
    tables = []
    total_rows = []
    
    # ВСЕГДА загружаем данные из базы для отображения
    measured_data = query_db(db_path, "measured_parameters")
    components_data = query_db(db_path, "components")
    
    if not measured_data.empty:
        total_measured = len(measured_data)
        total_rows.append(total_measured)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        df_pag = measured_data.iloc[start_idx:end_idx] if total_measured > 0 else pd.DataFrame()
        tables.append({
            'name': 'Измеренные параметры',
            'data': df_pag.to_html(classes='table table-striped table-sm', index=False) if not df_pag.empty else 'Таблица пуста.'
        })

    if not components_data.empty:
        total_components = len(components_data)
        total_rows.append(total_components)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        df_pag = components_data.iloc[start_idx:end_idx] if total_components > 0 else pd.DataFrame()
        tables.append({
            'name': 'Компоненты',
            'data': df_pag.to_html(classes='table table-striped table-sm', index=False) if not df_pag.empty else 'Таблица пуста.'
        })
    
    # Если есть данные в базе, показываем их
    if not measured_data.empty or not components_data.empty:
        show_data = True
        session['show_data'] = True

    # ВАЖНО: возвращаем render_template в конце функции
    return render_template(
        'index.html',
        segment='Главная',
        uploaded_files=uploaded_files,
        tables=tables,
        show_data=show_data,
        page=page,
        per_page=per_page,
        total_rows=total_rows
    )

@app.route('/', methods=['POST'])
def upload_file():
    uploaded_files = os.listdir(app.config['UPLOAD_FOLDER']) if os.path.exists(app.config['UPLOAD_FOLDER']) else []
    if 'file' not in request.files:
        flash('Файл не предоставлен.', 'danger')
        return jsonify({
            'success': False,
            'message': 'Файл не предоставлен.',
            'uploaded_files': uploaded_files
        })
    file = request.files['file']
    if file.filename == '':
        flash('Файл не выбран.', 'danger')
        return jsonify({
            'success': False,
            'message': 'Файл не выбран.',
            'uploaded_files': uploaded_files
        })
    if file and (file.filename.endswith('.csv') or file.filename.endswith('.xlsx')):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        try:
            messages, components_sheet_name, sheet_data = process_data_source(file_path, db_path)
            
            # Загружаем данные из базы для отображения
            measured_data = query_db(db_path, "measured_parameters")
            components_data = query_db(db_path, "components")
            
            if not sheet_data:
                flash('Файл не содержит данных для отображения.', 'warning')
            else:
                flash(f'Загружено {len(sheet_data)} листов из файла {file.filename}.', 'success')
            
            # Сохраняем в сессию
            session['sheet_data'] = [
                {'name': s['name'], 'data': s['data'].to_json(orient='records', force_ascii=False)} for s in sheet_data
            ]
            session['components_sheet_name'] = components_sheet_name
            session['show_data'] = True
            
            # Сохраняем данные из базы в сессию для немедленного отображения
            session['measured_data'] = measured_data.to_json(orient='records', force_ascii=False)
            session['components_data'] = components_data.to_json(orient='records', force_ascii=False)
            
            flash(f'Данные сохранены в сессию: {len(sheet_data)} листов.', 'info')
            
            # Возвращаем данные для обновления фронтенда
            return jsonify({
                'success': True,
                'message': 'Файл успешно загружен.',
                'uploaded_files': uploaded_files,
                'messages': messages,
                'measured_data': measured_data.head(20).to_html(classes='table table-striped table-sm', index=False) if not measured_data.empty else '',
                'components_data': components_data.head(20).to_html(classes='table table-striped table-sm', index=False) if not components_data.empty else '',
                'total_measured': len(measured_data),
                'total_components': len(components_data),
                'refresh_page': True
            })
        except Exception as e:
            flash(f'Ошибка обработки файла {file.filename}: {str(e)}', 'danger')
            return jsonify({
                'success': False,
                'message': f'Ошибка обработки файла: {str(e)}',
                'uploaded_files': uploaded_files
            })
    flash('Недопустимый формат файла.', 'danger')
    return jsonify({
        'success': False,
        'message': 'Недопустимый формат файла.',
        'uploaded_files': uploaded_files
    })

@app.route('/load_file', methods=['POST'])
def load_file():
    selected_file = request.form.get('selected_file')
    uploaded_files = os.listdir(app.config['UPLOAD_FOLDER']) if os.path.exists(app.config['UPLOAD_FOLDER']) else []
    if not selected_file:
        return jsonify({
            'success': False,
            'message': 'Файл не выбран.',
            'uploaded_files': uploaded_files
        })
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], selected_file)
    if not os.path.exists(file_path):
        return jsonify({
            'success': False,
            'message': f'Файл {selected_file} не найден.',
            'uploaded_files': uploaded_files
        })
    try:
        process_messages, components_sheet_name = process_data_source(file_path, db_path)
        for msg in process_messages:
            category = "danger" if "Error" in msg or "Warning" in msg else "success"
            flash(msg, category)
        
        measured_data = query_db(db_path, "measured_parameters")
        components_data = query_db(db_path, "components")
        graph = generate_graph_simple(measured_data)
        if measured_data.empty and components_data.empty:
            return jsonify({
                'success': False,
                'message': 'Данные обработаны, но таблицы пусты. Проверьте формат файла или названия столбцов.',
                'uploaded_files': uploaded_files
            })
        session['show_data'] = True
        session['data_loaded'] = True
        session['measured_data'] = measured_data.to_json()
        session['components_data'] = components_data.to_json()
        session['graph'] = graph
        session['components_sheet_name'] = components_sheet_name
        return jsonify({
            'success': True,
            'message': f'Данные из файла {selected_file} успешно загружены.',
            'measured_data': measured_data.head(10).to_html(classes='table table-striped table-sm', index=False),
            'components_data': components_data.head(10).to_html(classes='table table-striped table-sm', index=False),
            'graph': graph,
            'uploaded_files': uploaded_files,
            'components_sheet_name': components_sheet_name,
            'total_measured': len(measured_data),
            'total_components': len(components_data)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Ошибка обработки файла {selected_file}: {str(e)}',
            'uploaded_files': uploaded_files
        })

@app.route('/search', methods=['POST'])
def search():
    try:
        search_column = request.form.get('search_column')
        search_operator = request.form.get('search_operator', '=')
        search_value = request.form.get('search_value', '').strip()
        
        print(f"=== ПОИСК: column={search_column}, operator={search_operator}, value={search_value}")
        
        # Базовая проверка
        if not search_column or not search_value:
            return jsonify({
                'success': False,
                'message': 'Заполните все поля поиска'
            })
        
        # Получаем все данные
        all_measured_data = query_db(db_path, "measured_parameters")
        
        if all_measured_data.empty:
            return jsonify({
                'success': False,
                'message': 'Нет данных для поиска. Сначала загрузите файл.'
            })
        
        print(f"=== ВСЕХ ДАННЫХ: {len(all_measured_data)} строк")
        
        # Простая и надежная фильтрация
        filtered_data = simple_filter(all_measured_data, search_column, search_operator, search_value)
        
        print(f"=== НАЙДЕНО: {len(filtered_data)} строк")
        
        if filtered_data.empty:
            return jsonify({
                'success': True,
                'message': 'По вашему запросу ничего не найдено',
                'measured_data': '<div class="alert alert-info">По вашему запросу ничего не найдено</div>',
                'total_measured': 0
            })
        
        # Сохраняем ТОЛЬКО результаты поиска
        session['search_results'] = filtered_data.to_json(orient='records', force_ascii=False)
        session['search_performed'] = True
        session['show_data'] = True
        
        return jsonify({
            'success': True,
            'message': f'Найдено {len(filtered_data)} записей',
            'refresh_page': True  # Перезагружаем страницу чтобы показать результаты
        })
        
    except Exception as e:
        print(f"=== ОШИБКА ПОИСКА: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Ошибка при поиске: {str(e)}'
        })

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

@app.route('/add_data', methods=['POST'])
def add_data():
    try:
        data = {
            'composition': request.form.get('composition'),
            'density': float(request.form.get('density', '')) if request.form.get('density', '') else None,
            'kf': float(request.form.get('kf', '')) if request.form.get('kf', '') else None,
            'kt': float(request.form.get('kt', '')) if request.form.get('kt', '') else None,
            'h': float(request.form.get('h', '')) if request.form.get('h', '') else None,
            'mass_loss': float(request.form.get('mass_loss', '')) if request.form.get('mass_loss', '') else None,
            'tign': float(request.form.get('tign', '')) if request.form.get('tign', '') else None,
            'tb': float(request.form.get('tb', '')) if request.form.get('tb', '') else None,
            'tau_d1': float(request.form.get('tau_d1', '')) if request.form.get('tau_d1', '') else None,
            'tau_d2': float(request.form.get('tau_d2', '')) if request.form.get('tau_d2', '') else None,
            'tau_b': float(request.form.get('tau_b', '')) if request.form.get('tau_b', '') else None,
            'co2': float(request.form.get('co2', '')) if request.form.get('co2', '') else None,
            'co': float(request.form.get('co', '')) if request.form.get('co', '') else None,
            'so2': float(request.form.get('so2', '')) if request.form.get('so2', '') else None,
            'nox': float(request.form.get('nox', '')) if request.form.get('nox', '') else None,
            'q': float(request.form.get('q', '')) if request.form.get('q', '') else None,
            'ad': float(request.form.get('ad', '')) if request.form.get('ad', '') else None
        }
        df = pd.DataFrame([data])
        insert_data(db_path, "measured_parameters", df)
        measured_data = query_db(db_path, "measured_parameters")
        components_data = query_db(db_path, "components")
        graph = generate_graph_simple(measured_data)
        session['show_data'] = True
        session['data_loaded'] = True
        session['measured_data'] = measured_data.to_json()
        session['components_data'] = components_data.to_json()
        session['graph'] = graph
        return jsonify({
            'success': True,
            'message': 'Данные успешно добавлены.',
            'measured_data': measured_data.head(10).to_html(classes='table table-striped table-sm', index=False),
            'components_data': components_data.head(10).to_html(classes='table table-striped table-sm', index=False),
            'graph': graph,
            'uploaded_files': os.listdir(app.config['UPLOAD_FOLDER']) if os.path.exists(app.config['UPLOAD_FOLDER']) else [],
            'components_sheet_name': session.get('components_sheet_name', 'Таблица компонентов'),
            'total_measured': len(measured_data),
            'total_components': len(components_data)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Ошибка при добавлении данных: {str(e)}',
            'uploaded_files': os.listdir(app.config['UPLOAD_FOLDER']) if os.path.exists(app.config['UPLOAD_FOLDER']) else []
        })

@app.route('/tables')
def tables():
    uploaded_files = os.listdir(app.config['UPLOAD_FOLDER']) if os.path.exists(app.config['UPLOAD_FOLDER']) else []
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 10))
    
    search_performed = session.get('search_performed', False)
    tables = []
    total_rows = []
    
    # Обрабатываем результаты поиска
    if search_performed:
        try:
            measured_data_json = session.get('search_results', '[]')
            measured_data = pd.read_json(measured_data_json) if measured_data_json != '[]' else pd.DataFrame()
            
            if not measured_data.empty:
                total_measured = len(measured_data)
                total_rows.append(total_measured)
                start_idx = (page - 1) * per_page
                end_idx = start_idx + per_page
                df_pag = measured_data.iloc[start_idx:end_idx] if total_measured > 0 else pd.DataFrame()
                
                tables.append({
                    'name': 'Результаты поиска',
                    'data': df_pag.to_html(classes='table table-striped table-sm', index=False) if not df_pag.empty else '<div class="alert alert-info">Ничего не найдено</div>'
                })
        except Exception as e:
            print(f"Ошибка загрузки результатов поиска: {e}")
    
    # Если нет результатов поиска, показываем все данные
    if not tables:
        measured_data = query_db(db_path, "measured_parameters")
        components_data = query_db(db_path, "components")
        
        if not measured_data.empty:
            total_measured = len(measured_data)
            total_rows.append(total_measured)
            start_idx = (page - 1) * per_page
            end_idx = start_idx + per_page
            df_pag = measured_data.iloc[start_idx:end_idx] if total_measured > 0 else pd.DataFrame()
            tables.append({
                'name': 'Измеренные параметры',
                'data': df_pag.to_html(classes='table table-striped table-sm', index=False) if not df_pag.empty else 'Таблица пуста.'
            })

        if not components_data.empty:
            total_components = len(components_data)
            total_rows.append(total_components)
            start_idx = (page - 1) * per_page
            end_idx = start_idx + per_page
            df_pag = components_data.iloc[start_idx:end_idx] if total_components > 0 else pd.DataFrame()
            tables.append({
                'name': 'Компоненты',
                'data': df_pag.to_html(classes='table table-striped table-sm', index=False) if not df_pag.empty else 'Таблица пуста.'
            })
    
    show_data = len(tables) > 0

    return render_template(
        'tables.html',
        segment='Таблицы',
        uploaded_files=uploaded_files,
        tables=tables,
        show_data=show_data,
        page=page,
        per_page=per_page,
        total_rows=total_rows,
        search_performed=search_performed
    )

@app.route('/clear_search', methods=['POST'])
def clear_search():
    session.pop('search_results', None)
    session.pop('search_performed', None)
    return jsonify({
        'success': True,
        'message': 'Поиск очищен, показаны все данные'
    })

@app.route('/compare')
def compare():
    uploaded_files = os.listdir(app.config['UPLOAD_FOLDER']) if os.path.exists(app.config['UPLOAD_FOLDER']) else []
    measured_data = query_db(db_path, "measured_parameters")
    compositions = measured_data['composition'].tolist() if not measured_data.empty else []
    return render_template('compare.html', segment='Сравнительная таблица',uploaded_files=uploaded_files, compositions=compositions)

@app.route('/compare', methods=['POST'])
def compare_data():
    try:
        # Получаем все выбранные составы
        compositions = []
        i = 1
        while True:
            comp = request.form.get(f'comp{i}')
            if comp:
                compositions.append(comp)
                i += 1
            else:
                break
        
        print(f"=== СРАВНЕНИЕ: выбрано {len(compositions)} составов: {compositions}")
        
        if len(compositions) < 2:
            return jsonify({
                'success': False,
                'message': 'Выберите хотя бы два состава для сравнения.'
            })
        
        # Получаем данные из базы
        measured_data = query_db(db_path, "measured_parameters")
        
        if measured_data.empty:
            return jsonify({
                'success': False,
                'message': 'В базе данных нет измеренных параметров.'
            })
        
        # Фильтруем данные по выбранным составам
        comparison_data = pd.DataFrame()
        found_compositions = []
        
        for comp in compositions:
            comp_data = measured_data[measured_data['composition'] == comp]
            if not comp_data.empty:
                comparison_data = pd.concat([comparison_data, comp_data])
                found_compositions.append(comp)
        
        if comparison_data.empty:
            return jsonify({
                'success': False,
                'message': 'Выбранные составы не найдены в базе данных.'
            })
        
        # Если не все составы найдены, сообщаем об этом
        if len(found_compositions) != len(compositions):
            missing = set(compositions) - set(found_compositions)
            flash(f'Некоторые составы не найдены: {", ".join(missing)}', 'warning')
        
        # Получаем настройки фильтрации
        show_all = request.form.get('show_all') == 'on'
        show_diff = request.form.get('show_diff') == 'on'
        param_group = request.form.get('paramGroup', 'all')
        
        # Применяем фильтрацию параметров
        filtered_data = filter_parameters(comparison_data, param_group, show_diff and not show_all)
        
        # Если после фильтрации данных нет
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
            'comparison': filtered_data.to_html(
                classes='table table-striped table-sm comparison-table', 
                index=False, 
                escape=False,
                na_rep='N/A'
            ),
            'compositions': found_compositions,
            'stats': get_comparison_stats(filtered_data)
        })
        
    except Exception as e:
        print(f"=== ОШИБКА СРАВНЕНИЯ: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'message': f'Ошибка при сравнении: {str(e)}'
        })

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

@app.route('/ai_analysis')
def ai_analysis():
    uploaded_files = os.listdir(app.config['UPLOAD_FOLDER']) if os.path.exists(app.config['UPLOAD_FOLDER']) else []
    return render_template('ai_analysis.html',segment='ИИ-анализ', uploaded_files=uploaded_files)

@app.route('/ai_analysis', methods=['POST'])
def perform_ai_analysis():
    return jsonify({
        'success': True,
        'message': 'Анализ ИИ пока не реализован. Это заглушка.'
    })

@app.route('/create_graph', methods=['GET', 'POST'])
def create_graph():
    uploaded_files = os.listdir(app.config['UPLOAD_FOLDER']) if os.path.exists(app.config['UPLOAD_FOLDER']) else []
    measured_data = query_db(db_path, "measured_parameters")
    parameters = measured_data.columns.tolist() if not measured_data.empty else []
    
    # Получаем выбранный тип визуализации
    selected_viz_type = request.form.get('viz_type', 'matplotlib') if request.method == 'POST' else 'matplotlib'
    
    # Выбираем соответствующий список графиков
    if selected_viz_type == 'plotly':
        graphs = PLOTLY_GRAPHS
    elif selected_viz_type == 'seaborn':
        graphs = SEABORN_GRAPHS
    else:  # matplotlib
        graphs = MATPLOTLIB_GRAPHS

    if request.method == 'POST':
        try:
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
            
            stats = get_data_statistics(measured_data)

            if measured_data.empty:
                return jsonify({
                    'success': False,
                    'message': 'Нет данных для построения графика. Сначала загрузите файл.',
                    'stats': stats
                })
            
            # Выбор типа визуализации
            graph = None
            graph_message = ""
            graph_output_type = "matplotlib"
            
            if viz_type == 'plotly':
                graph, graph_message = generate_plotly_graph(
                    measured_data, x_param, y_param, graph_type,
                    z_param, color_param, size_param, animation_param,
                    theme, title, width, height, show_grid
                )
                graph_output_type = 'plotly'
                
            elif viz_type == 'seaborn':
                graph, graph_message = generate_seaborn_plot(
                    measured_data, x_param, y_param, graph_type, theme, color_param
                )
                graph_output_type = 'matplotlib'
                
            else:  # matplotlib
                graph, graph_message = generate_graph(
                    measured_data, x_param, y_param, graph_type,
                    z_param, color_param, size_param, animation_param,
                    theme, title, width, height, show_grid
                )
                graph_output_type = 'matplotlib'
            
            if not graph:
                return jsonify({
                    'success': False,
                    'message': graph_message or 'Не удалось создать график',
                    'stats': stats
                })
                
            return jsonify({
                'success': True,
                'message': graph_message,
                'graph': graph,
                'graph_type': graph_output_type,
                'stats': stats
            })
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({
                'success': False,
                'message': f'Ошибка при создании графика: {str(e)}',
                'stats': get_data_statistics(measured_data)
            })

    # GET запрос
    graph, _ = generate_graph(measured_data)
    stats = get_data_statistics(measured_data)
    
    return render_template(
        'create_graph.html',
        segment='Создание графика',
        uploaded_files=uploaded_files,
        parameters=parameters,
        graphs=graphs,
        viz_types=VIZ_TYPES,
        selected_viz_type=selected_viz_type,
        graph=graph,
        components_sheet_name=session.get('components_sheet_name', 'Таблица компонентов'),
        stats=stats
    )

@app.route('/<path:path>.map')
def ignore_map_files(path):
    return '', 204

if __name__ == '__main__':
    app.run(debug=True)