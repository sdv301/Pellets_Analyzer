"""Интерактивные графики Plotly (модуль подсистемы Charts)."""
from typing import Optional, List, Tuple, Any, Dict
import json
import logging
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from app.services.charts.palette import (
    COLOR_PALETTE,
    get_param_display_name,
    get_plotly_theme,
)
from app.services.charts.registry import ChartRegistry

logger = logging.getLogger(__name__)


def create_radar_chart(data: pd.DataFrame, color_param: Optional[str], template: str, title: Optional[str]) -> Optional[go.Figure]:
    """Создает радарную диаграмму."""
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 3:
        return None
    
    radar_cols = numeric_cols[:6]
    categories = data[color_param].unique() if color_param and color_param in data.columns else ['Все данные']
    
    fig = go.Figure()
    for category in categories:
        cat_data = data[data[color_param] == category] if color_param and color_param in data.columns else data
        values = [cat_data[col].mean() for col in radar_cols]
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=[get_param_display_name(col) for col in radar_cols],
            fill='toself',
            name=str(category)
        ))
    
    max_val = max([max(trace.r) for trace in fig.data]) if fig.data else 10.0
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, max_val * 1.1])),
        title=title or "Радарная диаграмма параметров",
        template=template,
        showlegend=False
    )
    return fig


def create_plotly_radar_chart(data: pd.DataFrame, color_param: Optional[str], template: str, title: Optional[str]) -> Optional[go.Figure]:
    """Алиас для create_radar_chart для обратной совместимости."""
    return create_radar_chart(data, color_param, template, title)


def create_animated_scatter(data: pd.DataFrame, x_param: str, y_param: str,
                            animation_param: str, template: str, title: Optional[str],
                            color_param: Optional[str] = None, size_param: Optional[str] = None) -> Optional[go.Figure]:
    """Создает анимированный scatter plot."""
    if animation_param not in data.columns:
        return None
    return px.scatter(
        data, x=x_param, y=y_param,
        animation_frame=animation_param,
        animation_group='composition' if 'composition' in data.columns else None,
        color=color_param if color_param and color_param in data.columns else None,
        size=size_param if size_param and size_param in data.columns else None,
        title=title or f"Анимация {get_param_display_name(y_param)} vs {get_param_display_name(x_param)} по {animation_param}",
        template=template
    )


# -------------------------------------------------------------------------
# Регистрация билдеров для Plotly
# -------------------------------------------------------------------------

@ChartRegistry.register('plotly', 'scatter')
def _build_scatter(df, x, y, ctx):
    color = 'composition' if 'composition' in df.columns else (ctx.get('color') if ctx.get('color') in df.columns else None)
    size = ctx.get('size') if ctx.get('size') in df.columns else None
    hover_data = {'composition': True} if 'composition' in df.columns else None
    title = ctx.get('title') or f"{get_param_display_name(y)} vs {get_param_display_name(x)}"

    fig = px.scatter(
        df, x=x, y=y, color=color, size=size,
        hover_name='composition' if 'composition' in df.columns else None,
        hover_data=hover_data, title=title, template=ctx['template']
    )
    htemplate = (
        "<b>%{hovertext}</b><br>"
        f"{get_param_display_name(x)}: %{{x:.2f}}<br>"
        f"{get_param_display_name(y)}: %{{y:.2f}}<br><extra></extra>"
    ) if 'composition' in df.columns else (
        f"{get_param_display_name(x)}: %{{x:.2f}}<br>"
        f"{get_param_display_name(y)}: %{{y:.2f}}<br><extra></extra>"
    )
    fig.update_traces(hovertemplate=htemplate)
    return fig


@ChartRegistry.register('plotly', 'line')
def _build_line(df, x, y, ctx):
    sorted_df = df.sort_values(by=x)
    color = 'composition' if 'composition' in sorted_df.columns else None
    title = ctx.get('title') or f"Линейный график: {get_param_display_name(y)} от {get_param_display_name(x)}"
    return px.line(sorted_df, x=x, y=y, color=color, title=title, template=ctx['template'])


@ChartRegistry.register('plotly', 'bar')
def _build_bar(df, x, y, ctx):
    color = 'composition' if 'composition' in df.columns else None
    title = ctx.get('title') or f"Столбчатая диаграмма: {get_param_display_name(y)}"
    return px.bar(df, x=x, y=y, color=color, title=title, template=ctx['template'])


@ChartRegistry.register('plotly', 'box')
def _build_box(df, x, y, ctx):
    title = ctx.get('title') or f"Распределение {get_param_display_name(y)}"
    if 'composition' in df.columns:
        fig = px.box(df, x='composition', y=y, color='composition', title=title, template=ctx['template'])
    else:
        fig = px.box(df, x=x, y=y, title=title, template=ctx['template'])
    return fig


@ChartRegistry.register('plotly', 'histogram')
def _build_histogram(df, x, y, ctx):
    color = 'composition' if 'composition' in df.columns else None
    title = ctx.get('title') or f"Гистограмма распределения {get_param_display_name(x)}"
    return px.histogram(df, x=x, color=color, nbins=30, title=title, template=ctx['template'])


@ChartRegistry.register('plotly', 'violin')
def _build_violin(df, x, y, ctx):
    x_col = 'composition' if 'composition' in df.columns else x
    title = ctx.get('title') or f"Violin Plot: {get_param_display_name(y)}"
    return px.violin(df, x=x_col, y=y, box=True, points="all", title=title, template=ctx['template'])


@ChartRegistry.register('plotly', 'pie')
def _build_pie(df, x, y, ctx):
    title = ctx.get('title') or f"Круговая диаграмма: {get_param_display_name(x)}"
    if y and y in df.columns:
        return px.pie(df, names=x, values=y, title=title, template=ctx['template'])
    return px.pie(df, names=x, title=title, template=ctx['template'])


@ChartRegistry.register('plotly', 'heatmap')
def _build_heatmap(df, x, y, ctx):
    numeric_data = df.select_dtypes(include=[np.number])
    title = ctx.get('title') or "Тепловая карта корреляций"
    if len(numeric_data.columns) >= 2:
        corr_matrix = numeric_data.corr()
        return px.imshow(
            corr_matrix,
            labels=dict(x="Параметр X", y="Параметр Y", color="Корреляция"),
            x=[get_param_display_name(c) for c in corr_matrix.columns],
            y=[get_param_display_name(c) for c in corr_matrix.columns],
            color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
            title=title, template=ctx['template']
        )
    return None


@ChartRegistry.register('plotly', 'radar')
def _build_radar(df, x, y, ctx):
    return create_radar_chart(df, ctx.get('color'), ctx['template'], ctx.get('title'))


@ChartRegistry.register('plotly', '3d_scatter')
def _build_3d(df, x, y, ctx):
    z = ctx.get('z')
    if not z or z not in df.columns:
        num_cols = df.select_dtypes(include=[np.number]).columns
        avail = [c for c in num_cols if c not in (x, y)]
        z = avail[0] if avail else x
    color = 'composition' if 'composition' in df.columns else None
    title = ctx.get('title') or f"3D Scatter: {get_param_display_name(x)}, {get_param_display_name(y)}, {get_param_display_name(z)}"
    return px.scatter_3d(df, x=x, y=y, z=z, color=color, title=title, template=ctx['template'])


@ChartRegistry.register('plotly', 'contour')
def _build_contour(df, x, y, ctx):
    title = ctx.get('title') or f"Контурный график: {get_param_display_name(x)} vs {get_param_display_name(y)}"
    return px.density_contour(df, x=x, y=y, title=title, template=ctx['template'])


@ChartRegistry.register('plotly', 'bubble')
def _build_bubble(df, x, y, ctx):
    size = ctx.get('size') if ctx.get('size') in df.columns else None
    color = 'composition' if 'composition' in df.columns else None
    title = ctx.get('title') or f"Пузырьковая диаграмма"
    return px.scatter(df, x=x, y=y, size=size, color=color, title=title, template=ctx['template'])


@ChartRegistry.register('plotly', 'parallel_coords')
def _build_parallel(df, x, y, ctx):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:5]
    if len(num_cols) >= 2:
        return px.parallel_coordinates(df, dimensions=num_cols, template=ctx['template'])
    return None


@ChartRegistry.register('plotly', 'ternary')
def _build_ternary(df, x, y, ctx):
    z = ctx.get('z') or 'density'
    if z in df.columns and x in df.columns and y in df.columns:
        return px.scatter_ternary(df, a=x, b=y, c=z, template=ctx['template'])
    return None


def generate_plotly_graph(
    data: pd.DataFrame,
    x_param: str = 'ad',
    y_param: str = 'q',
    graph_type: str = 'scatter',
    z_param: Optional[str] = None,
    color_param: Optional[str] = None,
    size_param: Optional[str] = None,
    animation_param: Optional[str] = None,
    theme: str = 'default',
    title: Optional[str] = None,
    width: int = 800,
    height: int = 600,
    show_grid: bool = True,
    selected_compositions: Optional[List[str]] = None,
) -> Tuple[Optional[str], Optional[str], List[str]]:
    """Создает интерактивные графики с использованием Plotly через ChartRegistry."""
    if data.empty or x_param not in data.columns:
        return None, "Нет данных или неверный параметр X для построения графика", []
    
    try:
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
        
        template = get_plotly_theme(theme)
        
        if graph_type not in ['pie', 'histogram', 'radar', 'heatmap']:
            if not y_param or y_param not in filtered_data.columns or filtered_data[y_param].dropna().empty:
                return None, f"Отсутствует или неверный параметр Y ({y_param})", available_compositions
        
        # Контекст для билдеров
        ctx = {
            'template': template,
            'color': color_param,
            'size': size_param,
            'z': z_param,
            'animation': animation_param,
            'title': title,
            'width': width,
            'height': height,
        }
        
        if animation_param and animation_param in filtered_data.columns:
            fig = create_animated_scatter(filtered_data, x_param, y_param, animation_param, template, title, color_param, size_param)
        else:
            builder = ChartRegistry.get('plotly', graph_type)
            if builder is None:
                builder = _build_scatter
            fig = builder(filtered_data, x_param, y_param, ctx)
        
        if fig is None:
            return None, "Недостаточно числовых данных для построения графика", available_compositions
        
        fig.update_layout(
            autosize=True,
            margin=dict(l=60, r=40, t=60, b=60),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.25,
                xanchor="center",
                x=0.5
            )
        )
        
        graph_json = pio.to_json(fig, validate=False, pretty=False)
        return graph_json, f"{graph_type.capitalize()} график создан успешно", available_compositions
        
    except Exception as e:
        logger.error(f"Ошибка при создании Plotly графика {graph_type}: {e}", exc_info=True)
        return None, f"Ошибка при создании Plotly графика {graph_type}: {str(e)}", available_compositions
