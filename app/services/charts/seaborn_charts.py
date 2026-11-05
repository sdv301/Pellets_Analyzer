"""Статистические графики Seaborn (модуль подсистемы Charts)."""
from typing import Optional, List, Tuple, Any
import base64
import io
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from app.services.charts.palette import (
    COLOR_PALETTE,
    get_param_display_name,
    get_seaborn_style,
)
from app.services.charts.registry import ChartRegistry

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# Регистрация билдеров для Seaborn
# -------------------------------------------------------------------------

@ChartRegistry.register('seaborn', 'scatter')
def _build_sb_scatter(df, x, y, ax, ctx):
    hue = 'composition' if ctx.get('use_comp_color') else ctx.get('color')
    if hue and hue in df.columns:
        sns.scatterplot(data=df, x=x, y=y, hue=hue, palette='viridis', s=50, ax=ax)
    else:
        sns.scatterplot(data=df, x=x, y=y, s=50, ax=ax)


@ChartRegistry.register('seaborn', 'line')
def _build_sb_line(df, x, y, ax, ctx):
    sorted_df = df.sort_values(by=x)
    hue = 'composition' if ctx.get('use_comp_color') else ctx.get('color')
    if hue and hue in sorted_df.columns:
        sns.lineplot(data=sorted_df, x=x, y=y, hue=hue, palette='viridis', marker='o', ax=ax)
    else:
        sns.lineplot(data=sorted_df, x=x, y=y, marker='o', ax=ax)


@ChartRegistry.register('seaborn', 'violin')
def _build_sb_violin(df, x, y, ax, ctx):
    hue = 'composition' if ctx.get('use_comp_color') else ctx.get('color')
    if hue and hue in df.columns:
        sns.violinplot(data=df, x=x, y=y, hue=hue, palette='viridis', ax=ax)
    else:
        sns.violinplot(data=df, x=x, y=y, ax=ax)


@ChartRegistry.register('seaborn', 'box')
def _build_sb_box(df, x, y, ax, ctx):
    hue = 'composition' if ctx.get('use_comp_color') else ctx.get('color')
    if hue and hue in df.columns:
        sns.boxplot(data=df, x=x, y=y, hue=hue, palette='viridis', ax=ax)
    else:
        sns.boxplot(data=df, x=x, y=y, ax=ax)


@ChartRegistry.register('seaborn', 'histogram')
def _build_sb_histogram(df, x, y, ax, ctx):
    hue = 'composition' if ctx.get('use_comp_color') else ctx.get('color')
    if hue and hue in df.columns:
        sns.histplot(data=df, x=x, hue=hue, kde=True, palette='viridis', multiple="layer", ax=ax)
    else:
        sns.histplot(data=df, x=x, kde=True, ax=ax)


@ChartRegistry.register('seaborn', 'bar')
def _build_sb_bar(df, x, y, ax, ctx):
    hue = 'composition' if ctx.get('use_comp_color') else ctx.get('color')
    if df[x].dtype == 'object' or len(df[x].unique()) <= 20:
        if hue and hue in df.columns:
            sns.barplot(data=df, x=x, y=y, hue=hue, palette='viridis', ax=ax)
        else:
            sns.barplot(data=df, x=x, y=y, palette='viridis', ax=ax)
    else:
        if hue and hue in df.columns:
            sns.histplot(data=df, x=x, hue=hue, kde=True, palette='viridis', ax=ax)
        else:
            sns.histplot(data=df, x=x, kde=True, ax=ax)


@ChartRegistry.register('seaborn', 'heatmap')
def _build_sb_heatmap(df, x, y, ax, ctx):
    numeric_data = df.select_dtypes(include=[np.number])
    if len(numeric_data.columns) >= 2:
        corr_matrix = numeric_data.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', ax=ax)
        return True
    return False


@ChartRegistry.register('seaborn', 'pie')
def _build_sb_pie(df, x, y, ax, ctx):
    pie_data = df[x].value_counts()
    if len(pie_data) > 10:
        main_cats = pie_data.head(9)
        other_sum = pie_data.iloc[9:].sum()
        if other_sum > 0:
            main_cats = pd.concat([main_cats, pd.Series([other_sum], index=['Другие'])])
        pie_data = main_cats
    colors = sns.color_palette('viridis', len(pie_data))
    ax.pie(pie_data.values, labels=pie_data.index, autopct='%1.1f%%', startangle=90, colors=colors)
    ax.axis('equal')


@ChartRegistry.register('seaborn', 'kde')
def _build_sb_kde(df, x, y, ax, ctx):
    hue = 'composition' if ctx.get('use_comp_color') else ctx.get('color')
    if hue and hue in df.columns:
        try:
            sns.kdeplot(data=df, x=x, y=y, hue=hue, palette='viridis', ax=ax, warn_singular=False)
            return
        except Exception:
            pass
    try:
        sns.kdeplot(data=df, x=x, y=y, ax=ax, warn_singular=False)
    except Exception:
        sns.kdeplot(data=df, x=x, ax=ax, warn_singular=False)


def generate_seaborn_plot(
    data: pd.DataFrame,
    x_param: str = 'ad',
    y_param: str = 'q',
    plot_type: str = 'scatter',
    theme: str = 'default',
    color_param: Optional[str] = None,
    selected_compositions: Optional[List[str]] = None,
) -> Tuple[Optional[str], Optional[str], List[str]]:
    """Создает статические графики с использованием Seaborn через ChartRegistry."""
    filtered_data = data.copy()
    available_compositions = []
    
    if 'composition' in data.columns:
        available_compositions = data['composition'].unique().tolist()
        if selected_compositions:
            filtered_data = data[data['composition'].isin(selected_compositions)]
            if filtered_data.empty:
                return None, "Нет данных для выбранных составов", available_compositions
    
    try:
        if plot_type not in ['pie', 'histogram', 'radar', 'heatmap']:
            if not y_param or y_param not in filtered_data.columns or filtered_data[y_param].dropna().empty:
                return None, f"Отсутствует или неверный параметр Y ({y_param})", available_compositions
        
        sns.set_theme(style=get_seaborn_style(theme))
        fig, ax = plt.subplots(figsize=(12, 8))
        
        use_composition_color = ('composition' in filtered_data.columns and 
                               not color_param and plot_type not in ['pie', 'heatmap'])
        
        ctx = {
            'color': color_param,
            'use_comp_color': use_composition_color,
            'theme': theme,
        }
        
        builder = ChartRegistry.get('seaborn', plot_type)
        if builder is None:
            builder = _build_sb_scatter
        
        ok = builder(filtered_data, x_param, y_param, ax, ctx)
        if ok is False and plot_type == 'heatmap':
            plt.close(fig)
            return None, "Для тепловой карты нужно как минимум 2 числовых параметра", available_compositions
        
        # Заголовки и подписи
        if plot_type == 'pie':
            ax.set_title(f"Распределение {get_param_display_name(x_param)}")
        elif plot_type == 'heatmap':
            ax.set_title("Тепловая карта корреляций")
        elif plot_type == 'histogram':
            ax.set_title(f"Гистограмма {get_param_display_name(x_param)}")
        elif plot_type == 'line':
            ax.set_title(f"Линейный график: {get_param_display_name(y_param)} vs {get_param_display_name(x_param)}")
        else:
            ax.set_title(f"{get_param_display_name(y_param)} vs {get_param_display_name(x_param)}")
        
        if plot_type != 'pie':
            ax.set_xlabel(get_param_display_name(x_param))
            if plot_type != 'histogram':
                ax.set_ylabel(get_param_display_name(y_param))
        
        if use_composition_color and plot_type not in ['pie', 'heatmap']:
            ax.legend(title='Составы', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        
        fig.tight_layout()
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        graph = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)
        
        return graph, "Seaborn график создан успешно", available_compositions
        
    except Exception as e:
        plt.close('all')
        logger.error(f"Ошибка при создании Seaborn графика: {e}", exc_info=True)
        return None, f"Ошибка при создании Seaborn графика: {str(e)}", available_compositions
