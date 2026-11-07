"""Построение графиков с использованием Matplotlib."""
import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns

from app.services.charts.palette import PARAM_NAMES, get_param_display_name, apply_theme, COLOR_PALETTE
from app.services.charts.registry import ChartRegistry


def generate_animated_graph(data, x_param, y_param, animation_param, theme, title, selected_compositions=None):
    """Создает анимированный график."""
    filtered_data = data.copy()
    available_comps = []
    if 'composition' in data.columns:
        available_comps = data['composition'].unique().tolist()
        if selected_compositions:
            filtered_data = data[data['composition'].isin(selected_compositions)]
            if filtered_data.empty:
                return None, "Нет данных для выбранных составов", available_comps
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        apply_theme(theme)
        animation_values = sorted(filtered_data[animation_param].unique())

        def update(frame):
            ax.clear()
            curr = animation_values[frame]
            f_data = filtered_data[filtered_data[animation_param] == curr]
            sc = ax.scatter(f_data[x_param], f_data[y_param], alpha=0.7, s=50, c=COLOR_PALETTE[0])
            ax.set_xlabel(PARAM_NAMES.get(x_param, x_param.capitalize()))
            ax.set_ylabel(PARAM_NAMES.get(y_param, y_param.capitalize()))
            ax.set_title(f'{title or "Анимированный график"} - {animation_param}: {curr}')
            ax.grid(True)
            return sc,

        anim = FuncAnimation(fig, update, frames=len(animation_values), interval=500, blit=False, repeat=True)
        buf = io.BytesIO()
        anim.save(buf, format='gif', writer='pillow', fps=2)
        buf.seek(0)
        graph = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)
        return graph, "Анимированный график создан успешно", available_comps
    except Exception as e:
        plt.close('all')
        return None, f"Ошибка при создании анимированного графика: {str(e)}", available_comps


@ChartRegistry.register('matplotlib', 'scatter')
def _plot_scatter(ax, df, x, y, ctx):
    if ctx.get('use_comp'):
        comps = df['composition'].unique()
        colors = (COLOR_PALETTE * (len(comps) // len(COLOR_PALETTE) + 1))[:len(comps)]
        for comp, c in zip(comps, colors):
            sub = df[df['composition'] == comp]
            ax.scatter(sub[x], sub[y], c=[c], label=comp, alpha=0.7, s=50)
    elif ctx.get('color') and ctx['color'] in df.columns:
        cp = ctx['color']
        if pd.api.types.is_numeric_dtype(df[cp]):
            sc = ax.scatter(df[x], df[y], c=df[cp], cmap='viridis', alpha=0.7, s=50)
            plt.colorbar(sc, label=PARAM_NAMES.get(cp, cp))
        else:
            uvs = df[cp].unique()
            colors = (COLOR_PALETTE * (len(uvs) // len(COLOR_PALETTE) + 1))[:len(uvs)]
            for val, c in zip(uvs, colors):
                sub = df[df[cp] == val]
                ax.scatter(sub[x], sub[y], c=[c], label=val, alpha=0.7, s=50)
            ax.legend(title=PARAM_NAMES.get(cp, cp), bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        ax.scatter(df[x], df[y], alpha=0.7, s=50, color=COLOR_PALETTE[0])

@ChartRegistry.register('matplotlib', 'line')
def _plot_line(ax, df, x, y, ctx):
    sorted_df = df.sort_values(by=x)
    if ctx.get('use_comp'):
        comps = sorted_df['composition'].unique()
        colors = (COLOR_PALETTE * (len(comps) // len(COLOR_PALETTE) + 1))[:len(comps)]
        for comp, c in zip(comps, colors):
            sub = sorted_df[sorted_df['composition'] == comp]
            ax.plot(sub[x], sub[y], color=c, label=comp, linewidth=2, marker='o')
    else:
        ax.plot(sorted_df[x], sorted_df[y], linewidth=2, marker='o', color=COLOR_PALETTE[0])

@ChartRegistry.register('matplotlib', 'histogram')
def _plot_hist(ax, df, x, y, ctx):
    if ctx.get('use_comp'):
        comps = df['composition'].unique()
        colors = (COLOR_PALETTE * (len(comps) // len(COLOR_PALETTE) + 1))[:len(comps)]
        for comp, c in zip(comps, colors):
            sub = df[df['composition'] == comp]
            ax.hist(sub[x].dropna(), bins=15, alpha=0.7, color=c, label=comp, edgecolor='black')
    else:
        ax.hist(df[x].dropna(), bins=20, alpha=0.7, edgecolor='black', color=COLOR_PALETTE[0])
    ax.set_xlabel(get_param_display_name(x))
    ax.set_ylabel('Частота')


@ChartRegistry.register('matplotlib', 'bar')
def _plot_bar(ax, df, x, y, ctx):
    if df[x].dtype == 'object' or len(df[x].unique()) <= 20:
        if ctx.get('use_comp'):
            grp = df.groupby(['composition', x])[y].mean().reset_index()
            ux, uc = grp[x].unique(), grp['composition'].unique()
            bw, xpos = 0.8 / max(len(uc), 1), np.arange(len(ux))
            colors = (COLOR_PALETTE * (len(uc) // len(COLOR_PALETTE) + 1))[:len(uc)]
            for i, (comp, c) in enumerate(zip(uc, colors)):
                cd = grp[grp['composition'] == comp]
                vals = [cd[cd[x] == xv][y].values[0] if len(cd[cd[x] == xv]) > 0 else 0 for xv in ux]
                ax.bar(xpos + i * bw, vals, bw, label=comp, alpha=0.7, color=c)
            ax.set_xticks(xpos + bw * (len(uc) - 1) / 2)
            ax.set_xticklabels([str(val) for val in ux], rotation=45)
        else:
            grp = df.groupby(x)[y].mean()
            ax.bar(range(len(grp)), grp.values, alpha=0.7, color=COLOR_PALETTE[0])
            ax.set_xticks(range(len(grp)))
            ax.set_xticklabels([str(label) for label in grp.index], rotation=45)
        ax.set_xlabel(get_param_display_name(x))
        ax.set_ylabel(get_param_display_name(y))
    else:
        _plot_hist(ax, df, x, y, ctx)

@ChartRegistry.register('matplotlib', 'box')
def _plot_box(ax, df, x, y, ctx):
    if ctx.get('use_comp'):
        labels = [c for c in df['composition'].unique()[:10] if not df[df['composition'] == c].empty]
        data_plot = [df[df['composition'] == c][y].dropna() for c in labels]
        if data_plot:
            try:
                bp = ax.boxplot(data_plot, tick_labels=labels, patch_artist=True)
            except TypeError:
                bp = ax.boxplot(data_plot, labels=labels, patch_artist=True)
            colors = (COLOR_PALETTE * (len(data_plot) // len(COLOR_PALETTE) + 1))[:len(data_plot)]
            for patch, c in zip(bp['boxes'], colors):
                patch.set_facecolor(c)
    else:
        cats = df[x].unique()[:10]
        data_plot = [df[df[x] == cat][y].dropna() for cat in cats]
        ax.boxplot(data_plot)
        ax.set_xticklabels([str(c) for c in cats], rotation=45)
    ax.set_xlabel(get_param_display_name(x) if not ctx.get('use_comp') else 'Составы')
    ax.set_ylabel(get_param_display_name(y))

@ChartRegistry.register('matplotlib', 'pie')
def _plot_pie(ax, df, x, y, ctx):
    pd_counts = df[x].value_counts()
    if len(pd_counts) > 10:
        main_c = pd_counts.head(9)
        other_sum = pd_counts.iloc[9:].sum()
        if other_sum > 0:
            main_c = pd.concat([main_c, pd.Series([other_sum], index=['Другие'])])
        pd_counts = main_c
    ax.pie(pd_counts.values, labels=pd_counts.index, autopct='%1.1f%%', startangle=90, colors=COLOR_PALETTE)
    ax.axis('equal')

@ChartRegistry.register('matplotlib', 'heatmap')
def _plot_heatmap(ax, df, x, y, ctx):
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) <= 1:
        return False
    cm = df[num_cols].corr()
    im = ax.imshow(cm, cmap='coolwarm', aspect='auto')
    ax.set_xticks(range(len(cm.columns)))
    ax.set_yticks(range(len(cm.columns)))
    ax.set_xticklabels(cm.columns, rotation=45)
    ax.set_yticklabels(cm.columns)
    plt.colorbar(im, ax=ax)
    return True

@ChartRegistry.register('matplotlib', '3d_scatter')
def _plot_3d(ax, df, x, y, ctx):
    z = ctx.get('z')
    if not z or z not in df.columns:
        return False
    c_arg, cmap_arg = COLOR_PALETTE[0], None
    cp = ctx.get('color')
    if cp and cp in df.columns:
        if pd.api.types.is_numeric_dtype(df[cp]):
            c_arg, cmap_arg = df[cp], 'viridis'
        else:
            c_arg, cmap_arg = pd.Categorical(df[cp]).codes, 'tab10'
    sc = ax.scatter(df[x], df[y], df[z], c=c_arg, cmap=cmap_arg, alpha=0.7)
    ax.set_xlabel(get_param_display_name(x))
    ax.set_ylabel(get_param_display_name(y))
    ax.set_zlabel(get_param_display_name(z))
    if cp and cp in df.columns and pd.api.types.is_numeric_dtype(df[cp]):
        plt.colorbar(sc, ax=ax, label=get_param_display_name(cp))
    return True


def generate_graph(data, x_param='ad', y_param='q', graph_type='scatter', z_param=None,
                   color_param=None, size_param=None, animation_param=None, theme='default',
                   title=None, width=800, height=600, show_grid=True, selected_compositions=None):
    """Генерирует статический график Matplotlib через ChartRegistry."""
    filtered_data = data.copy()
    available_comps = []
    if 'composition' in data.columns:
        available_comps = data['composition'].unique().tolist()
        if selected_compositions is not None:
            if len(selected_compositions) == 0:
                return None, "Нет выбранных составов для отображения", available_comps
            filtered_data = data[data['composition'].isin(selected_compositions)]
            if filtered_data.empty:
                return None, "Нет данных для выбранных составов", available_comps

    if filtered_data.empty or x_param not in filtered_data.columns:
        return None, "Нет данных для построения графика", available_comps

    try:
        if graph_type not in ['pie', 'histogram', 'radar', 'heatmap', '3d_scatter', 'animated_scatter']:
            if not y_param or y_param not in filtered_data.columns or filtered_data[y_param].dropna().empty:
                return None, f"Отсутствует или неверный параметр Y ({y_param})", available_comps

        if graph_type == 'animated_scatter' and animation_param and animation_param in filtered_data.columns:
            return generate_animated_graph(filtered_data, x_param, y_param, animation_param, theme, title)

        apply_theme(theme)
        fig = plt.figure(figsize=(width / 100, height / 100))
        ax = fig.add_subplot(111, projection='3d') if graph_type == '3d_scatter' else fig.add_subplot(111)

        use_comp = ('composition' in filtered_data.columns and not color_param and graph_type not in ['pie', 'heatmap'])
        ctx = {'use_comp': use_comp, 'color': color_param, 'z': z_param}

        builder = ChartRegistry.get('matplotlib', graph_type)
        if builder:
            res = builder(ax, filtered_data, x_param, y_param, ctx)
            if res is False:
                plt.close(fig)
                err = "Для 3D Scatter выберите корректный параметр Z" if graph_type == '3d_scatter' else "Для тепловой карты нужно больше числовых параметров"
                return None, err, available_comps
        else:
            ax.scatter(filtered_data[x_param], filtered_data[y_param], alpha=0.7, s=50, color=COLOR_PALETTE[0])

        if not graph_type.startswith('3d') and graph_type != 'pie':
            if graph_type != 'histogram':
                ax.set_xlabel(get_param_display_name(x_param))
                ax.set_ylabel(get_param_display_name(y_param))
            ax.grid(show_grid)

        t_title = title or (
            f'Распределение {get_param_display_name(x_param)}' if graph_type == 'pie'
            else f'Гистограмма {get_param_display_name(x_param)}' if graph_type == 'histogram'
            else f'Линейный график: {get_param_display_name(y_param)} vs {get_param_display_name(x_param)}' if graph_type == 'line'
            else f'{get_param_display_name(y_param)} vs {get_param_display_name(x_param)}'
        )
        ax.set_title(t_title)

        if use_comp and graph_type not in ['pie', 'heatmap', '3d_scatter', 'box']:
            handles, _ = ax.get_legend_handles_labels()
            if handles:
                ax.legend(title='Составы', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        graph = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)
        return graph, "График создан успешно", available_comps
    except Exception as e:
        plt.close('all')
        return None, f"Ошибка при создании графика: {str(e)}", available_comps


def handle_scatter_plot(ax, data, x_param, y_param, color_param):
    """Обрабатывает scatter plot."""
    _plot_scatter(ax, data, x_param, y_param, {'color': color_param})

def handle_bar_chart(ax, data, x_param, y_param):
    """Обрабатывает bar chart."""
    _plot_bar(ax, data, x_param, y_param, {})

def handle_heatmap(data, fig, ax, theme):
    """Обрабатывает heatmap."""
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
    """Обрабатывает 3D scatter plot."""
    _plot_3d(ax, data, x_param, y_param, {'z': z_param, 'color': color_param})

def generate_graph_simple(data, x_param='ad', y_param='q', graph_type='scatter'):
    """Простая версия для обратной совместимости."""
    graph, _, _ = generate_graph(data, x_param, y_param, graph_type)
    return graph
