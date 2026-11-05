"""Тесты визуализации и графиков (Фаза B6: Charts).

Проверяет идентичность, валидность и устойчивость построения:
1. Plotly графиков (scatter, line, bar, box, histogram, 3d_scatter, heatmap, radar, pie, violin).
2. Seaborn графиков (scatter, line, bar, box, violin, kde, heatmap).
3. Matplotlib графиков (scatter, bar, line, histogram, box, heatmap, 3d_scatter).
4. Статистических функций (get_data_statistics, filter_parameters, get_comparison_stats).
"""
import json
import base64
import pandas as pd
import numpy as np
import pytest

from app.services.gui import (
    COLOR_PALETTE,
    PARAM_NAMES,
    PARAM_UNITS,
    get_param_display_name,
    get_plotly_theme,
    get_data_statistics,
    filter_parameters,
    get_comparison_stats,
    generate_plotly_graph,
    generate_seaborn_plot,
    generate_graph,
)


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'composition': ['100% Опилки', '100% Опилки', '90% Опилки, 10% Торф', '90% Опилки, 10% Торф', '85% Опилки, 15% Картон'],
        'ad': [1.2, 1.4, 2.5, 2.8, 3.1],
        'q': [18.5, 18.2, 17.1, 17.3, 16.9],
        'density': [1100, 1120, 1050, 1080, 1020],
        'kf': [98.2, 97.9, 95.1, 95.4, 93.0],
        'kt': [88.0, 87.5, 82.0, 83.0, 80.5],
        'war': [8.0, 8.2, 9.5, 9.1, 10.0],
    })


def test_palette_and_param_display_name():
    assert len(COLOR_PALETTE) >= 8
    assert "#0161EF" in COLOR_PALETTE
    name = get_param_display_name('q')
    assert "МДж/кг" in name
    assert "теплота сгорания" in name.lower()


def test_get_data_statistics(sample_data):
    stats = get_data_statistics(sample_data)
    assert 'q' in stats
    assert 'ad' in stats
    assert stats['q']['count'] == 5
    assert stats['q']['min'] == 16.9
    assert stats['q']['max'] == 18.5
    assert stats['q']['mean'] > 17.0


def test_filter_parameters(sample_data):
    filtered = filter_parameters(sample_data, 'thermal')
    assert 'composition' in filtered.columns
    assert 'q' in filtered.columns
    assert 'density' not in filtered.columns


def test_get_comparison_stats(sample_data):
    stats = get_comparison_stats(sample_data)
    assert stats['compositions_count'] == 3
    assert stats['total_rows'] == 5


@pytest.mark.parametrize("graph_type", [
    'scatter', 'bar', 'line', 'box', 'histogram',
    '3d_scatter', 'heatmap', 'radar', 'pie', 'violin'
])
def test_plotly_graph_generation(sample_data, graph_type):
    z_param = 'density' if graph_type == '3d_scatter' else None
    graph_json, msg, comps = generate_plotly_graph(
        sample_data,
        x_param='ad',
        y_param='q',
        graph_type=graph_type,
        z_param=z_param,
        theme='default'
    )
    assert graph_json is not None
    assert "успешно" in msg.lower()
    assert len(comps) == 3
    parsed = json.loads(graph_json)
    assert 'data' in parsed
    assert 'layout' in parsed


@pytest.mark.parametrize("plot_type", [
    'scatter', 'line', 'bar', 'box', 'violin', 'kde', 'heatmap'
])
def test_seaborn_plot_generation(sample_data, plot_type):
    graph_b64, msg, comps = generate_seaborn_plot(
        sample_data,
        x_param='ad',
        y_param='q',
        plot_type=plot_type,
        theme='default'
    )
    assert graph_b64 is not None
    assert len(graph_b64) > 100
    assert "успешно" in msg.lower()
    decoded = base64.b64decode(graph_b64)
    assert decoded[:8] == b'\x89PNG\r\n\x1a\n'


@pytest.mark.parametrize("graph_type", [
    'scatter', 'bar', 'line', 'histogram', 'box', 'heatmap', '3d_scatter'
])
def test_matplotlib_graph_generation(sample_data, graph_type):
    z_param = 'density' if graph_type == '3d_scatter' else None
    graph_b64, msg, comps = generate_graph(
        sample_data,
        x_param='ad',
        y_param='q',
        graph_type=graph_type,
        z_param=z_param,
        theme='default'
    )
    assert graph_b64 is not None
    assert len(graph_b64) > 100
    assert "успешно" in msg.lower()
    decoded = base64.b64decode(graph_b64)
    assert decoded[:8] == b'\x89PNG\r\n\x1a\n'
