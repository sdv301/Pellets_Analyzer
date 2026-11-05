"""Сервис построения графиков и аналитики данных Pellets Analyzer.

Реэкспортирует функции из модульного пакета app.services.charts:
- palette: Палитры, названия параметров, стили и темы
- registry: Реестр типов графиков (ChartRegistry)
- stats: Статистические и фильтрационные функции
- plotly_charts: Интерактивные графики Plotly
- seaborn_charts: Статистические графики Seaborn
- matplotlib_charts: Статические и анимированные графики Matplotlib
"""

from app.services.charts import *  # noqa: F401, F403
from app.services.charts import __all__ as _charts_all

__all__ = _charts_all
