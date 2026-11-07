"""Реестр генераторов графиков (библиотека, тип графика -> функция)."""
from typing import Callable, Dict, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class ChartRegistry:
    """Центральный реестр генераторов графиков."""
    _registry: Dict[Tuple[str, str], Callable] = {}

    @classmethod
    def register(cls, library: str, graph_type: str):
        """Декоратор для регистрации генератора графика."""
        def decorator(func: Callable):
            key = (library.lower(), graph_type.lower())
            cls._registry[key] = func
            return func
        return decorator

    @classmethod
    def get(cls, library: str, graph_type: str) -> Optional[Callable]:
        """Возвращает зарегистрированную функцию построения графика."""
        return cls._registry.get((library.lower(), graph_type.lower()))

    @classmethod
    def list_supported(cls, library: Optional[str] = None) -> list:
        """Возвращает список поддерживаемых типов графиков."""
        if library:
            lib = library.lower()
            return [gt for (l, gt) in cls._registry if l == lib]
        return list(cls._registry.keys())


register_chart = ChartRegistry.register

