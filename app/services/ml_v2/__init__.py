"""Тонкий адаптер продукта к пакету pellet_ml из субмодуля lab/.

Само обучение здесь не дублируется. Нет субмодуля или мажор API не 1 —
режим v2 отказывается работать, legacy остаётся.
"""

__all__ = ["model_version"]


def model_version() -> str:
    from app.services.ml_v2.switch import model_version as _model_version

    return _model_version()
