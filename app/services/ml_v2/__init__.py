"""Честная модель v2, отдельная от legacy PelletMLSystem.

Артефакт читается из model_store/manifest.json. Прогнозы в обучающую
таблицу и в БД отсюда не пишутся.
"""

__all__ = ["model_version"]


def model_version() -> str:
    from app.services.ml_v2.switch import model_version as _model_version

    return _model_version()
