"""Поиск субмодуля lab/ и проверка мажора API_VERSION.

Путь: PELLET_LAB_PATH, затем LAB_PATH, затем каталог lab/ в корне продукта.
Импорт не тянет Flask. Если пакета нет, приложение всё равно стартует.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
PIN_FILE = Path(__file__).resolve().parent / "PIN"
EXPECTED_API_MAJOR = "1"

_state = {
    "checked": False,
    "available": False,
    "reason": "pellet_ml ещё не проверялся",
    "api_version": "",
    "version": "",
    "lab_path": "",
}


class LabUnavailable(RuntimeError):
    """v2 нельзя использовать. Legacy при этом не затрагивается."""


def reset_lab_state() -> None:
    _state.update(
        {
            "checked": False,
            "available": False,
            "reason": "pellet_ml ещё не проверялся",
            "api_version": "",
            "version": "",
            "lab_path": "",
        }
    )


def resolve_lab_path() -> Path | None:
    for key in ("PELLET_LAB_PATH", "LAB_PATH"):
        raw = os.environ.get(key, "").strip()
        if raw:
            return Path(raw).expanduser().resolve()
    candidate = REPO_ROOT / "lab"
    if (candidate / "pellet_ml" / "__init__.py").is_file():
        return candidate
    return None


def _ensure_on_path(lab_path: Path) -> None:
    location = str(lab_path)
    if location not in sys.path:
        sys.path.insert(0, location)


def api_major(version: str) -> str:
    return str(version).split(".", 1)[0]


def read_pin_file() -> str:
    if not PIN_FILE.is_file():
        return ""
    return PIN_FILE.read_text(encoding="utf-8").strip()


def expected_lab_commit() -> str:
    """Коммит, с которым должен совпасть артефакт.

    Явный PELLET_LAB_COMMIT важнее файла PIN. Иначе PIN, иначе gitlink.
    """
    override = os.environ.get("PELLET_LAB_COMMIT", "").strip()
    if override:
        return override
    pinned = read_pin_file()
    gitlink = _gitlink_commit()
    if pinned and gitlink and pinned != gitlink:
        logger.error(
            "Пин субмодуля %s и файл %s (%s) разошлись. v2 не будет загружен, пока их не сведут.",
            gitlink,
            PIN_FILE,
            pinned,
        )
        return ""
    return pinned or gitlink or ""


def _gitlink_commit() -> str:
    for spec in ("HEAD:lab", ":lab"):
        try:
            completed = subprocess.run(
                ["git", "rev-parse", spec],
                cwd=REPO_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError:
            return ""
        if completed.returncode == 0:
            sha = completed.stdout.strip()
            if sha and "fatal" not in sha:
                return sha
    return ""


def check_lab_at_startup() -> dict:
    """Импортирует pellet_ml и сверяет мажор API. Исключения наружу не выпускает."""
    reset_lab_state()
    lab_path = resolve_lab_path()
    if lab_path is None or not (lab_path / "pellet_ml" / "__init__.py").is_file():
        reason = (
            "субмодуль lab не инициализирован (нет pellet_ml). "
            "Режим v2 недоступен, legacy работает. "
            "Клонирование: git clone --recurse-submodules или git submodule update --init lab. "
            "Локальный путь можно задать переменной PELLET_LAB_PATH."
        )
        _state.update({"checked": True, "available": False, "reason": reason})
        logger.error(reason)
        return dict(_state)

    _ensure_on_path(lab_path)
    try:
        import pellet_ml
    except Exception as exc:
        reason = f"не удалось импортировать pellet_ml из {lab_path}: {exc}. v2 отключён, legacy работает."
        _state.update({"checked": True, "available": False, "reason": reason, "lab_path": str(lab_path)})
        logger.error(reason)
        return dict(_state)

    api_version = str(getattr(pellet_ml, "API_VERSION", ""))
    package_version = str(getattr(pellet_ml, "__version__", ""))
    if api_major(api_version) != EXPECTED_API_MAJOR:
        reason = (
            f"pellet_ml API_VERSION {api_version or 'не задан'}: мажор не {EXPECTED_API_MAJOR}. "
            "Режим v2 отключён, legacy продолжает работать. "
            "Нужен субмодуль с API 1.x (сейчас зафиксирован lab-v0.1.0)."
        )
        _state.update(
            {
                "checked": True,
                "available": False,
                "reason": reason,
                "api_version": api_version,
                "version": package_version,
                "lab_path": str(lab_path),
            }
        )
        logger.error(reason)
        return dict(_state)

    _state.update(
        {
            "checked": True,
            "available": True,
            "reason": "",
            "api_version": api_version,
            "version": package_version,
            "lab_path": str(lab_path),
        }
    )
    logger.info(
        "pellet_ml %s, API_VERSION %s, каталог %s. v2 загрузит артефакт при первом запросе.",
        package_version,
        api_version,
        lab_path,
    )
    return dict(_state)


def require_lab():
    """Возвращает модуль pellet_ml или поднимает LabUnavailable."""
    if not _state["checked"]:
        check_lab_at_startup()
    if not _state["available"]:
        raise LabUnavailable(_state["reason"])
    import pellet_ml

    return pellet_ml
