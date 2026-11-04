"""Контракт прогноза и артефакт модели для переноса в Pellets_Analyzer.

Выход на каждое свойство: {value, std, lo90, hi90, in_domain, model_name}.
Артефакт: joblib + manifest.json рядом.
"""
from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import joblib

from app.services.ml_v2.applicability import Z90

CONTRACT_KEYS = ("value", "std", "lo90", "hi90", "in_domain", "model_name")


def prediction_interval(value: float, std: float) -> Dict[str, float]:
    std = float(abs(std))
    return {
        "value": float(value),
        "std": std,
        "lo90": float(value) - Z90 * std,
        "hi90": float(value) + Z90 * std,
    }


def make_contract(
    *,
    value: float,
    std: float,
    in_domain: bool,
    model_name: str,
    informative: bool = True,
    note: str = "",
) -> Dict:
    payload = prediction_interval(value, std)
    payload.update(
        {
            "in_domain": bool(in_domain),
            "model_name": model_name,
            "informative": bool(informative),
            "note": note,
        }
    )
    missing = [key for key in CONTRACT_KEYS if key not in payload]
    if missing:
        raise RuntimeError(f"контракт прогноза неполный: {missing}")
    return payload


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def library_versions() -> Dict[str, str]:
    import numpy
    import sklearn

    versions = {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "scikit-learn": sklearn.__version__,
    }
    try:
        import xgboost

        versions["xgboost"] = xgboost.__version__
    except ImportError:
        versions["xgboost"] = ""
    return versions


def versions_compatible(saved: Dict[str, str], current: Optional[Dict[str, str]] = None) -> bool:
    current = current or library_versions()
    for key in ("scikit-learn", "numpy"):
        left = str(saved.get(key, "")).split(".")[:2]
        right = str(current.get(key, "")).split(".")[:2]
        if left != right:
            return False
    return True


def write_artifact(directory: Path, bundle: dict, manifest: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / "pellet_models.joblib"
    joblib.dump(bundle, model_path)
    manifest = dict(manifest)
    manifest["artifact"] = model_path.name
    manifest["created_at"] = datetime.now(timezone.utc).isoformat()
    manifest["versions"] = library_versions()
    (directory / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return model_path


def read_manifest(directory: Path) -> Optional[dict]:
    path = directory / "manifest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
