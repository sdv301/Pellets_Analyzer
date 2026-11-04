"""Загрузка артефакта v2 (joblib + manifest) и контракт прогноза.

Модели подобраны по уже посчитанной nested leave-one-recipe-out оценке
(data/honest_metrics.csv) и дообучены на всех source=real один раз.
density, kf, ad и любая цель с informative=false отвечают средним ± σ.
"""
from __future__ import annotations

import json
import logging
import math
import threading
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from app.services.ml_v2.applicability import assess, build_domain
from app.services.ml_v2.components_registry import (
    RECIPE_COMPONENTS,
    canonicalize_composition,
    match_component,
    parse_composition,
)
from app.services.ml_v2.dataset import HONEST_METRICS_CSV, load_real_recipes
from app.services.ml_v2.evaluation import fit_full, fraction_matrix
from app.services.ml_v2.model_contract import (
    file_sha256,
    make_contract,
    read_manifest,
    versions_compatible,
    write_artifact,
)
from app.services.ml_v2.properties import display_name, resolve_property_id

logger = logging.getLogger(__name__)

PACKAGE_DIR = Path(__file__).resolve().parent
MODEL_DIR = PACKAGE_DIR / "model_store"

_LOCK = threading.Lock()
_PREDICTOR: Optional["V2Predictor"] = None

INFORMATIVE_R2 = 0.2


def _plain(value):
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        return number
    return value


def _truthy_metric(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes")
    return bool(value)


class V2Predictor:
    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = Path(model_dir)
        self.manifest: Dict = {}
        self.models: Dict = {}
        self.scalers: Dict = {}
        self.bundles: Dict = {}
        self.feature_names: List[str] = []
        self.domain_state = None
        self.is_loaded = False

    def load(self) -> bool:
        manifest = read_manifest(self.model_dir)
        model_path = self.model_dir / "pellet_models.joblib"
        if manifest is None:
            logger.error("Отказ загрузки v2: нет %s", self.model_dir / "manifest.json")
            return False
        if not model_path.exists():
            logger.error("Отказ загрузки v2: нет %s", model_path)
            return False
        expected = manifest.get("sha256")
        if expected and file_sha256(model_path) != expected:
            logger.error("Отказ загрузки v2: sha256 joblib не совпадает с manifest")
            return False
        saved_versions = manifest.get("versions") or {}
        if saved_versions and not versions_compatible(saved_versions) and not allow_version_mismatch():
            logger.error("Отказ загрузки v2: версии numpy/scikit-learn в manifest не совпадают с окружением")
            return False
        import joblib

        bundle = joblib.load(model_path)
        saved_versions = bundle.get("versions") or saved_versions
        if saved_versions and not versions_compatible(saved_versions) and not allow_version_mismatch():
            logger.error("Отказ загрузки v2 joblib: несовместимые версии sklearn/numpy")
            return False
        self.manifest = manifest
        self.models = bundle.get("models") or {}
        self.scalers = bundle.get("scalers") or {}
        self.bundles = bundle.get("bundles") or {}
        self.feature_names = list(bundle.get("feature_names") or manifest.get("features") or [])
        self.domain_state = bundle.get("domain_state")
        self.is_loaded = True
        return True

    def metrics_for(self, target_property: str) -> Dict:
        metrics = (self.manifest.get("metrics") or {}).get(target_property) or {}
        return dict(metrics)

    def predict_contract(self, composition: Dict[str, float], target_property: str) -> Optional[Dict]:
        if not self.is_loaded and not self.load():
            raise FileNotFoundError("артефакт v2 не загружен: нужен manifest.json рядом с pellet_models.joblib")
        target_property = resolve_property_id(target_property)
        if isinstance(composition, str):
            composition = parse_composition(composition, strict=True)
        composition = canonicalize_composition(composition)
        unknown = [
            key
            for key, value in composition.items()
            if not str(key).startswith(("phys_", "synergy_"))
            and float(value) > 0
            and match_component(key) is None
        ]
        if unknown:
            raise ValueError("неизвестный компонент: " + ", ".join(unknown))

        domain = assess(composition, self.domain_state)
        metrics = self.metrics_for(target_property)
        bundle = self.bundles.get(target_property) or {}
        informative = bool(bundle.get("informative", metrics.get("informative", False)))
        note = domain.get("reason") or ""
        if not domain["in_domain"]:
            note = ("вне области применимости. " + note).strip()

        if not informative:
            mean = bundle.get("train_mean", metrics.get("train_mean"))
            std = bundle.get("train_std", metrics.get("train_std"))
            if mean is None:
                return None
            extra = "модель не информативна: nested R² не лучше порога или цель из density/kf/ad."
            return make_contract(
                value=float(mean),
                std=float(std or 0.0),
                in_domain=domain["in_domain"],
                model_name="baseline_mean",
                informative=False,
                note=(extra + " " + note).strip(),
            )

        if target_property not in self.models:
            return None
        features = self._feature_row(composition)
        scaler = self.scalers[target_property]
        scaled = scaler.transform([features])
        model = self.models[target_property]
        mu = float(bundle.get("mu") or 0.0)
        sd = float(bundle.get("sd") or 1.0)
        std = None
        if bundle.get("has_std"):
            try:
                pred_s, std_s = model.predict(scaled, return_std=True)
                value = float(np.asarray(pred_s).ravel()[0]) * sd + mu
                std = float(np.asarray(std_s).ravel()[0]) * sd
            except TypeError:
                value = float(np.asarray(model.predict(scaled)).ravel()[0]) * sd + mu
        else:
            value = float(np.asarray(model.predict(scaled)).ravel()[0]) * sd + mu
        if std is None:
            std = float(bundle.get("rmse_test") or metrics.get("rmse_test") or bundle.get("train_std") or 0.0)
        std *= float(bundle.get("std_scale") or metrics.get("std_scale") or 1.0)
        return make_contract(
            value=value,
            std=max(float(std), 0.0),
            in_domain=domain["in_domain"],
            model_name=str(bundle.get("model") or metrics.get("model") or type(model).__name__),
            informative=True,
            note=note.strip(),
        )

    def _feature_row(self, composition: Dict[str, float]) -> np.ndarray:
        canon = canonicalize_composition(composition)
        total = sum(
            float(v)
            for k, v in canon.items()
            if not str(k).startswith(("phys_", "synergy_"))
        )
        if total <= 0:
            raise ValueError("Сумма компонентов должна быть больше 0")
        if abs(total - 100.0) > 0.5:
            canon = {
                k: (float(v) / total * 100.0) if not str(k).startswith(("phys_", "synergy_")) else float(v)
                for k, v in canon.items()
            }
        return np.array([float(canon.get(name, 0.0)) / 100.0 for name in self.feature_names], dtype=float)


def allow_version_mismatch() -> bool:
    import os

    raw = os.environ.get("ML_ALLOW_VERSION_MISMATCH", "")
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def get_predictor() -> V2Predictor:
    global _PREDICTOR
    with _LOCK:
        if _PREDICTOR is None or not _PREDICTOR.is_loaded:
            predictor = V2Predictor()
            if not predictor.load():
                raise FileNotFoundError(
                    f"не удалось загрузить v2 из {MODEL_DIR}. Нужны manifest.json и pellet_models.joblib"
                )
            _PREDICTOR = predictor
        return _PREDICTOR


def reset_predictor() -> None:
    global _PREDICTOR
    with _LOCK:
        _PREDICTOR = None


def public_metrics(metrics: Dict) -> Dict:
    """Поля, которые можно показать в UI. In-sample R² сюда не входит."""
    r2_test = _plain(metrics.get("r2_test"))
    return {
        "metrics_kind": "nested_cv",
        "r2_test_nested": r2_test,
        "r2_score": r2_test,
        "cv_r2": r2_test,
        "mae": _plain(metrics.get("mae_test")),
        "rmse_test": _plain(metrics.get("rmse_test")),
        "baseline_r2": _plain(metrics.get("baseline_r2")),
        "coverage90": _plain(metrics.get("coverage90")),
        "ci95_lo": _plain(metrics.get("ci95_lo")),
        "ci95_hi": _plain(metrics.get("ci95_hi")),
        "n": _plain(metrics.get("n")),
        "informative": bool(metrics.get("informative")),
        "cv_scheme": metrics.get("cv_scheme") or "nested_loro_group",
    }


def v2_system_status(predictor: Optional[V2Predictor] = None) -> Dict:
    predictor = predictor or get_predictor()
    manifest = predictor.manifest or {}
    metrics_map = manifest.get("metrics") or {}
    model_metrics = {}
    for prop, metrics in metrics_map.items():
        shown = public_metrics(metrics)
        model_metrics[prop] = {
            "display_name": display_name(prop),
            "algorithm_used": metrics.get("model"),
            "algorithm_reason": (
                "Модель выбрана по nested leave-one-recipe-out CV (recipe_id, source=real). "
                "В интерфейсе показан R² test, не in-sample."
            ),
            "training_metrics": shown,
            "informative": shown["informative"],
        }
    return {
        "model_version": "v2",
        "is_trained": bool(predictor.is_loaded),
        "trained_models": list(metrics_map.keys()),
        "available_components": list(manifest.get("components") or RECIPE_COMPONENTS),
        "training_data_size": int(manifest.get("n_real") or 0),
        "metrics_kind": "nested_cv",
        "model_metrics": model_metrics,
        "target_properties_mapping": {prop: display_name(prop) for prop in metrics_map},
        "ml_optimizations_count": 0,
    }


def select_winners(metrics_frame: pd.DataFrame) -> Dict[str, dict]:
    scored = metrics_frame.dropna(subset=["r2_test"])
    winners = {}
    for target, group in scored.groupby("target"):
        best = group.sort_values("r2_test", ascending=False).iloc[0]
        winners[str(target)] = best.to_dict()
    return winners


def build_artifact(
    *,
    metrics_csv: Path = HONEST_METRICS_CSV,
    out_dir: Path = MODEL_DIR,
) -> Path:
    """Полный refit выбранных моделей и запись joblib + manifest.json."""
    frame = load_real_recipes()
    matrix, columns = fraction_matrix(frame)
    winners = select_winners(pd.read_csv(metrics_csv))
    if not winners:
        raise RuntimeError(f"в {metrics_csv} нет метрик")

    models = {}
    scalers = {}
    bundles = {}
    manifest_metrics = {}
    compositions = []
    for text in frame["composition"].astype(str):
        compositions.append(parse_composition(text, strict=True))
    domain = build_domain(compositions, columns)
    domain["seen"] = sorted(domain["seen"])

    for target, row in winners.items():
        if target not in frame.columns:
            continue
        y_all = pd.to_numeric(frame[target], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(y_all)
        if int(mask.sum()) < 15:
            continue
        y = y_all[mask]
        fitted = fit_full(str(row["model"]), matrix[mask], y)
        informative = _truthy_metric(row.get("informative")) and target not in {"density", "kf", "ad"}
        if informative and _plain(row.get("r2_test")) is not None and float(row["r2_test"]) < INFORMATIVE_R2:
            informative = False
        train_mean = float(np.mean(y))
        train_std = float(np.std(y))
        models[target] = fitted["estimator"]
        scalers[target] = fitted["scaler"]
        bundles[target] = {
            "mu": fitted["mu"],
            "sd": fitted["sd"],
            "scale_y": fitted["scale_y"],
            "has_std": fitted["has_std"],
            "model": fitted["model"],
            "std_scale": float(row.get("std_scale") or 1.0),
            "rmse_test": _plain(row.get("rmse_test")),
            "informative": informative,
            "train_mean": train_mean,
            "train_std": train_std,
        }
        manifest_metrics[target] = {
            "model": str(row["model"]),
            "n": _plain(row.get("n")),
            "n_groups": _plain(row.get("n_groups")),
            "r2_test": _plain(row.get("r2_test")),
            "r2_train": _plain(row.get("r2_train")),
            "baseline_r2": _plain(row.get("baseline_r2")),
            "mae_test": _plain(row.get("mae_test")),
            "rmse_test": _plain(row.get("rmse_test")),
            "coverage90": _plain(row.get("coverage90")),
            "std_scale": _plain(row.get("std_scale")),
            "ci95_lo": _plain(row.get("ci95_lo")),
            "ci95_hi": _plain(row.get("ci95_hi")),
            "has_std": _truthy_metric(row.get("has_std")),
            "cv_scheme": row.get("cv_scheme"),
            "informative": informative,
            "train_mean": train_mean,
            "train_std": train_std,
        }

    bundle = {
        "models": models,
        "scalers": scalers,
        "bundles": bundles,
        "feature_names": columns,
        "domain_state": domain,
    }
    manifest = {
        "model_version": "v2",
        "schema": "pellet-contract-v2",
        "source": (
            "Честные метрики — nested leave-one-recipe-out из снимка Tolko "
            "artifacts/honest_eval (прогон 2026-11-02). "
            "joblib — refit выбранной модели на всех source=real, без IQR, "
            "без «Опыт:»/«УЧИТЕЛЬ:» и без записи прогнозов обратно в выборку."
        ),
        "n_real": int(len(frame)),
        "cv_scheme": "nested_loro_group",
        "features": columns,
        "components": columns,
        "metrics": manifest_metrics,
        "non_informative_targets": ["density", "kf", "ad"],
        "notes": [
            "q_origin в выгрузке не установлен (stored_unverified).",
            "kf 0.98508 у «70% Опилки, 30% листья» переведён в 98.508; исходное значение в kf_raw.",
            "Прогнозы v2 не пишутся в measured_parameters и ml_predictions продукта.",
        ],
    }
    model_path = write_artifact(Path(out_dir), bundle, manifest)
    manifest_path = Path(out_dir) / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["sha256"] = file_sha256(model_path)
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("v2 artifact written to %s", model_path)
    return model_path
