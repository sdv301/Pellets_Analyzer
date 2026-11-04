"""Честная оценка: вложенная групповая CV по recipe_id.

Внешний цикл — leave-one-recipe-out (при <50 уникальных рецептурах).
Подбор гиперпараметров и масштабирование — только внутри train.
Метрики train и test считаются отдельно, рядом с baseline (среднее).

Запуск:
    python -m app.services.ml_v2.evaluation
    python -m app.services.ml_v2.evaluation --quick
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from app.services.ml_v2.components_registry import RECIPE_COMPONENTS, parse_composition
from app.services.ml_v2.dataset import load_real_recipes
from app.services.ml_v2.properties import NON_INFORMATIVE_TARGETS, TRAINABLE_TARGETS

SEED = 42

logger = logging.getLogger(__name__)

Z90 = 1.6448536269514722


class _PLS(BaseEstimator, RegressorMixin):
    def __init__(self, n_components: int = 2):
        self.n_components = n_components

    def fit(self, X, y):
        n_comp = int(min(self.n_components, X.shape[1], max(1, len(X) - 1)))
        self.model_ = PLSRegression(n_components=n_comp, scale=False)
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return np.asarray(self.model_.predict(X)).ravel()


class _GPR(BaseEstimator, RegressorMixin):
    def __init__(self, seed: int = 0, n_restarts: int = 0):
        self.seed = seed
        self.n_restarts = n_restarts

    def fit(self, X, y):
        kernel = C(1.0, (1e-2, 1e2)) * Matern(
            length_scale=2.0, length_scale_bounds=(0.1, 50.0), nu=2.5
        ) + WhiteKernel(0.1, (1e-4, 2.0))
        self.model_ = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=self.n_restarts,
            random_state=self.seed,
            normalize_y=False,
        )
        self.model_.fit(X, y)
        return self

    def predict(self, X, return_std: bool = False):
        return self.model_.predict(X, return_std=return_std)


class _Bayes(BayesianRidge):
    """Тонкая обёртка: predict(..., return_std=) уже есть у BayesianRidge."""


def group_folds(groups: Sequence, k: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Разбиение по группам. k ≥ числа групп → leave-one-group-out.

    Одна и та же группа не попадает в train и test одного фолда.
    """
    groups = np.asarray(groups)
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError("для групповой CV нужно минимум две группы")
    folds = int(min(k, len(unique)))
    rng = np.random.RandomState(seed)
    perm = rng.permutation(unique)
    bucket = {g: i % folds for i, g in enumerate(perm)}
    fold_id = np.array([bucket[g] for g in groups])
    splits = []
    for i in range(folds):
        te = np.where(fold_id == i)[0]
        tr = np.where(fold_id != i)[0]
        if len(te) == 0 or len(tr) == 0:
            continue
        splits.append((tr, te))
    return splits


def assert_groups_disjoint(splits, groups) -> None:
    groups = np.asarray(groups)
    for tr, te in splits:
        overlap = set(groups[tr]).intersection(set(groups[te]))
        if overlap:
            raise AssertionError(f"группа попала и в train, и в test: {overlap}")


def _model_specs(seed: int, names: Sequence[str]) -> Dict[str, Tuple[Callable, Optional[dict], bool]]:
    specs = {
        "ridge": (
            lambda: Ridge(),
            {"alpha": list(np.logspace(-3, 3, 7))},
            False,
        ),
        "bayes_ridge": (lambda: _Bayes(), None, True),
        "pls": (lambda: _PLS(), {"n_components": [1, 2, 3, 4]}, False),
        "gp": (lambda: _GPR(seed=seed, n_restarts=0), None, True),
        "gb": (
            lambda: GradientBoostingRegressor(random_state=seed, subsample=0.9),
            {"n_estimators": [40], "max_depth": [2, 3], "learning_rate": [0.05]},
            False,
        ),
    }
    missing = [n for n in names if n not in specs]
    if missing:
        raise KeyError(f"неизвестная модель: {missing}")
    return {n: specs[n] for n in names}


def _regression_metrics(y, pred) -> Tuple[float, float, float]:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y, pred)))
    return float(r2_score(y, pred)), float(mean_absolute_error(y, pred)), rmse


def nested_cv(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: Sequence,
    *,
    seed: int = SEED,
    outer_k: Optional[int] = None,
) -> Dict:
    """Вложенная групповая CV. Возвращает OOF-прогноз, σ и метрики train/test."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    n_groups = len(np.unique(groups))
    k_outer = outer_k if outer_k is not None else (10**6 if n_groups < 50 else 7)
    splits = group_folds(groups, k_outer, seed)
    assert_groups_disjoint(splits, groups)
    factory, grid, has_std = _model_specs(seed, [name])[name]

    oof = np.full(len(y), np.nan)
    std = np.full(len(y), np.nan)
    train_r2, train_mae, train_rmse = [], [], []

    for tr, te in splits:
        scaler = StandardScaler().fit(X[tr])
        xtr = scaler.transform(X[tr])
        xte = scaler.transform(X[te])
        mu = float(y[tr].mean())
        sd = float(y[tr].std()) + 1e-12
        ytr = (y[tr] - mu) / sd
        est = factory()
        if grid:
            inner = group_folds(groups[tr], min(5, max(2, len(np.unique(groups[tr])))), seed + 1000)
            search = GridSearchCV(
                est,
                grid,
                cv=inner,
                scoring="neg_mean_squared_error",
                n_jobs=1,
                refit=True,
            )
            search.fit(xtr, ytr)
            est = search.best_estimator_
        else:
            est.fit(xtr, ytr)
        std_te = None
        if has_std:
            try:
                pred_s, std_s = est.predict(xte, return_std=True)
                std_te = np.asarray(std_s, dtype=float) * sd
            except TypeError:
                pred_s = est.predict(xte)
        else:
            pred_s = est.predict(xte)
        pred_te = np.asarray(pred_s, dtype=float).ravel() * sd + mu
        pred_tr = np.asarray(est.predict(xtr), dtype=float).ravel() * sd + mu
        oof[te] = pred_te
        if std_te is not None:
            std[te] = np.asarray(std_te, dtype=float).ravel()
        r2_tr, mae_tr, rmse_tr = _regression_metrics(y[tr], pred_tr)
        train_r2.append(r2_tr)
        train_mae.append(mae_tr)
        train_rmse.append(rmse_tr)

    if np.isnan(oof).any():
        raise RuntimeError(f"OOF не заполнен для модели {name}")

    # Baseline: среднее train каждого внешнего фолда, собранное в OOF.
    base = np.full(len(y), np.nan)
    for tr, te in splits:
        base[te] = float(y[tr].mean())
    r2_te, mae_te, rmse_te = _regression_metrics(y, oof)
    b_r2, b_mae, b_rmse = _regression_metrics(y, base)
    coverage = None
    width = None
    std_scale = 1.0
    if has_std and not np.isnan(std).all():
        std_scale = conformal_scale(y, oof, std, level=0.9)
        std = std * std_scale
        inside = np.abs(y - oof) <= Z90 * std
        coverage = float(np.mean(inside))
        width = float(np.mean(2 * Z90 * std))
    ci_lo, ci_hi = bootstrap_r2_ci(y, oof, seed=seed)
    return {
        "model": name,
        "n": int(len(y)),
        "n_groups": int(n_groups),
        "r2_test": r2_te,
        "mae_test": mae_te,
        "rmse_test": rmse_te,
        "r2_train": float(np.mean(train_r2)),
        "mae_train": float(np.mean(train_mae)),
        "rmse_train": float(np.mean(train_rmse)),
        "baseline_r2": b_r2,
        "baseline_mae": b_mae,
        "baseline_rmse": b_rmse,
        "gap_r2": float(np.mean(train_r2) - r2_te),
        "ci95_lo": ci_lo,
        "ci95_hi": ci_hi,
        "coverage90": coverage,
        "width90": width,
        "std_scale": std_scale,
        "oof": oof,
        "std": std,
        "has_std": has_std,
        "cv_scheme": "nested_loro_group" if n_groups < 50 else "nested_group_k",
    }


def bootstrap_r2_ci(y, pred, *, n_boot: int = 1000, seed: int = SEED) -> Tuple[float, float]:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    rng = np.random.RandomState(seed)
    scores = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if np.std(y[idx]) < 1e-12:
            continue
        scores.append(r2_score(y[idx], pred[idx]))
    if len(scores) < 20:
        return float("nan"), float("nan")
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def conformal_scale(y, pred, std, level: float = 0.9) -> float:
    """Множитель σ, чтобы эмпирическое покрытие на OOF совпало с level.

    Это калибровка по тем же OOF-остаткам (слегка оптимистична), нужна когда
    сырой интервал GP/BayesRidge на новом семействе добавок слишком узкий.
    """
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    std = np.asarray(std, dtype=float)
    ok = np.isfinite(std) & (std > 1e-12)
    if ok.sum() < 5:
        return 1.0
    z = np.abs(y[ok] - pred[ok]) / std[ok]
    q = float(np.quantile(z, level))
    return max(q / Z90, 1e-6)


def fraction_matrix(frame: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    parsed = []
    for text in frame["composition"].astype(str):
        parsed.append(parse_composition(text, strict=True))
    present = [c for c in RECIPE_COMPONENTS if any(d.get(c, 0) > 0 for d in parsed)]
    rows = [[d.get(c, 0.0) / 100.0 for c in present] for d in parsed]
    return np.asarray(rows, dtype=float), present


def evaluate_frame(
    frame: pd.DataFrame,
    targets: Sequence[str],
    models: Sequence[str],
    *,
    seed: int = SEED,
) -> pd.DataFrame:
    X_all, _cols = fraction_matrix(frame)
    groups_all = frame["recipe_id"].astype(str).values
    rows = []
    for target in targets:
        if target not in frame.columns:
            rows.append({"target": target, "model": "-", "note": "нет колонки"})
            continue
        y_all = pd.to_numeric(frame[target], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(y_all)
        if int(mask.sum()) < 15:
            rows.append({"target": target, "model": "-", "n": int(mask.sum()), "note": "n<15"})
            continue
        X = X_all[mask]
        y = y_all[mask]
        groups = groups_all[mask]
        for name in models:
            logger.info("nested CV %s / %s (n=%s)", target, name, len(y))
            result = nested_cv(name, X, y, groups, seed=seed)
            result.pop("oof", None)
            result.pop("std", None)
            result["target"] = target
            result["informative"] = (
                target not in NON_INFORMATIVE_TARGETS and result["r2_test"] >= 0.2
            )
            rows.append(result)
    return pd.DataFrame(rows)


def fit_full(name: str, X: np.ndarray, y: np.ndarray, seed: int = SEED, scale_y: bool = True):
    """Модель на всей выборке (для артефакта прогноза), после того как метрика уже посчитана nested-CV."""
    factory, grid, has_std = _model_specs(seed, [name])[name]
    scaler = StandardScaler().fit(X)
    xs = scaler.transform(X)
    mu = float(np.mean(y))
    sd = float(np.std(y)) + 1e-12
    ys = (y - mu) / sd if scale_y else np.asarray(y, dtype=float)
    est = factory()
    if grid:
        est = GridSearchCV(est, grid, cv=min(5, max(2, len(y) // 8)), scoring="neg_mean_squared_error", n_jobs=1)
        est.fit(xs, ys)
        est = est.best_estimator_
    else:
        est.fit(xs, ys)
    return {
        "estimator": est,
        "scaler": scaler,
        "mu": mu,
        "sd": sd,
        "scale_y": scale_y,
        "has_std": has_std,
        "model": name,
    }


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def run_evaluation(
    *,
    quick: bool = False,
    targets: Optional[Sequence[str]] = None,
    models: Optional[Sequence[str]] = None,
    out_dir: Optional[Path] = None,
) -> pd.DataFrame:
    frame = load_real_recipes()
    chosen_targets = list(targets or (["q", "kt", "density", "kf"] if quick else TRAINABLE_TARGETS))
    chosen_models = list(models or (["ridge", "bayes_ridge"] if quick else ["ridge", "bayes_ridge", "pls", "gp"]))
    table = evaluate_frame(frame, chosen_targets, chosen_models)
    destination = out_dir or (_project_root() / "artifacts" / "honest_eval")
    destination.mkdir(parents=True, exist_ok=True)
    keep = [c for c in table.columns if c not in ("oof", "std")]
    table[keep].to_csv(destination / "metrics.csv", index=False)
    _write_summary(table, destination / "summary.md")
    return table


def _write_summary(table: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Честная оценка (nested group CV, только source=real)",
        "",
        "R² test — pooled out-of-fold. R² train — среднее in-sample по внешним фолдам.",
        "Выбор «лучшей» модели по тем же OOF слегка оптимистичен.",
        "",
        "| цель | n | модель | R² test | R² train | baseline | 95% CI | покрытие 90% |",
        "|---|---:|---|---:|---:|---:|---|---:|",
    ]
    if "r2_test" not in table.columns:
        path.write_text("Нет метрик.\n", encoding="utf-8")
        return
    scored = table.dropna(subset=["r2_test"]) if "r2_test" in table.columns else table
    if scored.empty:
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    for target, grp in scored.groupby("target"):
        best = grp.sort_values("r2_test", ascending=False).iloc[0]
        cov = best.get("coverage90")
        cov_s = "" if pd.isna(cov) else f"{float(cov):.3f}"
        ci = ""
        if pd.notna(best.get("ci95_lo")):
            ci = f"[{float(best['ci95_lo']):.2f}; {float(best['ci95_hi']):.2f}]"
        lines.append(
            f"| {target} | {int(best['n'])} | {best['model']} | {float(best['r2_test']):.3f} | "
            f"{float(best['r2_train']):.3f} | {float(best['baseline_r2']):.3f} | {ci} | {cov_s} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Честная nested group CV по recipe_id")
    parser.add_argument("--quick", action="store_true", help="Только q/kt/density/kf и ridge/bayes")
    parser.add_argument("--targets", nargs="*")
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = Path(args.out) if args.out else None
    table = run_evaluation(quick=args.quick, targets=args.targets, models=args.models, out_dir=out)
    cols = [c for c in ("target", "model", "n", "r2_test", "r2_train", "baseline_r2", "coverage90") if c in table.columns]
    print(table[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
