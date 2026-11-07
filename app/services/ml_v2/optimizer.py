"""Оптимизатор состава пеллет для модели v2.

Использует контракты v2 (V2Predictor), оценку области применимости (pellet_ml.domain),
учёт неопределённости (std / 90% доверительный интервал) и ограничения по компонентам.
Никогда не записывает прогноз в обучающую таблицу `measured_parameters`.
Результат оптимизации пишется только в журнал `ml_optimizations`.
Детерминирован благодаря параметру seed (по умолчанию 42).
"""
from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from app.database.database import insert_ml_optimization
from app.services.ml_v2.lab_import import require_lab
from app.services.ml_v2.predictor import get_predictor

logger = logging.getLogger(__name__)


def _project_to_simplex_bounds(
    x: np.ndarray,
    bounds: List[Tuple[float, float]],
    target_sum: float = 100.0,
    max_iter: int = 50,
) -> np.ndarray:
    """Проекция вектора x на симплекс с ограничениями bounds [min_i, max_i]."""
    n = len(x)
    mins = np.array([b[0] for b in bounds], dtype=float)
    maxs = np.array([b[1] for b in bounds], dtype=float)

    current = np.clip(x.astype(float), mins, maxs)
    current_sum = float(np.sum(current))
    if abs(current_sum - target_sum) < 1e-5:
        return current

    # Итеративное распределение остатка по незафиксированным компонентам
    for _ in range(max_iter):
        diff = target_sum - np.sum(current)
        if abs(diff) < 1e-4:
            break
        if diff > 0:
            free = maxs - current > 1e-4
            if not np.any(free):
                break
            step = diff / np.sum(free)
            current[free] = np.minimum(maxs[free], current[free] + step)
        else:
            free = current - mins > 1e-4
            if not np.any(free):
                break
            step = diff / np.sum(free)
            current[free] = np.maximum(mins[free], current[free] + step)

    return current


def _generate_candidate_pool(
    components: List[str],
    domain: Optional[Dict],
    constraints: Optional[Dict[str, Tuple[float, float]]] = None,
    available_components: Optional[List[str]] = None,
    seed: int = 42,
) -> List[np.ndarray]:
    """Генерация пула кандидатов составов, приоритизируя область применимости."""
    rng = np.random.RandomState(seed)
    n = len(components)
    bounds = [(0.0, 100.0) for _ in range(n)]

    avail_set = set(available_components) if available_components else set(components)
    for i, c in enumerate(components):
        if c not in avail_set:
            bounds[i] = (0.0, 0.0)

    if constraints:
        for c, (c_min, c_max) in constraints.items():
            if c in components:
                idx = components.index(c)
                bounds[idx] = (max(bounds[idx][0], float(c_min)), min(bounds[idx][1], float(c_max)))

    candidates: List[np.ndarray] = []

    # 1. Реальные рецептуры из области применимости (domain['matrix'])
    if domain and domain.get("matrix") is not None and len(domain["matrix"]) > 0:
        matrix = domain["matrix"]
        tau = float(domain.get("tau") or 28.28)
        for row in matrix:
            projected = _project_to_simplex_bounds(row, bounds)
            candidates.append(projected)

        # 2. Интерполяции между близкими реальными рецептурами
        n_rows = len(matrix)
        for i in range(n_rows):
            for j in range(i + 1, min(i + 8, n_rows)):
                dist = np.linalg.norm(matrix[i] - matrix[j])
                if dist <= tau:
                    for alpha in (0.25, 0.5, 0.75):
                        blend = alpha * matrix[i] + (1.0 - alpha) * matrix[j]
                        candidates.append(_project_to_simplex_bounds(blend, bounds))

    # 3. Случайные выборки Дирихле с проекцией на симплекс с ограничениями
    active_indices = [i for i, b in enumerate(bounds) if b[1] > 0.01]
    if not active_indices:
        active_indices = list(range(n))

    for _ in range(250):
        raw = np.zeros(n, dtype=float)
        # Смещённое распределение: разреженные составы (1-4 компонента)
        k_active = min(len(active_indices), rng.choice([1, 2, 3, 4]))
        chosen = rng.choice(active_indices, size=k_active, replace=False)
        dirichlet_weights = rng.dirichlet(np.ones(k_active) * 0.8)
        for c_idx, w in zip(chosen, dirichlet_weights):
            raw[c_idx] = w * 100.0
        projected = _project_to_simplex_bounds(raw, bounds)
        candidates.append(projected)

    # 4. Сетка вокруг активных компонентов
    if len(active_indices) <= 4:
        steps = [0.0, 10.0, 25.0, 50.0, 75.0, 90.0, 100.0]
        # Простая сетка для малого числа компонентов
        for s in steps:
            for idx in active_indices:
                point = np.zeros(n, dtype=float)
                point[idx] = s
                # Заполняем остаток первым доступным
                other = [o for o in active_indices if o != idx]
                if other:
                    point[other[0]] = max(0.0, 100.0 - s)
                candidates.append(_project_to_simplex_bounds(point, bounds))

    # Устраняем дубликаты
    unique_candidates = []
    seen = set()
    for cand in candidates:
        key = tuple(np.round(cand, 2))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(cand)

    return unique_candidates


def optimize_composition_v2(
    target_property: str,
    maximize: bool = True,
    constraints: Optional[Dict[str, Tuple[float, float]]] = None,
    available_components: Optional[List[str]] = None,
    db_path: Optional[str] = None,
    seed: int = 42,
    uncertainty_weight: float = 0.1,
) -> Dict[str, Any]:
    """Оптимизация состава пеллет по модели v2.

    Параметры:
        target_property: Целевой параметр (например, 'q', 'ad', 'density', 'tau_d2').
        maximize: True для максимизации, False для минимизации.
        constraints: Словарь {компонент: (min_%, max_%)}.
        available_components: Список доступных компонентов.
        db_path: Путь к БД SQLite для записи в ml_optimizations (не в measured_parameters).
        seed: Зерно генератора случайных чисел для детерминированности.
        uncertainty_weight: Вес штрафа за ширину доверительного интервала (std).
    """
    pellet_ml = require_lab()
    predictor = get_predictor()

    try:
        resolved_prop = pellet_ml.properties.resolve_property_id(target_property)
    except Exception:
        resolved_prop = target_property

    manifest = predictor.manifest or {}
    metrics_map = manifest.get("metrics") or {}
    if resolved_prop not in metrics_map:
        return {
            "success": False,
            "error": f"Свойство '{target_property}' не найдено в манифесте модели v2.",
            "model_version": "v2",
        }

    all_components = list(manifest.get("components") or [])
    if not all_components:
        return {
            "success": False,
            "error": "В манифесте модели v2 нет списка компонентов.",
            "model_version": "v2",
        }

    # Валидация ограничений
    if constraints:
        min_sum = 0.0
        max_sum = 0.0
        for comp, (c_min, c_max) in constraints.items():
            if c_min > c_max:
                return {
                    "success": False,
                    "error": f"Для компонента '{comp}' минимальное значение ({c_min}%) больше максимального ({c_max}%).",
                    "model_version": "v2",
                }
            if comp in all_components:
                min_sum += c_min
                max_sum += c_max
        if min_sum > 100.0:
            return {
                "success": False,
                "error": f"Ограничения несовместимы: минимальная сумма {min_sum:.1f}% превышает 100%.",
                "model_version": "v2",
            }
        if max_sum < 100.0 and len(constraints) == len(all_components):
            return {
                "success": False,
                "error": f"Ограничения несовместимы: максимальная сумма {max_sum:.1f}% меньше 100%.",
                "model_version": "v2",
            }

    domain = predictor.loaded.bundle.get("domain") if predictor.loaded else None

    # Генерация пула кандидатов
    candidates = _generate_candidate_pool(
        components=all_components,
        domain=domain,
        constraints=constraints,
        available_components=available_components,
        seed=seed,
    )

    if not candidates:
        return {
            "success": False,
            "error": "Не удалось сгенерировать допустимые кандидаты для заданных ограничений.",
            "model_version": "v2",
        }

    # Оценка кандидатов
    best_candidate: Optional[np.ndarray] = None
    best_contract: Optional[Dict] = None
    best_score = float("inf")

    for cand in candidates:
        comp_dict = {
            all_components[i]: float(cand[i])
            for i in range(len(all_components))
            if cand[i] > 0.01
        }
        try:
            contract = predictor.predict_contract(comp_dict, resolved_prop)
        except Exception as exc:
            logger.debug("Candidate predict failed: %s", exc)
            continue

        if not contract:
            continue

        val = contract["value"]
        std = contract.get("std", 0.0) or 0.0
        in_domain = bool(contract.get("in_domain", False))

        # Базовая цель
        base_score = -val if maximize else val

        # Штраф за неопределённость (предпочитаем более узкий доверительный интервал)
        uncertainty_penalty = uncertainty_weight * std

        # Штраф за выход из области применимости
        if not in_domain:
            domain_penalty = 10000.0
            if domain:
                info = pellet_ml.domain.assess(comp_dict, domain)
                dist = info.get("distance") or 0.0
                tau = info.get("tau") or 0.0
                domain_penalty += 100.0 * max(0.0, dist - tau)
        else:
            domain_penalty = 0.0

        total_score = base_score + uncertainty_penalty + domain_penalty
        if total_score < best_score:
            best_score = total_score
            best_candidate = cand
            best_contract = contract

    if best_candidate is None or best_contract is None:
        return {
            "success": False,
            "error": "Не удалось найти решение оптимизации.",
            "model_version": "v2",
        }

    # Очистка и нормализация финального состава
    raw_comp = {
        all_components[i]: round(float(best_candidate[i]), 2)
        for i in range(len(all_components))
        if best_candidate[i] >= 0.1
    }
    total_percent = sum(raw_comp.values())
    if total_percent > 0 and abs(total_percent - 100.0) > 0.01:
        # Корректируем самый большой компонент
        dominant = max(raw_comp, key=raw_comp.get)
        raw_comp[dominant] = round(raw_comp[dominant] + (100.0 - total_percent), 2)

    # Итоговый контракт на очищенном составе
    final_contract = predictor.predict_contract(raw_comp, resolved_prop) or best_contract
    optimal_value = round(float(final_contract["value"]), 2)
    display_name = pellet_ml.properties.display_name(resolved_prop)
    direction_str = "максимизации" if maximize else "минимизации"
    comp_str = ", ".join([f"{v:.1f}% {k}" for k, v in raw_comp.items()])

    domain_tag = " [в области применимости]" if final_contract.get("in_domain") else " [ВНИМАНИЕ: вне области применимости]"
    message = (
        f"Оптимальный состав (v2) для {direction_str} '{display_name}': {comp_str}. "
        f"Ожидаемое значение: {optimal_value:.2f}{domain_tag}."
    )

    result = {
        "success": True,
        "optimal_composition": raw_comp,
        "optimal_value": optimal_value,
        "std": round(float(final_contract.get("std", 0.0) or 0.0), 3),
        "lo90": round(float(final_contract.get("lo90", 0.0) or 0.0), 2),
        "hi90": round(float(final_contract.get("hi90", 0.0) or 0.0), 2),
        "in_domain": bool(final_contract.get("in_domain")),
        "model_name": final_contract.get("model_name"),
        "target_property": resolved_prop,
        "display_name": display_name,
        "message": message,
        "model_version": "v2",
        "contract": final_contract,
    }

    # Сохраняем результат в ml_optimizations (БЕЗ добавления в measured_parameters!)
    if db_path:
        try:
            optimization_data = {
                "target_property": resolved_prop,
                "maximize": maximize,
                "optimal_composition": raw_comp,
                "optimal_value": optimal_value,
                "constraints": constraints or {},
                "algorithm": f"v2_{final_contract.get('model_name', 'ensemble')}",
                "model_metrics": metrics_map.get(resolved_prop, {}),
            }
            insert_ml_optimization(db_path, optimization_data)
            logger.info("Результат оптимизации v2 сохранён в ml_optimizations (db=%s)", db_path)
        except Exception as exc:
            logger.warning("Не удалось сохранить результат оптимизации v2 в БД: %s", exc)

    return result
