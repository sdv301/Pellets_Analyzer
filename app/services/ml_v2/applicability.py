"""Область применимости: новая добавка и расстояние до обучающих рецептур."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from app.services.ml_v2.components_registry import RECIPE_COMPONENTS, canonicalize_composition, match_component

Z90 = 1.6448536269514722


def fraction_vector(composition: Dict[str, float], components: Sequence[str]) -> np.ndarray:
    canon = canonicalize_composition(composition)
    vec = np.array([float(canon.get(c, 0.0)) for c in components], dtype=float)
    total = float(vec.sum())
    if total > 1e-9:
        vec = vec / total * 100.0
    return vec


def build_domain(compositions: List[Dict[str, float]], components: Optional[Sequence[str]] = None) -> Dict:
    comps = list(components or RECIPE_COMPONENTS)
    if not compositions:
        return {
            "components": comps,
            "matrix": np.zeros((0, len(comps))),
            "seen": set(),
            "tau": 0.0,
        }
    matrix = np.vstack([fraction_vector(c, comps) for c in compositions])
    seen = set()
    for row in matrix:
        for name, value in zip(comps, row):
            if value > 1.0:
                seen.add(name)
    tau = _loo_distance_threshold(matrix)
    return {"components": comps, "matrix": matrix, "seen": seen, "tau": tau}


def _loo_distance_threshold(matrix: np.ndarray) -> float:
    n = len(matrix)
    if n < 3:
        return float("inf")
    dists = []
    for i in range(n):
        diff = matrix - matrix[i]
        dist = np.linalg.norm(diff, axis=1)
        dist[i] = np.inf
        dists.append(float(np.min(dist)))
    return float(np.percentile(dists, 95))


def assess(composition: Dict[str, float], domain: Optional[Dict]) -> Dict:
    """in_domain=False, если добавка новая (>1%) или состав дальше 95-го перцентиля LOO."""
    if not domain or domain.get("matrix") is None or len(domain.get("matrix", [])) == 0:
        return {
            "in_domain": True,
            "reason": "область применимости не откалибрована",
            "distance": None,
            "new_components": [],
        }
    canon = canonicalize_composition(composition)
    new = []
    for key, value in canon.items():
        if str(key).startswith(("phys_", "synergy_")):
            continue
        if float(value) <= 1.0:
            continue
        name = match_component(key) or key
        if name not in domain["seen"]:
            new.append(name)
    vec = fraction_vector(canon, domain["components"])
    dists = np.linalg.norm(domain["matrix"] - vec, axis=1)
    distance = float(np.min(dists)) if len(dists) else None
    tau = float(domain.get("tau") or 0.0)
    reasons = []
    if new:
        reasons.append("новая добавка: " + ", ".join(new))
    if distance is not None and np.isfinite(tau) and distance > tau + 1e-6:
        reasons.append(f"далеко от обучающих рецептур (d={distance:.2f} > τ={tau:.2f})")
    return {
        "in_domain": not reasons,
        "reason": "; ".join(reasons),
        "distance": distance,
        "new_components": new,
        "tau": tau,
    }


def extrapolation_penalty(composition: Dict[str, float], domain: Optional[Dict], weight: float = 0.05) -> float:
    """Штраф оптимизатору за выход за область применимости."""
    info = assess(composition, domain)
    if info["in_domain"]:
        return 0.0
    distance = info["distance"] or 0.0
    tau = info.get("tau") or 0.0
    extra = max(0.0, distance - tau)
    novelty = 25.0 * len(info["new_components"])
    return weight * (extra + novelty)
