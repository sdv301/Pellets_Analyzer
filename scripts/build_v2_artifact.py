#!/usr/bin/env python3
"""Собрать артефакт v2 пакетом pellet_ml и записать в manifest lab_commit.

Пример:
    python scripts/build_v2_artifact.py --out models/v2
    python scripts/build_v2_artifact.py --quick --out /tmp/pellet-quick

joblib в git не коммитится. Каталог models/ в .gitignore.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from app.services.ml_v2.lab_import import expected_lab_commit, require_lab, resolve_lab_path
from app.services.ml_v2.predictor import stamp_manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Сборка артефакта pellet_ml для продукта")
    parser.add_argument("--out", default=str(REPO / "models" / "v2"))
    parser.add_argument("--data", default="", help="CSV рецептур. По умолчанию lab/data/recipes.csv")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--targets", nargs="*")
    parser.add_argument("--models", nargs="*")
    args = parser.parse_args(argv)

    lab = resolve_lab_path()
    if lab is None:
        print("субмодуль lab не найден. Задайте PELLET_LAB_PATH или выполните git submodule update --init lab", file=sys.stderr)
        return 1
    pellet_ml = require_lab()
    data = Path(args.data) if args.data else lab / "data" / "recipes.csv"
    if not data.is_file():
        print(f"нет файла рецептур: {data}", file=sys.stderr)
        return 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pellet_ml.train_and_save(
        data,
        out,
        targets=args.targets,
        models=args.models,
        quick=args.quick,
    )
    commit = expected_lab_commit()
    payload = stamp_manifest(out, commit)
    print(f"{out / 'pellet_models.joblib'}")
    print(f"api_version={payload['api_version']} lab_commit={payload['lab_commit']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
