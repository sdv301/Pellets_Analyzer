#!/usr/bin/env python3
"""CLI скрипт для автономного обучения legacy ML моделей вне веб-воркеров.

Использование:
    python scripts/train_legacy.py [--algorithm random_forest] [--db-path data/pellets_data.db]
"""
import argparse
import sys
import os
import logging

# Обеспечиваем импорт из корня проекта
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.services.ai_integration import AIMLSystem
from app.services.ml_optimizer import PelletPropertyPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_legacy")


def main():
    parser = argparse.ArgumentParser(description="Автономное обучение legacy ML-моделей Pellets Analyzer.")
    parser.add_argument("--db-path", default="pellets_data.db", help="Путь к SQLite базе данных")
    parser.add_argument("--algorithm", default="random_forest", choices=["random_forest", "gradient_boosting", "xgboost", "linear", "auto"],
                        help="Алгоритм регрессии")
    args = parser.parse_args()

    db_path = args.db_path
    if not os.path.isabs(db_path):
        db_path = os.path.join(BASE_DIR, db_path)

    logger.info("Инициализация AIMLSystem для БД: %s", db_path)
    ml_system = AIMLSystem(db_path=db_path)

    target_properties = list(PelletPropertyPredictor().target_properties_mapping.keys())
    logger.info("Запуск обучения для свойств: %s (алгоритм: %s)", target_properties, args.algorithm)

    result = ml_system.train_models(target_properties=target_properties, algorithm=args.algorithm)
    if result.get("success"):
        logger.info("Обучение успешно завершено!")
        logger.info("Результаты: %s", result.get("metrics", {}))
        return 0
    else:
        logger.error("Ошибка при обучении: %s", result.get("error"))
        return 1


if __name__ == "__main__":
    sys.exit(main())
