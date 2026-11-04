"""
Тесты для модуля database.py — подключение, инициализация, CRUD.
"""
import os
import tempfile
import sqlite3
import pytest
import pandas as pd

from app.database.database import (
    init_db, insert_data, query_db,
    insert_ml_optimization, insert_ml_model_metrics,
    get_ml_optimizations, get_active_ml_models,
    add_ml_optimization_to_training_data,
    check_and_migrate_db,
    DatabaseConnection, get_db_connection,
)


class TestInitDB:
    """Тесты инициализации БД."""

    def test_init_db_creates_tables(self, db_path):
        """init_db создаёт все необходимые таблицы."""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()

        assert 'measured_parameters' in tables
        assert 'components' in tables
        assert 'ml_optimizations' in tables
        assert 'ml_model_metrics' in tables

    def test_init_db_is_idempotent(self, db_path):
        """init_db можно вызывать многократно без ошибок."""
        init_db(db_path)
        init_db(db_path)
        init_db(db_path)

        # Проверяем что таблицы существуют
        result = query_db(db_path, 'measured_parameters')
        assert isinstance(result, pd.DataFrame)

    def test_init_db_creates_indexes(self, db_path):
        """init_db создаёт индексы."""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
        indexes = [row[0] for row in cursor.fetchall()]
        conn.close()

        assert 'idx_composition' in indexes
        assert 'idx_component' in indexes


class TestInsertData:
    """Тесты вставки данных."""

    def test_insert_measured_parameters(self, db_path, sample_measured_data):
        """Вставка данных в measured_parameters."""
        insert_data(db_path, 'measured_parameters', sample_measured_data)

        result = query_db(db_path, 'measured_parameters')
        assert len(result) == 3
        assert 'composition' in result.columns
        assert 'density' in result.columns

    def test_insert_components(self, db_path, sample_components_data):
        """Вставка данных в components."""
        insert_data(db_path, 'components', sample_components_data)

        result = query_db(db_path, 'components')
        assert len(result) == 3
        assert 'component' in result.columns
        assert 'ro' in result.columns

    def test_insert_replaces_duplicates(self, db_path, sample_measured_data):
        """INSERT OR REPLACE заменяет дубликаты."""
        insert_data(db_path, 'measured_parameters', sample_measured_data)
        insert_data(db_path, 'measured_parameters', sample_measured_data)

        result = query_db(db_path, 'measured_parameters')
        assert len(result) == 3

    def test_insert_handles_missing_columns(self, db_path):
        """Вставка работает с неполным DataFrame."""
        df = pd.DataFrame([
            {'composition': '100% Опилки', 'q': 19.0}
        ])
        insert_data(db_path, 'measured_parameters', df)

        result = query_db(db_path, 'measured_parameters')
        assert len(result) == 1
        assert result.iloc[0]['q'] == 19.0
        assert pd.isna(result.iloc[0]['density'])


class TestQueryDB:
    """Тесты запросов из БД."""

    def test_query_all_data(self, db_path, sample_measured_data):
        """query_db возвращает все данные."""
        insert_data(db_path, 'measured_parameters', sample_measured_data)

        result = query_db(db_path, 'measured_parameters')
        assert len(result) == 3

    def test_query_empty_table(self, db_path):
        """query_db возвращает пустой DataFrame для пустой таблицы."""
        result = query_db(db_path, 'measured_parameters')
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_query_invalid_table(self, db_path):
        """query_db выбрасывает ошибку для недопустимой таблицы."""
        with pytest.raises(ValueError):
            query_db(db_path, 'invalid_table')


class TestMLOperations:
    """Тесты ML-операций с БД."""

    def test_insert_ml_optimization(self, db_path):
        """Сохранение ML оптимизации."""
        opt_data = {
            'target_property': 'q',
            'maximize': True,
            'optimal_composition': {'Опилки': 60.0, 'Солома': 40.0},
            'optimal_value': 18.5,
            'constraints': {},
            'algorithm': 'xgboost',
            'model_metrics': {'r2': 0.85}
        }
        insert_ml_optimization(db_path, opt_data)

        results = get_ml_optimizations(db_path)
        assert len(results) == 1
        assert results.iloc[0]['target_property'] == 'q'
        assert results.iloc[0]['optimal_value'] == 18.5

    def test_insert_ml_model_metrics(self, db_path):
        """Сохранение метрик ML модели."""
        metrics = {
            'target_property': 'q',
            'algorithm': 'xgboost',
            'r2_score': 0.85,
            'mae': 0.5,
            'cv_r2': 0.80,
            'feature_importance': {'Опилки': 0.6, 'Солома': 0.4},
            'training_data_size': 100
        }
        insert_ml_model_metrics(db_path, metrics)

        results = get_active_ml_models(db_path)
        assert len(results) == 1
        assert results.iloc[0]['r2_score'] == 0.85

    def test_add_ml_optimization_to_training_data(self, db_path):
        """Добавление оптимизации в тренировочные данные."""
        opt_data = {
            'target_property': 'q',
            'maximize': True,
            'optimal_composition': {'Опилки': 70.0, 'Солома': 30.0},
            'optimal_value': 19.0,
            'constraints': {},
            'algorithm': 'xgboost',
            'model_metrics': {}
        }
        add_ml_optimization_to_training_data(db_path, opt_data)

        result = query_db(db_path, 'measured_parameters')
        assert len(result) == 1
        assert 'Опилки' in result.iloc[0]['composition']


class TestDatabaseConnection:
    """Тесты класса DatabaseConnection."""

    def test_connection_context_manager(self, db_path):
        """DatabaseConnection работает как контекстный менеджер."""
        with DatabaseConnection(db_path=db_path) as conn:
            df = conn.query("SELECT * FROM measured_parameters")
            assert isinstance(df, pd.DataFrame)

    def test_connection_execute(self, db_path):
        """DatabaseConnection.execute выполняет запрос."""
        with DatabaseConnection(db_path=db_path) as conn:
            conn.execute("INSERT INTO components (component) VALUES (?)", ('Тест',))

        result = query_db(db_path, 'components')
        assert len(result) == 1

    def test_get_db_connection_factory(self, db_path):
        """Фабрика get_db_connection создаёт подключение."""
        conn = get_db_connection(db_path=db_path)
        assert isinstance(conn, DatabaseConnection)


class TestMigration:
    """Тесты миграции БД."""

    def test_migration_adds_missing_columns(self, db_path):
        """check_and_migrate_db добавляет недостающие колонки."""
        init_db(db_path)

        # Удаляем колонку для теста
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # Проверяем что колонки существуют
        cursor.execute("PRAGMA table_info(components)")
        columns = [row[1] for row in cursor.fetchall()]
        conn.close()

        assert 'ro' in columns
        assert 'cost_raw' in columns
        assert 'cost_crush' in columns
        assert 'cost_granule' in columns
