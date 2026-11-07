"""
Тесты для модуля ml_optimizer.py — ML-оптимизация.
"""
import pytest
import pandas as pd
import numpy as np

from app.services.ml_optimizer import (
    CompositionParser,
    PelletPropertyPredictor,
    MLCompositionOptimizer,
)


class TestCompositionParser:
    """Тесты парсера составов."""

    def setup_method(self):
        self.parser = CompositionParser()

    def test_parse_simple_composition(self):
        """Парсинг простого состава."""
        text = "60% Опилки, 30% Солома, 10% Картон"
        result = self.parser.parse_composition(text)

        assert 'Опилки' in result
        assert 'Солома' in result
        assert 'Картон' in result
        assert result['Опилки'] == 60.0
        assert result['Солома'] == 30.0
        assert result['Картон'] == 10.0

    def test_parse_empty_composition(self):
        """Парсинг пустого состава."""
        assert self.parser.parse_composition('') == {}
        assert self.parser.parse_composition(None) == {}

    def test_normalize_composition(self):
        """Нормализация состава до 100%."""
        composition = {'Опилки': 50.0, 'Солома': 25.0}  # Сумма = 75%
        normalized, warnings = self.parser.validate_and_normalize(composition)

        assert abs(sum(normalized.values()) - 100.0) < 0.1
        assert normalized['Опилки'] > 50.0  # Должно быть нормализовано

    def test_composition_to_text(self):
        """Конвертация состава в текст."""
        composition = {'Опилки': 60.0, 'Солома': 40.0}
        text = self.parser.composition_to_text(composition)

        assert '60.0% Опилки' in text
        assert '40.0% Солома' in text

    def test_composition_from_text(self):
        """Парсинг текста в состав."""
        text = "60.0% Опилки, 40.0% Солома"
        result = self.parser.composition_from_text(text)

        assert 'Опилки' in result
        assert 'Солома' in result


class TestPelletPropertyPredictor:
    """Тесты предсказателя свойств."""

    def setup_method(self):
        self.predictor = PelletPropertyPredictor()

    def test_prepare_features(self):
        """Подготовка признаков из данных."""
        data = pd.DataFrame([
            {'composition': '60% Опилки, 40% Солома', 'q': 18.0},
            {'composition': '50% Опилки, 50% Солома', 'q': 17.0},
        ])

        X, feature_names, valid_indices = self.predictor.prepare_features(data)

        assert len(X) == 2
        assert len(feature_names) == 2  # Опилки и Солома
        assert 'Опилки' in feature_names
        assert 'Солома' in feature_names

    def test_prepare_features_empty_data(self):
        """Подготовка признаков из пустых данных."""
        data = pd.DataFrame()
        X, feature_names, valid_indices = self.predictor.prepare_features(data)

        assert len(X) == 0
        assert len(feature_names) == 0

    def test_prepare_composition_for_prediction(self):
        """Подготовка состава для предсказания."""
        self.predictor.feature_names = ['Опилки', 'Солома']
        composition = {'Опилки': 60.0, 'Солома': 40.0}

        X = self.predictor.prepare_composition_for_prediction(composition)

        assert len(X) == 2
        assert X[0] == 60.0
        assert X[1] == 40.0

    def test_train_insufficient_data(self):
        """Обучение на недостаточных данных."""
        data = pd.DataFrame([
            {'composition': '60% Опилки, 40% Солома', 'q': 18.0},
        ])  # Только 1 запись, нужно минимум 8

        result = self.predictor.train(data, ['q'])

        assert result['success'] is False
        assert result['trained_count'] == 0


class TestMLCompositionOptimizer:
    """Тесты оптимизатора составов."""

    def setup_method(self):
        predictor = PelletPropertyPredictor()
        self.optimizer = MLCompositionOptimizer(predictor)

    def test_optimize_without_model(self):
        """Оптимизация без обученной модели."""
        result = self.optimizer.optimize_composition('q', maximize=True)

        assert result['success'] is False
        assert 'не обучена' in result['error'].lower() or 'модель' in result['error'].lower()
