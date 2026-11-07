"""
Тесты для модуля data_processor.py — обработка CSV/Excel.
"""
import os
import tempfile
import pytest
import pandas as pd

from app.services.data_processor import (
    normalize_column_name, validate_data, process_data_source,
    prepare_data_for_display
)


class TestNormalizeColumnName:
    """Тесты нормализации имён колонок."""

    def test_basic_normalization(self):
        """Базовая нормализация."""
        assert normalize_column_name('  Составы  ') == 'composition'

    def test_russian_to_english(self):
        """Русские названия -> английские."""
        assert normalize_column_name('ρ, кг/м3') == 'density'
        assert normalize_column_name('kf, %') == 'kf'
        assert normalize_column_name('kt, %') == 'kt'
        assert normalize_column_name('h, %') == 'h'
        assert normalize_column_name('тign, °c') == 'tign'
        assert normalize_column_name('тign, °с') == 'tign'  # русская 'с'
        assert normalize_column_name('tb, °c') == 'tb'
        assert normalize_column_name('τd1, с') == 'tau_d1'
        assert normalize_column_name('τd2, с') == 'tau_d2'
        assert normalize_column_name('τb, с') == 'tau_b'
        assert normalize_column_name('q, мдж/кг') == 'q'
        assert normalize_column_name('ad, %') == 'ad'

    def test_unknown_column(self):
        """Неизвестная колонка возвращается очищенной."""
        result = normalize_column_name('Some Unknown Column')
        assert 'some_unknown_column' == result

    def test_non_string_input(self):
        """Не-строковый ввод конвертируется."""
        result = normalize_column_name(123)
        assert result == '123'

    def test_special_characters(self):
        """Спецсимволы заменяются на подчёркивания."""
        result = normalize_column_name('col@name#with!special')
        assert '@' not in result
        assert '#' not in result


class TestValidateData:
    """Тесты валидации данных."""

    def test_valid_data_passes(self):
        """Валидные данные проходят без предупреждений."""
        df = pd.DataFrame([
            {'composition': 'Test', 'density': 1000.0, 'q': 18.0, 'ad': 3.0}
        ])
        expected = ['composition', 'density', 'q', 'ad']
        result, messages = validate_data(df, expected, 'test')

        assert len(messages) == 0
        assert 'composition' in result.columns

    def test_missing_columns_filled(self):
        """Отсутствующие колонки заполняются None."""
        df = pd.DataFrame([{'composition': 'Test'}])
        expected = ['composition', 'density', 'q']
        result, messages = validate_data(df, expected, 'test')

        assert 'density' in result.columns
        assert 'q' in result.columns
        assert len(messages) == 2  # 2 предупреждения

    def test_removes_extra_columns(self):
        """Лишние колонки удаляются."""
        df = pd.DataFrame([
            {'composition': 'Test', 'density': 1000.0, 'extra_col': 'value'}
        ])
        expected = ['composition', 'density']
        result, messages = validate_data(df, expected, 'test')

        assert 'extra_col' not in result.columns

    def test_removes_duplicates(self):
        """Дубликаты строк удаляются."""
        df = pd.DataFrame([
            {'composition': 'Test', 'density': 1000.0},
            {'composition': 'Test', 'density': 1000.0}
        ])
        expected = ['composition', 'density']
        result, messages = validate_data(df, expected, 'test')

        assert len(result) == 1

    def test_numeric_conversion(self):
        """Строковые значения конвертируются в числа."""
        df = pd.DataFrame([
            {'composition': 'Test', 'density': '1000.5', 'q': '18.0'}
        ])
        expected = ['composition', 'density', 'q']
        result, messages = validate_data(df, expected, 'test')

        assert result['density'].dtype == 'float64'
        assert result['q'].dtype == 'float64'


class TestProcessDataSource:
    """Тесты обработки файлов."""

    def test_csv_processing(self, db_path):
        """Обработка CSV-файла."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            f.write('composition,density,kf,kt,h,q,ad\n')
            f.write('60% Опилки,1000,95,90,5,18.0,3.0\n')
            f.write('50% Солома,980,88,85,7,16.5,5.0\n')
            csv_path = f.name

        try:
            messages, comp_sheet, sheet_data = process_data_source(csv_path, db_path)

            assert len(messages) > 0
            assert len(sheet_data) == 1
            assert sheet_data[0]['name'] == 'Измеренные параметры'
            assert len(sheet_data[0]['data']) == 2
        finally:
            os.unlink(csv_path)

    def test_unsupported_format(self, db_path):
        """Неподдерживаемый формат файла."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write('test data')
            txt_path = f.name

        try:
            messages, _, sheet_data = process_data_source(txt_path, db_path)
            assert any('не поддерживается' in msg for msg in messages)
            assert len(sheet_data) == 0
        finally:
            os.unlink(txt_path)

    def test_nonexistent_file(self, db_path):
        """Несуществующий файл."""
        messages, _, sheet_data = process_data_source('nonexistent.csv', db_path)
        assert any('Ошибка' in msg for msg in messages)


class TestPrepareDataForDisplay:
    """Тесты подготовки данных для отображения."""

    def test_english_to_russian(self):
        """Английские названия -> русские."""
        df = pd.DataFrame([
            {'composition': 'Test', 'density': 1000.0, 'q': 18.0, 'ad': 3.0}
        ])
        result = prepare_data_for_display(df, 'measured_parameters')

        assert 'Составы' in result.columns
        assert 'Плотность, кг/м3' in result.columns
        assert 'Теплота сгорания, МДж/кг' in result.columns

    def test_unknown_columns_preserved(self):
        """Неизвестные колонки сохраняются."""
        df = pd.DataFrame([
            {'composition': 'Test', 'custom_col': 'value'}
        ])
        result = prepare_data_for_display(df, 'measured_parameters')

        assert 'custom_col' in result.columns
