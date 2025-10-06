import pandas as pd
import os
import re
from database import insert_data, init_db

def normalize_column_name(col):
    if not isinstance(col, str):
        col = str(col)
    
    # Приводим к нижнему регистру и убираем пробелы
    col = col.strip().lower()
    col = re.sub(r'\s+', ' ', col)
    
    # Основные замены
    replacements = {
        'составы': 'composition',
        'компоненты': 'component', 
        'ρ': 'density',
        'ρ, кг/м3': 'density',
        'kf, %': 'kf',
        'kt, %': 'kt', 
        'h, %': 'h',
        'mass loss, %': 'mass_loss',
        'тign, °c': 'tign',  # ИСПРАВЛЕНО: было 'тign_с'
        'тign, °с': 'tign',  # ДОБАВЛЕНО: для русского 'с'
        'tb, °c': 'tb',
        'τd1, с': 'tau_d1',
        'τd2, с': 'tau_d2', 
        'τb, с': 'tau_b',
        'co2,': 'co2',
        'co,': 'co',
        'so2,': 'so2',
        'nox,': 'nox',
        'q, мдж/кг': 'q',
        'ad, %': 'ad',
        'war, %': 'war',
        'vd, %': 'vd',
        'cd, %': 'cd',
        'hd, %': 'hd', 
        'nd, %': 'nd',
        'sd, %': 'sd',
        'od, %': 'od'
    }
    
    # Ищем точное соответствие
    for key, value in replacements.items():
        if key in col:
            return value
    
    # Если не нашли, возвращаем очищенное название
    col = re.sub(r'[^\w]', '_', col)  # заменяем спецсимволы на _
    col = re.sub(r'_+', '_', col)     # убираем повторяющиеся _
    return col.strip('_')

def validate_data(df, expected_columns, table_name):
    messages = []
    df = df.loc[:, ~df.columns.duplicated()]  # Удаляем дублирующиеся столбцы
    
    # УДАЛЯЕМ ЛИШНИЕ СТОЛБЦЫ, которые не входят в expected_columns
    columns_to_keep = [col for col in df.columns if col in expected_columns]
    df = df[columns_to_keep]
    
    for col in expected_columns:
        if col not in df.columns:
            df[col] = None
            messages.append(f"Предупреждение: Столбец '{col}' отсутствует в {table_name}. Заполнен значениями None.")
    
    df = df.dropna(how='all')
    df = df.drop_duplicates()
    
    for col in expected_columns:
        if col in ('composition', 'component'):
            continue
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except Exception as e:
            messages.append(f"Ошибка преобразования столбца '{col}' в {table_name}: {str(e)}")
    
    return df, messages

def process_data_source(file_path, db_path):
    messages = []
    sheet_data = []
    components_sheet_name = 'Таблица компонентов'
    
    print(f"=== ОТЛАДКА: Начало обработки файла {file_path} ===")
    
    init_db(db_path)
    expected_measured_columns = [
        'composition', 'density', 'kf', 'kt', 'h', 'mass_loss', 
        'tign', 'tb', 'tau_d1', 'tau_d2', 'tau_b', 
        'co2', 'co', 'so2', 'nox', 'q', 'ad'
    ]
    expected_components_columns = [
        'component', 'war', 'ad', 'vd', 'q', 'cd', 'hd', 'nd', 'sd', 'od'
    ]

    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            print(f"=== ОТЛАДКА CSV: Прочитано {len(df)} строк ===")
            print(f"=== ОТЛАДКА CSV: Колонки до нормализации: {list(df.columns)}")
            
            df.columns = [normalize_column_name(col) for col in df.columns]
            print(f"=== ОТЛАДКА CSV: Колонки после нормализации: {list(df.columns)}")
            
            # Удаляем полностью пустые столбцы и дублирующиеся
            df = df.dropna(axis=1, how='all')
            df = df.loc[:, ~df.columns.duplicated()]
            
            messages.append(f"Успешно загружен CSV-файл: {os.path.basename(file_path)}")
            df, validation_messages = validate_data(df, expected_measured_columns, "measured_parameters")
            messages.extend(validation_messages)
            
            print(f"=== ОТЛАДКА CSV: DataFrame после валидации: {len(df)} строк, {len(df.columns)} колонок")
            if not df.empty:
                print(f"=== ОТЛАДКА CSV: Первые 3 строки данных:")
                print(df.head(3))
            
            if not df.empty:
                insert_data(db_path, "measured_parameters", df)
                messages.append("Данные измеренных параметров сохранены в базу данных")
                sheet_data.append({'name': 'Измеренные параметры', 'data': df})
            else:
                messages.append("CSV-файл не содержит валидных данных")
                sheet_data.append({'name': 'Измеренные параметры', 'data': pd.DataFrame()})
            return messages, components_sheet_name, sheet_data
        
        elif file_path.endswith('.xlsx'):
            xls = pd.ExcelFile(file_path)
            sheet_names = xls.sheet_names
            print(f"=== ОТЛАДКА Excel: Листы в файле: {sheet_names}")
            
            if not sheet_names:
                messages.append("Excel-файл не содержит листов")
                return messages, components_sheet_name, sheet_data
                
            for sheet_name in sheet_names:
                print(f"=== ОТЛАДКА Excel: Обрабатываем лист '{sheet_name}'")
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                print(f"=== ОТЛАДКА Excel: Прочитано {len(df)} строк ===")
                print(f"=== ОТЛАДКА Excel: Колонки до нормализации: {list(df.columns)}")
                
                # Пропускаем полностью пустые листы
                if df.empty or df.shape[0] == 0:
                    messages.append(f"Лист '{sheet_name}' пуст - пропускаем")
                    continue
                    
                df.columns = [normalize_column_name(col) for col in df.columns]
                print(f"=== ОТЛАДКА Excel: Колонки после нормализации: {list(df.columns)}")
                
                # Удаляем полностью пустые столбцы и дублирующиеся
                df = df.dropna(axis=1, how='all')
                df = df.loc[:, ~df.columns.duplicated()]
                print(f"=== ОТЛАДКА Excel: Колонки после очистки: {list(df.columns)}")
                
                messages.append(f"Успешно загружен лист '{sheet_name}' из Excel-файла: {os.path.basename(file_path)}")
                
                # Определяем тип данных по наличию ключевых столбцов
                table_name = None
                if 'composition' in df.columns:
                    print(f"=== ОТЛАДКА Excel: Определен как measured_parameters (есть composition)")
                    df, validation_messages = validate_data(df, expected_measured_columns, f"measured_parameters ({sheet_name})")
                    table_name = "measured_parameters"
                elif 'component' in df.columns:
                    print(f"=== ОТЛАДКА Excel: Определен как components (есть component)")
                    df, validation_messages = validate_data(df, expected_components_columns, f"components ({sheet_name})")
                    table_name = "components"
                else:
                    # Если не можем определить тип, пробуем оба варианта
                    print(f"=== ОТЛАДКА Excel: Не удалось определить тип - пробуем measured_parameters")
                    messages.append(f"Не удалось определить тип данных для листа '{sheet_name}' - пробуем как измеренные параметры")
                    df, validation_messages = validate_data(df, expected_measured_columns, f"measured_parameters ({sheet_name})")
                    table_name = "measured_parameters"
                
                messages.extend(validation_messages)
                print(f"=== ОТЛАДКА Excel: После валидации: {len(df)} строк")
                
                if not df.empty:
                    print(f"=== ОТЛАДКА Excel: Сохраняем в таблицу {table_name}")
                    print(f"=== ОТЛАДКА Excel: Первые 3 строки данных:")
                    print(df.head(3))
                    
                    insert_data(db_path, table_name, df)
                    messages.append(f"Данные листа '{sheet_name}' сохранены в таблицу {table_name}")
                    # Добавляем в sheet_data только если данные не пустые
                    sheet_data.append({'name': sheet_name, 'data': df})
                else:
                    messages.append(f"Лист '{sheet_name}' не содержит валидных данных после обработки")
            
            # Устанавливаем имя листа компонентов
            if len(sheet_names) > 1:
                components_sheet_name = sheet_names[1]
            else:
                components_sheet_name = 'Характеристики компонентов'
                
            if not sheet_data:
                messages.append("Excel-файл не содержит валидных данных для отображения")
            else:
                messages.append(f"Обработано {len(sheet_data)} листов из Excel-файла")
                
            print(f"=== ОТЛАДКА Excel: Итог - {len(sheet_data)} листов с данными")
            return messages, components_sheet_name, sheet_data
        
        else:
            messages.append(f"Ошибка: Формат файла {file_path} не поддерживается")
            return messages, components_sheet_name, sheet_data
            
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"=== ОШИБКА в process_data_source: {str(e)}")
        print(f"=== ДЕТАЛИ ОШИБКИ: {error_details}")
        messages.append(f"Ошибка обработки файла {file_path}: {str(e)}")
        return messages, components_sheet_name, sheet_data


def load_sample_data():
    measured_params_data = {
        'composition': [
            '100% Опилки', 
            '95% Опилки, 5% рисовая шелуха + солома', 
            '90% Опилки, 10% рисовая шелуха + солома', 
            '85% Опилки, 15% рисовая шелуха + солома', 
            '95% Опилки, 5% торф'
        ],
        'density': [967, 1017, 1024, 1029, 1051],
        'kf': [98.87, 98.96, 98.65, 98.16, 99.35],
        'kt': [97.96, 99.36, 99.51, 99.54, 99.6],
        'h': [3.81, 3.18, 3.32, 3.45, 2.78],
        'mass_loss': [99.99, 96.63, 94.86, 93.09, 97.02],
        'tign': [276, 292, 293, 298, 288],
        'tb': [602, 629, 647, 676, 713],
        'tau_d1': [5.24, 6.55, 6.9, 7.15, 5.21],
        'tau_d2': [40.86, 39.53, 40.49, 41.76, 39.85],
        'tau_b': [162, 193, 197, 202, 193],
        'co2': [5.56, 5.79, 5.84, 6, 6.47],
        'co': [1.56, 1.26, 1.14, 0.88, 1.22],
        'so2': [168, 6, 5, 4, 7],
        'nox': [26, 56, 63, 69, 55],
        'q': [19.79, 19.62, 19.45, 19.27, 19.39],
        'ad': [0.18, 0.55, 0.92, 1.29, 1.21]
    }

    components_data = {
        'component': [
            'Опилки', 'Солома', 'Картон', 'Подсолнечный жмых', 'Рисовая шелуха'
        ],
        'war': [10.8, 9.83, 5.02, 6.16, 6.34],
        'ad': [0.18, 4.33, 3.16, 4.09, 10.85],
        'vd': [66.48, 59.35, 95.84, 60.05, 80.45],
        'q': [19.79, 16.75, 16.73, 20.79, 15.95],
        'cd': [52.62, 48.42, 48.2, 54.22, 39.21],
        'hd': [6.82, 6.54, 6.5, 7.36, 5.3],
        'nd': [0.16, 1.15, 0.59, 4.01, 0.36],
        'sd': [0.58, 0.06, 0.05, 0.59, 0.09],
        'od': [39.64, 39.5, 41.5, 29.73, 44.2]
    }

    return pd.DataFrame(measured_params_data), pd.DataFrame(components_data)