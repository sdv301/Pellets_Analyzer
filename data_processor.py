import pandas as pd
import os
import re
from database import insert_data


def normalize_column_name(col):
    if not isinstance(col, str):
        col = str(col)
    col = col.strip().lower()
    col = re.sub(r'\s+', ' ', col)
    col = col.replace(', ', ',').replace(' ,', ',').replace('°с', '°c').replace('°C', '°c')
    if col.endswith(','):
        base_name = col.rstrip(',')
        if 'so2' in base_name or 'nox' in base_name:
            return f"{base_name}, ppm"
        return f"{base_name}, %"
    mappings = {
        'составы': 'composition',
        'компоненты': 'component',
        'q': 'q',
        'ad': 'ad',
        'ρ': 'density',
        'density': 'density',
        'kf': 'kf',
        'kt': 'kt',
        'h': 'h',  # Оставляем 'h' для measured_parameters
        'hd': 'hd',  # Добавляем для components
        'mass loss': 'mass_loss',
        'mass_loss': 'mass_loss',
        'тign': 'tign',
        'тign,°c': 'tign',  # Добавляем точное совпадение
        'tb': 'tb',
        'tau_d1': 'tau_d1',
        'τd1': 'tau_d1',
        'tau_d2': 'tau_d2',
        'τd2': 'tau_d2',
        'tau_b': 'tau_b',
        'τb': 'tau_b',
        'co2': 'co2',
        'co': 'co',
        'so2': 'so2',
        'nox': 'nox',
        'war': 'war',
        'vd': 'vd',
        'cd': 'cd',
        'nd': 'nd',
        'sd': 'sd',
        'od': 'od'
    }
    for key, value in mappings.items():
        if key in col:
            return value
    return col

def validate_data(df, expected_columns, table_name):
    messages = []
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        messages.append(f"Missing columns for {table_name}: {missing_cols}")
        return False, messages
    df = df.dropna(how='all')  # Удаляем пустые строки
    df = df.drop_duplicates()  # Удаляем дубликаты
    for col in expected_columns:
        if col in ('composition', 'component'):
            continue
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].isna().any():
                messages.append(f"Warning: Column {col} contains non-numeric values or NaNs. Filled with 0.")
                df[col] = df[col].fillna(0)
        except Exception as e:
            messages.append(f"Error in column {col}: {str(e)}")
            return False, messages
    return True, messages

def process_data_source(source_file, db_path, chunk_size=1000):
    messages = []
    try:
        if not os.path.exists(source_file):
            messages.append(f"Файл не найден: {source_file}")
            return messages

        file_ext = os.path.splitext(source_file)[1].lower()
        if file_ext not in ['.csv', '.xlsx']:
            messages.append(f"Неподдерживаемый формат файла: {file_ext}. Используйте .csv или .xlsx")
            return messages
        messages.append(f"Обработка файла: {source_file} с расширением {file_ext}")

        column_map = {
            'составы': 'composition',
            'ρ, кг/м3': 'density',
            'kf, %': 'kf',
            'kt, %': 'kt',
            'h, %': 'h',
            'mass loss, %': 'mass_loss',
            'тign, °c': 'tign',
            'тign,°c': 'tign',  # Добавляем вариант без пробела
            'tb, °c': 'tb',
            'τd1, с': 'tau_d1',
            'τd2, с': 'tau_d2',
            'τb, с': 'tau_b',
            'co2, %': 'co2',
            'co, %': 'co',
            'so2, ppm': 'so2',
            'nox, ppm': 'nox',
            'q, мдж/кг': 'q',
            'ad, %': 'ad',
            'компоненты': 'component',
            'war, %': 'war',
            'vd, %': 'vd',
            'cd, %': 'cd',
            'hd, %': 'hd',  # Исправляем для components
            'nd, %': 'nd',
            'sd, %': 'sd',
            'od, %': 'od'
        }

        measured_params_cols = ['composition', 'density', 'kf', 'kt', 'h', 'mass_loss', 'tign', 'tb', 
                               'tau_d1', 'tau_d2', 'tau_b', 'co2', 'co', 'so2', 'nox', 'q', 'ad']
        components_cols = ['component', 'war', 'ad', 'vd', 'q', 'cd', 'hd', 'nd', 'sd', 'od']

        def process_df(df, source_name):
            nonlocal messages
            # Нормализация столбцов
            df.columns = [normalize_column_name(col) for col in df.columns]
            # Удаляем дублирующиеся столбцы, оставляя первый
            df = df.loc[:, ~df.columns.duplicated()]
            df = df.rename(columns={k.lower(): v for k, v in column_map.items()})
            messages.append(f"Переименованные столбцы в {source_name}: {list(df.columns)}")

            # Проверяем, есть ли минимально необходимые столбцы
            if all(col in df.columns for col in ['composition', 'q', 'ad']) and len(set(measured_params_cols).intersection(df.columns)) >= 10:
                is_valid, validation_msgs = validate_data(df, measured_params_cols, "measured_parameters")
                messages.extend(validation_msgs)
                if is_valid:
                    insert_data(db_path, "measured_parameters", df[measured_params_cols])
                    messages.append(f"Данные успешно вставлены в measured_parameters из {source_name}")
                    # Проверка количества строк после вставки
                    conn = sqlite3.connect(db_path)
                    row_count = conn.execute(f"SELECT COUNT(*) FROM measured_parameters").fetchone()[0]
                    messages.append(f"Строк в measured_parameters после вставки: {row_count}")
                    conn.close()
                else:
                    messages.append(f"Ошибка вставки данных в measured_parameters из {source_name}")
            elif all(col in df.columns for col in ['component', 'q', 'ad']) and len(set(components_cols).intersection(df.columns)) >= 7:
                is_valid, validation_msgs = validate_data(df, components_cols, "components")
                messages.extend(validation_msgs)
                if is_valid:
                    insert_data(db_path, "components", df[components_cols])
                    messages.append(f"Данные успешно вставлены в components из {source_name}")
                    conn = sqlite3.connect(db_path)
                    row_count = conn.execute(f"SELECT COUNT(*) FROM components").fetchone()[0]
                    messages.append(f"Строк в components после вставки: {row_count}")
                    conn.close()
                else:
                    messages.append(f"Ошибка вставки данных в components из {source_name}")
            else:
                messages.append(f"Предупреждение: {source_name} не соответствует ожидаемой структуре. Требуемые столбцы: {measured_params_cols} или {components_cols}. Доступные: {list(df.columns)}")

        if file_ext == '.xlsx':
            try:
                xl = pd.ExcelFile(source_file, engine='openpyxl')
                for sheet_name in xl.sheet_names:
                    messages.append(f"Обработка листа: {sheet_name}")
                    df = pd.read_excel(source_file, sheet_name=sheet_name, engine='openpyxl')
                    messages.append(f"Обработка листа с {len(df)} строками в {sheet_name}")
                    process_df(df, f"лист {sheet_name}")
            except Exception as e:
                messages.append(f"Ошибка чтения Excel-листа: {str(e)}")
        elif file_ext == '.csv':
            try:
                for chunk in pd.read_csv(source_file, chunksize=chunk_size, encoding='utf-8'):
                    messages.append(f"Обработка части CSV с {len(chunk)} строками")
                    process_df(chunk, f"CSV {source_file}")
            except Exception as e:
                messages.append(f"Ошибка чтения CSV: {str(e)}")
        return messages
    except Exception as e:
        messages.append(f"Ошибка обработки {source_file}: {str(e)}")
        return messages

def validate_data(df, expected_columns, table_name):
    messages = []
    df.columns = [col.strip().lower() for col in df.columns]
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        messages.append(f"Отсутствуют столбцы для {table_name}: {missing_cols}. Доступные: {list(df.columns)}")
        return False, messages
    df = df.dropna(how='all')
    df = df.drop_duplicates()
    for col in expected_columns:
        if col in ('composition', 'component'):
            continue
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].isna().any():
                messages.append(f"Предупреждение в {col}: найдены нечисловые значения или NaN. Подробности: {df[col].isna().sum()} проблем. Заполнено нулями.")
                df[col] = df[col].fillna(0)
        except Exception as e:
            messages.append(f"Ошибка в столбце {col}: {str(e)}")
            return False, messages
    return True, messages

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