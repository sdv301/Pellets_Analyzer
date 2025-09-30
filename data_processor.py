import pandas as pd
import os
from database import insert_data

def process_data_source(source_file, db_path):
    """Обработка данных из файла и вставка в базу."""
    messages = []
    try:
        if not os.path.exists(source_file):
            messages.append(f"File not found: {source_file}")
            return messages
        
        # Определение типа файла
        if source_file.endswith('.xlsx'):
            xl = pd.ExcelFile(source_file)
            for sheet_name in xl.sheet_names:
                df = pd.read_excel(source_file, sheet_name=sheet_name)
                messages.append(f"Processing sheet: {sheet_name}")
                messages.append(f"Columns in sheet: {list(df.columns)}")

                # Очистка и исправление имен столбцов
                df.columns = [col.strip().replace(', ', ', %').replace(',', ', %') if col.endswith(',') or col.endswith(', ') else col for col in df.columns]
                df = df.loc[:, ~df.columns.str.contains('Unnamed|\\.1')]

                # Словарь для переименования столбцов
                column_map = {
                    'Составы': 'composition',
                    'ρ, кг/м3': 'density',
                    'Kf, %': 'kf',
                    'Kt, %': 'kt',
                    'H, %': 'h',
                    'Mass loss, %': 'mass_loss',
                    'Тign, °С': 'tign',
                    'Tb, °C': 'tb',
                    'τd1, с': 'tau_d1',
                    'τd2, с': 'tau_d2',
                    'τb, с': 'tau_b',
                    'CO2, %': 'co2',
                    'CO, %': 'co',
                    'SO2, ppm': 'so2',
                    'NOx, ppm': 'nox',
                    'Q, МДж/кг': 'q',
                    'Ad, %': 'ad',
                    'Компоненты': 'component',
                    'War, %': 'war',
                    'Vd, %': 'vd',
                    'Cd, %': 'cd',
                    'Hd, %': 'hd',
                    'Nd, %': 'nd',
                    'Sd, %': 'sd',
                    'Od, %': 'od'
                }

                # Переименование столбцов
                df = df.rename(columns=column_map)

                # Проверка структуры и вставка данных
                if 'composition' in df.columns:
                    expected_columns = ['composition', 'density', 'kf', 'kt', 'h', 'mass_loss', 'tign', 'tb', 'tau_d1', 'tau_d2', 'tau_b', 'co2', 'co', 'so2', 'nox', 'q', 'ad']
                    missing_cols = [col for col in expected_columns if col not in df.columns]
                    if missing_cols:
                        messages.append(f"Warning: Sheet {sheet_name} missing columns for measured_parameters: {missing_cols}")
                    else:
                        insert_data(db_path, "measured_parameters", df[expected_columns])
                        messages.append(f"Successfully inserted data into measured_parameters from {sheet_name}")
                elif 'component' in df.columns:
                    expected_columns = ['component', 'war', 'ad', 'vd', 'q', 'cd', 'hd', 'nd', 'sd', 'od']
                    missing_cols = [col for col in expected_columns if col not in df.columns]
                    if missing_cols:
                        messages.append(f"Warning: Sheet {sheet_name} missing columns for components: {missing_cols}")
                    else:
                        insert_data(db_path, "components", df[expected_columns])
                        messages.append(f"Successfully inserted data into components from {sheet_name}")
                else:
                    messages.append(f"Warning: Sheet {sheet_name} does not match expected structure.")

        elif source_file.endswith('.csv'):
            df = pd.read_csv(source_file)
            messages.append(f"Columns in CSV: {list(df.columns)}")
            df.columns = [col.strip().replace(', ', ', %').replace(',', ', %') if col.endswith(',') or col.endswith(', ') else col for col in df.columns]
            df = df.loc[:, ~df.columns.str.contains('Unnamed|\\.1')]
            df = df.rename(columns=column_map)
            if 'composition' in df.columns:
                expected_columns = ['composition', 'density', 'kf', 'kt', 'h', 'mass_loss', 'tign', 'tb', 'tau_d1', 'tau_d2', 'tau_b', 'co2', 'co', 'so2', 'nox', 'q', 'ad']
                missing_cols = [col for col in expected_columns if col not in df.columns]
                if missing_cols:
                    messages.append(f"Warning: CSV {source_file} missing columns for measured_parameters: {missing_cols}")
                else:
                    insert_data(db_path, "measured_parameters", df[expected_columns])
                    messages.append(f"Successfully inserted data into measured_parameters from {source_file}")
            elif 'component' in df.columns:
                expected_columns = ['component', 'war', 'ad', 'vd', 'q', 'cd', 'hd', 'nd', 'sd', 'od']
                missing_cols = [col for col in expected_columns if col not in df.columns]
                if missing_cols:
                    messages.append(f"Warning: CSV {source_file} missing columns for components: {missing_cols}")
                else:
                    insert_data(db_path, "components", df[expected_columns])
                    messages.append(f"Successfully inserted data into components from {source_file}")
            else:
                messages.append(f"Warning: CSV {source_file} does not match expected structure.")
        else:
            messages.append(f"Unsupported file format: {source_file}")
        return messages
    except Exception as e:
        messages.append(f"Error processing {source_file}: {str(e)}")
        return messages

def load_sample_data():
    """Тестовые данные для отладки."""
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