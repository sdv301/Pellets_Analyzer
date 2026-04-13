import pandas as pd
import os
import re
from app.database.database import insert_data, init_db

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

def prepare_data_for_display(df, table_type):
    """Подготавливает данные для отображения с русскими названиями колонок"""
    
    # Создаем копию DataFrame чтобы не менять оригинал
    df_display = df.copy()
    
    # Словарь замены заголовков
    header_replacements = {
        'composition': 'Составы',
        'density': 'Плотность, кг/м3',
        'kf': 'Ударопрочность, %',
        'kt': 'Устойчивость к колебательным нагрузкам, %', 
        'h': 'Гигроскопичность, %',
        'mass_loss': 'Потеря массы, %',
        'tign': 'Температура зажигания, °С',
        'tb': 'Температура выгорания, °C',
        'tau_d1': 'Задержка газофазного зажигания, C',
        'tau_d2': 'Задержка гетерогенного зажигания, C',
        'tau_b': 'Время горения, С',
        'co2': 'Концентрации диоксида углерода, %',
        'co': 'Концентрации монооксида углерода, %', 
        'so2': 'Концентрации оксидов серы, ppm',
        'nox': 'Концентрации оксидов азота, ppm',
        'q': 'Теплота сгорания, МДж/кг',
        'ad': 'Содержание золы на сухую массу, %',
        'component': 'Компоненты',
        'war': 'Влажность на аналитическую массу, %',
        'vd': 'Содержание летучих на сухую массу, %',
        'cd': 'Содержание углерода на сухую массу, %',
        'hd': 'Содержание водорода на сухую массу, %', 
        'nd': 'Содержание азота на сухую массу, %',
        'sd': 'Содержание серы на сухую массу, %',
        'od': 'Содержание кислорода на сухую массу, %'
    }
    
    # Переименовываем только те колонки, которые существуют в DataFrame
    existing_columns = {k: v for k, v in header_replacements.items() if k in df_display.columns}
    df_display = df_display.rename(columns=existing_columns)
    
    return df_display

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
            
            df.columns = [normalize_column_name(col) for col in df.columns]
            
            # Удаляем полностью пустые столбцы и дублирующиеся
            df = df.dropna(axis=1, how='all')
            df = df.loc[:, ~df.columns.duplicated()]
            
            messages.append(f"Успешно загружен CSV-файл: {os.path.basename(file_path)}")
            df, validation_messages = validate_data(df, expected_measured_columns, "measured_parameters")
            messages.extend(validation_messages)
            
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
            
            if not sheet_names:
                messages.append("Excel-файл не содержит листов")
                return messages, components_sheet_name, sheet_data
                
            for sheet_name in sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                
                # Пропускаем полностью пустые листы
                if df.empty or df.shape[0] == 0:
                    messages.append(f"Лист '{sheet_name}' пуст - пропускаем")
                    continue
                    
                df.columns = [normalize_column_name(col) for col in df.columns]
                
                # Удаляем полностью пустые столбцы и дублирующиеся
                df = df.dropna(axis=1, how='all')
                df = df.loc[:, ~df.columns.duplicated()]
                
                messages.append(f"Успешно загружен лист '{sheet_name}' из Excel-файла: {os.path.basename(file_path)}")
                
                # Определяем тип данных по наличию ключевых столбцов
                table_name = None
                if 'composition' in df.columns:
                    df, validation_messages = validate_data(df, expected_measured_columns, f"measured_parameters ({sheet_name})")
                    table_name = "measured_parameters"
                elif 'component' in df.columns:
                    df, validation_messages = validate_data(df, expected_components_columns, f"components ({sheet_name})")
                    table_name = "components"
                else:
                    messages.append(f"Не удалось определить тип данных для листа '{sheet_name}' - пробуем как измеренные параметры")
                    df, validation_messages = validate_data(df, expected_measured_columns, f"measured_parameters ({sheet_name})")
                    table_name = "measured_parameters"
                
                messages.extend(validation_messages)
                
                if not df.empty:
                    insert_data(db_path, table_name, df)
                    messages.append(f"Данные листа '{sheet_name}' сохранены в таблицу {table_name}")
                    
                    # Добавляем в sheet_data DataFrame (как для CSV)
                    sheet_data.append({'name': sheet_name, 'data': df})
                else:
                    messages.append(f"Лист '{sheet_name}' не содержит валидных данных после обработки")
            
            # Устанавливаем имя листа компонентов
            if len(sheet_names) > 1:
                components_sheet_name = sheet_names[1]
            else:
                components_sheet_name = 'Характеристики компонентов'
                
            if not sheet_data:
                messages.append("Еxcel-файл не содержит валидных данных для отображения")
            else:
                messages.append(f"Обработано {len(sheet_data)} листов из Excel-файла")
            return messages, components_sheet_name, sheet_data
        
        else:
            messages.append(f"Ошибка: Формат файла {file_path} не поддерживается")
            return messages, components_sheet_name, sheet_data
            
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        messages.append(f"Ошибка обработки файла {file_path}: {str(e)}")
        return messages, components_sheet_name, sheet_data