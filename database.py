import sqlite3
import pandas as pd
import json
from datetime import datetime

def init_db(db_path):
    """Инициализация базы данных и создание таблиц с индексами."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Существующие таблицы
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS measured_parameters (
        composition TEXT PRIMARY KEY,
        density REAL,
        kf REAL,
        kt REAL,
        h REAL,
        mass_loss REAL,
        tign REAL,
        tb REAL,
        tau_d1 REAL,
        tau_d2 REAL,
        tau_b REAL,
        co2 REAL,
        co REAL,
        so2 REAL,
        nox REAL,
        q REAL,
        ad REAL
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS components (
        component TEXT PRIMARY KEY,
        war REAL,
        ad REAL,
        vd REAL,
        q REAL,
        cd REAL,
        hd REAL,
        nd REAL,
        sd REAL,
        od REAL
    )
    ''')

    # НОВЫЕ ТАБЛИЦЫ ДЛЯ ML
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS ml_optimizations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        target_property TEXT NOT NULL,
        maximize BOOLEAN NOT NULL,
        optimal_composition_json TEXT NOT NULL,
        optimal_value REAL NOT NULL,
        constraints_json TEXT,
        algorithm TEXT,
        model_metrics_json TEXT,
        created_by TEXT DEFAULT 'ml_system'
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS ml_model_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        target_property TEXT NOT NULL,
        algorithm TEXT NOT NULL,
        r2_score REAL,
        mae REAL,
        cv_r2 REAL,
        feature_importance_json TEXT,
        training_data_size INTEGER,
        is_active BOOLEAN DEFAULT 1
    )
    ''')

    # Добавление индексов
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_composition ON measured_parameters(composition)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_component ON components(component)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_ml_timestamp ON ml_optimizations(timestamp)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_ml_property ON ml_optimizations(target_property)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_ml_metrics_property ON ml_model_metrics(target_property)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_ml_metrics_active ON ml_model_metrics(is_active)')

    conn.commit()
    conn.close()

def insert_data(db_path, table, data):
    """Пакетная вставка данных."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    if table == "measured_parameters":
        expected_columns = ['composition', 'density', 'kf', 'kt', 'h', 'mass_loss', 
                          'tign', 'tb', 'tau_d1', 'tau_d2', 'tau_b', 
                          'co2', 'co', 'so2', 'nox', 'q', 'ad']
        
        data = data[expected_columns] if all(col in data.columns for col in expected_columns) else data
        
        query = '''
        INSERT OR REPLACE INTO measured_parameters VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        records = [tuple(row) for _, row in data.iterrows()]
        cursor.executemany(query, records)
        
    elif table == "components":
        expected_columns = ['component', 'war', 'ad', 'vd', 'q', 'cd', 'hd', 'nd', 'sd', 'od']
        
        data = data[expected_columns] if all(col in data.columns for col in expected_columns) else data
        
        query = '''
        INSERT OR REPLACE INTO components VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        records = [tuple(row) for _, row in data.iterrows()]
        cursor.executemany(query, records)

    conn.commit()
    conn.close()

def insert_ml_optimization(db_path, optimization_data):
    """Сохраняет результат ML оптимизации в базу"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    query = '''
    INSERT INTO ml_optimizations 
    (target_property, maximize, optimal_composition_json, optimal_value, constraints_json, algorithm, model_metrics_json)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    '''
    
    cursor.execute(query, (
        optimization_data['target_property'],
        optimization_data['maximize'],
        json.dumps(optimization_data['optimal_composition']),
        optimization_data['optimal_value'],
        json.dumps(optimization_data.get('constraints', {})),
        optimization_data.get('algorithm', 'gradient_boosting'),
        json.dumps(optimization_data.get('model_metrics', {}))
    ))
    
    conn.commit()
    conn.close()

def insert_ml_model_metrics(db_path, metrics_data):
    """Сохраняет метрики ML модели в базу"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Деактивируем старые метрики для этого свойства
    cursor.execute(
        'UPDATE ml_model_metrics SET is_active = 0 WHERE target_property = ? AND algorithm = ?',
        (metrics_data['target_property'], metrics_data['algorithm'])
    )
    
    query = '''
    INSERT INTO ml_model_metrics 
    (target_property, algorithm, r2_score, mae, cv_r2, feature_importance_json, training_data_size)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    '''
    
    cursor.execute(query, (
        metrics_data['target_property'],
        metrics_data['algorithm'],
        metrics_data.get('r2_score'),
        metrics_data.get('mae'),
        metrics_data.get('cv_r2'),
        json.dumps(metrics_data.get('feature_importance', {})),
        metrics_data.get('training_data_size', 0)
    ))
    
    conn.commit()
    conn.close()

def get_ml_optimizations(db_path, limit=50):
    """Получает историю ML оптимизаций"""
    conn = sqlite3.connect(db_path)
    query = '''
    SELECT * FROM ml_optimizations 
    ORDER BY timestamp DESC 
    LIMIT ?
    '''
    df = pd.read_sql_query(query, conn, params=(limit,))
    conn.close()
    
    # Парсим JSON поля
    if not df.empty:
        df['optimal_composition'] = df['optimal_composition_json'].apply(json.loads)
        df['constraints'] = df['constraints_json'].apply(lambda x: json.loads(x) if x else {})
        df['model_metrics'] = df['model_metrics_json'].apply(lambda x: json.loads(x) if x else {})
    
    return df

def get_active_ml_models(db_path):
    """Получает активные ML модели"""
    conn = sqlite3.connect(db_path)
    query = '''
    SELECT * FROM ml_model_metrics 
    WHERE is_active = 1 
    ORDER BY timestamp DESC
    '''
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Парсим JSON поля
    if not df.empty:
        df['feature_importance'] = df['feature_importance_json'].apply(json.loads)
    
    return df

def add_ml_optimization_to_training_data(db_path, optimization_data, actual_properties=None):
    """Добавляет успешную ML оптимизацию в тренировочные данные"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Создаем текстовое представление состава
    composition_text = ", ".join([f"{v}% {k}" for k, v in optimization_data['optimal_composition'].items()])
    
    # Если есть реальные измерения, используем их, иначе оставляем NULL
    if actual_properties:
        query = '''
        INSERT OR REPLACE INTO measured_parameters 
        (composition, density, kf, kt, h, mass_loss, tign, tb, tau_d1, tau_d2, tau_b, co2, co, so2, nox, q, ad)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        cursor.execute(query, (
            composition_text,
            actual_properties.get('density'),
            actual_properties.get('kf'),
            actual_properties.get('kt'),
            actual_properties.get('h'),
            actual_properties.get('mass_loss'),
            actual_properties.get('tign'),
            actual_properties.get('tb'),
            actual_properties.get('tau_d1'),
            actual_properties.get('tau_d2'),
            actual_properties.get('tau_b'),
            actual_properties.get('co2'),
            actual_properties.get('co'),
            actual_properties.get('so2'),
            actual_properties.get('nox'),
            actual_properties.get('q'),
            actual_properties.get('ad')
        ))
    else:
        # Добавляем только состав, свойства будут NULL (для предсказания)
        query = '''
        INSERT OR IGNORE INTO measured_parameters (composition) VALUES (?)
        '''
        cursor.execute(query, (composition_text,))
    
    conn.commit()
    conn.close()


def query_db(db_path, table, query="SELECT * FROM {}", params=None):
    """Запрос данных из таблицы."""
    conn = sqlite3.connect(db_path)
    if params:
        df = pd.read_sql_query(query.format(table), conn, params=params)
    else:
        df = pd.read_sql_query(query.format(table), conn)
    conn.close()
    return df