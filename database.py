import sqlite3
import pandas as pd

def init_db(db_path):
    """Инициализация базы данных и создание таблиц."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Таблица measured_parameters
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

    # Таблица components
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

    conn.commit()
    conn.close()

def insert_data(db_path, table, data):
    """Вставка данных в указанную таблицу."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    if table == "measured_parameters":
        for _, row in data.iterrows():
            cursor.execute('''
            INSERT OR REPLACE INTO measured_parameters VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (row['composition'], row['density'], row['kf'], row['kt'], row['h'], row['mass_loss'], 
                  row['tign'], row['tb'], row['tau_d1'], row['tau_d2'], row['tau_b'], row['co2'], 
                  row['co'], row['so2'], row['nox'], row['q'], row['ad']))

    elif table == "components":
        for _, row in data.iterrows():
            cursor.execute('''
            INSERT OR REPLACE INTO components VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (row['component'], row['war'], row['ad'], row['vd'], row['q'], 
                  row['cd'], row['hd'], row['nd'], row['sd'], row['od']))

    conn.commit()
    conn.close()

def query_db(db_path, table, condition=None):
    """Запрос данных из таблицы."""
    conn = sqlite3.connect(db_path)
    query = f"SELECT * FROM {table}"
    if condition:
        query += f" WHERE {condition}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df