import sqlite3
import pandas as pd

def init_db(db_path):
    """Инициализация базы данных и создание таблиц с индексами."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

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

    # Добавление индексов
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_composition ON measured_parameters(composition)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_component ON components(component)')

    conn.commit()
    conn.close()

def insert_data(db_path, table, data):
    """Пакетная вставка данных."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    if table == "measured_parameters":
        query = '''
        INSERT OR REPLACE INTO measured_parameters VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        records = [tuple(row) for _, row in data.iterrows()]
        cursor.executemany(query, records)
    elif table == "components":
        query = '''
        INSERT OR REPLACE INTO components VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        records = [tuple(row) for _, row in data.iterrows()]
        cursor.executemany(query, records)

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