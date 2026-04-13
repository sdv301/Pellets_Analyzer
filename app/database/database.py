import sqlite3
import pandas as pd
import json
import os
import logging
from datetime import datetime
from typing import Optional, Dict, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# ============================================================
# КОНФИГУРАЦИЯ УДАЛЁННОГО ПОДКЛЮЧЕНИЯ
# Загружается из переменных окружения или из db_config.json
# ============================================================

def _load_db_config() -> Dict[str, Any]:
    """Загружает конфигурацию подключения к БД."""
    config = {
        'use_remote': False,
        'db_path': 'pellets_data.db',
        # SSH-туннель
        'ssh_host': os.environ.get('DB_SSH_HOST', ''),
        'ssh_port': int(os.environ.get('DB_SSH_PORT', '22')),
        'ssh_user': os.environ.get('DB_SSH_USER', ''),
        'ssh_password': os.environ.get('DB_SSH_PASSWORD', ''),
        'ssh_key_path': os.environ.get('DB_SSH_KEY_PATH', ''),
        'ssh_key_password': os.environ.get('DB_SSH_KEY_PASSWORD', ''),
        # Удалённая БД
        'remote_db_path': os.environ.get('DB_REMOTE_PATH', ''),
        # Синхронизация
        'auto_sync': True,
        'sync_interval': 300,  # 5 минут
    }
    
    # Загрузка из файла конфигурации (если существует)
    config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'db_config.json')
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                file_config = json.load(f)
                config.update(file_config)
        except Exception as e:
            logger.warning(f"Не удалось загрузить config из {config_file}: {e}")
    
    return config

# Глобальная конфигурация
_db_config = _load_db_config()
_remote_sync = None
_tunnel = None


class DatabaseConnection:
    """
    Менеджер подключений к базе данных с поддержкой SSH-туннеля.
    
    Поддерживает:
    - Локальное подключение к SQLite
    - Удалённое подключение через SSH-туннель (с синхронизацией файла)
    - Автоматическую синхронизацию при изменении данных
    
    Пример использования:
        # Локальная БД
        with DatabaseConnection() as conn:
            df = conn.query("SELECT * FROM components")
        
        # Удалённая БД через SSH
        with DatabaseConnection(use_remote=True, ssh_host='server.com', ...) as conn:
            df = conn.query("SELECT * FROM components")
    """
    
    def __init__(
        self,
        db_path: Optional[str] = None,
        use_remote: Optional[bool] = None,
        ssh_host: Optional[str] = None,
        ssh_port: Optional[int] = None,
        ssh_user: Optional[str] = None,
        ssh_password: Optional[str] = None,
        ssh_key_path: Optional[str] = None,
        remote_db_path: Optional[str] = None,
        auto_sync: Optional[bool] = None
    ):
        
        self.config = _db_config.copy()
        
        # Переопределение из параметров
        if db_path:
            self.config['db_path'] = db_path
        if use_remote is not None:
            self.config['use_remote'] = use_remote
        if ssh_host:
            self.config['ssh_host'] = ssh_host
        if ssh_port:
            self.config['ssh_port'] = ssh_port
        if ssh_user:
            self.config['ssh_user'] = ssh_user
        if ssh_password:
            self.config['ssh_password'] = ssh_password
        if ssh_key_path:
            self.config['ssh_key_path'] = ssh_key_path
        if remote_db_path:
            self.config['remote_db_path'] = remote_db_path
        if auto_sync is not None:
            self.config['auto_sync'] = auto_sync
        
        self._local_db_path = self.config['db_path']
        self._sync = None
        self._modified = False
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
    
    def connect(self) -> Dict[str, Any]:
        """
        Устанавливает подключение к БД.
        
        Returns:
            Dict с результатом подключения
        """
        if self.config['use_remote'] and self.config['remote_db_path']:
            return self._connect_remote()
        else:
            return self._connect_local()
    
    def _connect_local(self) -> Dict[str, Any]:
        """Подключение к локальной БД."""
        if not os.path.exists(self._local_db_path):
            logger.info(f"Создание новой БД: {self._local_db_path}")
            init_db(self._local_db_path)
        
        return {
            'success': True,
            'mode': 'local',
            'db_path': self._local_db_path,
            'message': f'Подключено к локальной БД: {self._local_db_path}'
        }
    
    def _connect_remote(self) -> Dict[str, Any]:
        """Подключение к удалённой БД через SSH с синхронизацией."""
        try:
            from app.utils.ssh_tunnel import SQLiteRemoteSync
            
            self._sync = SQLiteRemoteSync(
                ssh_host=self.config['ssh_host'],
                ssh_port=self.config['ssh_port'],
                ssh_user=self.config['ssh_user'],
                ssh_password=self.config['ssh_password'] or None,
                ssh_key_path=self.config['ssh_key_path'] or None,
                ssh_key_password=self.config['ssh_key_password'] or None,
                remote_db_path=self.config['remote_db_path'],
                local_db_path=self._local_db_path
            )
            
            logger.info(f"Синхронизация удалённой БД: {self.config['remote_db_path']}")
            if not self._sync.download():
                return {
                    'success': False,
                    'error': 'Не удалось скачать удалённую БД. Проверьте SSH-подключение.',
                    'mode': 'fallback_local'
                }
            
            return {
                'success': True,
                'mode': 'remote',
                'db_path': self._local_db_path,
                'remote_path': self.config['remote_db_path'],
                'message': f'Синхронизирована удалённая БД: {self.config["remote_db_path"]}'
            }
            
        except ImportError:
            return {
                'success': False,
                'error': 'Не установлены зависимости для SSH. Установите: pip install paramiko sshtunnel',
                'mode': 'error'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Ошибка подключения к удалённой БД: {str(e)}',
                'mode': 'error'
            }
    
    def close(self):
        """Закрывает подключение и синхронизирует изменения."""
        if self._sync and self.config['auto_sync'] and self._modified:
            logger.info("Синхронизация изменений с удалённой БД...")
            self._sync.upload()
        
        if self._sync:
            self._sync._disconnect_ssh()
            self._sync = None
    
    def get_connection(self) -> sqlite3.Connection:
        """Возвращает sqlite3.Connection для прямого использования."""
        return sqlite3.connect(self._local_db_path)
    
    def mark_modified(self):
        """Отмечает, что данные были изменены (нужна синхронизация)."""
        self._modified = True
    
    def query(self, sql: str, params: tuple = None) -> pd.DataFrame:
        """Выполняет SELECT-запрос."""
        conn = sqlite3.connect(self._local_db_path)
        try:
            if params:
                df = pd.read_sql_query(sql, conn, params=params)
            else:
                df = pd.read_sql_query(sql, conn)
            return df
        finally:
            conn.close()
    
    def execute(self, sql: str, params: tuple = None):
        """Выполняет INSERT/UPDATE/DELETE-запрос."""
        conn = sqlite3.connect(self._local_db_path)
        try:
            cursor = conn.cursor()
            if params:
                cursor.execute(sql, params)
            else:
                cursor.execute(sql)
            conn.commit()
            self.mark_modified()
        finally:
            conn.close()
    
    def executemany(self, sql: str, rows: list):
        """Выполняет массовую вставку/обновление."""
        conn = sqlite3.connect(self._local_db_path)
        try:
            cursor = conn.cursor()
            cursor.executemany(sql, rows)
            conn.commit()
            self.mark_modified()
        finally:
            conn.close()


def get_db_connection(**kwargs) -> DatabaseConnection:
    """
    Фабрика подключений к БД.
    
    Пример:
        with get_db_connection() as conn:
            df = conn.query("SELECT * FROM components")
    """
    return DatabaseConnection(**kwargs)


# ============================================================
# СТАРЫЕ ФУНКЦИИ (обратная совместимость)
# Теперь используют DatabaseConnection
# ============================================================

def _get_db_path(db_path: str = None) -> str:
    """Возвращает путь к БД с учётом конфигурации."""
    if db_path:
        return db_path
    return _db_config.get('db_path', 'pellets_data.db')

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
        od REAL,
        ro REAL,
        cost_raw REAL,
        cost_crush REAL,
        cost_granule REAL
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
    
    # Запуск миграции для существующих баз
    try:
        check_and_migrate_db(db_path)
    except Exception as e:
        print(f"⚠️ Ошибка при миграции БД: {e}")

def check_and_migrate_db(db_path):
    """Проверяет и добавляет недостающие колонки в существующие таблицы."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 1. Проверка таблицы components
    cursor.execute("PRAGMA table_info(components)")
    columns = [row[1] for row in cursor.fetchall()]
    
    new_columns = {
        'ro': 'REAL',
        'cost_raw': 'REAL',
        'cost_crush': 'REAL',
        'cost_granule': 'REAL'
    }
    
    for col, dtype in new_columns.items():
        if col not in columns:
            print(f"🔧 Миграция: Добавление колонки {col} в таблицу components")
            cursor.execute(f"ALTER TABLE components ADD COLUMN {col} {dtype}")
            
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
        
        # Добавляем недостающие колонки как None
        for col in expected_columns:
            if col not in data.columns:
                data[col] = None
        
        data = data[expected_columns]
        
        query = '''
        INSERT OR REPLACE INTO measured_parameters VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        records = [tuple(row) for _, row in data.iterrows()]
        cursor.executemany(query, records)
        
    elif table == "components":
        expected_columns = ['component', 'war', 'ad', 'vd', 'q', 'cd', 'hd', 'nd', 'sd', 'od', 
                           'ro', 'cost_raw', 'cost_crush', 'cost_granule']
        
        # Если в DataFrame нет новых колонок, заполняем их None/NaN
        for col in ['ro', 'cost_raw', 'cost_crush', 'cost_granule']:
            if col not in data.columns:
                data[col] = None
        
        data = data[expected_columns] if all(col in data.columns for col in expected_columns) else data
        
        query = '''
        INSERT OR REPLACE INTO components VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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


def update_ml_optimization_with_actual(db_path, optimization_id, actual_properties):
    """Обновляет запись ML оптимизации реальными измерениями и добавляет данные в тренировочный набор."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Получаем оптимизацию
        cursor.execute('SELECT * FROM ml_optimizations WHERE id = ?', (optimization_id,))
        row = cursor.fetchone()
        if not row:
            conn.close()
            return False
        
        # Получаем оптимальный состав из JSON
        optimal_composition = json.loads(row[5])  # optimal_composition_json
        composition_text = ", ".join([f"{v}% {k}" for k, v in optimal_composition.items()])
        
        # Добавляем реальные измерения в measured_parameters
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
        
        conn.commit()
        conn.close()
        return True
        
    except Exception as e:
        print(f"Ошибка обновления оптимизации: {e}")
        return False


ALLOWED_TABLES = {'measured_parameters', 'components', 'ml_optimizations', 'ml_model_metrics'}

def query_db(db_path, table, query="SELECT * FROM {}", params=None):
    """Запрос данных из таблицы."""
    if table not in ALLOWED_TABLES:
        raise ValueError(f"Недопустимое имя таблицы: {table}")
    conn = sqlite3.connect(db_path)
    if params:
        df = pd.read_sql_query(query.format(table), conn, params=params)
    else:
        df = pd.read_sql_query(query.format(table), conn)
    conn.close()
    return df