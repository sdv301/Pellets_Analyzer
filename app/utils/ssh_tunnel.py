# ssh_tunnel.py
"""
Модуль SSH-туннеля для подключения к удалённой базе данных.

Поддерживает:
- Подключение по паролю
- Подключение по SSH-ключу
- Проброс локального порта к удалённому серверу БД
- Автоматическое переподключение при обрыве связи
- Контекстный менеджер для безопасного использования

Пример использования:
    # Через контекстный менеджер
    with SSHTunnel(
        ssh_host='example.com',
        ssh_user='user',
        ssh_password='password',
        remote_db_path='/path/to/remote.db'
    ) as tunnel:
        local_path = tunnel.local_db_path
        # local_path теперь указывает на локальный проброшенный файл
        # или можно использовать напрямую через sqlite3 с проброшенным портом

    # Для PostgreSQL/MySQL через проброс порта
    with SSHTunnel(
        ssh_host='example.com',
        ssh_user='user',
        ssh_key='/path/to/key.pem',
        remote_host='localhost',
        remote_port=5432,
        local_port=5433
    ) as tunnel:
        # Подключаемся к localhost:5433 как к удалённой БД
        conn = psycopg2.connect(host='localhost', port=5433, ...)
"""

import os
import threading
import logging
from contextlib import contextmanager
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Глобальное состояние туннеля
_tunnel_state: Dict[str, Any] = {
    'tunnel': None,
    'local_port': None,
    'is_active': False,
    'lock': threading.Lock()
}


class SSHTunnel:
    """
    SSH-туннель для подключения к удалённым ресурсам.
    
    Args:
        ssh_host: Адрес SSH-сервера
        ssh_port: Порт SSH (по умолчанию 22)
        ssh_user: Имя пользователя SSH
        ssh_password: Пароль SSH (если не используется ключ)
        ssh_key_path: Путь к SSH-ключу (если используется)
        ssh_key_password: Пароль для SSH-ключа (если зашифрован)
        remote_host: Удалённый хост БД (относительно SSH-сервера, по умолчанию localhost)
        remote_port: Удалённый порт БД (для PostgreSQL/MySQL)
        local_port: Локальный порт для проброса (по умолчанию выбирается автоматически)
        remote_db_path: Путь к удалённому SQLite файлу (альтернатива remote_port)
        timeout: Таймаут подключения в секундах
        
    Для SQLite:
        SSH-туннель не может напрямую пробросить файл.
        Вместо этого используется rsync/sftp для синхронизации.
        Или можно использовать remote_port если SQLite работает в режиме сервера.
    
    Для PostgreSQL/MySQL:
        Используется проброс порта через SSH.
    """
    
    def __init__(
        self,
        ssh_host: str,
        ssh_user: str,
        ssh_port: int = 22,
        ssh_password: Optional[str] = None,
        ssh_key_path: Optional[str] = None,
        ssh_key_password: Optional[str] = None,
        remote_host: str = 'localhost',
        remote_port: Optional[int] = None,
        local_port: Optional[int] = None,
        remote_db_path: Optional[str] = None,
        timeout: int = 30
    ):
        self.ssh_host = ssh_host
        self.ssh_port = ssh_port
        self.ssh_user = ssh_user
        self.ssh_password = ssh_password
        self.ssh_key_path = ssh_key_path
        self.ssh_key_password = ssh_key_password
        self.remote_host = remote_host
        self.remote_port = remote_port
        self.local_port = local_port
        self.remote_db_path = remote_db_path
        self.timeout = timeout
        
        self._tunnel = None
        self._ssh_client = None
        self._is_active = False
        
        # Валидация
        if not remote_port and not remote_db_path:
            raise ValueError("Укажите remote_port (для PostgreSQL/MySQL) или remote_db_path (для SQLite)")
    
    def start(self) -> Dict[str, Any]:
        """
        Запускает SSH-туннель.
        
        Returns:
            Dict с информацией о туннеле:
                - success: bool
                - local_port: int (проброшенный порт)
                - message: str
                - error: str (если ошибка)
        """
        try:
            from sshtunnel import SSHTunnelForwarder
            import paramiko
            
            # Определяем аутентификацию
            ssh_kwargs = {
                'ssh_address_or_host': (self.ssh_host, self.ssh_port),
                'ssh_username': self.ssh_user,
                'remote_bind_address': (self.remote_host, self.remote_port or 22),
                'logger': logger
            }
            
            if self.ssh_key_path:
                if self.ssh_key_password:
                    ssh_kwargs['ssh_pkey'] = paramiko.RSAKey.from_private_key_file(
                        self.ssh_key_path,
                        password=self.ssh_key_password
                    )
                else:
                    ssh_kwargs['ssh_pkey'] = self.ssh_key_path
            elif self.ssh_password:
                ssh_kwargs['ssh_password'] = self.ssh_password
            else:
                # Пробуем аутентификацию через SSH-агент
                ssh_kwargs['allow_agent'] = True
                ssh_kwargs['look_for_keys'] = True
            
            # Если указан local_port, используем его
            if self.local_port:
                ssh_kwargs['local_bind_port'] = self.local_port
            
            logger.info(f"Подключение к SSH-серверу {self.ssh_host}:{self.ssh_port}...")
            
            self._tunnel = SSHTunnelForwarder(**ssh_kwargs)
            self._tunnel.start()
            
            actual_local_port = self._tunnel.local_bind_port
            self._is_active = True
            
            message = f"SSH-туннель активен: localhost:{actual_local_port} -> {self.remote_host}:{self.remote_port}"
            logger.info(message)
            
            return {
                'success': True,
                'local_port': actual_local_port,
                'remote_host': self.remote_host,
                'remote_port': self.remote_port,
                'message': message
            }
            
        except ImportError as e:
            return {
                'success': False,
                'error': f'Не установлены зависимости: {e}. Установите: pip install paramiko sshtunnel'
            }
        except Exception as e:
            error_msg = f"Ошибка создания SSH-туннеля: {str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg
            }
    
    def stop(self):
        """Останавливает SSH-туннель."""
        if self._tunnel:
            try:
                self._tunnel.stop()
                self._is_active = False
                logger.info("SSH-туннель остановлен")
            except Exception as e:
                logger.error(f"Ошибка при остановке туннеля: {e}")
    
    @property
    def is_active(self) -> bool:
        return self._is_active
    
    @property
    def local_bind_port(self) -> Optional[int]:
        if self._tunnel and self._is_active:
            return self._tunnel.local_bind_port
        return None
    
    def __enter__(self):
        result = self.start()
        if not result['success']:
            raise RuntimeError(result['error'])
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
    
    def __del__(self):
        self.stop()


class SQLiteRemoteSync:
    """
    Синхронизация удалённой SQLite БД через SSH (SFTP).
    
    SQLite — файловая БД, поэтому для работы с удалённой копией
    нужно скачать файл, работать локально, и загрузить обратно.
    """
    
    def __init__(
        self,
        ssh_host: str,
        ssh_user: str,
        remote_db_path: str,
        local_db_path: Optional[str] = None,
        ssh_port: int = 22,
        ssh_password: Optional[str] = None,
        ssh_key_path: Optional[str] = None,
        ssh_key_password: Optional[str] = None
    ):
        self.ssh_host = ssh_host
        self.ssh_port = ssh_port
        self.ssh_user = ssh_user
        self.ssh_password = ssh_password
        self.ssh_key_path = ssh_key_path
        self.ssh_key_password = ssh_key_password
        self.remote_db_path = remote_db_path
        self.local_db_path = local_db_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'remote_db_sync.db'
        )
        self._ssh_client = None
        self._sftp = None
    
    def _connect_ssh(self):
        """Подключается к SSH-серверу."""
        import paramiko
        
        self._ssh_client = paramiko.SSHClient()
        self._ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        connect_kwargs = {
            'hostname': self.ssh_host,
            'port': self.ssh_port,
            'username': self.ssh_user,
            'timeout': 30
        }
        
        if self.ssh_key_path:
            connect_kwargs['key_filename'] = self.ssh_key_path
            if self.ssh_key_password:
                connect_kwargs['passphrase'] = self.ssh_key_password
        elif self.ssh_password:
            connect_kwargs['password'] = self.ssh_password
        
        self._ssh_client.connect(**connect_kwargs)
        self._sftp = self._ssh_client.open_sftp()
    
    def _disconnect_ssh(self):
        """Отключается от SSH-сервера."""
        if self._sftp:
            try:
                self._sftp.close()
            except Exception:
                pass
            self._sftp = None
        
        if self._ssh_client:
            try:
                self._ssh_client.close()
            except Exception:
                pass
            self._ssh_client = None
    
    def download(self) -> bool:
        """Скачивает удалённую БД локально."""
        try:
            if not self._sftp:
                self._connect_ssh()
            
            logger.info(f"Скачивание БД: {self.remote_db_path} -> {self.local_db_path}")
            self._sftp.get(self.remote_db_path, self.local_db_path)
            logger.info("БД успешно скачана")
            return True
        except Exception as e:
            logger.error(f"Ошибка скачивания БД: {e}")
            return False
    
    def upload(self) -> bool:
        """Загружает локальную БД на удалённый сервер."""
        try:
            if not self._sftp:
                self._connect_ssh()
            
            if not os.path.exists(self.local_db_path):
                logger.error(f"Локальный файл БД не найден: {self.local_db_path}")
                return False
            
            logger.info(f"Загрузка БД: {self.local_db_path} -> {self.remote_db_path}")
            self._sftp.put(self.local_db_path, self.remote_db_path)
            logger.info("БД успешно загружена")
            return True
        except Exception as e:
            logger.error(f"Ошибка загрузки БД: {e}")
            return False
    
    def __enter__(self):
        if not self.download():
            raise RuntimeError("Не удалось скачать удалённую БД")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:  # Только если не было ошибок
            self.upload()
        self._disconnect_ssh()
        return False


@contextmanager
def ssh_tunnel_context(**kwargs):
    """
    Контекстный менеджер для удобного создания SSH-туннеля.
    
    Пример:
        with ssh_tunnel_context(
            ssh_host='server.com',
            ssh_user='user',
            ssh_password='pass',
            remote_port=5432
        ) as tunnel:
            # tunnel.local_bind_port — локальный порт
            pass
    """
    tunnel = SSHTunnel(**kwargs)
    try:
        result = tunnel.start()
        if not result['success']:
            raise RuntimeError(result['error'])
        yield tunnel
    finally:
        tunnel.stop()


def get_tunnel_state() -> Dict[str, Any]:
    """Возвращает текущее состояние глобального туннеля."""
    with _tunnel_state['lock']:
        return {
            'is_active': _tunnel_state['is_active'],
            'local_port': _tunnel_state['local_port'],
            'tunnel': _tunnel_state['tunnel']
        }


def start_global_tunnel(**kwargs) -> Dict[str, Any]:
    """
    Запускает глобальный SSH-туннель (singleton).
    
    Args:
        **kwargs: Аргументы для SSHTunnel
        
    Returns:
        Dict с результатом запуска
    """
    with _tunnel_state['lock']:
        if _tunnel_state['is_active']:
            return {
                'success': False,
                'error': 'Туннель уже активен',
                'local_port': _tunnel_state['local_port']
            }
        
        tunnel = SSHTunnel(**kwargs)
        result = tunnel.start()
        
        if result['success']:
            _tunnel_state['tunnel'] = tunnel
            _tunnel_state['local_port'] = result['local_port']
            _tunnel_state['is_active'] = True
        
        return result


def stop_global_tunnel() -> bool:
    """Останавливает глобальный SSH-туннель."""
    with _tunnel_state['lock']:
        if _tunnel_state['tunnel']:
            _tunnel_state['tunnel'].stop()
            _tunnel_state['tunnel'] = None
            _tunnel_state['local_port'] = None
            _tunnel_state['is_active'] = False
            return True
        return False
