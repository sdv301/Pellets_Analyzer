# Pellets Analyzer — Деплой

## Архитектура

- **Web**: Flask + Gunicorn (порт 5000)
- **Reverse Proxy**: Nginx (порт 80)
- **База данных**: SQLite (Docker volume)
- **Оркестрация**: Docker Compose

## Файлы деплоя

| Файл | Назначение |
|------|-----------|
| `docker-compose.prod.yml` | Production compose (web + nginx) |
| `nginx.prod.conf` | Nginx конфиг (HTTP, rate limiting, security headers) |
| `.env.production` | Шаблон переменных окружения |
| `setup-server.sh` | Начальная настройка сервера (Docker, UFW, Fail2Ban) |
| `deploy.sh` | Деплой на сервере (запуск контейнеров) |
| `full-deploy.sh` | Полный деплой с локальной машины (настройка + код) |
| `quick-deploy.sh` | Быстрый деплой только кода |
| `backup.sh` | Бэкап БД, загрузок, конфигов |

## Быстрый старт

### 1. Первый деплой

```bash
# С локальной машины (Windows — через Git Bash или WSL):
bash full-deploy.sh
```

Или вручную:

```bash
# 1. Настройка сервера
ssh root@<SERVER_IP>
# На сервере:
bash /tmp/setup-server.sh

# 2. Копирование файлов
rsync -avz --exclude='.git' --exclude='__pycache__' \
    --exclude='.env' --exclude='sessions/' --exclude='data/' \
    ./ root@<SERVER_IP>:/opt/pellets-analyzer/

# 3. Деплой
ssh root@<SERVER_IP>
cd /opt/pellets-analyzer
bash deploy.sh
```

### 2. Обновление кода

```bash
bash quick-deploy.sh
```

### 3. Бэкап

```bash
# На сервере:
bash backup.sh

# Автоматически (cron):
# 0 3 * * * /opt/pellets-analyzer/backup.sh
```

## Конфигурация

### .env (на сервере)

```bash
SECRET_KEY=<сгенерированный ключ>
FLASK_ENV=production
DATABASE_URL=sqlite:////app/data/pellets_data.db
MAIL_SERVER=smtp.gmail.com
MAIL_PORT=587
MAIL_USE_TLS=true
MAIL_USERNAME=your-email@gmail.com
MAIL_PASSWORD=your-app-password
```

### mail_config.json

```json
{
    "MAIL_SERVER": "smtp.gmail.com",
    "MAIL_PORT": 587,
    "MAIL_USE_TLS": true,
    "MAIL_USERNAME": "your-email@gmail.com",
    "MAIL_PASSWORD": "app-password"
}
```

## Полезные команды

```bash
# Логи приложения
docker-compose -f docker-compose.prod.yml logs -f web

# Логи nginx
docker-compose -f docker-compose.prod.yml logs -f nginx

# Статус контейнеров
docker-compose -f docker-compose.prod.yml ps

# Перезапуск
docker-compose -f docker-compose.prod.yml restart

# Остановка
docker-compose -f docker-compose.prod.yml down

# Пересборка
docker-compose -f docker-compose.prod.yml up -d --build

# Очистка Docker
docker system prune -af
```

## SSL (Let's Encrypt)

После привязки домена:

```bash
# Остановить nginx
docker-compose -f docker-compose.prod.yml stop nginx

# Получить сертификат
certbot certonly --standalone -d your-domain.com

# Скопировать сертификаты
mkdir -p /opt/pellets-analyzer/ssl
cp /etc/letsencrypt/live/your-domain.com/fullchain.pem /opt/pellets-analyzer/ssl/
cp /etc/letsencrypt/live/your-domain.com/privkey.pem /opt/pellets-analyzer/ssl/

# Обновить nginx.prod.conf для SSL (listen 443 + ssl_certificate)
# Пересобрать:
docker-compose -f docker-compose.prod.yml up -d --build
```

## Безопасность

Настроено:
- UFW (порты 22, 80)
- Fail2Ban
- Nginx rate limiting
- Security headers (X-Frame-Options, CSP, HSTS)
- SQL injection, XSS, CSRF защита

Рекомендации:
- Сменить пароль SSH
- Настроить SSH-ключи, отключить парольный доступ
- Включить автоматические обновления

## Решение проблем

```bash
# Контейнер не запускается
docker-compose -f docker-compose.prod.yml logs web

# Nginx ошибка
docker-compose -f docker-compose.prod.yml logs nginx

# Проверка nginx конфига
docker-compose -f docker-compose.prod.yml run --rm nginx nginx -t

# Порт 80 занят
ss -tlnp | grep :80
```
