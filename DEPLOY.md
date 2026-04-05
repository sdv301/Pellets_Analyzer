# ============================================================
# Pellets Analyzer — Руководство по деплою и VPS
# ============================================================

## 1. Email конфигурация

### Как настроить почту:

```bash
# 1. Скопируйте пример конфига
cp mail_config.example.json mail_config.json

# 2. Отредактируйте mail_config.json, указав свои данные
```

### Для Gmail:
1. Зайдите в Google Account → Security
2. Включите **2-Step Verification**
3. Создайте **App Password**: Account → Security → App Passwords
4. Вставьте полученный пароль в `MAIL_PASSWORD`

### Для других провайдеров:
| Провайдер | SMTP Server | Port | TLS |
|-----------|-------------|------|-----|
| Gmail | smtp.gmail.com | 587 | ✅ |
| Yandex | smtp.yandex.ru | 587 | ✅ |
| Mail.ru | smtp.mail.ru | 587 | ✅ |
| Outlook | smtp-mail.outlook.com | 587 | ✅ |

---

## 2. Характеристики VPS

### Минимальные требования:
| Параметр | Значение | Почему |
|----------|----------|--------|
| CPU | 2 ядра | Flask + обработка данных |
| RAM | 2 GB | ML-модели (XGBoost, scipy) |
| SSD | 20 GB | База данных + загрузки |
| OS | Ubuntu 22.04 LTS | Стабильность, Docker |

### Рекомендуемые требования (для продакшена):
| Параметр | Значение | Почему |
|----------|----------|--------|
| CPU | 4 ядра | Параллельная обработка ML |
| RAM | 4 GB | Больше данных в памяти |
| SSD | 40 GB | Логи, бэкапы, рост БД |
| OS | Ubuntu 22.04 LTS | Долгосрочная поддержка |

### Рекомендуемые VPS-провайдеры:
- **Timeweb Cloud** (Россия) — от 300₽/мес
- **Selectel** (Россия) — от 500₽/мес
- **DigitalOcean** — от $6/мес
- **Hetzner** — от €4/мес

---

## 3. Docker — использовать или нет?

### ✅ Преимущества Docker:
1. **Изоляция** — все зависимости внутри контейнера
2. **Воспроизводимость** — одинаковое окружение на любом сервере
3. **Безопасность** — приложение работает в изолированной среде
4. **Масштабируемость** — легко добавить реплики
5. **Простой деплой** — одна команда `docker-compose up -d`

### ❌ Когда Docker не нужен:
- Простой сервер для 1-2 пользователей
- Ограниченные ресурсы (Docker добавляет ~10% оверхеда)
- Нужен прямой доступ к системным библиотекам

### 🏆 Рекомендация: **Использовать Docker**
Для вашего проекта (Flask + ML + SQLite) Docker — лучший выбор:
```bash
# На сервере:
git clone <your-repo>
cd Pellets_Analyzer
cp .env.example .env
# Отредактируйте .env
docker-compose up -d
```

---

## 4. Лицензия для сайта

### Нужна ли лицензия?
- **MIT License** (уже добавлена) — разрешает свободное использование кода
- Для **коммерческого использования** — нужна лицензия на ПО
- Для **сайта** — добавьте страницу с условиями использования

### Что добавить на сайт:
1. [`LICENSE`](LICENSE) — файл лицензии в корне проекта
2. Страница `/terms` — условия использования сайта
3. Страница `/privacy` — политика конфиденциальности
4. Футер — ссылка на лицензию

### Для коммерческой версии:
```
© 2025 Pellets Analyzer. Все права защищены.
Использование без разрешения запрещено.
```

---

## 5. Защита от атак

### Реализованная защита:

| Тип атаки | Защита | Статус |
|-----------|--------|--------|
| **SQL Injection** | Параметризованные запросы + валидация | ✅ `security.py` |
| **XSS** | Санитизация входных данных | ✅ `security.py` |
| **CSRF** | Токены для POST-запросов | ✅ `security.py` |
| **Brute Force** | Блокировка после 5 попыток | ✅ `security.py` |
| **Rate Limiting** | Ограничение частоты запросов | ✅ `security.py` + nginx |
| **DDoS (базовый)** | Nginx rate limiting | ✅ `nginx.conf` |
| **Clickjacking** | X-Frame-Options | ✅ `security.py` + nginx |
| **MIME Sniffing** | X-Content-Type-Options | ✅ `security.py` + nginx |
| **File Upload** | Валидация расширений + размера | ✅ `security.py` |
| **IP Blocking** | Чёрный список IP | ✅ `security.py` |

### Дополнительные меры (на сервере):

```bash
# 1. Fail2Ban — блокировка подозрительных IP
sudo apt install fail2ban
sudo systemctl enable fail2ban

# 2. UFW Firewall
sudo ufw allow 22/tcp   # SSH
sudo ufw allow 80/tcp   # HTTP
sudo ufw allow 443/tcp  # HTTPS
sudo ufw enable

# 3. SSL сертификат (бесплатный Let's Encrypt)
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com

# 4. Автоматические обновления
sudo apt install unattended-upgrades
sudo dpkg-reconfigure unattended-upgrades

# 5. Отключение root-доступа по SSH
sudo nano /etc/ssh/sshd_config
# PermitRootLogin no
sudo systemctl restart sshd
```

---

## 6. Быстрый деплой на VPS

### Пошаговая инструкция:

```bash
# === 1. Подключение к серверу ===
ssh user@your-server-ip

# === 2. Установка Docker ===
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
exit  # Переподключитесь

# === 3. Клонирование проекта ===
git clone https://github.com/your-username/Pellets_Analyzer.git
cd Pellets_Analyzer

# === 4. Настройка конфигурации ===
cp .env.example .env
nano .env  # Укажите SECRET_KEY, MAIL_PASSWORD и т.д.

cp mail_config.example.json mail_config.json
nano mail_config.json  # Укажите SMTP данные

# === 5. Запуск ===
docker-compose up -d

# === 6. Проверка ===
docker-compose ps
docker-compose logs -f

# === 7. Настройка Nginx + SSL (опционально) ===
# Раскомментируйте nginx в docker-compose.yml
mkdir ssl
# Положите сертификлы в ssl/
docker-compose up -d nginx
```

---

## 7. Мониторинг и обслуживание

### Логи:
```bash
# Логи приложения
docker-compose logs -f web

# Логи безопасности
tail -f security.log

# Логи nginx
docker-compose logs -f nginx
```

### Бэкапы:
```bash
# Бэкап базы данных
docker-compose exec web cp /app/data/pellets_data.db /app/data/backup_$(date +%Y%m%d).db

# Бэкап загрузок
tar -czf uploads_backup_$(date +%Y%m%d).tar.gz Uploads/
```

### Обновление:
```bash
git pull
docker-compose down
docker-compose up -d --build
```
