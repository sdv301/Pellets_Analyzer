# Лаборатория как субмодуль

`lab/` — git-субмодуль [sdv301/Tolko_model](https://github.com/sdv301/Tolko_model.git). В продукте зафиксирован конкретный коммит. Частые коммиты в лаборатории не меняют прод, пока пин не сдвинут отдельной командой и не попал в PR продукта.

Сейчас пин: тег `lab-v0.1.0`, коммит `959e545d1649c1acfa4ef108a64a45d843ca20f4`, пакет `pellet_ml` 0.1.0, `API_VERSION` 1.0.0. Тот же коммит записан в `app/services/ml_v2/PIN`.

HTTPS в `.gitmodules`:

```text
https://github.com/sdv301/Tolko_model.git
```

SSH, если так удобнее на сервере (подставьте в `git remote` внутри `lab/` или в `submodule.lab.url`):

```text
git@github.com:sdv301/Tolko_model.git
```

Продукт импортирует только `pellet_ml`. Своего дубля обучения больше нет. `MODEL_VERSION` по умолчанию `legacy`. Откат режима — убрать переменную. Откат пина — вернуть gitlink, см. ниже.

## Как устроена работа

1. Опыты и модели меняются в репозитории Tolko_model.
2. Когда контракт готов, в лаборатории ставится тег (`lab-v0.1.0`, следующий — `lab-v0.2.0`, …). Ломающее изменение публичных функций или полей прогноза поднимает мажор `API_VERSION` (`2.0.0`). Продукт принимает только мажор 1 и пишет в лог, что v2 отключён. Legacy при этом стартует как раньше.
3. В продукте: `scripts/update_lab.sh <тег-или-коммит>` (на Windows — `scripts/update_lab.ps1`). Скрипт забирает лабораторию, прогоняет `lab/tests/test_public_api.py` и тесты продукта, собирает артефакт `python -m pellet_ml build` в `models/v2/`, сравнивает прогнозы с `tests/fixtures/v2_baseline.json`. Если всё зелёное, в индекс попадают gitlink `lab` и файл `PIN`. Если нет — `lab` возвращается на прежний коммит, индекс пина не меняется.
4. Владелец смотрит diff и открывает PR продукта. Деплой подхватывает новый пин.
5. Образ Docker собирает артефакт заново внутри Python 3.11. joblib в git не коммитится: его можно воспроизвести из пина и `lab/data/recipes.csv`.

Добавление необязательного поля контракта — минор `API_VERSION`, пин можно сдвигать без правок продукта. Другой смысл `predict` / `load_models` или отказ от `manifest.json` — мажор, продукт такую лабораторию не загрузит, пока адаптер не обновят отдельно.

## Клонирование

```bash
git clone --recurse-submodules https://github.com/sdv301/Pellets_Analyzer.git
```

Уже склонированный репозиторий:

```bash
git submodule update --init lab
```

PowerShell:

```powershell
git clone --recurse-submodules https://github.com/sdv301/Pellets_Analyzer.git
cd Pellets_Analyzer
git submodule update --init lab
```

Без доступа к Tolko_model каталог `lab/` пустой. Приложение всё равно запускается в режиме legacy. В логе будет сообщение, что v2 недоступен. Локально можно указать распакованный снимок, не коммитя его:

```powershell
$env:PELLET_LAB_PATH = "C:\src\Tolko_model"
$env:PYTHONPATH = $env:PELLET_LAB_PATH
```

То же самое в bash: `export PELLET_LAB_PATH=...` и `export PYTHONPATH="$PELLET_LAB_PATH"`.

## Обновление пина

Bash:

```bash
bash scripts/update_lab.sh lab-v0.2.0
git status
git commit -m "Bump lab submodule to lab-v0.2.0"
```

PowerShell из корня репозитория:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\update_lab.ps1 lab-v0.2.0
git status
git commit -m "Bump lab submodule to lab-v0.2.0"
```

Скрипт сам делает `git fetch` внутри `lab/` и `git add lab` плюс `app/services/ml_v2/PIN`. Коммит не создаёт.

Если прогнозы уехали дальше допуска (`abs_tol` 0.05 и `rel_tol` 0.01 в снимке) или сменилось имя модели, скрипт откатывает субмодуль. Осознанное изменение: на нужном коммите лаборатории пересобрать артефакт, обновить `tests/fixtures/v2_baseline.json` отдельным коммитом, затем снова запустить скрипт.

Проверка руками:

```bash
python -m pytest lab/tests/test_public_api.py -q
python scripts/build_v2_artifact.py --out models/v2
python scripts/compare_v2_baseline.py
```

## Откат

Режим модели, без отката кода:

```bash
# legacy — значение по умолчанию
unset MODEL_VERSION
```

Пин, если bump ещё не закоммичен:

```bash
git checkout 959e545d1649c1acfa4ef108a64a45d843ca20f4 -- lab
# и вернуть прежнее содержимое app/services/ml_v2/PIN
```

Пин, если bump уже в истории продукта (историю не переписывать):

```bash
git revert <sha-коммита-с-пином>
```

Либо новый коммит, в котором `lab` снова указывает на старый SHA, а `PIN` содержит тот же SHA. Образ после этого пересобрать: артефакт внутри образа должен совпасть с пином, иначе загрузчик v2 откажется и оставит legacy.

## Сервер и Docker

На сервере один раз:

```bash
git clone --recurse-submodules <url продукта>
# или в существующем клоне
git submodule update --init lab
docker compose build
docker compose up -d
```

`docker compose build` копирует `lab/` в образ и выполняет `python scripts/build_v2_artifact.py` на Python 3.11. Каталог `models/v2/` в git не входит. `MODEL_VERSION` в compose по умолчанию `legacy`. Чтобы включить v2: `MODEL_VERSION=v2`.

Без `git submodule update --init` контекст сборки не содержит `lab/`, и `docker compose build` остановится на `COPY lab/`.

## Deploy key для CI

Репозиторий лаборатории приватный. В GitHub Actions секрет не хранится в git.

1. На машине владельца: `ssh-keygen -t ed25519 -f lab_deploy_key -N "" -C "pellets-analyzer-ci"`.
2. Открыть Tolko_model → Settings → Deploy keys → Add deploy key. Вставить `lab_deploy_key.pub`. Галочку Allow write access не ставить.
3. Открыть Pellets_Analyzer → Settings → Secrets and variables → Actions → New repository secret. Имя `LAB_DEPLOY_KEY`. Значение — содержимое приватного файла `lab_deploy_key` целиком, включая строки BEGIN и END.
4. Приватный файл удалить с машины, в репозиторий его не класть.

Workflow `.github/workflows/tests.yml` по этому ключу делает `git submodule update --init lab` через SSH и гоняет тесты на Python 3.11. Запасной секрет — `LAB_READ_TOKEN` (fine-grained PAT только с чтением Tolko_model). Если заданы оба, используется ключ.

Локальный снимок архива для отладки в репозиторий не коммитится.
