@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion

:: Устанавливаем зеленый цвет текста для индикации успешной работы
color 0A

echo ===========================================================
echo            Pellets Analyzer - Установка и Запуск           
echo ===========================================================
echo.

:: Шаг 1
echo [^|^|^|       ] 30%% ШАГ 1: Проверка системы и окружения...
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    
    :: Быстрая проверка наличия основных библиотек
    python -c "import flask; import flask_session; import matplotlib; import numpy; import pandas; import plotly; import requests; import seaborn; import openpyxl; import scipy; import sklearn" >nul 2>&1
    if !errorlevel! equ 0 (
        echo    -- Окружение уже настроено и готово к работе.
        goto launch
    ) else (
        echo    -- Найдены не все библиотеки. Потребуется доустановка.
    )
) else (
    echo    -- Создание виртуального окружения ^(пожалуйста, подождите^) ...
    python -m venv venv
    if !errorlevel! neq 0 (
        echo.
        color 0C
        echo [ОШИБКА] Не удалось создать окружение. Убедитесь, что установлен Python!
        pause
        exit /b !errorlevel!
    )
    call venv\Scripts\activate.bat
)

:: Шаг 2
echo.
echo [^|^|^|^|^|^|    ] 60%% ШАГ 2: Установка необходимых библиотек...
echo    -- Процесс может занять несколько минут. Идет скачивание...
if exist "requirements.txt" (
    pip install -r requirements.txt
    if !errorlevel! neq 0 (
        echo.
        color 0C
        echo [ОШИБКА] Проблема при установке библиотек. Проверьте подключение к интернету.
        pause
        exit /b !errorlevel!
    )
) else (
    echo [ВНИМАНИЕ] Файл requirements.txt не найден!
)

:launch
:: Шаг 3
echo.
echo [^|^|^|^|^|^|^|^|^|^|] 100%% ГОТОВО! Программа запускается...
echo ===========================================================
echo Пожалуйста, не закрывайте это окно, пока работаете в программе!
echo ===========================================================
:: Возвращаем стандартный цвет перед запуском программы
color 0F
python main.py

if !errorlevel! neq 0 (
    echo.
    color 0C
    echo [ОШИБКА] Программа завершила работу с ошибкой.
    pause
)

pause
