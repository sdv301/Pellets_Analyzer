@echo off
setlocal enabledelayedexpansion
color 0A

echo ============================================================
echo            Pellets Analyzer - Setup and Launch
echo ============================================================
echo.

:: ============================================================
:: Step 0: Python check
:: ============================================================
python --version >nul 2>&1
if !errorlevel! neq 0 (
    color 0C
    echo [ERROR] Python not found!
    echo.
    echo Please install Python 3.11+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [OK] Python found: !PYTHON_VERSION!
echo.

:: ============================================================
:: Step 1: Virtual environment check
:: ============================================================
if not exist "venv" (
    echo [INFO] Creating virtual environment...
    python -m venv venv
    if !errorlevel! neq 0 (
        color 0C
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created.
) else (
    echo [OK] Virtual environment already exists.
)
echo.

:: ============================================================
:: Step 2: Activate environment
:: ============================================================
call venv\Scripts\activate.bat
if !errorlevel! neq 0 (
    color 0C
    echo [ERROR] Failed to activate virtual environment.
    pause
    exit /b 1
)
echo [OK] Virtual environment activated.
echo.

:: ============================================================
:: Step 3: Install dependencies
:: ============================================================
if exist "requirements.txt" (
    echo [INFO] Installing/updating dependencies...
    pip install -r requirements.txt --quiet
    if !errorlevel! neq 0 (
        color 0C
        echo [ERROR] Failed to install dependencies.
        echo Check your internet connection and try again.
        pause
        exit /b 1
    )
    echo [OK] Dependencies installed.
) else (
    color 0E
    echo [WARNING] requirements.txt not found!
)
echo.

:: ============================================================
:: Step 4: Configuration check
:: ============================================================
if not exist "mail_config.json" (
    if exist "mail_config.example.json" (
        echo [INFO] Creating mail_config.json from example...
        copy mail_config.example.json mail_config.json >nul
        echo [OK] mail_config.json created.
        echo       Please edit it with your SMTP credentials.
    )
) else (
    echo [OK] mail_config.json found.
)

if not exist "db_config.json" (
    if exist "db_config.example.json" (
        echo [INFO] Creating db_config.json from example...
        copy db_config.example.json db_config.json >nul
        echo [OK] db_config.json created.
    )
) else (
    echo [OK] db_config.json found.
)
echo.

:: ============================================================
:: Step 5: Create necessary directories
:: ============================================================
if not exist "Uploads" mkdir Uploads
if not exist "sessions" mkdir sessions
echo [OK] Directories ready.
echo.

:: ============================================================
:: Step 6: Launch application
:: ============================================================
echo ============================================================
echo  Starting Pellets Analyzer...
echo  Open your browser: http://localhost:5000
echo  Press Ctrl+C to stop
echo ============================================================
echo.
python main.py

if !errorlevel! neq 0 (
    color 0C
    echo.
    echo ============================================================
    echo [ERROR] Application crashed! Check the log above.
    echo ============================================================
    pause
)

pause
