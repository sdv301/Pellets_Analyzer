@echo off
chcp 65001 > nul
echo ==============================================
echo  Pellets Analyzer - Setup and Launch Script
echo ==============================================

echo.
echo [1/3] Creating virtual environment (venv)...
if not exist "venv" (
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment. Make sure Python is installed.
        pause
        exit /b %errorlevel%
    )
    echo Virtual environment created successfully.
) else (
    echo Virtual environment already exists.
)

echo.
echo [2/3] Activating virtual environment and installing dependencies...
call venv\Scripts\activate.bat
if exist "requirements.txt" (
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to install dependencies.
        pause
        exit /b %errorlevel%
    )
) else (
    echo [WARNING] requirements.txt not found.
)

echo.
echo [3/3] Launching the application...
echo The application will run locally. Please do not close this window.
python main.py

pause
