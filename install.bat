@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion

echo ==============================================
echo  Pellets Analyzer - Setup and Launch Script
echo ==============================================

:: Check if venv exists and is valid
if exist "venv\Scripts\activate.bat" (
    echo [1/3] Checking existing environment...
    call venv\Scripts\activate.bat
    
    :: Fast check if main libraries are already installed
    python -c "import flask, flask_session, matplotlib, numpy, pandas, plotly, requests, seaborn, openpyxl, scipy, sklearn" >nul 2>&1
    if !errorlevel! equ 0 (
        echo [INFO] All dependencies are already installed. Skipping installation...
        goto launch
    ) else (
        echo [INFO] Some dependencies are missing. Proceeding with installation...
    )
) else (
    echo.
    echo [1/3] Creating virtual environment (venv)...
    python -m venv venv
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to create virtual environment. Make sure Python is installed.
        pause
        exit /b !errorlevel!
    )
    echo Virtual environment created successfully.
    call venv\Scripts\activate.bat
)

echo.
echo [2/3] Installing/Updating dependencies...
if exist "requirements.txt" (
    pip install -r requirements.txt
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to install dependencies.
        pause
        exit /b !errorlevel!
    )
) else (
    echo [WARNING] requirements.txt not found.
)

:launch
echo.
echo [3/3] Launching the application...
echo The application will run locally. Please do not close this window.
python main.py

if !errorlevel! neq 0 (
    echo.
    echo [ERROR] Application crashed or finished with error.
    pause
)

pause

