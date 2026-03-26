@echo off
setlocal enabledelayedexpansion
color 0A

echo ===========================================================
echo            Pellets Analyzer - Setup and Launch           
echo ===========================================================

:: Step 1: Venv check
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if !errorlevel! neq 0 (
        color 0C
        echo ERROR: Python not found! Install Python and add to PATH.
        pause
        exit /b
    )
)

:: Step 2: Activate and Install
call venv\Scripts\activate.bat
if exist "requirements.txt" (
    echo Checking libraries...
    pip install -r requirements.txt 
)

:: Step 3: Run
echo Launching program...
echo ===========================================================
python main.py

if !errorlevel! neq 0 (
    color 0C
    echo ERROR: Program crashed.
    pause
)
pause
