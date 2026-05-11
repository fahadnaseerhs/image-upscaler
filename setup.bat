@echo off
echo ===================================================
echo Antigravity Setup Script (Windows)
echo ===================================================

echo Checking for Python...
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Python is not installed or not in your PATH. Please install Python 3.10+ and try again.
    pause
    exit /b 1
)

echo.
echo Creating virtual environment (venv)...
if not exist "venv\Scripts\python.exe" (
    python -m venv venv
    if %ERRORLEVEL% neq 0 (
        echo Failed to create virtual environment.
        pause
        exit /b 1
    )
) else (
    echo Virtual environment already exists.
)

echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Installing dependencies from requirements.txt...
pip install -r requirements.txt

echo.
echo ===================================================
echo Setup Complete!
echo ===================================================
echo To run the application, double-click on run.bat
echo OR manually activate the environment and run app.py:
echo   1. venv\Scripts\activate
echo   2. python app.py
echo.
pause
