@echo off
if not exist "venv\Scripts\activate.bat" (
    echo Virtual environment not found. Running setup first...
    call setup.bat
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Starting the application...
python app.py
pause
