#!/bin/bash
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Running setup first..."
    chmod +x setup.sh
    ./setup.sh
    if [ $? -ne 0 ]; then
        exit 1
    fi
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Starting the application..."
python3 app.py
