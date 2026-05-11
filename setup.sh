#!/bin/bash
echo "==================================================="
echo "Antigravity Setup Script (Linux/macOS)"
echo "==================================================="

# Check for Python 3
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed or not in your PATH. Please install Python 3.10+ and try again."
    exit 1
fi

echo ""
echo "Creating virtual environment (venv)..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "Failed to create virtual environment."
        exit 1
    fi
else
    echo "Virtual environment already exists."
fi

echo ""
echo "Activating virtual environment..."
source venv/bin/activate

echo ""
echo "Upgrading pip..."
pip install --upgrade pip

echo ""
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo ""
echo "==================================================="
echo "Setup Complete!"
echo "==================================================="
echo "To run the application, execute: ./run.sh"
echo "OR manually activate the environment and run app.py:"
echo "  1. source venv/bin/activate"
echo "  2. python app.py"
echo ""
