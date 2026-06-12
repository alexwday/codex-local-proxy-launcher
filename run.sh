#!/bin/bash
# Run Kilo-Launcher

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

echo "Installing dependencies..."
pip install -q -r src/requirements.txt
pip install -q rbc_security 2>/dev/null || true

if [ ! -f "src/.env" ]; then
    echo "No .env file found. Copying from .env.example..."
    if [ -f "src/.env.example" ]; then
        cp src/.env.example src/.env
        echo "Please edit src/.env with your endpoint, auth, and model settings, then run again."
        exit 1
    fi
fi

python src/app.py
