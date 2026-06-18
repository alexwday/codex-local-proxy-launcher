#!/bin/bash
# Run Codex Local Proxy Launcher.

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
        echo "Created src/.env with default settings. Edit it for your upstream endpoint/auth as needed."
    fi
fi

python src/app.py
