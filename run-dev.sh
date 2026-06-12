#!/bin/bash
# Run Kilo-Launcher in development mode with placeholder responses.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

echo "Installing dependencies..."
pip install -q -r src/requirements.txt 2>/dev/null

export PROXY_PORT="${PROXY_PORT:-5050}"
export DEV_MODE=true
export USE_PLACEHOLDER_MODE=true
export MODEL_OPTIONS="${MODEL_OPTIONS:-gpt-5.4,gpt-5.4-mini}"
export SKIP_SSL_VERIFY=true
export AUTO_OPEN_BROWSER=true

echo "Starting Kilo-Launcher in development mode..."
python src/app.py
