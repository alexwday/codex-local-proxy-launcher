#!/bin/bash
# Run Codex Local Proxy Launcher in development mode with placeholder responses.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

echo "Installing dependencies..."
pip install -q -r src/requirements.txt 2>/dev/null

export CODEX_PROXY_PORT="${CODEX_PROXY_PORT:-5051}"
export DEV_MODE=true
export USE_PLACEHOLDER_MODE=true
export CODEX_MODEL_OPTIONS="${CODEX_MODEL_OPTIONS:-gpt-5.5,gpt-5.4,gpt-5.4-mini}"
export CODEX_DEFAULT_MODEL="${CODEX_DEFAULT_MODEL:-gpt-5.5}"
export SKIP_SSL_VERIFY=true
export AUTO_OPEN_BROWSER=true

echo "Starting Codex Local Proxy Launcher in development mode..."
python src/app.py
