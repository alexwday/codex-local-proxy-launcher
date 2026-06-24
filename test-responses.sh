#!/bin/bash
# Smoke-test the configured Responses API endpoint using the launcher env.

set -euo pipefail

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

python src/test_responses_endpoint.py "$@"
