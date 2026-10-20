#!/usr/bin/env bash
# Runner script for the Navier-Stokes STEM Exhibition

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d ".venv" ]; then
    echo "Virtual environment .venv not found. Creating and installing..."
    /opt/homebrew/bin/python3.12 -m venv .venv
    ./.venv/bin/pip install -r requirements.txt
fi

PYTHON="./.venv/bin/python3"

MODE="${1:---dev-mode}"

echo "=========================================================="
echo " Starting Navier-Stokes CFD Interactive Exhibition App"
echo " Mode: $MODE"
echo "=========================================================="

"$PYTHON" app/main.py "$MODE"
