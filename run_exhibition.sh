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

# Usage: ./run_exhibition.sh [--dev-mode | --monitor-mode] [--objects <id> ...] [--screens N ...]
#                            [--follow <host-ip>[:port]] [--no-remote] ...
#   Two computers: run the host with its --objects, then on the other Mac the remaining --objects plus --follow <host-ip>.
# Needs the OpenFOAM results in assets/cache/ (python solver/import_results.py); extra flags go to app/main.py.
if [ "$#" -eq 0 ]; then
    set -- --dev-mode
fi
MODE="$*"

echo "=========================================================="
echo " Starting Navier-Stokes CFD Interactive Exhibition App"
echo " Mode: $MODE"
echo "=========================================================="

"$PYTHON" app/main.py "$@"
