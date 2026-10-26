#!/usr/bin/env bash
# Fetch finished OpenFOAM runs from the Mac mini into runs/full/ on this machine.
# Each machine keeps its own queue file (queue_<machine>.json), so this never overwrites
# the laptop's queue. Afterwards run: ./.venv/bin/python solver/import_results.py
# Set MINI_HOST to change the source (default mac-mini.local).
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p runs/full
rsync -av "${MINI_HOST:-mac-mini.local}:navflow/runs/full/" runs/full/
