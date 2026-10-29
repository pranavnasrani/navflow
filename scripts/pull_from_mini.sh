#!/usr/bin/env bash
# Fetch finished OpenFOAM runs from the Mac mini into runs/full/ on this machine.
# Each machine keeps its own queue file (queue_<machine>.json), so this never overwrites
# the laptop's queue. Afterwards run: ./.venv/bin/python solver/import_results.py
# Set MINI_HOST to change the source (default the mini's Tailscale IP, 100.114.120.115).
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p runs/full
# Never copy queue_laptop.json: a run on the mini started without --machine mini writes that name,
# and it would overwrite the laptop's real queue.
rsync -av --exclude 'queue_laptop.json' "${MINI_HOST:-100.114.120.115}:navflow/runs/full/" runs/full/
