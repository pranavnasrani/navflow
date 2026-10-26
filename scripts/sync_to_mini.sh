#!/usr/bin/env bash
# Copy this project to the Mac mini so it can run OpenFOAM batches too.
# Skips the virtualenv (rebuild it on the mini), the run results and git history.
# Set MINI_HOST to change the target (default mac-mini.local).
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
rsync -av --exclude .venv --exclude runs --exclude .git ./ "${MINI_HOST:-mac-mini.local}:navflow/"
