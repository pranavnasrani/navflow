#!/usr/bin/env bash
# Copy this project to the Mac mini so it can run OpenFOAM batches too.
# Skips the virtualenv (rebuild it on the mini), the run results, git history and local clutter
# (__pycache__, .DS_Store, .superpowers).
# Set MINI_HOST to change the target (default the mini's Tailscale IP, 100.114.120.115).
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
rsync -av --exclude .venv --exclude runs --exclude .git \
  --exclude '__pycache__' --exclude '.DS_Store' --exclude '.superpowers' \
  --exclude 'assets/models/incoming' ./ "${MINI_HOST:-100.114.120.115}:navflow/"
