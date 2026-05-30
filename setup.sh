#!/usr/bin/env bash
# macOS/Linux setup: creates .venv and installs dependencies.
# Usage (from project root):  source setup.sh   (use 'source' so the venv stays active)
set -e

python3 setup_env.py
source .venv/bin/activate
echo "Environment ready and activated."
