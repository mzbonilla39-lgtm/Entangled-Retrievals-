#!/usr/bin/env bash
set -euo pipefail

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install pytest httpx

echo "Development environment ready. Activate with: source .venv/bin/activate"