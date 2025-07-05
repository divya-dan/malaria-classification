#!/usr/bin/env bash
set -euo pipefail

# 1) Create venv if it does not exist
if [[ ! -d "venv" ]]; then
  echo "Creating new venv folder..."
  python3 -m venv venv
else
  echo "venv already exists. Skipping creation."
fi

# 2) Activate venv for this session
#    Note: this only affects the current shell
source venv/bin/activate

# 3) Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# 4) Install dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo
echo "✅ All dependencies installed into $(pwd)/venv"
echo "To activate this environment in future sessions, run:"
echo "  source venv/bin/activate"
