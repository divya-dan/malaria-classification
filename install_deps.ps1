<#
.SYNOPSIS
  Create a Python venv and install dependencies on Windows.

.DESCRIPTION
  - Creates a virtual environment in .\venv\
  - Uses the venv’s pip to upgrade itself and install from requirements.txt
#>

# Stop on any error
$ErrorActionPreference = "Stop"
# 1) Create venv if it does not exist
if (-not (Test-Path .\venv)) {
    Write-Host "Creating new venv folder..."
    python -m venv venv
} else {
    Write-Host "venv already exists. Skipping creation."
}

# 2) Install packages using the venv’s pip
$venvPip = Join-Path -Path $PWD -ChildPath "venv\Scripts\pip.exe"

Write-Host "Upgrading pip..."
& $venvPip install --upgrade pip

Write-Host "Installing dependencies from requirements.txt..."
& $venvPip install -r requirements.txt

Write-Host "All dependencies installed into $(Resolve-Path .\venv)"

# Activate the virtual environment for the current session
$venvActivate = Join-Path -Path $PWD -ChildPath "venv\Scripts\Activate.ps1"
Write-Host "Activating the virtual environment..."
& $venvActivate
