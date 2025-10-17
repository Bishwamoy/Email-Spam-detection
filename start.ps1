
# PowerShell launcher (double-click friendly)
# Ensures venv, installs deps, and runs one_click.py
# Allow direct run without requiring user to 'activate' the venv

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here

# Ensure venv
if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
  Write-Host "[i] Creating virtual environment..." -ForegroundColor Yellow
  python -m venv .venv
}

# Use venv python directly (no activation needed)
$py = ".\.venv\Scripts\python.exe"

# Upgrade pip + install deps
& $py -m pip install --upgrade pip
& $py -m pip install -r requirements.txt

# Run
& $py one_click.py
