@echo off
REM Windows CMD launcher: ensures venv, installs deps, runs one_click_plus_classify.py
setlocal enabledelayedexpansion
cd /d %~dp0

if not exist ".venv\Scripts\python.exe" (
  echo [i] Creating virtual environment...
  py -3 -m venv .venv 2>nul || python -m venv .venv
)

set PY=.venv\Scripts\python.exe
"%PY%" -m pip install --upgrade pip
"%PY%" -m pip install -r requirements.txt

"%PY%" one_click.py
pause
