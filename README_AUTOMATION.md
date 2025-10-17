
# One‑Click Automation & Auto‑Venv

## TL;DR (Windows)
- Double‑click **start.bat** (or right‑click **start.ps1** → Run)
- It will **auto‑create/activate** a local `.venv`, install requirements, and run the full pipeline.
- No manual activation needed.

## What it does
1. Ensures `.venv` exists and uses `.venv\Scripts\python.exe` directly
2. Installs dependencies from `requirements.txt`
3. Runs `one_click.py`, which:
   - fetches emails (asks IMAP host/folders + App Password at runtime)
   - balances dataset (if scripts exist)
   - trains the model (prefers `*_v2.py` if present)
   - evaluates the model
   - runs live prediction on newest INBOX emails
   - optionally applies mailbox actions (flag/move)

## Ask-once design
When `one_click.py` starts, it prompts for host/folders/limits and proceeds end‑to‑end.
It will reuse the same venv next time — you only **run start.bat** again.

## Notes
- You can skip "activation" entirely; the launcher calls the venv's Python directly.
- If PowerShell execution is blocked, run once as Administrator:
  `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`
