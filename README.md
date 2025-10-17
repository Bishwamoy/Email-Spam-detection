# Email Spam Detection (IMAP + ML, one-click)

End‑to‑end project that:
1) **Fetches emails over IMAP** (Gmail or any IMAP server) from your **INBOX** and **Spam** folders
2) **Builds a labeled dataset** automatically (`label=1` for Spam folder, `0` for Inbox)
3) **Trains an accurate ensemble** (TF‑IDF + LinearSVC + Calibrated Logistic + handcrafted features)
4) **Evaluates** with PR/ROC/Calibration plots
5) **Runs live prediction** on *new* INBOX emails (optionally writes an IMAP flag or moves them to Spam on request)

> Credentials are asked at runtime (email + App Password). **Nothing is stored** unless you explicitly pass `--save-creds` (off by default).

---

## Quick start (Windows PowerShell, VS Code friendly)

```powershell
# 1) Create & activate a venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) Install deps
pip install --upgrade pip
pip install -r requirements.txt

# 3) Fetch a small dataset (e.g., 500 from INBOX, 500 from Spam)
python fetch_emails.py --host imap.gmail.com --inbox INBOX --spam "[Gmail]/Spam" --limit-inbox 500 --limit-spam 500 --out data/dataset.csv

# 4) Train (holds out a test split, calibrates, finds best threshold)
python train_tfidf_ensemble.py --data data/dataset.csv --text-col text --label-col label --subject-col subject --from-col from --reply-to-col reply_to

# 5) Evaluate (prints metrics + saves plots under artifacts/plots/)
python evaluate.py --model artifacts/model.joblib --vectorizer artifacts/vectorizer.joblib --test-csv artifacts/test.csv

# 6) Predict on *new* INBOX mail (no changes to mailbox by default)
python predict_live.py --host imap.gmail.com --inbox INBOX --limit 200 --output artifacts/live_preds.csv
```

### Optional (apply actions to mailbox)
Add `--apply "flag:\Seen"` to mark predicted spam as seen, or `--apply "move:[Gmail]/Spam"` to move them to Spam.  
(Use with caution; test first without `--apply`.)

```powershell
python predict_live.py --host imap.gmail.com --inbox INBOX --limit 200 --output artifacts/live_preds.csv --apply "move:[Gmail]/Spam" --min-prob 0.97
```

---

## Gmail notes
- Create a **16‑character App Password** (Google Account → Security → 2‑Step Verification → App Passwords → “Mail” on “Windows Computer”).  
- IMAP must be ON in Gmail settings → Forwarding and POP/IMAP.  
- Login with your **email** and that **App Password** (not your normal password).

## Privacy & Safety
- The scripts **prompt** for credentials and keep them in memory only.  
- Pass `--save-creds` only if you understand the risk; it saves to `.env` in plain text (off by default).

---

## Folder layout
- `fetch_emails.py` — IMAP fetcher → CSV (INBOX = ham, Spam folder = spam)
- `train_tfidf_ensemble.py`, `evaluate.py`, `predict.py` — ML pipeline
- `predict_live.py` — fetch latest INBOX, run model, (optionally) act on results
- `utils_text.py` — normalization & handcrafted features
- `imap_utils.py` — IMAP helpers (robust parsing, HTML→text, actions)
- `requirements.txt` — deps (classical ML + IMAP)
- `run_one_click.ps1` — guided pipeline: fetch → train → eval → live predict

---

## Tuning tips
- Prefer **char 3–5 n‑grams** + **word 1–2** (already set) for obfuscations.
- Use `--target-precision` in `evaluate.py` if false positives are costly.
- Re‑train weekly; new spam patterns drift.
