param(
  [string]$HostName = "imap.gmail.com",
  [string]$Inbox = "INBOX",
  [string]$SpamFolder = "[Gmail]/Spam",
  [int]$InboxLimit = 500,
  [int]$SpamLimit = 500
)

Write-Host "=== Email Spam Detector: guided run ===" -ForegroundColor Cyan

# 1) venv
if (-not (Test-Path ".\.venv")) {
  Write-Host "[+] Creating venv..." -ForegroundColor Yellow
  python -m venv .venv
}
Write-Host "[+] Activating venv..." -ForegroundColor Yellow
.\.venv\Scripts\Activate.ps1

# 2) deps
Write-Host "[+] Installing requirements..." -ForegroundColor Yellow
pip install --upgrade pip
pip install -r requirements.txt

# 3) fetch
Write-Host "[+] Fetching emails (this will prompt for your email + app password)..." -ForegroundColor Yellow
python fetch_emails.py --host $HostName --inbox $Inbox --spam $SpamFolder --limit-inbox $InboxLimit --limit-spam $SpamLimit --out data/dataset.csv

# 4) train
Write-Host "[+] Training model..." -ForegroundColor Yellow
python train_tfidf_ensemble.py --data data/dataset.csv --text-col text --label-col label --subject-col subject --from-col from --reply-to-col reply_to

# 5) evaluate
Write-Host "[+] Evaluating..." -ForegroundColor Yellow
python evaluate.py --model artifacts/model.joblib --vectorizer artifacts/vectorizer.joblib --test-csv artifacts/test.csv

# 6) live predict (no mailbox changes; CSV only)
Write-Host "[+] Live prediction on newest INBOX emails (no changes to mailbox)..." -ForegroundColor Yellow
python predict_live.py --host $HostName --inbox $Inbox --limit 200 --output artifacts/live_preds.csv

Write-Host "=== Done. Check 'artifacts/' for models, plots, and live_preds.csv ===" -ForegroundColor Green
