# Email Spam Detection (IMAP + ML, one-click)

End‑to‑end project that:
1) **Fetches emails over IMAP** (Gmail or any IMAP server) from your **INBOX** and **Spam** folders
2) **Builds a labeled dataset** automatically (`label=1` for Spam folder, `0` for Inbox)
3) **Trains an accurate ensemble** (TF‑IDF + LinearSVC + Calibrated Logistic + handcrafted features)
4) **Evaluates** with PR/ROC/Calibration plots
5) **Runs live prediction** on *new* INBOX emails (optionally writes an IMAP flag or moves them to Spam on request)

> Credentials are asked at runtime (email + App Password). **Nothing is stored** unless you explicitly pass `--save-creds` (off by default).

if you already inside a folder the go to the Email-Spam-detection folder first by:- 
cd(Email -Spam-detection) 

Quick start (Windows)
.\start.bat

That’s it. The script will create a virtual environment, install dependencies, and launch the interactive pipeline.

What does start.bat do?
Bootstraps Python env
Creates .venv if missing and upgrades pip.
Installs requirements
Uses requirements.txt.
Runs the workflow
Launches one_click_plus_classify.py (the “one-click” pipeline).

What is this for?
A batch email spam detection workflow for Gmail:
Fetches recent INBOX (ham) and Spam (spam) via IMAP.
Trains a TF-IDF ensemble spam model on your mailbox.
Evaluates metrics (F1, ROC, confusion matrix).
Predicts on INBOX only (no cheating with Spam folder).
Writes clean Excel/CSV/TXT reports.
Auto-classifies the exported Inbox_Mails.* into Inbox_Mails_Classified.*.
All results are stored under outputs/<your_email_bucket>/.
What will you be asked when it runs?
During one_click_plus_classify.py:
IMAP host: imap.gmail.com
Inbox folder: INBOX
Spam folder: [Gmail]/Spam
Fetch limits: e.g., 50 (INBOX) and 50 (Spam)
Mailbox action: press Enter to skip (no changes to mailbox)
Email for outputs: the Gmail you’re using (used to name the output folder)
Gmail address + App Password (16 chars) when prompted by the fetch/predict steps
Make sure IMAP is enabled in Gmail and you created an App Password (Google Account → Security → App passwords).

Where do results go?
outputs/
  <your_email_bucket>/
    Inbox_Mails.[csv|txt|xlsx]
    Spam_Mails.[csv|txt|xlsx]
    Live_Predictions.[csv|txt|xlsx]
    Inbox_Mails_Classified.[csv|txt|xlsx]  <-- final decisions (Spam / Not Spam)
runs/
  <timestamp>/
    data/…, artifacts_v2/… (model, vectorizer, threshold), artifacts/test.csv, plots

Re-run classification later (no IMAP)
If you already fetched and trained once, you can re-classify the inbox file offline:
.\.venv\Scripts\python.exe classify_inbox_file.py

You’ll be prompted for:
Artifacts folder: e.g., runs\YYYY-MM-DD_HH-MM-SS\artifacts_v2
Email: used to locate outputs/<bucket>/
Path to Inbox_Mails.[csv|xlsx|txt]: press Enter to auto-discover, or paste a path
Output: outputs/<bucket>/Inbox_Mails_Classified.[csv|txt|xlsx].

How it works (in short)
Fetch & Label: INBOX=ham, Spam=spam → dataset.csv
Balance: Mitigates class imbalance → dataset_balanced.csv
Train: TF-IDF + ensemble (stored in artifacts_v2/)
Evaluate: F1/ROC/CM and plots
Predict: On INBOX only → Live_Predictions.*
Format: Human-readable Excel/CSV/TXT
Classify file: Applies trained model to Inbox_Mails.* → Inbox_Mails_Classified.*

Common hiccups
AUTHENTICATIONFAILED: Wrong email/app password or IMAP disabled.
openpyxl missing: Auto-installed; CSV/TXT still produced even if XLSX fails.
Model mismatch: Always point classify_inbox_file.py to the same artifacts_v2 that produced your outputs.
