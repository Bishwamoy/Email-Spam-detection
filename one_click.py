import os, sys, subprocess, json, re, html
from pathlib import Path
from datetime import datetime
import shutil

ROOT = Path(__file__).resolve().parent

def run(cmd_list, cwd):
    print("\n$ " + " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd_list))
    proc = subprocess.run(cmd_list, cwd=cwd, text=True, shell=False)
    if proc.returncode != 0:
        sys.exit(proc.returncode)

def ask(prompt, default=None):
    if default is None:
        val = input(prompt + ": ").strip()
        return val
    val = input(f"{prompt} [{default}]: ").strip()
    return val or str(default)

# ---------------- helpers for readable outputs ----------------
def _mime_decode(s: str) -> str:
    """Decode RFC-2047 subjects/names like '=?UTF-8?Q?...?='."""
    if s is None:
        return ""
    try:
        from email.header import decode_header, make_header
        return str(make_header(decode_header(str(s))))
    except Exception:
        return str(s)

def _preview_text(s, n=160):
    if s is None:
        return ""
    s = html.unescape(str(s))
    # strip markdown links [text](url) -> text
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)
    # strip HTML tags
    s = re.sub(r"<[^>]+>", " ", s)
    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return (s[:n] + "…") if len(s) > n else s

def _split_from(val):
    from email.utils import parseaddr
    name, email = parseaddr(val or "")
    return _mime_decode(name).strip(), (email or "").strip()

def _fmt_date_ist(iso):
    try:
        import pandas as pd
        from zoneinfo import ZoneInfo
        ts = pd.to_datetime(iso, utc=True, errors="coerce")
        if ts is pd.NaT:
            return str(iso)
        ts = ts.tz_convert(ZoneInfo("Asia/Kolkata"))
        return ts.strftime("%d %b %Y, %I:%M %p IST")
    except Exception:
        return str(iso)

def _detect(colnames, *cands, default=None):
    lut = {c.lower(): c for c in colnames}
    for c in cands:
        if c.lower() in lut:
            return lut[c.lower()]
    return default

def _bucket_from_email(email: str) -> str:
    """
    Turn an email into a safe folder name like 'user_gmail_com'.
    If empty, return 'default'.
    """
    if not email:
        return "default"
    email = email.strip().lower()
    # replace @ and dots with underscores; drop other non-alnum chars
    import re as _re
    bucket = _re.sub(r'[^a-z0-9]+', '_', email.replace('@', '_').replace('.', '_')).strip('_')
    return bucket or "default"

def _ensure_openpyxl():
    try:
        import openpyxl  # noqa: F401
        return True
    except Exception:
        try:
            # silent-ish install into the current venv
            subprocess.run([sys.executable, "-m", "pip", "install", "openpyxl", "-q"],
                           check=True, text=True)
            import openpyxl  # noqa: F401
            return True
        except Exception:
            return False

def _to_fileset(df, basepath: Path):
    """Write CSV, TSV (.txt), and XLSX (auto-installs openpyxl if missing)."""
    import pandas as pd
    base = str(basepath)

    # CSV
    csv_path = Path(base + ".csv")
    df.to_csv(csv_path, index=False, encoding="utf-8")

    # TSV for Notepad readability
    txt_path = Path(base + ".txt")
    df.to_csv(txt_path, index=False, sep="\t", encoding="utf-8")

    # XLSX
    xlsx_path = Path(base + ".xlsx")
    if _ensure_openpyxl():
        try:
            df.to_excel(xlsx_path, index=False)
        except Exception:
            # if writing still fails, ignore
            pass
    # return the three paths (some may not exist if xlsx failed)
    return [txt_path, csv_path, xlsx_path]

# (open_in_notepad / open_in_excel helpers left in the file originally are no longer used)

def write_readable_files(run_dir: Path, out_dir: Path):
    import pandas as pd
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- DATASET (split Inbox vs Spam) ----------
    ds = run_dir / "data" / "dataset.csv"
    inbox_paths = spam_paths = None
    if ds.exists() and ds.stat().st_size > 0:
        df = pd.read_csv(ds)
        cols = list(df.columns)

        date_c  = _detect(cols, "date", "received_at", "time")
        from_c  = _detect(cols, "from", "sender")
        subj_c  = _detect(cols, "subject", "title")
        text_c  = _detect(cols, "text", "body", "snippet")
        label_c = _detect(cols, "label", "is_spam", "spam")

        Date = df[date_c].map(_fmt_date_ist) if date_c in df else ""
        if from_c in df:
            nm, em = zip(*df[from_c].map(_split_from))
        else:
            nm, em = ([""]*len(df), [""]*len(df))
        Subject = (df[subj_c].map(_mime_decode) if subj_c in df else "")
        Preview = (df[text_c].map(_preview_text) if text_c in df else "")

        # enforce order Name -> Subject -> Date
        readable = pd.DataFrame({
            "From Name": nm,
            "Subject": Subject,
            "Date": Date,
            "From Email": em,
            "Preview": Preview,
        })

        if label_c in df:
            inbox = readable[df[label_c].astype(int)==0].reset_index(drop=True)
            spam  = readable[df[label_c].astype(int)==1].reset_index(drop=True)
        else:
            inbox, spam = readable.copy(), readable.iloc[0:0].copy()

        inbox_paths = _to_fileset(inbox, out_dir / "Inbox_Mails")
        spam_paths  = _to_fileset(spam,  out_dir / "Spam_Mails")
        print(f"[✓] Saved: {(out_dir/'Inbox_Mails.csv')}  and  {(out_dir/'Spam_Mails.csv')}")
    else:
        print("[i] dataset.csv missing or empty; wrote nothing for dataset.")

    # ---------- LIVE PREDICTIONS ----------
    lp = run_dir / "artifacts" / "live_preds.csv"
    live_paths = None
    if lp.exists() and lp.stat().st_size > 0:
        df = pd.read_csv(lp)
        cols = list(df.columns)

        date_c  = _detect(cols, "date", "received_at", "time")
        from_c  = _detect(cols, "from", "sender")
        subj_c  = _detect(cols, "subject", "title")
        text_c  = _detect(cols, "text", "body", "snippet")
        prob_c  = _detect(cols, "prob_spam", "probability", "score")
        pred_c  = _detect(cols, "pred", "prediction", "label")
        uid_c   = _detect(cols, "uid", "id")

        Date = df[date_c].map(_fmt_date_ist) if date_c in df else ""
        if from_c in df:
            nm, em = zip(*df[from_c].map(_split_from))
        else:
            nm, em = ([""]*len(df), [""]*len(df))
        Subject  = (df[subj_c].map(_mime_decode) if subj_c in df else "")
        Preview  = (df[text_c].map(_preview_text) if text_c in df else "")
        SpamProb = (df[prob_c].astype(float).round(3) if prob_c in df else "")
        PredLab  = (df[pred_c].map({1:"Spam",0:"Not Spam"}) if pred_c in df else "")
        UID      = df[uid_c] if uid_c in df else ""

        # enforce order Name -> Subject -> Date
        preds = pd.DataFrame({
            "From Name": nm,
            "Subject": Subject,
            "Date": Date,
            "From Email": em,
            "Preview": Preview,
            "Spam_Prob": SpamProb,
            "Prediction": PredLab,
            "UID": UID,
        })

        live_paths = _to_fileset(preds, out_dir / "Live_Predictions")
        print(f"[✓] Saved: {(out_dir/'Live_Predictions.csv')}")
    else:
        print("[i] live_preds.csv missing or empty; wrote nothing for predictions.")

    # return the written file triplets (txt, csv, xlsx)
    paths = []
    if inbox_paths: paths += inbox_paths
    if spam_paths:  paths += spam_paths
    if live_paths:  paths += live_paths
    return paths

def main():
    print("=== One-Click Email Spam Detector (per-run folders) ===")
    host = ask("IMAP host", "imap.gmail.com")
    inbox = ask("Inbox folder", "INBOX")
    spamf = ask("Spam folder", "[Gmail]/Spam")
    lim_in = int(ask("Fetch limit for INBOX", 400))
    lim_sp = int(ask("Fetch limit for Spam", 400))

    # Per-run working area
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    RUN = ROOT / "runs" / run_id
    (RUN / "data").mkdir(parents=True, exist_ok=True)
    (RUN / "artifacts").mkdir(parents=True, exist_ok=True)
    (RUN / "artifacts_v2").mkdir(parents=True, exist_ok=True)
    print(f"[i] Run folder: {RUN}")

    py = sys.executable

    # 1) Fetch
    run([py, str(ROOT/"fetch_emails.py"),
         "--host", host, "--inbox", inbox, "--spam", spamf,
         "--limit-inbox", str(lim_in), "--limit-spam", str(lim_sp),
         "--out", "data/dataset.csv", "--save-creds"], cwd=RUN)

    # 2) Balance
    if (ROOT/"balance_data_v2.py").exists():
        run([py, str(ROOT/"balance_data_v2.py"),
             "--in", "data/dataset.csv", "--out", "data/dataset_balanced.csv"], cwd=RUN)
        data_csv = "data/dataset_balanced.csv"
    elif (ROOT/"balance_data.py").exists():
        run([py, str(ROOT/"balance_data.py"),
             "--in", "data/dataset.csv", "--out", "data/dataset_balanced.csv"], cwd=RUN)
        data_csv = "data/dataset_balanced.csv"
    else:
        print("[i] No balance script found. Using raw dataset.csv")
        data_csv = "data/dataset.csv"

    # 3) Train (prefer v2)
    if (ROOT/"train_tfidf_ensemble_v2.py").exists():
        run([py, str(ROOT/"train_tfidf_ensemble_v2.py"),
             "--data", data_csv, "--text-col", "text", "--label-col", "label",
             "--subject-col", "subject", "--from-col", "from", "--reply-to-col", "reply_to"], cwd=RUN)
        # copy v2 artifacts into RUN/artifacts for evaluate
        try:
            shutil.copy(RUN/"artifacts_v2/model.joblib",      RUN/"artifacts/model.joblib")
            shutil.copy(RUN/"artifacts_v2/vectorizer.joblib", RUN/"artifacts/vectorizer.joblib")
            shutil.copy(RUN/"artifacts_v2/threshold.json",    RUN/"artifacts/threshold.json")
            shutil.copy(RUN/"artifacts_v2/test.csv",          RUN/"artifacts/test.csv")
        except Exception as e:
            print("[i] Could not copy v2 artifacts into artifacts/:", e)
    else:
        run([py, str(ROOT/"train_tfidf_ensemble.py"),
             "--data", data_csv, "--text-col", "text", "--label-col", "label",
             "--subject-col", "subject", "--from-col", "from", "--reply-to-col", "reply_to"], cwd=RUN)

    # 4) Evaluate
    if (ROOT/"evaluate.py").exists():
        run([py, str(ROOT/"evaluate.py"),
             "--model", "artifacts/model.joblib",
             "--vectorizer", "artifacts/vectorizer.joblib",
             "--test-csv", "artifacts/test.csv"], cwd=RUN)

    # 5) Live Predict
    print("\nOptional mailbox action examples: flag:\\Seen  OR  move:[Gmail]/Spam")
    action = ask("Enter mailbox action (or leave blank to skip applying)", "")
    apply_parts = ["--apply", action] if action else []
    run([py, str(ROOT/"predict_live.py"),
         "--host", host, "--inbox", inbox, "--limit", "200",
         "--output", "artifacts/live_preds.csv", "--save-creds", *apply_parts], cwd=RUN)

    # 6) Build readable outputs in ROOT/outputs (per account)
    account_for_outputs = ask("Save outputs under which email (use the same email you logged in with)", "")
    OUT = ROOT / "outputs" / _bucket_from_email(account_for_outputs)

    try:
        write_readable_files(RUN, OUT)
    except Exception as e:
        print(f"[i] Could not build readable outputs: {e}")

    print("\n=== All done! ===")
    print(f"- Easy outputs (fixed names): {OUT}")
    print("  - Inbox_Mails.[csv|txt|xlsx]")
    print("  - Spam_Mails.[csv|txt|xlsx]")
    print("  - Live_Predictions.[csv|txt|xlsx]")
    print(f"- Technical run data:        {RUN}")

if __name__ == "__main__":
    main()
