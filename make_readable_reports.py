import argparse, os, re, pandas as pd
from pathlib import Path
from email.utils import parseaddr
from datetime import datetime, timezone
try:
    from zoneinfo import ZoneInfo  # py3.9+
except Exception:
    ZoneInfo = None

def fmt_date_local(d, tzname="Asia/Kolkata"):
    if pd.isna(d):
        return ""
    try:
        ts = pd.to_datetime(d, utc=True, errors="coerce")
        if ts is pd.NaT:
            return str(d)
        if ZoneInfo is not None:
            tz = ZoneInfo(tzname)
            ts = ts.tz_convert(tz) if ts.tzinfo else ts.tz_localize(timezone.utc).astimezone(tz)
        else:
            # Fallback: no tz conversion, just show ISO
            return ts.strftime("%d %b %Y, %I:%M %p")
        return ts.strftime("%d %b %Y, %I:%M %p IST")
    except Exception:
        return str(d)

def split_from(val):
    if pd.isna(val):
        return "", ""
    name, email = parseaddr(str(val))
    return name.strip(), email.strip()

def preview_text(t, limit=200):
    if pd.isna(t):
        return ""
    s = str(t)
    # strip html tags & compress whitespace
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return (s[:limit] + "…") if len(s) > limit else s

def make_readable_dataset(df, tzname="Asia/Kolkata"):
    df = df.copy()
    df["Date"] = df["date"].map(lambda x: fmt_date_local(x, tzname))
    df[["From Name","From Email"]] = df["from"].map(split_from).apply(pd.Series)
    df["Subject"] = df.get("subject","").fillna("")
    df["Preview"] = df.get("text","").map(lambda x: preview_text(x, 200))
    cols = ["Date","From Name","From Email","Subject","Preview"]
    inbox = df[df["label"]==0][cols].reset_index(drop=True)
    spam  = df[df["label"]==1][cols].reset_index(drop=True)
    return inbox, spam

def make_readable_preds(df, tzname="Asia/Kolkata"):
    df = df.copy()
    df["Date"] = df["date"].map(lambda x: fmt_date_local(x, tzname))
    df[["From Name","From Email"]] = df["from"].map(split_from).apply(pd.Series)
    df["Subject"] = df.get("subject","").fillna("")
    df["Preview"] = df.get("text","").map(lambda x: preview_text(x, 200))
    # nicer labels
    df["Prediction"] = df.get("pred",0).map({1:"Spam", 0:"Not Spam"})
    df["Spam_Prob"]  = df.get("prob_spam",0.0).map(lambda x: round(float(x), 3))
    df["UID"] = df.get("uid","")
    cols = ["Date","From Name","From Email","Subject","Preview","Spam_Prob","Prediction","UID"]
    return df[cols].reset_index(drop=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="Path to a single run folder (contains data/ and artifacts/).")
    ap.add_argument("--tz", default="Asia/Kolkata")
    args = ap.parse_args()

    RUN = Path(args.run_dir)
    out_dir = RUN / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dataset → split into inbox/spam readable CSVs
    ds_path = RUN / "data" / "dataset.csv"
    if ds_path.exists():
        df_ds = pd.read_csv(ds_path)
        if {"date","from","text","label"}.issubset(df_ds.columns):
            inbox, spam = make_readable_dataset(df_ds, tzname=args.tz)
            inbox.to_csv(out_dir/"inbox_readable.csv", index=False, encoding="utf-8")
            spam.to_csv(out_dir/"spam_readable.csv",  index=False, encoding="utf-8")
            print(f"[✓] Wrote: {out_dir/'inbox_readable.csv'}")
            print(f"[✓] Wrote: {out_dir/'spam_readable.csv'}")
        else:
            print("[i] dataset.csv missing required columns; skipping split.")
    else:
        print("[i] dataset.csv not found; skipping dataset reports.")

    # Live predictions → readable CSV
    lp_path = RUN / "artifacts" / "live_preds.csv"
    if lp_path.exists():
        df_lp = pd.read_csv(lp_path)
        if {"date","from","text"}.issubset(df_lp.columns):
            preds = make_readable_preds(df_lp, tzname=args.tz)
            preds.to_csv(out_dir/"live_preds_readable.csv", index=False, encoding="utf-8")
            print(f"[✓] Wrote: {out_dir/'live_preds_readable.csv'}")
        else:
            print("[i] live_preds.csv missing expected columns; skipping predictions report.")
    else:
        print("[i] live_preds.csv not found; skipping predictions report.")

if __name__ == "__main__":
    main()
