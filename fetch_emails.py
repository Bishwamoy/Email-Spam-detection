import argparse, os, sys, pandas as pd
from imap_utils import prompt_creds, connect_imap, fetch_folder
os.makedirs("data", exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--inbox", default="INBOX")
    ap.add_argument("--spam", default="[Gmail]/Spam")
    ap.add_argument("--limit-inbox", type=int, default=500)
    ap.add_argument("--limit-spam", type=int, default=500)
    ap.add_argument("--out", default="data/dataset.csv")
    ap.add_argument("--save-creds", action="store_true")
    args = ap.parse_args()

    user, pwd = prompt_creds(save=args.save_creds)
    client = connect_imap(args.host, user=user, pwd=pwd)

    print(f"[+] Fetching ham from {args.inbox} ...")
    df_in = fetch_folder(client, args.inbox, limit=args.limit_inbox)
    df_in["label"] = 0

    print(f"[+] Fetching spam from {args.spam} ...")
    df_sp = fetch_folder(client, args.spam, limit=args.limit_spam)
    df_sp["label"] = 1

    df = pd.concat([df_in, df_sp], ignore_index=True)
    df.to_csv(args.out, index=False, encoding="utf-8")
    print(f"[✓] Wrote dataset: {args.out} (rows={len(df)})")

if __name__ == "__main__":
    main()
