import argparse, os, json, pandas as pd, numpy as np
from joblib import load
from imap_utils import prompt_creds, connect_imap, fetch_folder, apply_action
from utils_text import normalize_text, handcrafted_features

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--inbox", default="INBOX")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--model", default="artifacts/model.joblib")
    ap.add_argument("--vectorizer", default="artifacts/vectorizer.joblib")
    ap.add_argument("--output", default="artifacts/live_preds.csv")
    ap.add_argument("--threshold", default="auto")
    ap.add_argument("--min-prob", type=float, default=0.98, help="Only apply actions above this prob.")
    ap.add_argument("--apply", default=None, help='Optional: "flag:\\Seen" or "move:[Gmail]/Spam"')
    ap.add_argument("--save-creds", action="store_true")
    args = ap.parse_args()

    # Load model bundle & threshold
    bundle = load(args.model)
    vec = load(args.vectorizer)
    import json, os
    with open(os.path.join(os.path.dirname(args.model), "threshold.json")) as f:
        th = json.load(f)["threshold"]
    if args.threshold != "auto":
        th = float(args.threshold)

    # Fetch latest INBOX emails
    user, pwd = prompt_creds(save=args.save_creds)
    client = connect_imap(args.host, user=user, pwd=pwd)
    df = fetch_folder(client, args.inbox, limit=args.limit, search_criteria='ALL')

    # Vectorize
    df["norm"] = df["text"].astype(str).map(normalize_text)
    from scipy.sparse import hstack
    Xw = vec["word"].transform(df["norm"])
    Xc = vec["char"].transform(df["norm"])
    X = hstack([Xw, Xc]).tocsr()

    H = None
    if bundle.get("use_handcrafted", False):
        import numpy as np
        H = np.vstack([handcrafted_features(t,f,r) for t,f,r in zip(df["text"], df.get("from",""), df.get("reply_to",""))])

    svc_p = bundle["cal_svc"].predict_proba(X)[:,1]
    lr_p  = bundle["logreg"].predict_proba(X)[:,1]
    feats = [svc_p, lr_p]
    if H is not None:
        feats.append(H)
    META = np.column_stack(feats)
    if hasattr(bundle["meta"], "predict_proba"):
        probs = bundle["meta"].predict_proba(META)[:,1]
    else:
        d = bundle["meta"].decision_function(META)
        probs = (d - d.min())/(d.max()-d.min()+1e-12)

    preds = (probs >= th).astype(int)
    df_out = df.copy()
    df_out["prob_spam"] = probs
    df_out["pred"] = preds
    df_out.to_csv(args.output, index=False, encoding="utf-8")
    print(f"[✓] Wrote predictions: {args.output}")

    # Optional mailbox action
    if args.apply:
        uids_to_act = df_out.loc[(df_out["pred"]==1)&(df_out["prob_spam"]>=args.min_prob), "uid"].tolist()
        if uids_to_act:
            apply_action(client, uids_to_act, args.apply)
            print(f"[✓] Applied '{args.apply}' to {len(uids_to_act)} messages (prob >= {args.min_prob}).")
        else:
            print("[i] No messages met the prob threshold for action.")

if __name__ == "__main__":
    main()
