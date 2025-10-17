import argparse, json, os, pandas as pd, numpy as np
from joblib import load
from utils_text import normalize_text, handcrafted_features

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--vectorizer", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--threshold", default="auto")  # "auto" or float
    args = ap.parse_args()

    bundle = load(args.model)
    vec = load(args.vectorizer)
    df = pd.read_csv(args.input)
    if "text" not in df.columns:
        raise ValueError("Input CSV must have a 'text' column.")
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

    if args.threshold == "auto":
        with open(os.path.join(os.path.dirname(args.model), "threshold.json")) as f:
            th = json.load(f)["threshold"]
    else:
        th = float(args.threshold)

    df_out = df.copy()
    df_out["prob_spam"] = probs
    df_out["pred"] = (probs >= th).astype(int)
    cols = ["text","prob_spam","pred"] + [c for c in df.columns if c not in ("text","norm")]
    df_out[cols].to_csv(args.output, index=False)
    print(f"Wrote predictions to {args.output}")

if __name__ == "__main__":
    main()
