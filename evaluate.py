import argparse, json, os
import pandas as pd, numpy as np
from joblib import load
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt
from utils_text import normalize_text, handcrafted_features

def plot_pr(y_true, probs, out):
    p, r, _ = precision_recall_curve(y_true, probs)
    plt.figure()
    plt.plot(r, p, linewidth=2)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision‑Recall")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(out); plt.close()

def plot_roc(y_true, probs, out):
    fpr, tpr, _ = roc_curve(y_true, probs)
    plt.figure()
    plt.plot(fpr, tpr, linewidth=2)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve (AUC=%.4f)"%auc(fpr,tpr))
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(out); plt.close()

def plot_calibration(y_true, probs, out):
    bins = np.linspace(0,1,11)
    idx = np.digitize(probs, bins)-1
    df = pd.DataFrame({"y":y_true, "p":probs, "b":idx})
    g = df.groupby("b").agg(y_mean=("y","mean"), p_mean=("p","mean"))
    plt.figure()
    plt.plot([0,1],[0,1], linestyle='--')
    plt.plot(g["p_mean"], g["y_mean"], linewidth=2)
    plt.xlabel("Predicted probability"); plt.ylabel("Empirical rate")
    plt.title("Calibration")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(out); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--vectorizer", required=True)
    ap.add_argument("--test-csv", required=True)
    ap.add_argument("--target-precision", type=float, default=None)
    args = ap.parse_args()

    bundle = load(args.model)
    vec = load(args.vectorizer)
    df = pd.read_csv(args.test_csv)
    y = df["label"].values
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

    # Load threshold (or compute by target precision)
    with open(os.path.join(os.path.dirname(args.model), "threshold.json")) as f:
        th = json.load(f)["threshold"]
    if args.target_precision is not None:
        from sklearn.metrics import precision_recall_curve
        p, r, t = precision_recall_curve(y, probs)
        chosen = th
        for pi, ri, ti in zip(p[:-1], r[:-1], t):
            if pi >= args.target_precision:
                chosen = float(ti); break
        th = chosen

    y_pred = (probs >= th).astype(int)
    print("Threshold used:", th)
    print(confusion_matrix(y, y_pred))
    print(classification_report(y, y_pred, digits=4))

    os.makedirs("artifacts/plots", exist_ok=True)
    plot_pr(y, probs, "artifacts/plots/pr_curve.png")
    plot_roc(y, probs, "artifacts/plots/roc_curve.png")
    plot_calibration(y, probs, "artifacts/plots/calibration.png")
    print("Saved plots under artifacts/plots/.")

if __name__ == "__main__":
    main()
