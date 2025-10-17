import os, json, argparse, warnings
warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score
from scipy.sparse import hstack
from joblib import dump
from utils_text import normalize_text, handcrafted_features, HANDCRAFTED_NAMES

def read_csv(path, text_col, label_col, subject_col=None, from_col=None, reply_col=None):
    df = pd.read_csv(path)
    if subject_col and subject_col in df.columns:
        subj = df[subject_col].fillna("").astype(str)
        txt = (subj + " " + df[text_col].fillna("").astype(str)).str.strip()
    else:
        txt = df[text_col].fillna("").astype(str)
    meta = {}
    meta["from"] = df[from_col].astype(str).fillna("") if from_col and from_col in df.columns else pd.Series([""]*len(df))
    meta["reply_to"] = df[reply_col].astype(str).fillna("") if reply_col and reply_col in df.columns else pd.Series([""]*len(df))
    y = df[label_col].astype(int).values
    X = pd.DataFrame({"text": txt, "from": meta["from"], "reply_to": meta["reply_to"]})
    return X, y

def build_vectorizers(word_ngrams=(1,2), char_ngrams=(3,5), max_features=300000, min_df=2, sublinear_tf=True):
    wv = TfidfVectorizer(ngram_range=word_ngrams, analyzer="word", min_df=min_df,
                         max_features=max_features, sublinear_tf=sublinear_tf)
    cv = TfidfVectorizer(ngram_range=char_ngrams, analyzer="char", min_df=min_df,
                         max_features=max_features//2, sublinear_tf=sublinear_tf)
    return wv, cv

def optimize_threshold(y_true, probs):
    p, r, th = precision_recall_curve(y_true, probs)
    best_t, best_f1 = 0.5, -1.0
    for pi, ri, ti in zip(p[:-1], r[:-1], th):
        f1 = 2*pi*ri/(pi+ri+1e-12)
        if f1 > best_f1:
            best_f1, best_t = f1, ti
    return float(best_t), float(best_f1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--subject-col", default=None)
    ap.add_argument("--from-col", dest="from_col", default=None)
    ap.add_argument("--reply-to-col", dest="reply_col", default=None)
    ap.add_argument("--random-seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs("artifacts_v2/plots", exist_ok=True)

    X, y = read_csv(args.data, args.text_col, args.label_col, args.subject_col, args.from_col, args.reply_col)
    X["norm"] = X["text"].astype(str).map(normalize_text)

    # Try stratified split; if it fails (minority < 2), fall back to non-stratified
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15,
                                                            random_state=args.random_seed, stratify=y)
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15,
                                                            random_state=args.random_seed, stratify=None)

    word_vec, char_vec = build_vectorizers()
    Xw = word_vec.fit_transform(X_train["norm"])
    Xc = char_vec.fit_transform(X_train["norm"])
    X_train_sparse = hstack([Xw, Xc]).tocsr()

    # Handcrafted features
    H_train = np.vstack([handcrafted_features(t,f,r)
                         for t,f,r in zip(X_train["text"], X_train["from"], X_train["reply_to"])])

    # Base models
    svc = LinearSVC(C=2.0, class_weight="balanced")
    svc.fit(X_train_sparse, y_train)

    logreg = LogisticRegression(C=3.0, class_weight="balanced", max_iter=2000, n_jobs=-1)
    logreg.fit(X_train_sparse, y_train)

    # Calibrate SVC (scikit-learn 1.4+: use estimator=)
    cal_svc = CalibratedClassifierCV(estimator=svc, method="sigmoid", cv=5)
    cal_svc.fit(X_train_sparse, y_train)

    # Meta training data
    svc_p_tr = cal_svc.predict_proba(X_train_sparse)[:,1]
    lr_p_tr  = logreg.predict_proba(X_train_sparse)[:,1]
    META_TRAIN = np.column_stack([svc_p_tr, lr_p_tr, H_train])

    meta = LogisticRegression(max_iter=2000, n_jobs=-1)
    meta.fit(META_TRAIN, y_train)

    # Test set
    Xw_t = word_vec.transform(X_test["norm"])
    Xc_t = char_vec.transform(X_test["norm"])
    X_test_sparse = hstack([Xw_t, Xc_t]).tocsr()
    H_test = np.vstack([handcrafted_features(t,f,r)
                        for t,f,r in zip(X_test["text"], X_test["from"], X_test["reply_to"])])

    svc_p = cal_svc.predict_proba(X_test_sparse)[:,1]
    lr_p  = logreg.predict_proba(X_test_sparse)[:,1]
    META_TEST = np.column_stack([svc_p, lr_p, H_test])

    if hasattr(meta, "predict_proba"):
        y_prob = meta.predict_proba(META_TEST)[:,1]
    else:
        d = meta.decision_function(META_TEST)
        y_prob = (d - d.min())/(d.max()-d.min()+1e-12)

    th, best_f1 = optimize_threshold(y_test, y_prob)
    y_pred = (y_prob >= th).astype(int)
    f1 = f1_score(y_test, y_pred)
    try:
        roc = roc_auc_score(y_test, y_prob)
    except Exception:
        roc = float("nan")

    # Save bundle
    bundle = {
        "word": word_vec, "char": char_vec,
        "cal_svc": cal_svc, "logreg": logreg, "meta": meta,
        "use_handcrafted": True,
        "handcrafted_names": HANDCRAFTED_NAMES
    }
    os.makedirs("artifacts_v2", exist_ok=True)
    dump(bundle, os.path.join("artifacts_v2","model.joblib"))
    dump({"word": word_vec, "char": char_vec}, os.path.join("artifacts_v2","vectorizer.joblib"))

    with open(os.path.join("artifacts_v2","threshold.json"), "w") as f:
        json.dump({"threshold": th, "best_f1": best_f1}, f, indent=2)

    pd.concat([X_test.reset_index(drop=True), pd.Series(y_test, name="label")], axis=1).to_csv(
        os.path.join("artifacts_v2","test.csv"), index=False, encoding="utf-8"
    )

    metrics = {"f1": float(f1), "roc_auc": float(roc), "threshold": float(th)}
    with open(os.path.join("artifacts_v2","train_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("=== Held-out Test Metrics (v2) ===")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
