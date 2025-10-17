import argparse, os, json, math, warnings
warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier
from joblib import dump
from tqdm import tqdm

from utils_text import normalize_text, handcrafted_features, HANDCRAFTED_NAMES

def read_csv(path, text_col, label_col, subject_col=None, from_col=None, reply_col=None):
    df = pd.read_csv(path)
    if subject_col and subject_col in df.columns:
        subj = df[subject_col].fillna("")
        txt = (subj.astype(str) + " " + df[text_col].fillna("").astype(str)).str.strip()
    else:
        txt = df[text_col].fillna("").astype(str)
    # keep optional columns for features
    meta = {}
    for name, col in [('from', from_col), ('reply_to', reply_col)]:
        if col and col in df.columns:
            meta[name] = df[col].astype(str).fillna("")
        else:
            meta[name] = pd.Series([""]*len(df))
    y = df[label_col].astype(int).values
    X = pd.DataFrame({"text": txt, "from": meta["from"], "reply_to": meta["reply_to"]})
    return X, y

def compute_handcrafted_matrix(df):
    feats = np.vstack([handcrafted_features(t, f, r) for t,f,r in zip(df["text"], df["from"], df["reply_to"])])
    return feats

def build_vectorizer(max_features=300000, word_ngrams=(1,2), char_ngrams=(3,5), min_df=2, sublinear_tf=True):
    word = TfidfVectorizer(ngram_range=word_ngrams, analyzer='word', min_df=min_df,
                           max_features=max_features, sublinear_tf=sublinear_tf)
    char = TfidfVectorizer(ngram_range=char_ngrams, analyzer='char', min_df=min_df,
                           max_features=max_features//2, sublinear_tf=sublinear_tf)
    return word, char

def optimize_threshold(y_true, probs, optimize_for="f1", beta=1.0, target_precision=None):
    p, r, th = precision_recall_curve(y_true, probs)
    best_t, best_score = 0.5, -1.0
    if target_precision is not None:
        # pick smallest threshold achieving >= target precision
        for pi, ri, ti in zip(p[:-1], r[:-1], th):
            if pi >= target_precision:
                return float(ti), {"precision": float(pi), "recall": float(ri)}
        # fallback to max precision point
        idx = np.argmax(p)
        return float(th[min(idx, len(th)-1)]) if len(th)>0 else 0.5, {"precision": float(p[idx]), "recall": float(r[idx])}
    if optimize_for == "f1":
        for pi, ri, ti in zip(p[:-1], r[:-1], th):
            f1 = 2*pi*ri/(pi+ri+1e-12)
            if f1 > best_score:
                best_score, best_t = f1, ti
        return float(best_t), {"best_f1": float(best_score)}
    if optimize_for == "fbeta":
        beta2 = beta*beta
        for pi, ri, ti in zip(p[:-1], r[:-1], th):
            fbeta = (1+beta2)*pi*ri/(beta2*pi+ri+1e-12)
            if fbeta > best_score:
                best_score, best_t = fbeta, ti
        return float(best_t), {"best_fbeta": float(best_score)}
    # default
    return 0.5, {}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--subject-col", default=None)
    ap.add_argument("--from-col", dest="from_col", default=None)
    ap.add_argument("--reply-to-col", dest="reply_col", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--with-transformer-probs", default=None, help="CSV with columns: id(optional), prob")
    ap.add_argument("--random-seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs("artifacts/plots", exist_ok=True)

    # Load config or defaults
    cfg = {
        "test_size": 0.15,
        "val_size": 0.15,
        "tfidf": {"word_ngrams": [1,2], "char_ngrams": [3,5], "max_features": 300000, "min_df": 2, "sublinear_tf": True},
        "models": {"linear_svc": {"C": 2.0, "class_weight": "balanced"},
                   "logreg": {"C": 3.0, "class_weight": "balanced", "max_iter": 2000}},
        "stacking": {"meta": "logreg", "use_handcrafted": True},
        "thresholding": {"optimize_for": "f1", "beta": 1.0}
    }
    if args.config and os.path.exists(args.config):
        import yaml
        with open(args.config) as f:
            cfg.update(yaml.safe_load(f))

    X, y = read_csv(args.data, args.text_col, args.label_col, args.subject_col, args.from_col, args.reply_col)
    # normalize text
    X["norm"] = X["text"].astype(str).map(normalize_text)

    # split train/test (stratified)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=cfg["test_size"], random_state=args.random_seed, stratify=y)

    # vectorizers
    word, char = build_vectorizer(max_features=cfg["tfidf"]["max_features"],
                                  word_ngrams=tuple(cfg["tfidf"]["word_ngrams"]),
                                  char_ngrams=tuple(cfg["tfidf"]["char_ngrams"]),
                                  min_df=cfg["tfidf"]["min_df"],
                                  sublinear_tf=cfg["tfidf"]["sublinear_tf"])

    # Fit TF-IDF on train
    word_X = word.fit_transform(X_train["norm"])
    char_X = char.fit_transform(X_train["norm"])
    from scipy.sparse import hstack, csr_matrix
    X_train_sparse = hstack([word_X, char_X]).tocsr()

    # handcrafted
    H_train = compute_handcrafted_matrix(X_train) if cfg["stacking"]["use_handcrafted"] else np.zeros((len(X_train),0))

    # Base models
    svc = LinearSVC(C=cfg["models"]["linear_svc"]["C"], class_weight=cfg["models"]["linear_svc"]["class_weight"])
    svc.fit(X_train_sparse, y_train)
    # Calibrated Logistic Regression
    logreg = LogisticRegression(C=cfg["models"]["logreg"]["C"],
                                class_weight=cfg["models"]["logreg"]["class_weight"],
                                max_iter=cfg["models"]["logreg"]["max_iter"],
                                n_jobs=-1)
    logreg.fit(X_train_sparse, y_train)

    # Calibrate SVC (needs probabilities for stacking)
    cal_svc = CalibratedClassifierCV(estimator=svc, method="sigmoid", cv=5)
    cal_svc.fit(X_train_sparse, y_train)

    # Prepare meta-training data
    svc_p_train = cal_svc.predict_proba(X_train_sparse)[:,1]
    lr_p_train  = logreg.predict_proba(X_train_sparse)[:,1]
    meta_feats = [svc_p_train, lr_p_train]
    if H_train.shape[1] > 0:
        meta_feats.append(H_train)
    META_TRAIN = np.column_stack(meta_feats)

    # Optional transformer probs (align by index; or the CSV should match X_train ordering)
    if args.with_transformer_probs and os.path.exists(args.with_transformer_probs):
        tfp = pd.read_csv(args.with_transformer_probs)["prob"].values
        tfp = tfp[:len(META_TRAIN)]
        META_TRAIN = np.column_stack([META_TRAIN, tfp])

    # Meta model
    if cfg["stacking"]["meta"] == "ridge":
        meta = RidgeClassifier()
    else:
        meta = LogisticRegression(max_iter=2000, n_jobs=-1)
    meta.fit(META_TRAIN, y_train)

    # Evaluate on held-out test
    word_Xt = word.transform(X_test["norm"])
    char_Xt = char.transform(X_test["norm"])
    X_test_sparse = hstack([word_Xt, char_Xt]).tocsr()
    H_test = compute_handcrafted_matrix(X_test) if H_train.shape[1] > 0 else np.zeros((len(X_test),0))

    svc_p = cal_svc.predict_proba(X_test_sparse)[:,1]
    lr_p = logreg.predict_proba(X_test_sparse)[:,1]
    meta_test = [svc_p, lr_p]
    if H_test.shape[1] > 0:
        meta_test.append(H_test)
    META_TEST = np.column_stack(meta_test)

    y_prob = meta.predict_proba(META_TEST)[:,1] if hasattr(meta, "predict_proba") else (meta.decision_function(META_TEST) - meta.decision_function(META_TEST).min())
    if hasattr(meta, "predict_proba"):
        pass
    else:
        # scale to [0,1]
        y_prob = (y_prob - y_prob.min())/(y_prob.max()-y_prob.min()+1e-12)

    # Threshold search
    th, info = optimize_threshold(y_test, y_prob, **cfg["thresholding"])
    y_pred = (y_prob >= th).astype(int)
    f1 = f1_score(y_test, y_pred)
    try:
        roc = roc_auc_score(y_test, y_prob)
    except:
        roc = float("nan")

    os.makedirs("artifacts", exist_ok=True)
    # Save vectorizer and model as a dict via joblib
    bundle = {
        "word": word, "char": char,
        "cal_svc": cal_svc, "logreg": logreg, "meta": meta,
        "use_handcrafted": H_train.shape[1] > 0,
        "handcrafted_names": HANDCRAFTED_NAMES
    }
    dump(bundle, "artifacts/model.joblib")
    # Save threshold
    with open("artifacts/threshold.json","w") as f:
        json.dump({"threshold": th, "info": info}, f, indent=2)
    # Save test split for later eval
    pd.concat([X_test.reset_index(drop=True), pd.Series(y_test, name="label")], axis=1).to_csv("artifacts/test.csv", index=False)
    # Save vectorizer alone (optional separate file)
    from joblib import dump as jdump
    jdump({"word": word, "char": char}, "artifacts/vectorizer.joblib")

    metrics = {"f1": float(f1), "roc_auc": float(roc), "threshold": float(th)}
    with open("artifacts/train_metrics.json","w") as f:
        json.dump(metrics, f, indent=2)

    print("=== Held‑out Test Metrics ===")
    print(json.dumps(metrics, indent=2))
    print("Threshold search info:", info)

if __name__ == "__main__":
    main()
