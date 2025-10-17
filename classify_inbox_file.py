import os, sys, re, html, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def ask(prompt, default=None):
    if default is None:
        return input(prompt + ": ").strip()
    val = input(f"{prompt} [{default}]: ").strip()
    return val or str(default)

def _bucket_from_email(email: str) -> str:
    if not email: return "default"
    email = email.strip().lower()
    import re as _re
    bucket = _re.sub(r'[^a-z0-9]+', '_', email.replace('@', '_').replace('.', '_')).strip('_')
    return bucket or "default"

def _ensure_openpyxl():
    try:
        import openpyxl  # noqa
        return True
    except Exception:
        try:
            import subprocess
            subprocess.run([sys.executable, "-m", "pip", "install", "openpyxl", "-q"], check=True, text=True)
            import openpyxl  # noqa
            return True
        except Exception:
            return False

# -------- artifact + model helpers --------
def _looks_like_model(x):
    return hasattr(x, "predict") or hasattr(x, "predict_proba") or hasattr(x, "decision_function")

def _looks_like_vectorizer(x):
    return hasattr(x, "transform")

def _expected_n_features(model):
    """
    Try to read how many input features the model expects.
    Handles LinearSVC within CalibratedClassifierCV, plain LinearSVC, etc.
    """
    # calibrated wrapper?
    base = getattr(model, "base_estimator", None)
    if base is None and hasattr(model, "calibrated_classifiers_"):
        c0 = model.calibrated_classifiers_[0]
        base = getattr(c0, "base_estimator", None)

    candidates = [model]
    if base is not None:
        candidates.append(base)

    for m in candidates:
        nfi = getattr(m, "n_features_in_", None)
        if isinstance(nfi, int):
            return nfi
        # sklearn linear models sometimes store coef_ shape
        coef = getattr(m, "coef_", None)
        if coef is not None and hasattr(coef, "shape"):
            try:
                return coef.shape[1]
            except Exception:
                pass
    return None  # unknown; we'll just try combos

def _load_artifacts(art_dir: Path):
    """
    Return (model, vectorizers_bundle, threshold_float).
    vectorizers_bundle is a dict that may contain keys: word, char, vectorizer
    """
    from joblib import load
    art_dir = Path(art_dir)
    model_path = art_dir / "model.joblib"
    vec_path   = art_dir / "vectorizer.joblib"
    thr_path   = art_dir / "threshold.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}")

    obj = load(model_path)
    model = None
    bundle = {"word": None, "char": None, "vectorizer": None}
    threshold = 0.5

    if _looks_like_model(obj):
        model = obj
    else:
        if isinstance(obj, dict):
            print(f"[i] model.joblib is a dict with keys: {list(obj.keys())}")
            # model
            for pick in ("pipeline","model","estimator","clf","best_model","cal_svc","logreg"):
                if pick in obj and _looks_like_model(obj[pick]):
                    model = obj[pick]
                    print(f"[i] Picked model from key: {pick}")
                    break
            if model is None:
                # fallback: first model-like value
                for k,v in obj.items():
                    if _looks_like_model(v):
                        print(f"[i] Picked model from key: {k}")
                        model = v
                        break
            # vectorizers
            for k in ("vectorizer","word","char"):
                if k in obj and _looks_like_vectorizer(obj[k]):
                    bundle[k] = obj[k]
            # threshold possibly present in dict under 'threshold' or nested meta
            if "threshold" in obj:
                try: threshold = float(obj["threshold"])
                except Exception: pass
            if "meta" in obj and isinstance(obj["meta"], dict):
                t = obj["meta"].get("threshold")
                if t is not None:
                    try: threshold = float(t)
                    except Exception: pass
        else:
            # try attributes
            for k in dir(obj):
                if k.startswith("_"): continue
                v = getattr(obj,k)
                if model is None and _looks_like_model(v):
                    print(f"[i] Picked model from attribute: {k}")
                    model = v
                if bundle["vectorizer"] is None and _looks_like_vectorizer(v):
                    bundle["vectorizer"] = v

    # Separate vectorizer file support
    if bundle["vectorizer"] is None and vec_path.exists():
        try:
            from joblib import load as _load
            vec = _load(vec_path)
            if _looks_like_vectorizer(vec):
                print("[i] Loaded vectorizer.joblib")
                bundle["vectorizer"] = vec
        except Exception:
            pass

    # threshold.json
    if thr_path.exists():
        try:
            threshold = float(json.loads(thr_path.read_text()).get("threshold", threshold))
        except Exception:
            pass

    if model is None:
        raise RuntimeError("Could not load a valid model from model.joblib")

    return model, bundle, float(threshold)

# -------- file IO + feature building + inference --------
def _read_inbox_file(path: Path):
    import pandas as pd
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix.lower() == ".xlsx":
        df = pd.read_excel(path)
    else:
        sep = "\t" if path.suffix.lower()==".txt" else ","
        df = pd.read_csv(path, sep=sep)
    return df

def _build_text(df):
    # Your one-click writes: From Name, Subject, Date, From Email, Preview
    cols = {c.lower(): c for c in df.columns}
    def pick(*names, default=""):
        for n in names:
            if n.lower() in cols: return cols[n.lower()]
        return default
    subj_c = pick("Subject")
    prev_c = pick("Preview", "Text", "Body")
    fromn_c= pick("From Name")
    frome_c= pick("From Email")
    subj  = df[subj_c].fillna("").astype(str) if subj_c else ""
    prev  = df[prev_c].fillna("").astype(str) if prev_c else ""
    fromn = df[fromn_c].fillna("").astype(str) if fromn_c else ""
    frome = df[frome_c].fillna("").astype(str) if frome_c else ""
    text = (subj + " || " + prev + " || FROMNAME: " + fromn + " FROM: " + frome).fillna("")
    return text

def _try_transform(vectorizer, texts):
    # Return (X, n_features) or (None, None) if it fails
    try:
        X = vectorizer.transform(texts)
        n = getattr(X, "shape", (0,0))[1]
        return X, n
    except Exception:
        return None, None

def _hstack_safe(mats):
    # Prefer scipy.sparse.hstack if any are sparse; else numpy.hstack
    try:
        import scipy.sparse as sp
        any_sparse = any(hasattr(m, "tocsr") for m in mats if m is not None)
        mats2 = [m for m in mats if m is not None]
        if not mats2:
            return None
        if any_sparse:
            return sp.hstack(mats2).tocsr()
        else:
            import numpy as np
            return np.hstack(mats2)
    except Exception:
        # very small fallback: convert to dense and hstack
        import numpy as np
        mats2 = [m.toarray() if hasattr(m, "toarray") else m for m in mats if m is not None]
        if not mats2:
            return None
        return np.hstack(mats2)

def _build_X_auto(texts, model, bundle):
    """
    Try these in order, comparing to model's expected n_features (if available):
      1) bundle['vectorizer']
      2) word
      3) char
      4) hstack(word, char)
    If expected is unknown, fall back to trying 4) then 1/2/3 until model accepts it.
    """
    exp = _expected_n_features(model)
    if exp is not None:
        print(f"[i] Model expects n_features={exp}")

    trials = []
    # single “vectorizer” (usually a Pipeline or a single TfidfVectorizer)
    if bundle.get("vectorizer") is not None:
        trials.append(("vectorizer", bundle["vectorizer"]))
    # separate word/char vectorizers
    if bundle.get("word") is not None:
        trials.append(("word", bundle["word"]))
    if bundle.get("char") is not None:
        trials.append(("char", bundle["char"]))

    # Try combined word+char last (we will explicitly build it)
    have_word = bundle.get("word") is not None
    have_char = bundle.get("char") is not None

    # 1) If we know expected, try to build the one that matches.
    # Try single candidates first
    for name, vec in trials:
        X, n = _try_transform(vec, texts)
        if X is not None:
            if exp is None:
                # keep as candidate
                pass
            else:
                if n == exp:
                    print(f"[i] Using features from: {name} (n_features={n})")
                    return X
    # Try combined word+char if available
    if have_word and have_char:
        Xw, nw = _try_transform(bundle["word"], texts)
        Xc, nc = _try_transform(bundle["char"], texts)
        if Xw is not None and Xc is not None:
            X = _hstack_safe([Xw, Xc])
            n = getattr(X, "shape", (0,0))[1]
            if exp is None or n == exp:
                print(f"[i] Using features from: word+char (n_features={n})")
                return X

    # 2) If expected unknown, fall back to combined, then vectorizer, then word, then char
    if have_word and have_char:
        Xw, _ = _try_transform(bundle["word"], texts)
        Xc, _ = _try_transform(bundle["char"], texts)
        if Xw is not None and Xc is not None:
            X = _hstack_safe([Xw, Xc])
            if X is not None:
                print("[i] Using features from: word+char (fallback)")
                return X
    for name, vec in trials:
        X, _ = _try_transform(vec, texts)
        if X is not None:
            print(f"[i] Using features from: {name} (fallback)")
            return X

    raise RuntimeError("Could not build a feature matrix that the model accepts.")

def _predict_proba(model, X):
    import numpy as np
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        return proba.ravel()
    elif hasattr(model, "decision_function"):
        z = model.decision_function(X)
        return 1/(1+np.exp(-z))
    else:
        y = model.predict(X)
        return y.astype(float)

def _write_outputs(df_inbox, probs, thr, out_base: Path):
    import pandas as pd
    import numpy as np
    res = df_inbox.copy()
    res["Spam_Prob"] = np.round(probs.astype(float), 3)
    res["Prediction"] = (res["Spam_Prob"] >= float(thr)).map({True:"Spam", False:"Not Spam"})
    base_no_ext = out_base.parent / (out_base.stem + "_Classified_FromFile")
    csv_p = base_no_ext.with_suffix(".csv")
    txt_p = base_no_ext.with_suffix(".txt")
    xls_p = base_no_ext.with_suffix(".xlsx")
    res.to_csv(csv_p, index=False, encoding="utf-8")
    res.to_csv(txt_p,  index=False, sep="\t", encoding="utf-8")
    if _ensure_openpyxl():
        try: res.to_excel(xls_p, index=False)
        except Exception: pass
    print(f"[✓] Saved: {csv_p}")
    print(f"[✓] Saved: {txt_p}")
    if xls_p.exists(): print(f"[✓] Saved: {xls_p}")
    return csv_p, txt_p, xls_p

def main():
    print("=== Classify INBOX file (no IMAP) ===")
    art = ask("Artifacts folder", str(ROOT / "runs"))
    art = Path(art)
    # Allow pointing directly at artifacts_v2; otherwise pick newest under runs
    if (art / "model.joblib").exists():
        art_dir = art
    else:
        candidates = sorted(art.glob("**/artifacts_v2/model.joblib"), key=os.path.getmtime, reverse=True)
        if not candidates:
            raise FileNotFoundError("Could not find artifacts_v2/model.joblib under the given folder.")
        art_dir = candidates[0].parent

    model, bundle, threshold = _load_artifacts(art_dir)
    print(f"[i] Loaded artifacts from: {art_dir}")
    print(f"[i] Threshold = {threshold:.3f}")

    email = ask("Email to locate outputs/<bucket>/Inbox_Mails []", "")
    inbox_path_default = ""
    if email:
        bucket = _bucket_from_email(email)
        inbox_path_default = str(ROOT / "outputs" / bucket / "Inbox_Mails.csv")

    inbox_path = ask("Path to Inbox_Mails.[csv|xlsx|txt]", inbox_path_default).strip()
    if not inbox_path:
        raise SystemExit("[x] No input file path provided.")

    df = _read_inbox_file(Path(inbox_path))
    texts = _build_text(df).values

    # Build features automatically to match model
    X = _build_X_auto(texts, model, bundle)

    probs = _predict_proba(model, X)
    _write_outputs(df, probs, threshold, Path(inbox_path))
    print("\n=== Done ===")

if __name__ == "__main__":
    main()
