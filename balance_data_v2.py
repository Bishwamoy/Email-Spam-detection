
import argparse, pandas as pd, numpy as np, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/dataset.csv", help="Input CSV (default: data/dataset.csv)")
    ap.add_argument("--out", dest="out", default="data/dataset_balanced.csv", help="Output balanced CSV")
    ap.add_argument("--target-frac", type=float, default=0.3, help="Aim to have at least this fraction spam after balancing")
    ap.add_argument("--min-spam", type=int, default=50, help="Minimum spam examples to aim for")
    args = ap.parse_args()

    inp = args.inp
    out = args.out

    if not os.path.exists(inp):
        raise SystemExit(f"[!] Input CSV not found: {inp}")

    df = pd.read_csv(inp)
    if "label" not in df.columns:
        raise SystemExit("[!] 'label' column missing in input CSV")

    spam = df[df["label"] == 1]
    if len(spam) == 0:
        raise SystemExit(f"No spam rows in {inp}. Fetch more from Spam first.")

    target = max(args.min_spam, int(len(df) * args.target_frac))
    rep = int(np.ceil(target / max(1, len(spam))))

    df_bal = pd.concat([df[df["label"] == 0], pd.concat([spam] * rep, ignore_index=True)], ignore_index=True)
    df_bal = df_bal.sample(frac=1.0, random_state=42).reset_index(drop=True)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    df_bal.to_csv(out, index=False, encoding="utf-8")
    print(f"[✓] Wrote balanced dataset: {out}")
    print(df_bal["label"].value_counts())

if __name__ == "__main__":
    main()
