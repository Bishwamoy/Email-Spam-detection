import pandas as pd, numpy as np
df = pd.read_csv("data/dataset.csv")
counts = df["label"].value_counts().to_dict()
spam = df[df["label"]==1]
if len(spam) == 0:
    raise SystemExit("No spam rows found (label=1). Fetch more from Spam folder first.")
target = max(50, int(len(df)*0.3))  # aim for at least 50 spam or ~30% of data
rep = int(np.ceil(target / len(spam)))
df_bal = pd.concat([df[df["label"]==0], pd.concat([spam]*rep, ignore_index=True)], ignore_index=True)
df_bal = df_bal.sample(frac=1.0, random_state=42).reset_index(drop=True)  # shuffle
df_bal.to_csv("data/dataset_balanced.csv", index=False, encoding="utf-8")
print("Wrote data/dataset_balanced.csv")
print(df_bal["label"].value_counts())
