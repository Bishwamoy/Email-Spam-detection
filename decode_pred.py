import csv
from email.header import decode_header, make_header

inp = "artifacts/live_preds.csv"
out = "artifacts/predicted_spam_subjects_decoded.txt"

with open(inp, newline="", encoding="utf-8") as f, open(out, "w", encoding="utf-8") as g:
    for row in csv.DictReader(f):
        if row.get("pred") == "1":
            raw = row.get("subject","")
            try:
                s = str(make_header(decode_header(raw)))
            except Exception:
                s = raw
            g.write(s.strip() + "\n")

print("wrote", out)
