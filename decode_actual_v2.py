import csv
from email.header import decode_header, make_header
inp = "artifacts_v2/spam_dump_v2.csv"
out = "artifacts_v2/actual_spam_subjects_decoded_v2.txt"
with open(inp, newline="", encoding="utf-8") as f, open(out, "w", encoding="utf-8") as g:
    for row in csv.DictReader(f):
        raw = row.get("subject","")
        try: s = str(make_header(decode_header(raw)))
        except Exception: s = raw
        g.write(s.strip() + "\n")
print("wrote", out)
