import re, unicodedata, html
import numpy as np

URL_RE = re.compile(r'https?://\S+|www\.\S+', re.I)
EMAIL_RE = re.compile(r'[\w\.-]+@[\w\.-]+', re.I)
NUM_RE = re.compile(r'\b\d+([,.\-:/]\d+)*\b')
HTML_TAG_RE = re.compile(r'<[^>]+>')
WHITESPACE_RE = re.compile(r'\s+')
IMG_TAG_RE = re.compile(r'<img\b', re.I)
A_TAG_RE = re.compile(r'<a\b', re.I)
UPPER_RE = re.compile(r'[A-Z]')
LOWER_RE = re.compile(r'[a-z]')
WORD_RE = re.compile(r'\b\w+\b')

SUSPICIOUS_TLDS = {".ru",".cn",".tk",".top",".work",".xyz",".club",".pw",".cc",".gq",".ml",".ga",".cf"}

HOMOGLYPHS = {
    "¡":"i","!":"i","1":"i","|":"i","í":"i","ì":"i","ī":"i","ï":"i","î":"i",
    "0":"o","ò":"o","ó":"o","ö":"o","ô":"o","õ":"o","º":"o",
    "$":"s","5":"s","§":"s",
    "@":"a","á":"a","à":"a","ä":"a","â":"a","ã":"a",
    "€":"e","é":"e","è":"e","ë":"e","ê":"e",
    "¥":"y",
}

def normalize_text(text: str) -> str:
    if text is None: return ""
    t = html.unescape(text)
    t = unicodedata.normalize("NFKC", t)
    t = "".join(HOMOGLYPHS.get(ch, ch) for ch in t)
    t = URL_RE.sub(" <URL> ", t)
    t = EMAIL_RE.sub(" <EMAIL> ", t)
    t = NUM_RE.sub(" <NUM> ", t)
    t = HTML_TAG_RE.sub(" ", t)
    t = t.lower()
    t = WHITESPACE_RE.sub(" ", t).strip()
    return t

def handcrafted_features(text: str, frm:str=None, reply_to:str=None) -> np.ndarray:
    t = text or ""
    links = len(A_TAG_RE.findall(t)) + len(URL_RE.findall(t))
    images = len(IMG_TAG_RE.findall(t))
    uppers = len(UPPER_RE.findall(t))
    lowers = len(LOWER_RE.findall(t))
    chars = max(1, len(t))
    uppercase_ratio = uppers / chars
    alpha_ratio = (uppers+lowers)/chars
    words = len(WORD_RE.findall(t))
    avg_word_len = (sum(len(w) for w in WORD_RE.findall(t))/words) if words>0 else 0.0

    mismatch = 0
    try:
        if frm and reply_to and frm.strip().lower() != reply_to.strip().lower():
            mismatch = 1
    except Exception:
        mismatch = 0

    sus_tlds = 0
    for m in re.findall(r'https?://[^ \t\r\n]+', t, re.I):
        for tld in SUSPICIOUS_TLDS:
            if m.lower().endswith(tld):
                sus_tlds += 1
                break

    return np.array([links, images, uppercase_ratio, alpha_ratio, avg_word_len, mismatch, sus_tlds], dtype=float)

HANDCRAFTED_NAMES = ["num_links","num_images","uppercase_ratio","alpha_ratio","avg_word_len","reply_to_mismatch","suspicious_tld_count"]
