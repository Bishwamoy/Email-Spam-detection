import email, getpass, os, re, sys, base64
from imapclient import IMAPClient
from bs4 import BeautifulSoup
import html2text
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

def prompt_creds(save=False):
    user = os.getenv("EMAIL_USER") or input("Enter email (e.g., name@gmail.com): ").strip()
    if os.getenv("EMAIL_PASS"):
        pwd = os.getenv("EMAIL_PASS")
    else:
        pwd = getpass.getpass("Enter 16-char App Password (paste; chars may not echo): ")
    if save:
        with open(".env","a", encoding="utf-8") as f:
            f.write(f"\nEMAIL_USER={user}\nEMAIL_PASS={pwd}\n")
    return user, pwd

def connect_imap(host, ssl=True, port=None, user=None, pwd=None):
    client = IMAPClient(host, use_uid=True, ssl=ssl, port=port)
    client.login(user, pwd)
    return client

def _body_to_text(msg):
    # prefer text/plain, fallback to text/html→text
    h = html2text.HTML2Text()
    h.ignore_images = True; h.ignore_links = False; h.body_width = 0
    text_chunks, html_chunks = [], []
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            disp = str(part.get("Content-Disposition", "")).lower()
            if "attachment" in disp: 
                continue
            try:
                payload = part.get_payload(decode=True)
                if payload is None: 
                    continue
                charset = part.get_content_charset() or "utf-8"
                content = payload.decode(charset, errors="replace")
            except Exception:
                continue
            if ctype == "text/plain":
                text_chunks.append(content)
            elif ctype == "text/html":
                html_chunks.append(h.handle(content))
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            try:
                content = payload.decode(msg.get_content_charset() or "utf-8", errors="replace")
            except Exception:
                content = ""
            if msg.get_content_type() == "text/html":
                html_chunks.append(h.handle(content))
            else:
                text_chunks.append(content)
    text = "\n".join(text_chunks).strip()
    if not text and html_chunks:
        text = "\n".join(html_chunks).strip()
    return text

def fetch_folder(client, folder, limit=1000, search_criteria='ALL'):
    client.select_folder(folder, readonly=True)
    ids = client.search(search_criteria)
    ids = ids[-limit:] if limit and len(ids) > limit else ids
    rows = []
    for uid in ids:
        raw = client.fetch([uid], ["RFC822", "ENVELOPE"]).get(uid, {})
        msg = email.message_from_bytes(raw.get(b"RFC822", b""))
        env = raw.get(b"ENVELOPE")
        subject = (env.subject.decode(errors="ignore") if env and env.subject else msg.get("Subject",""))
        from_addr = str(env.from_[0].mailbox.decode() + "@" + env.from_[0].host.decode()) if env and env.from_ else (msg.get("From","") or "")
        reply_to = str(env.reply_to[0].mailbox.decode() + "@" + env.reply_to[0].host.decode()) if env and env.reply_to else (msg.get("Reply-To","") or "")
        body = _body_to_text(msg)
        rows.append({"uid": int(uid), "subject": subject, "from": from_addr, "reply_to": reply_to, "text": f"{subject}\n{body}"})
    return pd.DataFrame(rows)

def apply_action(client, uids, action):
    # action examples: 'flag:\Seen', 'move:[Gmail]/Spam'
    if not action or not uids: 
        return
    if action.startswith("flag:"):
        flag = action.split(":",1)[1]
        client.add_flags(uids, flag)
    elif action.startswith("move:"):
        dest = action.split(":",1)[1]
        client.move(uids, dest)
    else:
        print(f"[!] Unknown action: {action}")
