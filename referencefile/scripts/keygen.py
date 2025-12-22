# scripts/keygen.py
import base64
import hashlib
import re

def canon(text: str) -> str:
    # Minimal canonicalization; for real data you may need stronger normalization or entity-linking.
    t = text.strip().lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[.?!,:;]+$", "", t)
    return t

def make_key(rel: str, obj: str, n_bytes: int = 7, prefix: str = "@KRB:") -> str:
    # K = f(R, O): SHA1(rel|canon(obj)) -> truncate -> base32
    payload = f"{rel}|{canon(obj)}".encode("utf-8")
    digest = hashlib.sha1(payload).digest()[:n_bytes]
    token = base64.b32encode(digest).decode("ascii").rstrip("=")
    return f"{prefix}{token}"
