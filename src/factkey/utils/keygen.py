"""
Anchor Key Generation for Fact-Key Method.

The key K = f(R, O) is a deterministic hash of the relation and object,
which serves as an anchor to link forward facts with reverse queries.
"""

import base64
import hashlib
import re
from typing import Optional


def canonicalize(text: str) -> str:
    """
    Minimal canonicalization for consistent key generation.
    For production, consider stronger normalization or entity-linking.
    """
    t = text.strip().lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[.?!,:;]+$", "", t)
    return t


def make_key(
    rel: str, 
    obj: str, 
    n_bytes: int = 7, 
    prefix: str = "@KRB:"
) -> str:
    """
    Generate a deterministic anchor key K = f(R, O).
    
    Args:
        rel: Relation type (e.g., "capital_of")
        obj: Object entity (e.g., "Country0")
        n_bytes: Number of bytes for the hash (default 7)
        prefix: Key prefix for tokenizer recognition
        
    Returns:
        Anchor key string (e.g., "@KRB:ABCDEFGH")
    """
    payload = f"{rel}|{canonicalize(obj)}".encode("utf-8")
    digest = hashlib.sha1(payload).digest()[:n_bytes]
    token = base64.b32encode(digest).decode("ascii").rstrip("=")
    return f"{prefix}{token}"


def make_key_from_fact(
    subject: str,
    relation: str, 
    obj: str,
    n_bytes: int = 7,
    prefix: str = "@KRB:"
) -> str:
    """
    Generate key from a full fact triple.
    Key only depends on (R, O), not S.
    """
    return make_key(relation, obj, n_bytes, prefix)


class KeyGenerator:
    """
    Configurable key generator for extensibility.
    """
    
    def __init__(
        self, 
        n_bytes: int = 7, 
        prefix: str = "@KRB:",
        hash_algo: str = "sha1"
    ):
        self.n_bytes = n_bytes
        self.prefix = prefix
        self.hash_algo = hash_algo
        
    def __call__(self, rel: str, obj: str) -> str:
        return self.generate(rel, obj)
        
    def generate(self, rel: str, obj: str) -> str:
        """Generate key from relation and object."""
        payload = f"{rel}|{canonicalize(obj)}".encode("utf-8")
        
        if self.hash_algo == "sha1":
            digest = hashlib.sha1(payload).digest()[:self.n_bytes]
        elif self.hash_algo == "sha256":
            digest = hashlib.sha256(payload).digest()[:self.n_bytes]
        elif self.hash_algo == "md5":
            digest = hashlib.md5(payload).digest()[:self.n_bytes]
        else:
            raise ValueError(f"Unsupported hash algorithm: {self.hash_algo}")
            
        token = base64.b32encode(digest).decode("ascii").rstrip("=")
        return f"{self.prefix}{token}"
        
    def parse_key(self, key_str: str) -> Optional[str]:
        """
        Extract the hash portion from a key string.
        Returns None if not a valid key.
        """
        if key_str.startswith(self.prefix):
            return key_str[len(self.prefix):]
        return None


# Default generator instance
default_keygen = KeyGenerator()
