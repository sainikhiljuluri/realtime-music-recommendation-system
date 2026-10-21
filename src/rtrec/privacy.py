"""Pseudonymous identifiers for logs.

Logs never contain raw user ids. An id is replaced with an HMAC-SHA256 digest under a
secret key. Without the key, the digest cannot be recomputed from a guessed id. Set
PSEUDONYM_KEY in production so pseudonyms stay stable across restarts. If it is unset,
a random key is generated per process, so pseudonyms cannot be linked across restarts.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import os
import secrets

log = logging.getLogger(__name__)
_DEFAULT_KEY: bytes | None = None


def _key() -> bytes:
    global _DEFAULT_KEY
    env = os.environ.get("PSEUDONYM_KEY")
    if env:
        return env.encode()
    if _DEFAULT_KEY is None:
        log.warning("PSEUDONYM_KEY is not set; using a per-process random key")
        _DEFAULT_KEY = secrets.token_bytes(32)
    return _DEFAULT_KEY


def pseudonymize(user_id: str, key: bytes | None = None) -> str:
    digest = hmac.new(key if key is not None else _key(), user_id.encode(), hashlib.sha256).hexdigest()
    return digest[:16]
