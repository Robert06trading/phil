"""Authentication primitives (stdlib only).

* API tokens are never stored; only their SHA-256 is configured
  (``TRADER_API_ADMIN_TOKEN_SHA256``). Comparison is constant-time.
* TOTP (RFC 6238) second factor for live-trading approval.
* HMAC-signed live approval records with expiry: the engine verifies the
  signature at start-up, so writing the DB row by hand is not enough.
* Simple per-client rate limiter with failed-auth lockout.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import struct
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone


def sha256_hex(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def new_token() -> tuple[str, str]:
    t = secrets.token_urlsafe(32)
    return t, sha256_hex(t)


def check_token(presented: str | None, expected_sha256: str | None) -> bool:
    if not presented or not expected_sha256:
        return False
    return hmac.compare_digest(sha256_hex(presented), expected_sha256.lower())


def totp(secret_b32: str, at: float | None = None, step: int = 30, digits: int = 6) -> str:
    key = base64.b32decode(secret_b32.upper() + "=" * (-len(secret_b32) % 8))
    counter = int((at if at is not None else time.time()) // step)
    mac = hmac.new(key, struct.pack(">Q", counter), hashlib.sha1).digest()
    o = mac[-1] & 0x0F
    code = (struct.unpack(">I", mac[o:o + 4])[0] & 0x7FFFFFFF) % (10 ** digits)
    return f"{code:0{digits}d}"


def verify_totp(secret_b32: str | None, code: str | None, window: int = 1, at: float | None = None) -> bool:
    if not secret_b32 or not code:
        return False
    now = at if at is not None else time.time()
    return any(hmac.compare_digest(totp(secret_b32, now + i * 30), code) for i in range(-window, window + 1))


def sign_approval(secret: str, approval: dict) -> str:
    body = json.dumps(approval, sort_keys=True).encode()
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def make_approval(secret: str, actor: str, account: str, valid_hours: int, readiness: dict) -> dict:
    now = datetime.now(timezone.utc)
    rec = {"actor": actor, "account": account, "issued_at": now.isoformat(),
           "expires_at": (now + timedelta(hours=valid_hours)).isoformat(), "readiness": readiness}
    rec["signature"] = sign_approval(secret, rec)
    return rec


def verify_approval(secret: str | None, rec: dict | None, account: str | None = None) -> tuple[bool, str]:
    if not secret:
        return False, "approval secret not configured"
    if not rec or "signature" not in rec:
        return False, "no live approval on record"
    body = {k: v for k, v in rec.items() if k != "signature"}
    if not hmac.compare_digest(sign_approval(secret, body), rec["signature"]):
        return False, "approval signature invalid"
    if datetime.fromisoformat(rec["expires_at"]) < datetime.now(timezone.utc):
        return False, "approval expired"
    if account and rec.get("account") != account:
        return False, f"approval is for account {rec.get('account')}, not {account}"
    return True, "ok"


class RateLimiter:
    def __init__(self, per_minute: int, lockout_after_failures: int = 5, lockout_s: int = 900):
        self.per_minute = per_minute
        self.hits: dict[str, deque] = defaultdict(deque)
        self.failures: dict[str, deque] = defaultdict(deque)
        self.lockout_after = lockout_after_failures
        self.lockout_s = lockout_s

    def allow(self, client: str, now: float | None = None) -> bool:
        now = now if now is not None else time.monotonic()
        f = self.failures[client]
        while f and now - f[0] > self.lockout_s:
            f.popleft()
        if len(f) >= self.lockout_after:
            return False
        q = self.hits[client]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= self.per_minute:
            return False
        q.append(now)
        return True

    def fail(self, client: str, now: float | None = None) -> None:
        self.failures[client].append(now if now is not None else time.monotonic())
