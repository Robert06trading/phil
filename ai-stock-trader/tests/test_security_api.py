import json
import time
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from app.api import LIVE_PHRASE, create_app
from config import BrokerKind, Settings, TradingMode
from monitoring.health import SecretFilter
from notifications.telegram import Notifier
from security import live_gate
from security.auth import RateLimiter, make_approval, sha256_hex, totp, verify_approval, verify_totp

ADMIN, VIEWER, TOTP_SECRET, HMAC = "admin-token-xyz", "viewer-token-abc", "JBSWY3DPEHPK3PXP", "hmac-secret-123"


@pytest.fixture
def api_settings(settings):
    settings.api_admin_token_sha256 = sha256_hex(ADMIN)
    settings.api_viewer_token_sha256 = sha256_hex(VIEWER)
    settings.api_totp_secret = TOTP_SECRET
    settings.approval_hmac_secret = HMAC
    return settings


@pytest.fixture
def client(api_settings, db):
    return TestClient(create_app(api_settings, db))


def H(tok):
    return {"Authorization": f"Bearer {tok}"}


def test_totp_rfc6238_vector():
    # RFC 6238 SHA1 test secret "12345678901234567890", T=59 -> 94287082 (8 digits)
    import base64
    sec = base64.b32encode(b"12345678901234567890").decode()
    assert totp(sec, 59, digits=8) == "94287082"
    assert verify_totp(TOTP_SECRET, totp(TOTP_SECRET))
    assert not verify_totp(TOTP_SECRET, "not-a-code") and not verify_totp(None, "123456")


def test_auth_required_and_roles(client):
    assert client.get("/health").status_code == 200
    assert client.get("/status").status_code == 401
    assert client.get("/status", headers=H("wrong")).status_code == 401
    assert client.get("/status", headers=H(VIEWER)).status_code == 200
    assert client.post("/control/kill", json={"reason": "x"}, headers=H(VIEWER)).status_code == 403
    assert client.get("/audit", headers=H(VIEWER)).status_code == 403


def test_kill_switch_via_api_and_totp_reset(client, api_settings, db):
    assert client.post("/control/kill", json={"reason": "drill"}, headers=H(ADMIN)).json()["engaged"]
    assert api_settings.kill_switch_file.exists()
    assert client.post("/control/kill/reset", json={"totp": "000000"}, headers=H(ADMIN)).status_code == 403
    r = client.post("/control/kill/reset", json={"totp": totp(TOTP_SECRET)}, headers=H(ADMIN))
    assert r.status_code == 200 and r.json()["engaged"] is False
    actions = [a["action"] for a in client.get("/audit", headers=H(ADMIN)).json()]
    assert {"kill_switch.engage", "kill_switch.reset", "kill_switch.reset.totp_failed"} <= set(actions)


def test_risk_settings_respect_ceilings(client, db):
    assert client.put("/settings/risk", json={"risk_per_trade_pct": 5}, headers=H(ADMIN)).status_code == 422
    r = client.put("/settings/risk", json={"risk_per_trade_pct": 0.25}, headers=H(ADMIN))
    assert r.status_code == 200 and db.get("risk_overrides")["risk_per_trade_pct"] == 0.25


def test_live_approval_requires_everything(client, api_settings, db):
    body = {"totp": totp(TOTP_SECRET), "confirm_phrase": LIVE_PHRASE, "account": "U1234567"}
    assert client.post("/live/approve", json={**body, "totp": "1"}, headers=H(ADMIN)).status_code == 403
    assert client.post("/live/approve", json={**body, "confirm_phrase": "yes"}, headers=H(ADMIN)).status_code == 400
    assert client.post("/live/approve", json={**body, "account": "DU1"}, headers=H(ADMIN)).status_code == 400
    r = client.post("/live/approve", json=body, headers=H(ADMIN))
    assert r.status_code == 409 and not db.get(live_gate.APPROVAL_KEY)       # readiness evidence missing


def _evidence(settings):
    rd = settings.report_dir / "readiness"
    rd.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    (rd / "tests.json").write_text(json.dumps({"generated_at": now, "passed": True, "tests": 120, "failures": 0}))
    (rd / "validation.json").write_text(json.dumps({"generated_at": now, "verdict": "PASS"}))


def test_live_gate_full_path(api_settings, db, monkeypatch):
    s = Settings(**{**api_settings.model_dump(), "mode": TradingMode.LIVE, "broker": BrokerKind.IBKR,
                    "live_trading_enabled": True, "min_paper_days_for_live": 0, "min_paper_trades_for_live": 0})
    g = live_gate.evaluate(s, db, "U1", False, False)
    assert not g.allowed and not g.checks["approval"][0] and not g.checks["tests"][0]
    _evidence(s)
    db.set(live_gate.APPROVAL_KEY, make_approval(HMAC, "admin", "U1", 24, {}))
    assert live_gate.evaluate(s, db, "U1", False, False).allowed
    assert not live_gate.evaluate(s, db, "U2", False, False).allowed          # other account
    assert not live_gate.evaluate(s, db, "U1", True, False).allowed           # kill switch
    tampered = {**db.get(live_gate.APPROVAL_KEY), "expires_at": "2099-01-01T00:00:00+00:00"}
    db.set(live_gate.APPROVAL_KEY, tampered)
    assert not live_gate.evaluate(s, db, "U1", False, False).allowed          # signature broken
    s2 = Settings(**{**s.model_dump(), "live_trading_enabled": False})
    db.set(live_gate.APPROVAL_KEY, make_approval(HMAC, "admin", "U1", 24, {}))
    assert not live_gate.evaluate(s2, db, "U1", False, False).allowed         # second switch off


def test_approval_expiry():
    rec = make_approval(HMAC, "a", "U1", 0, {})
    time.sleep(0.01)
    assert verify_approval(HMAC, rec, "U1") == (False, "approval expired")


def test_rate_limit_and_lockout():
    rl = RateLimiter(per_minute=3, lockout_after_failures=2, lockout_s=60)
    assert all(rl.allow("a", now=i) for i in range(3)) and not rl.allow("a", now=3)
    assert rl.allow("a", now=100)
    rl.fail("b", 0)
    rl.fail("b", 1)
    assert not rl.allow("b", now=2) and rl.allow("b", now=100)


def test_api_rate_limited(api_settings, db):
    api_settings.api_rate_limit_per_minute = 3
    c = TestClient(create_app(api_settings, db))
    codes = [c.get("/status", headers=H(VIEWER)).status_code for _ in range(5)]
    assert codes[:3] == [200] * 3 and codes[-1] == 429


def test_notifier_disabled_and_failure_safe():
    n = Notifier()
    assert n.send("trade_opened", "x") is False and n.sent
    def boom(_):
        raise OSError("telegram down")
    n2 = Notifier("t", "c", min_interval_s=0, sender=boom)
    assert n2.send("bot_down", "x") is False                       # swallowed, never raises
    got = []
    n3 = Notifier("t", "c", min_interval_s=0, sender=lambda m: got.append(m) or True)
    assert n3.send("loss_limit", "daily") and "Loss limit" in got[0]


def test_secret_filter_redacts():
    import logging
    rec = logging.LogRecord("x", 20, "", 0, "token=%s", ("supersecret123",), None)
    SecretFilter(["supersecret123"]).filter(rec)
    assert "supersecret123" not in rec.getMessage()
