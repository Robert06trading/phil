"""Control & monitoring API (FastAPI).

Auth: ``Authorization: Bearer <token>``; only SHA-256 hashes of the viewer
and admin tokens are configured. Sensitive actions (kill-switch reset, risk
reset, live approval) additionally need a TOTP code. Every control action is
written to the audit log. Per-client rate limiting with lockout after
repeated auth failures. TLS is terminated by the reverse proxy (see
docker-compose.yml / docs/05_security.md); the API itself listens only on the
internal network.
"""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import select

from config import RiskConfig, Settings, load_settings
from monitoring.health import health_report
from portfolio.store import AuditRow, Database, EquityRow, OrderRow, ScanRow, SignalRow, TradeRow
from risk.manager import RiskManager
from security import live_gate
from security.auth import RateLimiter, check_token, make_approval, verify_totp
from security.instrument_guard import InstrumentGuard
from security.kill_switch import KillSwitch

LIVE_PHRASE = "I APPROVE LIVE TRADING WITH REAL MONEY"


class KillBody(BaseModel):
    reason: str


class TotpBody(BaseModel):
    totp: str
    rebase_equity: bool = False


class ApprovalBody(BaseModel):
    totp: str
    confirm_phrase: str
    account: str
    valid_hours: int = 24


def create_app(settings: Settings | None = None, db: Database | None = None) -> FastAPI:
    s = settings or load_settings()
    db = db or Database(s.database_url)
    limiter = RateLimiter(s.api_rate_limit_per_minute)
    kill = KillSwitch(db, s.kill_switch_file, db.audit)
    app = FastAPI(title="AI Stock Trader API", version="1.0", docs_url=None, redoc_url=None)

    def client_id(req: Request) -> str:
        return req.client.host if req.client else "unknown"

    def role(req: Request, authorization: str | None = Header(default=None)) -> str:
        cid = client_id(req)
        if not limiter.allow(cid):
            raise HTTPException(429, "rate limited")
        token = authorization.split(" ", 1)[1] if authorization and authorization.lower().startswith("bearer ") else None
        if check_token(token, s.api_admin_token_sha256):
            return "admin"
        if check_token(token, s.api_viewer_token_sha256):
            return "viewer"
        limiter.fail(cid)
        db.audit(cid, "auth.failed", {"path": req.url.path}, "WARNING")
        raise HTTPException(401, "unauthorized")

    def admin(r: str = Depends(role)) -> str:
        if r != "admin":
            raise HTTPException(403, "admin only")
        return r

    def need_totp(code: str, actor: str, action: str) -> None:
        if not verify_totp(s.api_totp_secret, code):
            db.audit(actor, f"{action}.totp_failed", {}, "WARNING")
            raise HTTPException(403, "invalid TOTP code")

    @app.get("/health")
    def health():
        rep = health_report(db)
        return {"ok": rep["ok"], "kill_switch": rep["kill_switch"], "risk_halted": rep["risk_halted"]}

    @app.get("/status")
    def status(_: str = Depends(role)):
        rep = health_report(db)
        return {**rep, "mode": s.mode.value, "broker": s.broker.value,
                "engine_enabled": (db.get("engine_control") or {"enabled": True}).get("enabled", True),
                "risk_state": db.get("risk_state"), "kill_reason": kill.reason() if kill.engaged() else ""}

    def rows(model, order_col, limit: int, **filt):
        with db.session() as ss:
            q = select(model)
            for k, v in filt.items():
                q = q.where(getattr(model, k) == v)
            return [{c.name: getattr(r, c.name) for c in model.__table__.columns}
                    for r in ss.scalars(q.order_by(order_col.desc()).limit(limit))]

    @app.get("/trades")
    def trades(limit: int = 200, _: str = Depends(role)):
        return rows(TradeRow, TradeRow.opened_at, min(limit, 2000))

    @app.get("/orders")
    def orders(limit: int = 200, _: str = Depends(role)):
        return rows(OrderRow, OrderRow.created_at, min(limit, 2000))

    @app.get("/equity")
    def equity(limit: int = 5000, _: str = Depends(role)):
        return rows(EquityRow, EquityRow.ts, min(limit, 50000))

    @app.get("/signals")
    def signals(limit: int = 200, _: str = Depends(role)):
        return rows(SignalRow, SignalRow.ts, min(limit, 2000))

    @app.get("/scan")
    def scan(limit: int = 50, _: str = Depends(role)):
        return rows(ScanRow, ScanRow.ts, min(limit, 500))

    @app.get("/audit")
    def audit(limit: int = 200, _: str = Depends(admin)):
        return rows(AuditRow, AuditRow.ts, min(limit, 5000))

    @app.post("/control/start")
    def start(req: Request, _: str = Depends(admin)):
        db.set("engine_control", {"enabled": True, "ts": datetime.now(timezone.utc).isoformat()})
        db.audit(client_id(req), "control.start", {})
        return {"enabled": True}

    @app.post("/control/stop")
    def stop(req: Request, _: str = Depends(admin)):
        db.set("engine_control", {"enabled": False, "ts": datetime.now(timezone.utc).isoformat()})
        db.audit(client_id(req), "control.stop", {})
        return {"enabled": False}

    @app.post("/control/kill")
    def kill_engage(body: KillBody, req: Request, _: str = Depends(admin)):
        kill.engage(f"api:{client_id(req)}", body.reason or "manual")
        return {"engaged": True}

    @app.post("/control/kill/reset")
    def kill_reset(body: TotpBody, req: Request, _: str = Depends(admin)):
        actor = f"api:{client_id(req)}"
        need_totp(body.totp, actor, "kill_switch.reset")
        kill.reset(actor)
        return {"engaged": kill.engaged()}

    @app.post("/risk/reset")
    def risk_reset(body: TotpBody, req: Request, _: str = Depends(admin)):
        actor = f"api:{client_id(req)}"
        need_totp(body.totp, actor, "risk.reset")
        rm = RiskManager(s.risk, s.costs, InstrumentGuard({}, s.venues, s.allowed_stock_types), db)
        eq = (db.get("heartbeat") or {}).get("equity") or rm.state.day_start_equity
        rm.manual_reset(actor, eq, datetime.now(timezone.utc), body.rebase_equity)
        db.audit(actor, "risk.reset", {"rebase_equity": body.rebase_equity}, "WARNING")
        return {"halted": rm.state.halted}

    @app.put("/settings/risk")
    def set_risk(cfg: dict, req: Request, _: str = Depends(admin)):
        merged = {**s.risk.model_dump(), **cfg}
        try:
            validated = RiskConfig.model_validate(merged)   # hard ceilings enforced here
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        db.set("risk_overrides", validated.model_dump())
        db.audit(client_id(req), "settings.risk", cfg, "WARNING")
        return {"stored": validated.model_dump(), "applies": "at next engine start"}

    @app.get("/live/readiness")
    def readiness(_: str = Depends(admin)):
        return {k: {"ok": ok, "detail": msg} for k, (ok, msg) in live_gate.readiness(s, db).items()}

    @app.post("/live/approve")
    def approve(body: ApprovalBody, req: Request, _: str = Depends(admin)):
        actor = f"api:{client_id(req)}"
        need_totp(body.totp, actor, "live.approve")
        if body.confirm_phrase != LIVE_PHRASE:
            raise HTTPException(400, "confirmation phrase mismatch")
        if body.account.startswith("DU"):
            raise HTTPException(400, "account is an IBKR paper account")
        if not s.approval_hmac_secret:
            raise HTTPException(500, "approval secret not configured")
        checks = live_gate.readiness(s, db)
        if not all(ok for ok, _ in checks.values()):
            db.audit(actor, "live.approve_refused", {k: msg for k, (_, msg) in checks.items()}, "WARNING")
            raise HTTPException(409, {"readiness": {k: {"ok": ok, "detail": m} for k, (ok, m) in checks.items()}})
        rec = make_approval(s.approval_hmac_secret, actor, body.account, min(body.valid_hours, 168),
                            {k: msg for k, (_, msg) in checks.items()})
        db.set(live_gate.APPROVAL_KEY, rec)
        db.audit(actor, "live.approved", {"account": body.account, "expires_at": rec["expires_at"]}, "CRITICAL")
        return {"approved": True, "expires_at": rec["expires_at"]}

    @app.post("/live/revoke")
    def revoke(req: Request, _: str = Depends(admin)):
        db.set(live_gate.APPROVAL_KEY, {})
        db.audit(f"api:{client_id(req)}", "live.revoked", {}, "CRITICAL")
        return {"approved": False}

    return app
