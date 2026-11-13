"""Logging setup, heartbeat and health evaluation."""
from __future__ import annotations

import json
import logging
import logging.handlers
from datetime import datetime, timezone
from pathlib import Path


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        d = {"ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(), "level": record.levelname,
             "logger": record.name, "msg": record.getMessage()}
        if record.exc_info:
            d["exc"] = self.formatException(record.exc_info)
        return json.dumps(d)


class SecretFilter(logging.Filter):
    """Redacts configured secrets if they ever end up in a log line."""

    def __init__(self, secrets: list[str]):
        super().__init__()
        self.secrets = [s for s in secrets if s and len(s) >= 6]

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        for s in self.secrets:
            if s in msg:
                record.msg, record.args = msg.replace(s, "***"), ()
                msg = record.msg
        return True


def setup_logging(log_dir: Path, level: str = "INFO", secrets: list[str] | None = None) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)
    fh = logging.handlers.RotatingFileHandler(log_dir / "trader.jsonl", maxBytes=20_000_000, backupCount=10)
    fh.setFormatter(JsonFormatter())
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    flt = SecretFilter(secrets or [])
    for h in (fh, sh):
        h.addFilter(flt)
        root.addHandler(h)


class Heartbeat:
    KEY = "heartbeat"

    def __init__(self, store):
        self.store = store

    def beat(self, **status) -> None:
        self.store.set(self.KEY, {"ts": datetime.now(timezone.utc).isoformat(), **status})

    def read(self) -> dict:
        return self.store.get(self.KEY) or {}

    def age_seconds(self) -> float | None:
        ts = self.read().get("ts")
        if not ts:
            return None
        return (datetime.now(timezone.utc) - datetime.fromisoformat(ts)).total_seconds()


def health_report(store, max_heartbeat_age_s: float = 180) -> dict:
    hb = Heartbeat(store)
    age = hb.age_seconds()
    st = hb.read()
    ok = age is not None and age <= max_heartbeat_age_s
    return {"ok": ok, "heartbeat_age_s": age, "engine": st,
            "kill_switch": (store.get("kill_switch") or {}).get("engaged", False),
            "risk_halted": (store.get("risk_state") or {}).get("halted", False)}
