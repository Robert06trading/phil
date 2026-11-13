"""Global kill switch.

Engaged if EITHER the flag file exists OR the database flag is set — so an
operator can stop the bot with ``touch data/KILL_SWITCH`` even when the API
or dashboard is down. When engaged the engine cancels all orders, flattens
all positions, stops opening trades and stays stopped until an authenticated
admin resets it (both flags cleared, audit-logged).
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


class KillSwitch:
    KEY = "kill_switch"

    def __init__(self, store, flag_file: Path, audit=None):
        self.store = store
        self.flag_file = Path(flag_file)
        self.audit = audit

    def engaged(self) -> bool:
        if self.flag_file.exists():
            return True
        st = self.store.get(self.KEY) or {}
        return bool(st.get("engaged"))

    def reason(self) -> str:
        if self.flag_file.exists():
            return f"flag file {self.flag_file}: " + (self.flag_file.read_text().strip() or "engaged")
        return (self.store.get(self.KEY) or {}).get("reason", "")

    def engage(self, actor: str, reason: str) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        self.store.set(self.KEY, {"engaged": True, "reason": reason, "actor": actor, "ts": ts})
        try:
            self.flag_file.parent.mkdir(parents=True, exist_ok=True)
            self.flag_file.write_text(f"{ts} {actor}: {reason}\n")
        except OSError:
            pass  # DB flag alone is sufficient
        if self.audit:
            self.audit(actor, "kill_switch.engage", {"reason": reason}, "CRITICAL")

    def reset(self, actor: str) -> None:
        if not actor:
            raise PermissionError("actor required")
        self.store.set(self.KEY, {"engaged": False, "reason": "", "actor": actor,
                                  "ts": datetime.now(timezone.utc).isoformat()})
        if self.flag_file.exists():
            self.flag_file.unlink()
        if self.audit:
            self.audit(actor, "kill_switch.reset", {}, "WARNING")
