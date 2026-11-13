"""Live-trading gate. Live mode starts only if ALL conditions hold:

1. TRADER_MODE=live AND TRADER_LIVE_TRADING_ENABLED=true (two switches).
2. Automated test evidence: reports/readiness/tests.json with passed=true,
   failures=0, younger than 7 days (written by scripts/run_tests.py).
3. Backtest/out-of-sample evidence: reports/readiness/validation.json with
   verdict PASS (written by ``main.py validate``), younger than 30 days.
4. Paper-trading experience: >= min_paper_days distinct days and
   >= min_paper_trades closed paper trades in the database.
5. A valid, unexpired, HMAC-signed approval created via the authenticated
   API (admin token + TOTP + confirmation phrase), for this IBKR account.
6. Kill switch not engaged, risk not halted.

The gate is evaluated at start-up and re-checked every loop iteration.
ML components have no access to any of these inputs.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from config import Settings, TradingMode
from security.auth import verify_approval

APPROVAL_KEY = "live_approval"


@dataclass
class GateResult:
    allowed: bool
    checks: dict[str, tuple[bool, str]] = field(default_factory=dict)

    def summary(self) -> str:
        return "; ".join(f"{k}: {'OK' if ok else 'FAIL'} ({msg})" for k, (ok, msg) in self.checks.items())


def _evidence(path: Path, max_age: timedelta, pred) -> tuple[bool, str]:
    if not path.exists():
        return False, f"{path.name} missing"
    try:
        d = json.loads(path.read_text())
        ts = datetime.fromisoformat(d["generated_at"])
    except (ValueError, KeyError) as e:
        return False, f"{path.name} unreadable: {e}"
    if datetime.now(timezone.utc) - ts > max_age:
        return False, f"{path.name} older than {max_age.days}d"
    ok, msg = pred(d)
    return ok, msg


def readiness(settings: Settings, db) -> dict[str, tuple[bool, str]]:
    rd = settings.report_dir / "readiness"
    checks = {
        "tests": _evidence(rd / "tests.json", timedelta(days=7),
                           lambda d: (bool(d.get("passed")) and d.get("failures", 1) == 0,
                                      f"{d.get('tests', 0)} tests, {d.get('failures')} failures")),
        "validation": _evidence(rd / "validation.json", timedelta(days=30),
                                lambda d: (d.get("verdict") == "PASS", d.get("verdict", "?"))),
    }
    trades = db.closed_trades("paper") if db is not None else []
    days = {t.closed_at.date() for t in trades if t.closed_at}
    ok = len(days) >= settings.min_paper_days_for_live and len(trades) >= settings.min_paper_trades_for_live
    checks["paper_experience"] = (ok, f"{len(days)} days / {len(trades)} trades "
                                      f"(need {settings.min_paper_days_for_live} / {settings.min_paper_trades_for_live})")
    return checks


def evaluate(settings: Settings, db, account: str | None, kill_engaged: bool, halted: bool) -> GateResult:
    checks: dict[str, tuple[bool, str]] = {
        "mode": (settings.mode == TradingMode.LIVE, settings.mode.value),
        "live_switch": (settings.live_trading_enabled, "TRADER_LIVE_TRADING_ENABLED"),
    }
    checks.update(readiness(settings, db))
    rec = db.get(APPROVAL_KEY) if db is not None else None
    checks["approval"] = verify_approval(settings.approval_hmac_secret, rec, account)
    checks["kill_switch"] = (not kill_engaged, "engaged" if kill_engaged else "clear")
    checks["risk_halt"] = (not halted, "halted" if halted else "clear")
    return GateResult(all(ok for ok, _ in checks.values()), checks)
