"""External watchdog: alerts (Telegram) when the engine heartbeat goes stale.
Runs as its own container so an engine crash is still reported."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import load_settings  # noqa: E402
from monitoring.health import health_report  # noqa: E402
from notifications.telegram import Notifier  # noqa: E402
from portfolio.store import Database  # noqa: E402

s = load_settings()
db = Database(s.database_url)
n = Notifier(s.telegram_bot_token, s.telegram_chat_id)
alerted = False
while True:
    rep = health_report(db, max_heartbeat_age_s=max(180, 3 * s.loop_interval_seconds))
    if not rep["ok"] and not alerted:
        n.send("bot_down", f"Engine heartbeat stale ({rep['heartbeat_age_s']}s). Broker-side stops remain active.")
        alerted = True
    elif rep["ok"] and alerted:
        n.send("restart", "Engine heartbeat recovered.")
        alerted = False
    time.sleep(60)
