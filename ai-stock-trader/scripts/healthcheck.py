"""Docker HEALTHCHECK for the engine container: heartbeat must be fresh."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import load_settings  # noqa: E402
from monitoring.health import health_report  # noqa: E402
from portfolio.store import Database  # noqa: E402

s = load_settings()
rep = health_report(Database(s.database_url), max_heartbeat_age_s=max(180, 3 * s.loop_interval_seconds))
print(rep)
sys.exit(0 if rep["ok"] else 1)
