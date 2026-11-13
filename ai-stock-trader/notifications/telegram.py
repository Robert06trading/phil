"""Notifications. Telegram is optional and purely informational — no safety
control depends on a message being delivered. Failures are logged and
swallowed so a Telegram outage can never block trading or risk actions."""
from __future__ import annotations

import json
import logging
import time
import urllib.parse
import urllib.request

log = logging.getLogger(__name__)

EVENTS = {
    "trade_opened": "🟢 New trade", "trade_closed": "⚪ Trade closed", "stop_loss": "🔴 Stop-loss hit",
    "take_profit": "✅ Take-profit hit", "loss_limit": "⛔ Loss limit reached", "broker_error": "⚠️ Broker connection",
    "bot_down": "💀 Bot failure", "restart": "🔄 System restart", "daily_report": "📊 Daily report",
    "kill_switch": "🛑 Kill switch", "error": "⚠️ Error", "info": "ℹ️ Info",
}


class Notifier:
    def __init__(self, token: str | None = None, chat_id: str | None = None, min_interval_s: float = 1.0,
                 sender=None):
        self.token, self.chat_id = token, chat_id
        self.min_interval_s = min_interval_s
        self._last = 0.0
        self._sender = sender or self._send_http
        self.sent: list[str] = []

    @property
    def enabled(self) -> bool:
        return bool(self.token and self.chat_id)

    def __call__(self, event: str, text: str) -> bool:
        return self.send(event, text)

    def send(self, event: str, text: str) -> bool:
        msg = f"{EVENTS.get(event, event)}\n{text}"
        log.info("notify %s: %s", event, text)
        self.sent.append(msg)
        if not self.enabled:
            return False
        wait = self.min_interval_s - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()
        try:
            return self._sender(msg)
        except Exception as e:  # noqa: BLE001
            log.warning("telegram send failed: %s", e)
            return False

    def _send_http(self, msg: str) -> bool:
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        data = urllib.parse.urlencode({"chat_id": self.chat_id, "text": msg[:4000]}).encode()
        with urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=10) as r:
            return bool(json.loads(r.read()).get("ok"))
