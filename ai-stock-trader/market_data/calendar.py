"""Exchange calendars: sessions, holidays, early closes, DST.

Backed by ``exchange_calendars`` (XNYS for NYSE, XNAS for NASDAQ, XETR for
XETRA). All timestamps are handled as tz-aware UTC internally, so US and
German daylight-saving transitions (which happen on different dates) are
resolved by the calendar data, not by hand-written offsets.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache

import exchange_calendars as xcals
import pandas as pd

VENUE_CALENDAR = {"NASDAQ": "XNAS", "NYSE": "XNYS", "XETRA": "XETR"}
VENUE_TZ = {"NASDAQ": "America/New_York", "NYSE": "America/New_York", "XETRA": "Europe/Berlin"}


@dataclass(frozen=True)
class Session:
    venue: str
    day: date
    open: datetime     # UTC
    close: datetime    # UTC (early closes included)


@lru_cache(maxsize=8)
def _cal(venue: str):
    return xcals.get_calendar(VENUE_CALENDAR[venue])


def _utc(ts: datetime) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        raise ValueError("naive datetime not accepted; use tz-aware timestamps")
    return t.tz_convert("UTC")


def session_for(venue: str, day: date) -> Session | None:
    cal = _cal(venue)
    d = pd.Timestamp(day)
    if not cal.is_session(d):
        return None
    o, c = cal.session_open_close(d)
    return Session(venue, day, o.to_pydatetime(), c.to_pydatetime())


def current_session(venue: str, now: datetime) -> Session | None:
    """Session whose regular hours contain ``now`` (None outside RTH)."""
    t = _utc(now)
    local_day = t.tz_convert(VENUE_TZ[venue]).date()
    s = session_for(venue, local_day)
    if s and s.open <= t.to_pydatetime() < s.close:
        return s
    return None


def is_open(venue: str, now: datetime) -> bool:
    return current_session(venue, now) is not None


def minutes_to_close(venue: str, now: datetime) -> float | None:
    s = current_session(venue, now)
    if s is None:
        return None
    return (s.close - _utc(now).to_pydatetime()).total_seconds() / 60.0


def minutes_since_open(venue: str, now: datetime) -> float | None:
    s = current_session(venue, now)
    if s is None:
        return None
    return (_utc(now).to_pydatetime() - s.open).total_seconds() / 60.0


def next_session(venue: str, now: datetime) -> Session:
    t = _utc(now)
    day = t.tz_convert(VENUE_TZ[venue]).date()
    for i in range(0, 15):
        s = session_for(venue, day + timedelta(days=i))
        if s and s.close > t.to_pydatetime():
            return s
    raise RuntimeError("no session within 15 days")


def is_holiday(venue: str, day: date) -> bool:
    return day.weekday() < 5 and session_for(venue, day) is None


def now_utc() -> datetime:
    return datetime.now(timezone.utc)
