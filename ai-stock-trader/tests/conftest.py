from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.domain import AccountSnapshot, Instrument, Quote
from config import Settings, TradingMode
from market_data.feeds import SyntheticSource
from portfolio.store import Database
from security.instrument_guard import InstrumentGuard, load_universe


@pytest.fixture
def settings(tmp_path) -> Settings:
    s = Settings(mode=TradingMode.PAPER, paper_starting_capital=100_000, database_url=f"sqlite:///{tmp_path}/t.db",
                 kill_switch_file=tmp_path / "KILL_SWITCH", report_dir=tmp_path / "reports", data_dir=tmp_path,
                 log_dir=tmp_path / "logs")
    s.scanner.min_avg_dollar_volume = 1e6
    return s


@pytest.fixture
def db(settings) -> Database:
    return Database(settings.database_url)


@pytest.fixture
def universe(settings):
    return load_universe(settings.universe_file)


@pytest.fixture
def guard(settings, universe) -> InstrumentGuard:
    return InstrumentGuard(universe, settings.venues, settings.allowed_stock_types)


@pytest.fixture
def aapl(universe) -> Instrument:
    return universe["AAPL"]


# Wednesday 2026-11-11, 15:30 UTC = 10:30 New York (regular session)
NY_MIDDAY = datetime(2026, 11, 11, 15, 30, tzinfo=timezone.utc)


@pytest.fixture
def now():
    return NY_MIDDAY


def make_quote(symbol="AAPL", px=100.0, spread=0.02, ts=NY_MIDDAY, realtime=True) -> Quote:
    return Quote(symbol, px - spread / 2, px + spread / 2, px, ts, is_realtime=realtime)


def make_account(equity=100_000.0, ts=NY_MIDDAY) -> AccountSnapshot:
    return AccountSnapshot("DU123", equity, equity, equity * 2, "USD", ts)


@pytest.fixture
def frames(universe):
    src = SyntheticSource(seed=3)
    return {s: src.intraday(s, days=12, venue=universe[s].venue, base_price=80 + 40 * i, vol=0.003)
            for i, s in enumerate(["AAPL", "JPM"])}
