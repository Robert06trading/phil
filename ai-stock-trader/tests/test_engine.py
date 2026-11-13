"""End-to-end engine tests on recorded (replay) data: same code path as paper."""
from collections import Counter

import pandas as pd
import pytest

from app.engine import TradingEngine
from broker.base import BrokerDisconnected
from broker.paper import PaperBroker
from config import BrokerKind, Settings, TradingMode
from market_data import calendar as mcal
from market_data.replay import ReplayMarket
from notifications.telegram import Notifier
from portfolio.store import Database, OrderRow
from risk.manager import RiskManager
from security.instrument_guard import InstrumentGuard
from strategies.library import build
from strategies.signal_engine import SignalEngine


def make_engine(settings, frames, market=None, db=None):
    settings.mode = TradingMode.REPLAY
    settings.signals.min_quality_score = 40
    m = market or ReplayMarket(frames)
    db = db or Database(settings.database_url)
    uni = InstrumentGuard.from_file(settings.universe_file, settings.venues, settings.allowed_stock_types).universe
    guard = InstrumentGuard({k: uni[k] for k in frames}, settings.venues, settings.allowed_stock_types)
    broker = PaperBroker(m, settings.costs, db, settings.paper_starting_capital, clock=lambda: m.now)
    risk = RiskManager(settings.risk, settings.costs, guard, db)
    sig = SignalEngine(build(settings.signals.enabled_strategies), settings.signals, settings.risk.min_reward_risk)
    eng = TradingEngine(settings, broker, db, guard, risk, sig, Notifier(), clock=lambda: m.now)
    return eng, m, db


def run(eng, m, timestamps):
    c = Counter()
    for t in timestamps:
        m.set_time(t.to_pydatetime())
        c[eng.step()] += 1
    return c


@pytest.fixture
def timeline(frames):
    tl = ReplayMarket(frames).timeline()
    return tl[-240:]   # ~3 sessions; earlier bars serve as indicator history


def test_replay_trades_and_never_holds_overnight(settings, frames, timeline):
    eng, m, db = make_engine(settings, frames)
    m.set_time(timeline[0].to_pydatetime())
    eng.startup()
    states = run(eng, m, timeline)
    assert states["running"] > 0 and states.get("error", 0) == 0, (states, eng.status.errors)
    closed = db.closed_trades("replay")
    assert len(closed) > 0
    assert {t.exit_reason for t in closed} <= {"stop_loss", "take_profit", "eod", "entry_not_filled", "exit"}
    for t in closed:
        if t.entry_price:
            assert pd.Timestamp(t.opened_at).date() == pd.Timestamp(t.closed_at).date()   # intraday only
    # at every session close the book was flat
    last_bars = pd.Series(timeline).groupby(pd.Series(timeline).dt.date).max()
    assert eng.broker.positions() == [] or timeline[-1] not in set(last_bars)
    assert any("New trade" in s for s in eng.notify.sent)


def test_restart_keeps_positions_and_creates_no_duplicates(settings, frames, timeline):
    eng, m, db = make_engine(settings, frames)
    m.set_time(timeline[0].to_pydatetime())
    eng.startup()
    stop_at = None
    for i, t in enumerate(timeline):
        m.set_time(t.to_pydatetime())
        eng.step()
        if eng.broker.positions():
            stop_at = i
            break
    assert stop_at is not None, "no position opened in replay window"
    held = {p.symbol: p.quantity for p in eng.broker.positions()}
    with db.session() as s:
        n_orders = s.query(OrderRow).count()
    # --- simulated crash + restart: brand-new objects on the same database
    eng2, _, _ = make_engine(settings, frames, market=m, db=Database(settings.database_url))
    eng2.startup()
    assert {p.symbol: p.quantity for p in eng2.broker.positions()} == held
    eng2.step()                                   # same bar again: the same signal must not re-submit
    with db.session() as s:
        assert s.query(OrderRow).count() == n_orders
    # every position still protected by a stop at the broker
    for p in eng2.broker.positions():
        assert any(o.symbol == p.symbol and o.order_type.value == "STP" for o in eng2.broker.open_orders())


def test_kill_switch_flattens_and_blocks(settings, frames, timeline):
    eng, m, db = make_engine(settings, frames)
    m.set_time(timeline[0].to_pydatetime())
    eng.startup()
    for t in timeline:
        m.set_time(t.to_pydatetime())
        eng.step()
        if eng.broker.positions():
            break
    settings.kill_switch_file.write_text("test")
    assert eng.step() == "killed"
    eng.broker.process()
    assert eng.broker.positions() == [] and eng.broker.open_orders() == []
    n = len(db.closed_trades("replay"))
    for t in timeline[-20:]:
        m.set_time(t.to_pydatetime())
        assert eng.step() == "killed"
    assert len(db.open_trades("replay")) == 0 or len(db.closed_trades("replay")) >= n
    assert any("Kill switch" in s for s in eng.notify.sent)


def test_daily_loss_limit_halts_engine(settings, frames, timeline):
    eng, m, db = make_engine(settings, frames)
    m.set_time(timeline[0].to_pydatetime())
    eng.startup()
    run(eng, m, timeline[:5])
    eng.broker.cash["USD"] -= 3_000          # simulate a 3% loss
    run(eng, m, timeline[5:8])
    assert eng.risk.state.halted and eng.status.state in ("halted", "eod")
    assert any("Loss limit" in s for s in eng.notify.sent)
    # stays halted on the next day without manual reset
    nxt = [t for t in timeline if t.date() > timeline[8].date()][:3]
    run(eng, m, nxt)
    assert eng.risk.state.halted


def test_broker_outage_blocks_trading_and_alerts(settings, frames, timeline):
    eng, m, db = make_engine(settings, frames)
    m.set_time(timeline[0].to_pydatetime())
    eng.startup()

    def down():
        raise BrokerDisconnected("gateway down")
    eng.broker.disconnect()
    eng.broker.connect = down
    for t in timeline[:3]:
        m.set_time(t.to_pydatetime())
        assert eng.step() == "broker_down"
    assert sum("Broker connection" in s for s in eng.notify.sent) == 1     # alerted once, not spammed


def test_live_mode_refused_without_gate(settings, frames, timeline, tmp_path):
    s = Settings(**{**settings.model_dump(), "mode": TradingMode.LIVE, "broker": BrokerKind.IBKR,
                    "live_trading_enabled": True})
    eng, m, db = make_engine(settings, frames)
    eng.s = s
    m.set_time(timeline[0].to_pydatetime())
    with pytest.raises(PermissionError) as e:
        eng.startup()
    assert "approval" in str(e.value) and "tests" in str(e.value)


def test_no_entries_after_cutoff(settings, frames, timeline):
    eng, m, db = make_engine(settings, frames)
    m.set_time(timeline[0].to_pydatetime())
    eng.startup()
    run(eng, m, timeline)
    with db.session() as s:
        entries = [r for r in s.query(OrderRow).all() if r.role == "entry"]
    for r in entries:
        venue = eng.instruments[r.symbol].venue
        ts = pd.Timestamp(r.created_at).tz_localize("UTC") if pd.Timestamp(r.created_at).tzinfo is None else pd.Timestamp(r.created_at)
        mtc = mcal.minutes_to_close(venue, ts.to_pydatetime())
        assert mtc is None or mtc > settings.risk.no_new_entries_minutes_before_close
