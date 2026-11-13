
from datetime import datetime, timedelta, timezone
from dataclasses import replace

import pytest

from app.domain import Direction, OrderRole, OrderSide, OrderState, OrderStatus, OrderType, Position, TradeCandidate
from risk.costs import commission
from risk.manager import MemoryStateStore, PreTradeContext, RiskManager

from .conftest import NY_MIDDAY, make_account, make_quote


def cand(direction=Direction.LONG, entry=100.0, stop=99.0, target=102.5, symbol="AAPL", ts=NY_MIDDAY):
    return TradeCandidate(symbol, direction, entry, stop, target, 80.0, ["momentum"], ts)


@pytest.fixture
def rm(settings, guard):
    return RiskManager(settings.risk, settings.costs, guard, MemoryStateStore())


def ctx(inst, now=NY_MIDDAY, **kw):
    base = dict(now=now, account=make_account(), positions=[], open_orders=[], quote=make_quote(ts=now),
                instrument=inst, broker_connected=True, data_fresh=True, fx_to_base=1.0)
    base.update(kw)
    return PreTradeContext(**base)


def test_sizing_uses_stop_distance_and_costs(rm, aapl):
    rm.cfg = rm.cfg.model_copy(update={"max_position_notional_pct": 50.0})
    d = rm.evaluate(cand(), ctx(aapl))
    assert d.approved
    # 0.5% of 100k = 500 risk; per-share risk 1.00 + costs -> slightly below 500 shares
    assert 450 <= d.quantity < 500
    assert d.risk_amount <= 500.0 + 1e-6


def test_fx_conversion_for_eur_stock(rm, universe):
    sap = universe["SAP"]
    now = datetime(2026, 11, 11, 10, 0, tzinfo=timezone.utc)      # 11:00 Frankfurt
    q = make_quote("SAP", 100.0, ts=now)
    d = rm.evaluate(cand(symbol="SAP", ts=now), ctx(sap, now=now, quote=q, fx_to_base=1.10,
                                                    account=make_account(ts=now)))
    assert d.approved
    assert d.risk_amount <= 500.0 + 1e-6 and d.quantity < 455   # EUR risk converted into USD budget


@pytest.mark.parametrize("override,msg", [
    ({"kill_switch_engaged": True}, "kill switch"),
    ({"broker_connected": False}, "broker"),
    ({"data_fresh": False}, "stale"),
    ({"quote": None}, "stale"),
])
def test_rejections_infrastructure(rm, aapl, override, msg):
    d = rm.evaluate(cand(), ctx(aapl, **override))
    assert not d.approved and msg in " ".join(d.reasons)


def test_rejects_wide_spread(rm, aapl):
    d = rm.evaluate(cand(), ctx(aapl, quote=make_quote(spread=0.5)))
    assert not d.approved and "spread" in d.reasons[0]


def test_rejects_market_closed_and_near_close(rm, aapl):
    sat = datetime(2026, 11, 14, 16, 0, tzinfo=timezone.utc)
    assert "closed" in rm.evaluate(cand(), ctx(aapl, now=sat)).reasons[0]
    late = datetime(2026, 11, 11, 20, 45, tzinfo=timezone.utc)    # 15:45 NY
    assert "no new entries" in rm.evaluate(cand(), ctx(aapl, now=late, quote=make_quote(ts=late))).reasons[0]
    early = datetime(2026, 11, 11, 14, 32, tzinfo=timezone.utc)
    assert "opening" in rm.evaluate(cand(), ctx(aapl, now=early, quote=make_quote(ts=early))).reasons[0]


def test_rejects_rr_below_two(rm, aapl):
    d = rm.evaluate(cand(target=101.5), ctx(aapl))
    assert not d.approved and "R:R" in d.reasons[0]


def test_rejects_invalid_geometry(rm, aapl):
    assert not rm.evaluate(cand(stop=101.0, target=103), ctx(aapl)).approved
    assert not rm.evaluate(cand(Direction.SHORT, stop=99.0, target=97), ctx(aapl)).approved


def test_rejects_duplicates_and_max_positions(rm, aapl):
    pos = [Position("AAPL", 10, 100.0)]
    assert "already open" in rm.evaluate(cand(), ctx(aapl, positions=pos)).reasons[0]
    o = OrderState("X", "1", "AAPL", OrderSide.BUY, 10, 0, None, OrderStatus.SUBMITTED, OrderType.LIMIT, OrderRole.ENTRY)
    assert "open order" in rm.evaluate(cand(), ctx(aapl, open_orders=[o])).reasons[0]
    three = [Position(s, 10, 50.0) for s in ("MSFT", "NVDA", "AMZN")]
    assert "max open positions" in rm.evaluate(cand(), ctx(aapl, positions=three)).reasons[0]


def test_daily_loss_limit_halts_persistently(settings, guard, aapl):
    store = MemoryStateStore()
    rm = RiskManager(settings.risk, settings.costs, guard, store)
    rm.check_loss_limits(NY_MIDDAY, 100_000)
    assert rm.check_loss_limits(NY_MIDDAY + timedelta(minutes=5), 98_100) is None
    reason = rm.check_loss_limits(NY_MIDDAY + timedelta(minutes=10), 97_990)
    assert reason and "daily" in reason and rm.state.halted
    # a restart (new instance, same store) is still halted
    rm2 = RiskManager(settings.risk, settings.costs, guard, store)
    assert rm2.state.halted
    assert not rm2.evaluate(cand(), ctx(aapl)).approved
    # next day does NOT auto-unhalt
    rm2.check_loss_limits(NY_MIDDAY + timedelta(days=1), 97_990)
    assert rm2.state.halted
    with pytest.raises(PermissionError):
        rm2.manual_reset("", 97_990, NY_MIDDAY)
    rm2.manual_reset("admin", 97_990, NY_MIDDAY + timedelta(days=1))
    assert not rm2.state.halted


def test_weekly_loss_limit(settings, guard):
    rm = RiskManager(settings.risk, settings.costs, guard, MemoryStateStore())
    mon = datetime(2026, 11, 9, 15, 0, tzinfo=timezone.utc)
    eq = 100_000.0
    rm.check_loss_limits(mon, eq)
    for d in range(4):            # lose 1.4% per day: never a daily breach, weekly after ~4 days
        day = mon + timedelta(days=d)
        rm.check_loss_limits(day, eq)
        eq *= 0.986
        r = rm.check_loss_limits(day + timedelta(hours=1), eq)
    assert rm.state.halted and "weekly" in rm.state.halt_reason
    assert r is not None


def test_no_martingale_risk_only_shrinks_after_losses(rm, aapl):
    base = rm.evaluate(cand(), ctx(aapl)).quantity
    for _ in range(5):
        rm.record_trade_result(-100)
    reduced = rm.evaluate(cand(), ctx(aapl)).quantity
    assert reduced < base and rm.effective_risk_pct() <= rm.cfg.risk_per_trade_pct
    rm.record_trade_result(+100)
    assert rm.evaluate(cand(), ctx(aapl)).quantity == base     # back to normal, never above


def test_portfolio_heat_cap(rm, aapl):
    rm.state.open_risk = {"MSFT": 1000.0, "NVDA": 400.0}       # 1.4% already at risk, cap 1.5%
    d = rm.evaluate(cand(), ctx(aapl))
    assert d.quantity <= 100 and d.risk_amount <= 100.0 + 1e-6


def test_notional_and_buying_power_caps(rm, aapl):
    d = rm.evaluate(cand(), ctx(aapl))
    assert d.approved and d.quantity == 250 and "notional" in d.reasons[0]     # 25% of equity at $100
    d = rm.evaluate(cand(), ctx(aapl, account=replace(make_account(), buying_power=5_000)))
    assert d.quantity <= 0.95 * 5_000 / 100


def test_commission_model(settings):
    assert commission(settings.costs, "USD", 10, 100) == pytest.approx(0.35)       # minimum
    assert commission(settings.costs, "USD", 1000, 100) == pytest.approx(3.5)
    assert commission(settings.costs, "EUR", 10, 100) == pytest.approx(1.25)
    assert commission(settings.costs, "EUR", 100, 100) == pytest.approx(5.0)


def test_shared_store_halt_visible_after_refresh(settings, guard):
    store = MemoryStateStore()
    a = RiskManager(settings.risk, settings.costs, guard, store)
    b = RiskManager(settings.risk, settings.costs, guard, store)
    a.halt("test")
    b.refresh()
    assert b.state.halted

