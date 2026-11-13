"""Paper broker, order manager, protection, reconciliation, kill switch and
restart consistency."""
from datetime import timedelta

import pandas as pd
import pytest

from app.domain import (BracketRequest, Direction, OrderRequest, OrderRole, OrderSide, OrderStatus, OrderType, Position,
                        TradeCandidate)
from broker.base import BrokerError
from broker.paper import PaperBroker
from execution.order_manager import OrderManager
from risk.manager import RiskDecision, RiskManager
from security.kill_switch import KillSwitch

from .conftest import NY_MIDDAY, make_quote


class FakeMarket:
    def __init__(self):
        self.px: dict[str, float] = {}
        self.now = NY_MIDDAY

    def quote(self, inst):
        p = self.px.get(inst.symbol)
        return make_quote(inst.symbol, p, spread=0.02, ts=self.now) if p else None

    def bars(self, inst, tf, days):
        return pd.DataFrame()

    def fx_rate(self, a, b):
        return 1.0 if a == b else 1.1


@pytest.fixture
def market():
    m = FakeMarket()
    m.px["AAPL"] = 100.0
    return m


@pytest.fixture
def broker(market, settings, db):
    b = PaperBroker(market, settings.costs, db, 100_000, clock=lambda: market.now)
    b.connect()
    return b


@pytest.fixture
def om(broker, db, settings, guard, universe, market):
    rm = RiskManager(settings.risk, settings.costs, guard, db)
    notes = []
    o = OrderManager(broker, db, rm, "paper", universe, "USD", notify=lambda e, t: notes.append((e, t)),
                     clock=lambda: market.now)
    o.notes = notes
    return o


def cand(ts=NY_MIDDAY):
    return TradeCandidate("AAPL", Direction.LONG, 100.0, 99.0, 102.0, 80, ["breakout"], ts)


def test_paper_requires_configured_capital(market, settings, db):
    with pytest.raises(BrokerError):
        PaperBroker(market, settings.costs, db, None)


def test_bracket_take_profit_and_oca(om, broker, market, aapl):
    om.submit(cand(), RiskDecision(True, 100, risk_amount=100), aapl, 100.01)
    assert broker.positions()[0].quantity == 100
    market.px["AAPL"] = 102.5
    broker.process()
    assert broker.positions() == []
    states = {o["state"].client_order_id: o["state"].status for o in broker.orders.values()}
    cid = cand().client_order_id
    assert states[cid + "-TP"] == OrderStatus.FILLED and states[cid + "-SL"] == OrderStatus.CANCELLED
    rep = om.sync()
    assert rep.closed_trades[0]["reason"] == "take_profit" and rep.closed_trades[0]["pnl"] > 0
    assert any(e == "take_profit" for e, _ in om.notes)


def test_stop_loss_with_gap_fills_at_worse_price(om, broker, market, aapl):
    om.submit(cand(), RiskDecision(True, 100, risk_amount=100), aapl, 100.01)
    market.px["AAPL"] = 97.0          # gap through the 99 stop
    broker.process()
    f = [x for x in broker.fills if x.client_order_id.endswith("-SL")][0]
    assert f.price < 99.0
    rep = om.sync()
    assert rep.closed_trades[0]["reason"] == "stop_loss" and rep.closed_trades[0]["pnl"] < -200


def test_duplicate_submission_refused(om, aapl):
    om.submit(cand(), RiskDecision(True, 10, risk_amount=10), aapl, 100.01)
    with pytest.raises(BrokerError):
        om.submit(cand(), RiskDecision(True, 10, risk_amount=10), aapl, 100.01)


def test_unapproved_never_submitted(om, aapl):
    with pytest.raises(ValueError):
        om.submit(cand(), RiskDecision(False, 10), aapl, 100.0)


def test_children_sized_to_partial_entry(broker, market, aapl):
    br = BracketRequest(OrderRequest("E1", "AAPL", OrderSide.BUY, 100, OrderType.LIMIT, OrderRole.ENTRY, limit_price=99.0),
                        OrderRequest("E1-SL", "AAPL", OrderSide.SELL, 100, OrderType.STOP, OrderRole.STOP, stop_price=98.0,
                                     parent_id="E1", oca_group="O"),
                        OrderRequest("E1-TP", "AAPL", OrderSide.SELL, 100, OrderType.LIMIT, OrderRole.TAKE_PROFIT,
                                     limit_price=103.0, parent_id="E1", oca_group="O"))
    broker.place_bracket(br, aapl)
    assert broker.positions() == []                     # limit 99 not reached
    # simulate a partial fill then cancellation of the remainder
    broker._fill("E1", broker.orders["E1"]["state"], aapl, 40, 99.0)
    broker.cancel_order("E1")
    broker.process()
    assert broker.orders["E1-SL"]["state"].quantity == 40 and broker.orders["E1-SL"]["state"].status == OrderStatus.SUBMITTED


def test_state_survives_restart(om, broker, market, settings, db, aapl):
    om.submit(cand(), RiskDecision(True, 50, risk_amount=50), aapl, 100.01)
    b2 = PaperBroker(market, settings.costs, db, 999_999, clock=lambda: market.now)   # capital ignored: state exists
    b2.connect()
    assert b2.positions()[0].quantity == 50
    assert {o.client_order_id[-3:] for o in b2.open_orders()} == {"-SL", "-TP"}
    assert b2.cash == broker.cash


def test_ensure_protection_places_missing_stop(om, broker, market, aapl):
    om.submit(cand(), RiskDecision(True, 50, risk_amount=50), aapl, 100.01)
    cid = cand().client_order_id
    broker.orders[cid + "-SL"]["state"].status = OrderStatus.CANCELLED   # stop vanished at broker
    rep = om.ensure_protection()
    assert rep.protective_placed and any(o.order_type == OrderType.STOP and o.status == OrderStatus.SUBMITTED
                                         for o in broker.open_orders())


def test_unknown_broker_position_is_flattened(om, broker, market, aapl):
    broker.pos["AAPL"] = Position("AAPL", 30, 100.0, "USD")
    rep = om.ensure_protection()
    broker.process()
    assert "AAPL" in rep.flattened and broker.positions() == []


def test_flatten_all(om, broker, market, aapl):
    om.submit(cand(), RiskDecision(True, 50, risk_amount=50), aapl, 100.01)
    om.flatten_all("eod")
    broker.process()
    assert broker.positions() == [] and broker.open_orders() == []
    rep = om.sync()
    assert rep.closed_trades[0]["reason"] == "eod"


def test_kill_switch_file_and_db(settings, db):
    k = KillSwitch(db, settings.kill_switch_file, db.audit)
    assert not k.engaged()
    settings.kill_switch_file.write_text("ops")
    assert k.engaged() and "ops" in k.reason()
    with pytest.raises(PermissionError):
        k.reset("")
    k.reset("admin")
    assert not k.engaged()
    k.engage("api", "test")
    assert k.engaged() and settings.kill_switch_file.exists()


def test_entry_timeout_cancels(om, broker, market, aapl):
    c = cand()
    om.submit(c, RiskDecision(True, 50, risk_amount=50), aapl, 99.0)   # limit below ask -> stays working
    assert broker.positions() == []
    market.now = NY_MIDDAY + timedelta(minutes=15)
    om.sync(entry_timeout=timedelta(minutes=10))
    broker.process()
    assert broker.orders[c.client_order_id]["state"].status == OrderStatus.CANCELLED
    rep = om.sync()
    assert rep.closed_trades and rep.closed_trades[0]["reason"] == "entry_not_filled"
