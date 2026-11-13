"""IBKR adapter against a fake gateway (no network). Uses the real ib_async
contract/order classes so field names are checked against the library."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from app.domain import BracketRequest, Instrument, OrderRequest, OrderRole, OrderSide, OrderStatus, OrderType
from broker.base import BrokerDisconnected, BrokerError
from broker.ibkr import IBKRBroker
from security.instrument_guard import InstrumentNotAllowed


class FakeIB:
    def __init__(self, accounts=("DU111",), fail_connect=False, stock_type="COMMON"):
        self.connected = False
        self.accounts = list(accounts)
        self.fail_connect = fail_connect
        self.stock_type = stock_type
        self._trades = []
        self._id = 100
        self.client = SimpleNamespace(getReqId=self._next)

    def _next(self):
        self._id += 1
        return self._id

    def isConnected(self):
        return self.connected

    def connect(self, *a, **k):
        if self.fail_connect:
            raise ConnectionRefusedError("no gateway")
        self.connected = True

    def disconnect(self):
        self.connected = False

    def managedAccounts(self):
        return self.accounts

    def reqMarketDataType(self, t):
        self.mdt = t

    def qualifyContracts(self, c):
        c.conId = 265598
        if not c.primaryExchange:
            c.primaryExchange = c.exchange
        return [c]

    def reqContractDetails(self, c):
        return [SimpleNamespace(stockType=self.stock_type, longName="Test Corp")]

    def placeOrder(self, contract, order):
        if not order.orderId:
            order.orderId = self._next()
        t = SimpleNamespace(contract=contract, order=order, orderStatus=SimpleNamespace(status="Submitted", filled=0,
                                                                                      avgFillPrice=0.0),
                            isDone=lambda: False)
        self._trades.append(t)
        return t

    def trades(self):
        return self._trades

    def openTrades(self):
        return self._trades

    def sleep(self, s):
        pass

    def cancelOrder(self, order):
        self.cancelled = order.orderRef

    def reqHistoricalData(self, c, **kw):
        start = datetime(2026, 11, 11, 14, 30, tzinfo=timezone.utc)
        return [SimpleNamespace(date=start + timedelta(minutes=5 * i), open=1, high=2, low=0.5, close=1.5, volume=100)
                for i in range(3)]


AAPL = Instrument("AAPL", "NASDAQ", "USD", "STK", "COMMON")


def leg(cid, side, typ, role, **kw):
    return OrderRequest(cid, "AAPL", side, 10, typ, role, **kw)


def bracket():
    return BracketRequest(leg("C1", OrderSide.BUY, OrderType.LIMIT, OrderRole.ENTRY, limit_price=100.0),
                          leg("C1-SL", OrderSide.SELL, OrderType.STOP, OrderRole.STOP, stop_price=99.0,
                              parent_id="C1", oca_group="OCA-C1"),
                          leg("C1-TP", OrderSide.SELL, OrderType.LIMIT, OrderRole.TAKE_PROFIT, limit_price=102.0,
                              parent_id="C1", oca_group="OCA-C1"))


def test_paper_mode_refuses_live_account():
    b = IBKRBroker("h", 1, 1, None, expect_paper=True, ib=FakeIB(accounts=["U999"]))
    with pytest.raises(BrokerError, match="NOT an IBKR paper"):
        b.connect()
    assert not b.is_connected()


def test_live_mode_refuses_paper_account():
    b = IBKRBroker("h", 1, 1, None, expect_paper=False, ib=FakeIB(accounts=["DU1"]))
    with pytest.raises(BrokerError, match="paper account"):
        b.connect()


def test_multiple_accounts_require_explicit_choice():
    b = IBKRBroker("h", 1, 1, None, expect_paper=True, ib=FakeIB(accounts=["DU1", "DU2"]))
    with pytest.raises(BrokerError, match="multiple"):
        b.connect()


def test_reconnect_backoff():
    b = IBKRBroker("h", 1, 1, None, expect_paper=True, ib=FakeIB(fail_connect=True))
    with pytest.raises(BrokerDisconnected):
        b.connect()
    with pytest.raises(BrokerDisconnected, match="backoff"):
        b.connect()


def test_bracket_is_atomic_and_tagged():
    ib = FakeIB()
    b = IBKRBroker("h", 1, 1, None, expect_paper=True, ib=ib)
    b.connect()
    states = b.place_bracket(bracket(), AAPL)
    parent, tp, sl = (t.order for t in ib._trades)
    assert [s.role for s in states] == [OrderRole.ENTRY, OrderRole.STOP, OrderRole.TAKE_PROFIT]
    assert parent.orderRef == "C1" and sl.orderRef == "C1-SL" and tp.orderRef == "C1-TP"
    assert tp.parentId == parent.orderId == sl.parentId
    assert (parent.transmit, tp.transmit, sl.transmit) == (False, False, True)
    assert sl.orderType == "STP" and sl.auxPrice == 99.0 and tp.lmtPrice == 102.0
    assert parent.outsideRth is False and parent.tif == "DAY" and parent.account == "DU111"
    with pytest.raises(BrokerError, match="duplicate"):
        b.place_bracket(bracket(), AAPL)


def test_qualify_reports_broker_stock_type_for_guard(guard):
    b = IBKRBroker("h", 1, 1, None, expect_paper=True, ib=FakeIB(stock_type="ETF"))
    b.connect()
    q = b.qualify(AAPL)
    assert q.stock_type == "ETF" and q.venue == "NASDAQ" and q.con_id == 265598
    with pytest.raises(InstrumentNotAllowed):
        guard.check(q)


def test_bars_are_restamped_to_close_time():
    b = IBKRBroker("h", 1, 1, None, expect_paper=True, ib=FakeIB())
    b.connect()
    df = b.bars(AAPL, "5m", 1)
    assert df.index[0] == datetime(2026, 11, 11, 14, 35, tzinfo=timezone.utc)


def test_order_state_mapping():
    ib = FakeIB()
    b = IBKRBroker("h", 1, 1, None, expect_paper=True, ib=ib)
    b.connect()
    b.place_order(leg("X1", OrderSide.SELL, OrderType.MARKET, OrderRole.EXIT), AAPL)
    ib._trades[0].orderStatus = SimpleNamespace(status="Submitted", filled=4, avgFillPrice=100.5)
    st = b.open_orders()[0]
    assert st.status == OrderStatus.PARTIALLY_FILLED and st.filled == 4 and st.remaining == 6
    b.cancel_order("X1")
    assert ib.cancelled == "X1"
