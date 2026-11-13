"""Interactive Brokers adapter (TWS / IB Gateway) via ``ib_async`` — the
maintained successor of ib_insync, wrapping IBKR's official TWS API.

Prerequisites (see docs/03_broker_and_data.md):
* IB Gateway or TWS running with API enabled (port 4002 paper / 4001 live).
* Real-time market data subscriptions for NASDAQ/NYSE (US) and XETRA.
* Trading permissions for US and German stocks.
This module cannot be exercised without a running gateway; the unit tests
use a fake ``IB`` object, and ``scripts/check_ibkr_connection.py`` performs
a read-only end-to-end check against your gateway.

Safety properties:
* Every order carries ``orderRef = client_order_id`` -> after a restart open
  orders and executions are re-associated; duplicates are refused.
* Entries are marketable LIMIT orders (never naked market orders) with
  broker-side attached stop-loss and take-profit (bracket, OCA).
* ``expect_paper`` enforces that paper mode only ever talks to an IBKR
  paper account (ids starting with "DU") and live mode never does.
"""
from __future__ import annotations

import logging
import math
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

from app.domain import (AccountSnapshot, BracketRequest, Fill, Instrument, OrderRequest, OrderRole, OrderSide,
                        OrderState, OrderStatus, OrderType, Position, Quote)
from broker.base import Broker, BrokerDisconnected, BrokerError
from security.instrument_guard import IBKR_EXCHANGE_TO_VENUE

log = logging.getLogger(__name__)

BAR_SIZE = {"1m": "1 min", "5m": "5 mins", "15m": "15 mins", "1h": "1 hour", "1d": "1 day"}
BAR_DELTA = {"1m": timedelta(minutes=1), "5m": timedelta(minutes=5), "15m": timedelta(minutes=15),
             "1h": timedelta(hours=1), "1d": timedelta(days=1)}
STATUS_MAP = {
    "PendingSubmit": OrderStatus.PENDING_SUBMIT, "ApiPending": OrderStatus.PENDING_SUBMIT,
    "PreSubmitted": OrderStatus.SUBMITTED, "Submitted": OrderStatus.SUBMITTED,
    "Filled": OrderStatus.FILLED, "Cancelled": OrderStatus.CANCELLED, "ApiCancelled": OrderStatus.CANCELLED,
    "PendingCancel": OrderStatus.SUBMITTED, "Inactive": OrderStatus.INACTIVE,
}


def _ib_module():
    import ib_async  # imported lazily so the rest of the system works without it
    return ib_async


class IBKRBroker(Broker):
    name = "ibkr"

    def __init__(self, host: str, port: int, client_id: int, account: str | None, expect_paper: bool,
                 market_data_type: int = 1, ib=None, max_backoff_s: float = 60.0):
        self.host, self.port, self.client_id = host, port, client_id
        self.account_id = account
        self.expect_paper = expect_paper
        self.is_paper = expect_paper
        self.market_data_type = market_data_type
        self.ib = ib
        self._contracts: dict[str, object] = {}
        self._tickers: dict[str, object] = {}
        self._backoff = 1.0
        self._max_backoff = max_backoff_s
        self._next_attempt = 0.0

    # ------------------------------------------------------------ connection
    def connect(self) -> None:
        if self.ib is None:
            self.ib = _ib_module().IB()
        if self.ib.isConnected():
            return
        now = time.monotonic()
        if now < self._next_attempt:
            raise BrokerDisconnected(f"reconnect backoff {self._next_attempt - now:.0f}s")
        try:
            self.ib.connect(self.host, self.port, clientId=self.client_id, account=self.account_id or "", timeout=10)
        except Exception as e:  # noqa: BLE001 - any connect failure -> backoff
            self._next_attempt = now + self._backoff
            self._backoff = min(self._backoff * 2, self._max_backoff)
            raise BrokerDisconnected(f"IBKR connect failed: {e}") from e
        self._backoff = 1.0
        accounts = self.ib.managedAccounts()
        if not self.account_id:
            if len(accounts) != 1:
                self.ib.disconnect()
                raise BrokerError(f"multiple accounts {accounts}: set TRADER_IBKR_ACCOUNT")
            self.account_id = accounts[0]
        if self.account_id not in accounts:
            self.ib.disconnect()
            raise BrokerError(f"account {self.account_id} not managed by this login")
        is_paper_acct = self.account_id.startswith("DU")
        if self.expect_paper and not is_paper_acct:
            self.ib.disconnect()
            raise BrokerError("paper mode refused: connected account is NOT an IBKR paper account")
        if not self.expect_paper and is_paper_acct:
            self.ib.disconnect()
            raise BrokerError("live mode refused: connected account is a paper account")
        self.ib.reqMarketDataType(self.market_data_type)
        log.info("connected to IBKR account %s (paper=%s)", self.account_id, is_paper_acct)

    def disconnect(self) -> None:
        if self.ib is not None and self.ib.isConnected():
            self.ib.disconnect()

    def is_connected(self) -> bool:
        return bool(self.ib is not None and self.ib.isConnected())

    def _require(self) -> None:
        if not self.is_connected():
            raise BrokerDisconnected("IBKR not connected")

    # ------------------------------------------------------------ contracts
    def _contract(self, inst: Instrument):
        if inst.symbol in self._contracts:
            return self._contracts[inst.symbol]
        m = _ib_module()
        if inst.venue == "XETRA":
            c = m.Stock(inst.symbol, "IBIS", "EUR")
        else:
            c = m.Stock(inst.symbol, "SMART", "USD", primaryExchange="NASDAQ" if inst.venue == "NASDAQ" else "NYSE")
        if inst.con_id:
            c.conId = inst.con_id
        self._require()
        q = self.ib.qualifyContracts(c)
        if not q:
            raise BrokerError(f"cannot qualify {inst.symbol}")
        self._contracts[inst.symbol] = q[0]
        return q[0]

    def qualify(self, inst: Instrument) -> Instrument:
        c = self._contract(inst)
        details = self.ib.reqContractDetails(c)
        if len(details) != 1:
            raise BrokerError(f"{inst.symbol}: ambiguous contract details ({len(details)})")
        d = details[0]
        venue = IBKR_EXCHANGE_TO_VENUE.get((c.primaryExchange or c.exchange or "").upper(), "UNKNOWN")
        return Instrument(c.symbol, venue, c.currency, c.secType, (d.stockType or "").upper() or None, c.conId,
                          d.longName or "")

    # ------------------------------------------------------------ data
    def quote(self, inst: Instrument) -> Quote | None:
        self._require()
        c = self._contract(inst)
        t = self._tickers.get(inst.symbol)
        if t is None:
            t = self.ib.reqMktData(c, "", False, False)
            self._tickers[inst.symbol] = t
            self.ib.sleep(1.0)
        self.ib.sleep(0)
        bid, ask, last = _num(t.bid), _num(t.ask), _num(t.last)
        if bid is None or ask is None:
            return None
        ts = t.time or datetime.now(timezone.utc)
        realtime = getattr(t, "marketDataType", 1) == 1
        return Quote(inst.symbol, bid, ask, last if last is not None else (bid + ask) / 2, ts,
                     _num(t.bidSize) or 0.0, _num(t.askSize) or 0.0, is_realtime=realtime)

    def bars(self, inst: Instrument, timeframe: str, lookback_days: int) -> pd.DataFrame:
        self._require()
        c = self._contract(inst)
        raw = self.ib.reqHistoricalData(c, endDateTime="", durationStr=f"{max(lookback_days, 1)} D",
                                        barSizeSetting=BAR_SIZE[timeframe], whatToShow="TRADES",
                                        useRTH=True, formatDate=2)
        if not raw:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        delta = BAR_DELTA[timeframe]
        idx = [pd.Timestamp(b.date).tz_convert("UTC") + delta if isinstance(b.date, datetime)
               else pd.Timestamp(b.date, tz="UTC") for b in raw]   # IB stamps bar START -> convert to close
        df = pd.DataFrame({"open": [b.open for b in raw], "high": [b.high for b in raw], "low": [b.low for b in raw],
                           "close": [b.close for b in raw], "volume": [float(b.volume) for b in raw]},
                          index=pd.DatetimeIndex(idx))
        now = pd.Timestamp.now(tz="UTC")
        df = df[df.index <= now]  # drop the still-forming bar
        df.attrs["synthetic"] = False
        return df

    def fx_rate(self, from_ccy: str, to_ccy: str) -> float:
        if from_ccy == to_ccy:
            return 1.0
        m = _ib_module()
        pair = from_ccy + to_ccy
        inverse = False
        if pair not in ("EURUSD",):
            pair, inverse = to_ccy + from_ccy, True
        key = f"FX:{pair}"
        t = self._tickers.get(key)
        if t is None:
            # data subscription only — a Forex contract is never passed to placeOrder
            t = self.ib.reqMktData(m.Forex(pair), "", False, False)
            self._tickers[key] = t
            self.ib.sleep(1.0)
        px = _num(t.midpoint()) if hasattr(t, "midpoint") else None
        if not px:
            raise BrokerError(f"no FX quote {pair}")
        return 1 / px if inverse else px

    # ------------------------------------------------------------ account
    def account(self) -> AccountSnapshot:
        self._require()
        vals = {(v.tag, v.currency): v.value for v in self.ib.accountSummary(self.account_id)}
        base = next((cur for (tag, cur) in vals if tag == "NetLiquidation"), "USD")

        def g(tag: str) -> float:
            return float(vals.get((tag, base), 0) or 0)

        return AccountSnapshot(self.account_id, g("NetLiquidation"), g("TotalCashValue"), g("BuyingPower"), base,
                               datetime.now(timezone.utc), g("RealizedPnL"), g("UnrealizedPnL"))

    def positions(self) -> list[Position]:
        self._require()
        out = []
        for p in self.ib.positions(self.account_id):
            if p.contract.secType != "STK":
                log.error("non-stock position at broker: %s %s", p.contract.secType, p.contract.symbol)
            out.append(Position(p.contract.symbol, int(p.position), float(p.avgCost), p.contract.currency))
        return [p for p in out if p.quantity != 0]

    def _trades_by_ref(self) -> dict[str, object]:
        return {t.order.orderRef: t for t in self.ib.trades() if t.order.orderRef}

    def open_orders(self) -> list[OrderState]:
        self._require()
        return [_to_state(t) for t in self.ib.openTrades() if t.order.orderRef]

    # ------------------------------------------------------------ orders
    def _ib_order(self, req: OrderRequest):
        m = _ib_module()
        action = req.side.value
        if req.order_type == OrderType.MARKET:
            o = m.MarketOrder(action, req.quantity)
        elif req.order_type == OrderType.LIMIT:
            o = m.LimitOrder(action, req.quantity, _tick(req.limit_price))
        elif req.order_type == OrderType.STOP:
            o = m.StopOrder(action, req.quantity, _tick(req.stop_price))
        else:
            o = m.StopLimitOrder(action, req.quantity, _tick(req.limit_price), _tick(req.stop_price))
        o.orderRef = req.client_order_id
        o.tif = req.tif
        o.outsideRth = False
        o.account = self.account_id
        if req.oca_group:
            o.ocaGroup, o.ocaType = req.oca_group, 1
        return o

    def place_order(self, req: OrderRequest, inst: Instrument) -> OrderState:
        self._require()
        req.validate()
        if req.client_order_id in self._trades_by_ref():
            raise BrokerError(f"duplicate orderRef {req.client_order_id}")
        trade = self.ib.placeOrder(self._contract(inst), self._ib_order(req))
        self.ib.sleep(0.2)
        return _to_state(trade, req.role)

    def place_bracket(self, br: BracketRequest, inst: Instrument) -> list[OrderState]:
        """Parent + attached stop + take-profit; only the last leg transmits,
        so IBKR receives the three orders atomically."""
        self._require()
        br.validate()
        existing = self._trades_by_ref()
        for leg in (br.entry, br.stop, br.take_profit):
            if leg.client_order_id in existing:
                raise BrokerError(f"duplicate orderRef {leg.client_order_id}")
        c = self._contract(inst)
        parent = self._ib_order(br.entry)
        parent.orderId = self.ib.client.getReqId()
        parent.transmit = False
        tp = self._ib_order(br.take_profit)
        sl = self._ib_order(br.stop)
        for child in (tp, sl):
            child.parentId = parent.orderId
            child.ocaGroup, child.ocaType = br.stop.oca_group or br.entry.client_order_id, 1
        tp.transmit, sl.transmit = False, True
        trades = [self.ib.placeOrder(c, parent), self.ib.placeOrder(c, tp), self.ib.placeOrder(c, sl)]
        self.ib.sleep(0.5)
        return [_to_state(trades[0], OrderRole.ENTRY), _to_state(trades[2], OrderRole.STOP),
                _to_state(trades[1], OrderRole.TAKE_PROFIT)]

    def cancel_order(self, client_order_id: str) -> None:
        self._require()
        t = self._trades_by_ref().get(client_order_id)
        if t is not None and not t.isDone():
            self.ib.cancelOrder(t.order)

    def modify_quantity(self, client_order_id: str, quantity: int) -> None:
        self._require()
        t = self._trades_by_ref().get(client_order_id)
        if t is None or t.isDone():
            raise BrokerError(f"order {client_order_id} not modifiable")
        t.order.totalQuantity = quantity
        self.ib.placeOrder(t.contract, t.order)

    def fills_since(self, since: datetime) -> list[Fill]:
        self._require()
        out = []
        for f in self.ib.fills():
            if f.time < since or not f.execution.orderRef:
                continue
            comm = f.commissionReport.commission if f.commissionReport else 0.0
            out.append(Fill(f.execution.orderRef, f.contract.symbol,
                            OrderSide.BUY if f.execution.side == "BOT" else OrderSide.SELL,
                            int(f.execution.shares), float(f.execution.price), float(comm or 0.0), f.time,
                            f.execution.execId))
        return out


def _num(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) or v <= 0 else v


def _tick(px: float | None) -> float | None:
    """Round to a valid tick (0.01 >= $1 for US; XETRA tick tables vary by
    price/liquidity band — IBKR rejects invalid ticks, which fails safe)."""
    return None if px is None else round(px + 1e-9, 2)


def _to_state(trade, role: OrderRole | None = None) -> OrderState:
    o, s = trade.order, trade.orderStatus
    ot = {"MKT": OrderType.MARKET, "LMT": OrderType.LIMIT, "STP": OrderType.STOP}.get(o.orderType, OrderType.STOP_LIMIT)
    if role is None:
        role = OrderRole.ENTRY if not o.parentId else (OrderRole.STOP if ot == OrderType.STOP else OrderRole.TAKE_PROFIT)
    filled = int(s.filled or 0)
    status = STATUS_MAP.get(s.status, OrderStatus.SUBMITTED)
    if status == OrderStatus.SUBMITTED and filled > 0:
        status = OrderStatus.PARTIALLY_FILLED
    return OrderState(o.orderRef, str(o.orderId), trade.contract.symbol,
                      OrderSide(o.action), int(o.totalQuantity), filled, float(s.avgFillPrice or 0) or None,
                      status, ot, role, _num(o.lmtPrice), _num(o.auxPrice), None)
