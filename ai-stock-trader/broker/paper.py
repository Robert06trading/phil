"""Local paper broker: simulated execution against a market data provider.

* Market data comes from a ``MarketDataProvider`` — in PAPER mode that is the
  IBKR adapter (real-time quotes); in REPLAY mode a recorded-data replay
  clearly marked as such. This broker never fabricates prices.
* Fill model: market orders fill at the opposite side of the book plus
  slippage; limits fill when the book crosses; stops trigger on the
  touching side and fill with slippage (gaps fill at the worse price).
* Bracket children activate only after the entry fills and are sized to the
  filled quantity; OCA siblings cancel each other.
* Full state (cash, positions, orders, fills) is persisted to the state
  store after every change, so a restart neither forgets positions nor
  re-sends orders.

Recommended alternative: an IBKR *paper account* via ``IBKRBroker`` (port
4002) gives broker-side order handling identical to live.
"""
from __future__ import annotations

import itertools
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Callable, Protocol

import pandas as pd

from app.domain import (AccountSnapshot, BracketRequest, Fill, Instrument, OrderRequest, OrderRole, OrderSide,
                        OrderState, OrderStatus, OrderType, Position, Quote)
from broker.base import Broker, BrokerError
from config import CostConfig
from risk.costs import commission


class MarketDataProvider(Protocol):
    def quote(self, inst: Instrument) -> Quote | None: ...
    def bars(self, inst: Instrument, timeframe: str, lookback_days: int) -> pd.DataFrame: ...
    def fx_rate(self, from_ccy: str, to_ccy: str) -> float: ...


class PaperBroker(Broker):
    name = "paper"
    is_paper = True
    STATE_KEY = "paper_broker"

    def __init__(self, market: MarketDataProvider, costs: CostConfig, store, starting_capital: float | None,
                 base_currency: str = "USD", clock: Callable[[], datetime] | None = None):
        self.market = market
        self.costs = costs
        self.store = store
        self.base = base_currency
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self._connected = False
        self._ids = itertools.count(1)
        raw = store.get(self.STATE_KEY)
        if raw:
            self._load(raw)
        else:
            if not starting_capital or starting_capital <= 0:
                raise BrokerError("paper starting capital not configured (TRADER_PAPER_STARTING_CAPITAL)")
            self.cash: dict[str, float] = {base_currency: float(starting_capital)}
            self.pos: dict[str, Position] = {}
            self.orders: dict[str, dict] = {}
            self.fills: list[Fill] = []
            self.realized_today: float = 0.0
            self._save()

    # ------------------------------------------------------------ persistence
    def _save(self) -> None:
        self.store.set(self.STATE_KEY, {
            "cash": self.cash,
            "pos": {k: asdict(v) for k, v in self.pos.items()},
            "orders": {k: {"state": _ser_order(v["state"]), "inst": asdict(v["inst"]), "oca": v["oca"],
                           "triggered": v["triggered"]} for k, v in self.orders.items()},
            "fills": [{**asdict(f), "ts": f.ts.isoformat(), "side": f.side.value} for f in self.fills[-500:]],
            "realized_today": self.realized_today,
            "day": getattr(self, "_day", None),
        })

    def _load(self, raw: dict) -> None:
        self.cash = raw["cash"]
        self.pos = {k: Position(**v) for k, v in raw["pos"].items()}
        self.orders = {k: {"state": _de_order(v["state"]), "inst": Instrument(**v["inst"]), "oca": v["oca"],
                           "triggered": v["triggered"]} for k, v in raw["orders"].items()}
        self.fills = [Fill(**{**f, "ts": datetime.fromisoformat(f["ts"]), "side": OrderSide(f["side"])})
                      for f in raw["fills"]]
        self.realized_today = raw.get("realized_today", 0.0)
        self._day = raw.get("day")

    # ------------------------------------------------------------ connection
    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------ account
    def fx_rate(self, from_ccy: str, to_ccy: str) -> float:
        return self.market.fx_rate(from_ccy, to_ccy)

    def account(self) -> AccountSnapshot:
        today = self.clock().date().isoformat()
        if getattr(self, "_day", None) != today:
            if getattr(self, "_day", None) is not None:
                self.realized_today = 0.0
            self._day = today
        equity = sum(v * self.fx_rate(c, self.base) for c, v in self.cash.items())
        unreal = 0.0
        gross = 0.0
        for p in self.pos.values():
            q = self.market.quote(Instrument(p.symbol, "", p.currency, stock_type="COMMON"))
            px = q.mid if q else p.market_price or p.avg_price
            p.market_price = px
            fx = self.fx_rate(p.currency, self.base)
            equity += p.quantity * px * fx
            unreal += p.unrealized_pnl * fx
            gross += abs(p.quantity * px * fx)
        return AccountSnapshot("PAPER-LOCAL", equity, self.cash.get(self.base, 0.0), max(equity * 2 - gross, 0.0),
                               self.base, self.clock(), self.realized_today, unreal)

    def positions(self) -> list[Position]:
        return [p for p in self.pos.values() if p.quantity != 0]

    def open_orders(self) -> list[OrderState]:
        return [o["state"] for o in self.orders.values() if not o["state"].status.is_terminal]

    def qualify(self, inst: Instrument) -> Instrument:
        return inst   # the local paper broker cannot verify; the static universe is authoritative

    def quote(self, inst: Instrument) -> Quote | None:
        return self.market.quote(inst)

    def bars(self, inst: Instrument, timeframe: str, lookback_days: int) -> pd.DataFrame:
        return self.market.bars(inst, timeframe, lookback_days)

    # ------------------------------------------------------------ orders
    def _register(self, req: OrderRequest, inst: Instrument, status: OrderStatus) -> OrderState:
        if req.client_order_id in self.orders:
            raise BrokerError(f"duplicate client order id {req.client_order_id}")
        req.validate()
        st = OrderState(req.client_order_id, f"P{next(self._ids)}", req.symbol, req.side, req.quantity, 0, None,
                        status, req.order_type, req.role, req.limit_price, req.stop_price, req.parent_id)
        self.orders[req.client_order_id] = {"state": st, "inst": inst, "oca": req.oca_group, "triggered": False}
        return st

    def place_order(self, req: OrderRequest, inst: Instrument) -> OrderState:
        if not self._connected:
            raise BrokerError("not connected")
        st = self._register(req, inst, OrderStatus.SUBMITTED)
        self._save()
        self.process()
        return st

    def place_bracket(self, br: BracketRequest, inst: Instrument) -> list[OrderState]:
        if not self._connected:
            raise BrokerError("not connected")
        br.validate()
        out = [self._register(br.entry, inst, OrderStatus.SUBMITTED),
               self._register(br.stop, inst, OrderStatus.PENDING_SUBMIT),
               self._register(br.take_profit, inst, OrderStatus.PENDING_SUBMIT)]
        self._save()
        self.process()
        return out

    def cancel_order(self, client_order_id: str) -> None:
        o = self.orders.get(client_order_id)
        if o and not o["state"].status.is_terminal:
            o["state"].status = OrderStatus.CANCELLED
            for child in self.orders.values():   # cancelling a parent cancels unfilled children
                if child["state"].parent_id == client_order_id and o["state"].filled == 0:
                    child["state"].status = OrderStatus.CANCELLED
            self._save()

    def modify_quantity(self, client_order_id: str, quantity: int) -> None:
        o = self.orders.get(client_order_id)
        if not o or o["state"].status.is_terminal:
            raise BrokerError("order not modifiable")
        o["state"].quantity = quantity
        self._save()

    def fills_since(self, since: datetime) -> list[Fill]:
        return [f for f in self.fills if f.ts >= since]

    # ------------------------------------------------------------ matching
    def process(self) -> list[Fill]:
        """Match active orders against current quotes. Call on every tick/bar."""
        new: list[Fill] = []
        for cid, o in list(self.orders.items()):
            st: OrderState = o["state"]
            if st.status.is_terminal:
                continue
            if st.parent_id:
                parent = self.orders.get(st.parent_id)
                if parent is None:
                    continue
                pst = parent["state"]
                if pst.filled == 0:
                    if pst.status.is_terminal:
                        st.status = OrderStatus.CANCELLED
                    continue
                if pst.status != OrderStatus.FILLED and not pst.status.is_terminal:
                    continue  # wait until the entry is complete
                st.quantity = pst.filled
                if st.status == OrderStatus.PENDING_SUBMIT:
                    st.status = OrderStatus.SUBMITTED
            q = self.market.quote(o["inst"])
            if q is None or q.bid <= 0 or q.ask <= 0:
                continue
            px = self._match_price(st, q, o)
            if px is None:
                continue
            new.append(self._fill(cid, st, o["inst"], st.remaining, px))
            if o["oca"]:
                for sib_id, sib in self.orders.items():
                    if sib_id != cid and sib["oca"] == o["oca"] and not sib["state"].status.is_terminal:
                        sib["state"].status = OrderStatus.CANCELLED
        if new:
            self._save()
        return new

    def _match_price(self, st: OrderState, q: Quote, o: dict) -> float | None:
        slip = self.costs.slippage_bps / 1e4
        buy = st.side == OrderSide.BUY
        if st.order_type == OrderType.MARKET:
            return q.ask * (1 + slip) if buy else q.bid * (1 - slip)
        if st.order_type in (OrderType.STOP, OrderType.STOP_LIMIT) and not o["triggered"]:
            hit = q.ask >= st.stop_price if buy else q.bid <= st.stop_price
            if not hit:
                return None
            o["triggered"] = True
            if st.order_type == OrderType.STOP:
                return max(q.ask, st.stop_price) * (1 + slip) if buy else min(q.bid, st.stop_price) * (1 - slip)
        if st.order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT):
            if buy and q.ask <= st.limit_price:
                return min(q.ask, st.limit_price)
            if not buy and q.bid >= st.limit_price:
                return max(q.bid, st.limit_price)
        return None

    def _fill(self, cid: str, st: OrderState, inst: Instrument, qty: int, price: float) -> Fill:
        comm = commission(self.costs, inst.currency, qty, price)
        signed = qty if st.side == OrderSide.BUY else -qty
        ccy = inst.currency
        self.cash[ccy] = self.cash.get(ccy, 0.0) - signed * price - comm
        p = self.pos.get(st.symbol) or Position(st.symbol, 0, 0.0, ccy)
        if p.quantity == 0 or (p.quantity > 0) == (signed > 0):
            new_q = p.quantity + signed
            p.avg_price = (p.avg_price * abs(p.quantity) + price * qty) / abs(new_q)
            p.quantity = new_q
        else:
            closed = min(abs(signed), abs(p.quantity))
            pnl = (price - p.avg_price) * closed * (1 if p.quantity > 0 else -1)
            self.realized_today += pnl * self.fx_rate(ccy, self.base) - comm * self.fx_rate(ccy, self.base)
            p.quantity += signed
            if p.quantity != 0 and abs(signed) > closed:
                p.avg_price = price
        p.market_price = price
        self.pos[st.symbol] = p
        prev_val = (st.avg_fill_price or 0) * st.filled
        st.filled += qty
        st.avg_fill_price = (prev_val + price * qty) / st.filled
        st.status = OrderStatus.FILLED if st.filled >= st.quantity else OrderStatus.PARTIALLY_FILLED
        f = Fill(cid, st.symbol, st.side, qty, price, comm, self.clock(), f"PX{len(self.fills)+1}")
        self.fills.append(f)
        return f

    def reset_day(self) -> None:
        self.realized_today = 0.0
        self._save()


def _ser_order(o: OrderState) -> dict:
    d = asdict(o)
    for k in ("side", "status", "order_type", "role"):
        d[k] = d[k].value
    return d


def _de_order(d: dict) -> OrderState:
    return OrderState(**{**d, "side": OrderSide(d["side"]), "status": OrderStatus(d["status"]),
                         "order_type": OrderType(d["order_type"]), "role": OrderRole(d["role"])})
