"""Order manager, position manager and execution monitor.

Responsibilities
* submit: deterministic client ids + write-ahead persistence + broker
  bracket (entry, stop-loss, take-profit). The same candidate can never be
  sent twice (DB primary key + broker orderRef check).
* sync: pull order states and fills from the broker, detect partial fills,
  resize protective legs, close trades and book P&L.
* ensure_protection: every open position must have a live broker-side stop
  covering its full size; otherwise one is placed immediately (or the
  position is flattened if price is already through the stop).
* reconcile: broker = source of truth. Unknown broker positions are flattened
  and reported; internal trades without a broker position are closed from
  executions.
* flatten_all: cancel everything and close every position (EOD, kill switch,
  loss limit).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from app.domain import (BracketRequest, Direction, Instrument, OrderRequest, OrderRole, OrderSide, OrderState,
                        OrderStatus, OrderType, Position, TradeCandidate)
from broker.base import Broker, BrokerError
from portfolio.store import Database, TradeRow
from risk.manager import RiskDecision, RiskManager

log = logging.getLogger(__name__)
ENTRY_LIMIT_OFFSET_BPS = 5.0     # marketable limit: never chase more than 5 bps through the book


@dataclass
class SyncReport:
    closed_trades: list[dict] = field(default_factory=list)
    resized: list[str] = field(default_factory=list)
    protective_placed: list[str] = field(default_factory=list)
    flattened: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


class OrderManager:
    def __init__(self, broker: Broker, db: Database, risk: RiskManager, mode: str, instruments: dict[str, Instrument],
                 base_currency: str, notify=None, clock=None):
        self.broker = broker
        self.db = db
        self.risk = risk
        self.mode = mode
        self.instruments = instruments
        self.base = base_currency
        self.notify = notify or (lambda *a, **k: None)
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    # ------------------------------------------------------------ submit
    def submit(self, cand: TradeCandidate, decision: RiskDecision, inst: Instrument, ref_price: float) -> list[OrderState]:
        if not decision.approved or decision.quantity < 1:
            raise ValueError("refusing to submit an unapproved trade")
        cid = cand.client_order_id
        with self.db.session() as s:
            if s.scalar(select(TradeRow).where(TradeRow.trade_key == cid)) is not None:
                raise BrokerError(f"duplicate trade {cid} (already in DB)")
        long = cand.direction is Direction.LONG
        entry_side, exit_side = (OrderSide.BUY, OrderSide.SELL) if long else (OrderSide.SELL, OrderSide.BUY)
        off = ENTRY_LIMIT_OFFSET_BPS / 1e4
        limit = ref_price * (1 + off) if long else ref_price * (1 - off)
        qty = decision.quantity
        br = BracketRequest(
            OrderRequest(cid, cand.symbol, entry_side, qty, OrderType.LIMIT, OrderRole.ENTRY, limit_price=round(limit, 2)),
            OrderRequest(cid + "-SL", cand.symbol, exit_side, qty, OrderType.STOP, OrderRole.STOP,
                         stop_price=round(cand.stop, 2), parent_id=cid, oca_group="OCA-" + cid),
            OrderRequest(cid + "-TP", cand.symbol, exit_side, qty, OrderType.LIMIT, OrderRole.TAKE_PROFIT,
                         limit_price=round(cand.target, 2), parent_id=cid, oca_group="OCA-" + cid),
        )
        br.validate()
        # write-ahead: intent is durable before anything leaves the process
        with self.db.Session.begin() as s:
            s.add(TradeRow(trade_key=cid, mode=self.mode, symbol=cand.symbol, direction=cand.direction.value,
                           strategies=",".join(cand.strategies), quantity=qty, stop_price=cand.stop,
                           target_price=cand.target, quality_score=cand.quality_score,
                           risk_amount=decision.risk_amount, status="open", opened_at=self.clock(),
                           meta={"reasons": cand.reasons, "risk_notes": decision.reasons}))
        for leg in (br.entry, br.stop, br.take_profit):
            self._persist(leg, OrderStatus.PENDING_SUBMIT)
        try:
            states = self.broker.place_bracket(br, inst)
        except Exception as e:
            self.db.audit("system", "order.submit_failed", {"cid": cid, "error": str(e)}, "ERROR")
            raise
        for st in states:
            self._persist_state(st)
        self.risk.record_entry(cand.symbol, decision.risk_amount)
        self.db.audit("system", "order.submitted", {"cid": cid, "symbol": cand.symbol, "qty": qty,
                                                    "dir": cand.direction.value, "stop": cand.stop,
                                                    "target": cand.target})
        self.notify("trade_opened", f"{cand.direction.value} {qty} {cand.symbol} @~{ref_price:.2f} "
                                    f"SL {cand.stop:.2f} TP {cand.target:.2f} ({','.join(cand.strategies)})")
        return states

    def _persist(self, req: OrderRequest, status: OrderStatus) -> None:
        self.db.upsert_order(self.mode, client_order_id=req.client_order_id, symbol=req.symbol, side=req.side.value,
                             role=req.role.value, order_type=req.order_type.value, quantity=req.quantity,
                             limit_price=req.limit_price, stop_price=req.stop_price, parent_id=req.parent_id,
                             status=status.value, created_at=self.clock())

    def _persist_state(self, st: OrderState) -> None:
        self.db.upsert_order(self.mode, client_order_id=st.client_order_id, broker_order_id=st.broker_order_id,
                             symbol=st.symbol, side=st.side.value, role=st.role.value,
                             order_type=st.order_type.value, quantity=st.quantity, filled=st.filled,
                             avg_fill_price=st.avg_fill_price, status=st.status.value)

    # ------------------------------------------------------------ sync
    def sync(self, entry_timeout: timedelta = timedelta(minutes=10)) -> SyncReport:
        rep = SyncReport()
        broker_orders = {o.client_order_id: o for o in self.broker.open_orders()}
        positions = {p.symbol: p for p in self.broker.positions()}
        now = self.clock()
        fills = self.broker.fills_since(now - timedelta(days=1))
        filled_by_cid: dict[str, int] = {}
        for f in fills:
            filled_by_cid[f.client_order_id] = filled_by_cid.get(f.client_order_id, 0) + f.quantity

        for row in self.db.open_order_rows(self.mode):
            st = broker_orders.get(row.client_order_id)
            if st is not None:
                self._persist_state(st)
                continue
            # not open at broker any more: decide terminal status from executions
            done = filled_by_cid.get(row.client_order_id, 0)
            status = OrderStatus.FILLED if done >= row.quantity else OrderStatus.CANCELLED
            if row.status == OrderStatus.PENDING_SUBMIT.value and done == 0 and \
                    (now - _aware(row.created_at)) < timedelta(seconds=30):
                continue  # may still be in flight
            self.db.upsert_order(self.mode, client_order_id=row.client_order_id, status=status.value, filled=done)

        for tr in self.db.open_trades(self.mode):
            cid = tr.trade_key
            entry_filled = filled_by_cid.get(cid, 0)
            entry = broker_orders.get(cid)
            # stale, unfilled entry -> cancel
            if entry is not None and entry.filled == 0 and now - _aware(tr.opened_at) > entry_timeout:
                self.broker.cancel_order(cid)
                rep.problems.append(f"{tr.symbol}: entry not filled in time, cancelled")
            # partial fill finished (entry no longer working) -> resize protective legs
            if entry is None and 0 < entry_filled < tr.quantity:
                for leg in (cid + "-SL", cid + "-TP"):
                    lo = broker_orders.get(leg)
                    if lo is not None and lo.quantity != entry_filled:
                        self.broker.modify_quantity(leg, entry_filled)
                        rep.resized.append(leg)
                with self.db.Session.begin() as s:
                    s.get(TradeRow, tr.id).quantity = entry_filled
            pos = positions.get(tr.symbol)
            if entry is None and entry_filled == 0 and pos is None:
                self._close_trade(tr, None, "entry_not_filled", 0.0, fills, rep)
            elif entry_filled > 0 and pos is None and entry is None:
                self._close_from_fills(tr, fills, rep)
        return rep

    def _close_from_fills(self, tr: TradeRow, fills, rep: SyncReport) -> None:
        cid = tr.trade_key
        mine = [f for f in fills if f.client_order_id == cid or f.client_order_id.startswith(cid + "-")]
        inst = self.instruments[tr.symbol]
        fx = self.broker.fx_rate(inst.currency, self.base)
        cash = sum((-1 if f.side == OrderSide.BUY else 1) * f.quantity * f.price for f in mine)
        comm = sum(f.commission for f in mine)
        exits = [f for f in mine if f.client_order_id != cid]
        reason = "unknown"
        if exits:
            last = max(exits, key=lambda f: f.ts)
            tail = last.client_order_id[len(cid):]
            if tail.startswith("-X"):
                reason = tail[2:].split("-")[0].lower() or "exit"   # eod / kill / limit / stopgap / manual
            elif tail.startswith("-SL"):
                reason = "stop_loss"
            elif tail.startswith("-TP"):
                reason = "take_profit"
        exit_px = (sum(f.quantity * f.price for f in exits) / sum(f.quantity for f in exits)) if exits else None
        entry_f = [f for f in mine if f.client_order_id == cid]
        entry_px = sum(f.quantity * f.price for f in entry_f) / sum(f.quantity for f in entry_f) if entry_f else None
        pnl = (cash - comm) * fx
        self._close_trade(tr, exit_px, reason, pnl, fills, rep, entry_px, comm * fx)

    def _close_trade(self, tr: TradeRow, exit_px, reason: str, pnl: float, fills, rep: SyncReport,
                     entry_px=None, comm: float = 0.0) -> None:
        with self.db.Session.begin() as s:
            row = s.get(TradeRow, tr.id)
            row.status, row.exit_reason, row.exit_price, row.pnl = "closed", reason, exit_px, round(pnl, 2)
            row.closed_at, row.commission = self.clock(), comm
            if entry_px:
                row.entry_price = entry_px
        self.risk.release_risk(tr.symbol)
        if reason != "entry_not_filled":
            self.risk.record_trade_result(pnl)
            event = {"stop_loss": "stop_loss", "take_profit": "take_profit"}.get(reason, "trade_closed")
            self.notify(event, f"{tr.symbol} closed ({reason}) P&L {pnl:+.2f} {self.base}")
        rep.closed_trades.append({"symbol": tr.symbol, "reason": reason, "pnl": pnl})
        self.db.audit("system", "trade.closed", {"symbol": tr.symbol, "reason": reason, "pnl": pnl})

    # ------------------------------------------------------------ protection
    def ensure_protection(self, rep: SyncReport | None = None) -> SyncReport:
        rep = rep or SyncReport()
        orders = [o for o in self.broker.open_orders() if not o.status.is_terminal]
        open_trades = {t.symbol: t for t in self.db.open_trades(self.mode)}
        for p in self.broker.positions():
            exit_side = OrderSide.SELL if p.quantity > 0 else OrderSide.BUY
            stops = [o for o in orders if o.symbol == p.symbol and o.side == exit_side
                     and o.order_type in (OrderType.STOP, OrderType.STOP_LIMIT)]
            covered = sum(o.remaining for o in stops)
            if covered >= abs(p.quantity):
                continue
            tr = open_trades.get(p.symbol)
            if tr is None:
                rep.problems.append(f"{p.symbol}: unknown broker position {p.quantity} -> flatten")
                self._exit_position(p, "UNKNOWN")
                rep.flattened.append(p.symbol)
                self.notify("error", f"Unknown position {p.symbol} {p.quantity} flattened")
                continue
            if p.symbol not in self.instruments:
                rep.problems.append(f"{p.symbol}: no instrument for protection")
                continue
            inst = self.instruments[p.symbol]
            q = self.broker.quote(inst)
            through = q is not None and ((p.quantity > 0 and q.bid <= tr.stop_price) or
                                         (p.quantity < 0 and q.ask >= tr.stop_price))
            if through:
                self._exit_position(p, "STOPGAP")
                rep.flattened.append(p.symbol)
                continue
            missing = abs(p.quantity) - covered
            cid = f"{tr.trade_key}-SLX{int(self.clock().timestamp())}"
            req = OrderRequest(cid, p.symbol, exit_side, missing, OrderType.STOP, OrderRole.STOP,
                               stop_price=round(tr.stop_price, 2))
            self._persist(req, OrderStatus.PENDING_SUBMIT)
            self._persist_state(self.broker.place_order(req, inst))
            rep.protective_placed.append(cid)
            self.db.audit("system", "protection.placed", {"symbol": p.symbol, "qty": missing}, "WARNING")
        return rep

    def _exit_position(self, p: Position, tag: str) -> None:
        inst = self.instruments.get(p.symbol) or Instrument(p.symbol, "UNKNOWN", p.currency, stock_type="COMMON")
        tr = next((t for t in self.db.open_trades(self.mode) if t.symbol == p.symbol), None)
        base = tr.trade_key if tr else f"ORPHAN-{p.symbol}"
        cid = f"{base}-X{tag}-{int(self.clock().timestamp())}"
        req = OrderRequest(cid, p.symbol, OrderSide.SELL if p.quantity > 0 else OrderSide.BUY, abs(p.quantity),
                           OrderType.MARKET, OrderRole.EXIT)
        self._persist(req, OrderStatus.PENDING_SUBMIT)
        self._persist_state(self.broker.place_order(req, inst))

    # ------------------------------------------------------------ flatten
    def cancel_all(self) -> int:
        n = 0
        for o in self.broker.open_orders():
            if not o.status.is_terminal:
                self.broker.cancel_order(o.client_order_id)
                n += 1
        return n

    def cancel_entries(self) -> int:
        n = 0
        for o in self.broker.open_orders():
            if o.role == OrderRole.ENTRY and not o.status.is_terminal and o.filled == 0:
                self.broker.cancel_order(o.client_order_id)
                n += 1
        return n

    def flatten_all(self, reason: str) -> list[str]:
        tag = {"eod": "EOD", "kill_switch": "KILL", "loss_limit": "LIMIT"}.get(reason, "MANUAL")
        self.cancel_all()
        closed = []
        for p in self.broker.positions():
            self._exit_position(p, tag)
            closed.append(p.symbol)
        self.db.audit("system", "flatten_all", {"reason": reason, "symbols": closed}, "WARNING")
        return closed


def _aware(ts: datetime) -> datetime:
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def opposite(d: Direction) -> OrderSide:
    return OrderSide.SELL if d is Direction.LONG else OrderSide.BUY
