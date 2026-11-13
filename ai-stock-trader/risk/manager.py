"""Risk manager — the final authority before any order.

Every order (paper, live, backtest) passes ``RiskManager.evaluate``. The
checks are deliberately redundant with checks elsewhere (defence in depth).
Nothing outside this module can change the limits at runtime: limits come
from ``RiskConfig`` (hard ceilings validated in config.py) and the risk state
can only be made *stricter* automatically; un-halting requires
``manual_reset`` with an operator id, which is audit-logged by the caller.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Protocol

from app.domain import AccountSnapshot, Direction, Instrument, OrderState, Position, Quote, TradeCandidate
from config import CostConfig, RiskConfig
from market_data import calendar as mcal
from risk.costs import round_trip_cost_per_share
from security.instrument_guard import InstrumentGuard, InstrumentNotAllowed


class StateStore(Protocol):
    def get(self, key: str) -> dict | None: ...
    def set(self, key: str, value: dict) -> None: ...


class MemoryStateStore:
    def __init__(self) -> None:
        self._d: dict[str, dict] = {}

    def get(self, key: str) -> dict | None:
        v = self._d.get(key)
        return dict(v) if v is not None else None

    def set(self, key: str, value: dict) -> None:
        self._d[key] = dict(value)


@dataclass
class RiskState:
    day: str = ""
    day_start_equity: float = 0.0
    week: str = ""
    week_start_equity: float = 0.0
    halted: bool = False
    halt_reason: str = ""
    loss_streak: int = 0
    trades_today: int = 0
    open_risk: dict[str, float] = field(default_factory=dict)  # symbol -> risk in base ccy


@dataclass
class PreTradeContext:
    now: datetime
    account: AccountSnapshot
    positions: list[Position]
    open_orders: list[OrderState]
    quote: Quote | None
    instrument: Instrument
    broker_connected: bool
    data_fresh: bool
    fx_to_base: float                       # 1 unit of instrument ccy in base ccy
    kill_switch_engaged: bool = False


@dataclass
class RiskDecision:
    approved: bool
    quantity: int = 0
    reasons: list[str] = field(default_factory=list)
    risk_amount: float = 0.0               # base currency at stop, incl. est. costs

    def reject(self, why: str) -> "RiskDecision":
        self.approved = False
        self.reasons.append(why)
        return self


class LimitBreached(Exception):
    pass


class RiskManager:
    STATE_KEY = "risk_state"

    def __init__(self, cfg: RiskConfig, costs: CostConfig, guard: InstrumentGuard, store: StateStore,
                 allow_extended_hours: bool = False):
        self.cfg = cfg
        self.costs = costs
        self.guard = guard
        self.store = store
        self.allow_extended_hours = allow_extended_hours
        raw = store.get(self.STATE_KEY)
        self.state = RiskState(**raw) if raw else RiskState()

    # ----------------------------------------------------------- state
    def refresh(self) -> None:
        """Re-read persisted state (the API may have reset a halt)."""
        raw = self.store.get(self.STATE_KEY)
        if raw:
            self.state = RiskState(**raw)

    def _save(self) -> None:
        self.store.set(self.STATE_KEY, asdict(self.state))

    def roll_periods(self, now: datetime, equity: float) -> None:
        """Set day/week start equity on the first observation of a new period."""
        d = now.date()
        iso = d.isocalendar()
        week = f"{iso[0]}-W{iso[1]:02d}"
        changed = False
        if self.state.day != d.isoformat():
            self.state.day, self.state.day_start_equity, self.state.trades_today = d.isoformat(), equity, 0
            changed = True
        if self.state.week != week:
            self.state.week, self.state.week_start_equity = week, equity
            changed = True
        if changed:
            self._save()

    def check_loss_limits(self, now: datetime, equity: float) -> str | None:
        """Update periods and halt (persistently) if a loss limit is hit.
        Returns the halt reason if newly halted."""
        self.roll_periods(now, equity)
        if self.state.halted:
            return None
        day_loss = (self.state.day_start_equity - equity) / self.state.day_start_equity * 100 if self.state.day_start_equity else 0
        week_loss = (self.state.week_start_equity - equity) / self.state.week_start_equity * 100 if self.state.week_start_equity else 0
        reason = None
        if day_loss >= self.cfg.max_daily_loss_pct:
            reason = f"daily loss {day_loss:.2f}% >= {self.cfg.max_daily_loss_pct}%"
        elif week_loss >= self.cfg.max_weekly_loss_pct:
            reason = f"weekly loss {week_loss:.2f}% >= {self.cfg.max_weekly_loss_pct}%"
        if reason:
            self.halt(reason)
        return reason

    def halt(self, reason: str) -> None:
        self.state.halted = True
        self.state.halt_reason = reason
        self._save()

    def manual_reset(self, operator: str, equity: float, now: datetime, rebase_equity: bool = False) -> None:
        """Only path out of a halt (audit-logged by the API layer). By default
        the day/week anchors are kept, so a reset on the same day re-halts on
        the next check while the loss is still beyond the limit — trading then
        effectively resumes with the next period. ``rebase_equity=True`` is an
        explicit operator decision to grant a fresh daily budget today."""
        if not operator:
            raise PermissionError("operator id required")
        self.state.halted = False
        self.state.halt_reason = f"reset by {operator} at {now.isoformat()}"
        if rebase_equity:
            self.state.day_start_equity = equity
        self._save()

    def record_trade_result(self, pnl: float) -> None:
        self.state.loss_streak = self.state.loss_streak + 1 if pnl < 0 else 0
        self._save()

    def record_entry(self, symbol: str, risk_amount: float) -> None:
        self.state.trades_today += 1
        self.state.open_risk[symbol] = risk_amount
        self._save()

    def release_risk(self, symbol: str) -> None:
        if self.state.open_risk.pop(symbol, None) is not None:
            self._save()

    # ----------------------------------------------------------- sizing
    def effective_risk_pct(self) -> float:
        pct = self.cfg.risk_per_trade_pct
        if self.state.loss_streak >= self.cfg.loss_streak_reduce_after:
            pct *= self.cfg.loss_streak_risk_factor   # reduce only — never increase (no martingale)
        return min(pct, self.cfg.risk_per_trade_pct)

    def size(self, cand: TradeCandidate, equity: float, buying_power: float, fx: float, currency: str,
             spread_bps: float | None = None) -> tuple[int, float, list[str]]:
        notes: list[str] = []
        risk_ps = abs(cand.entry - cand.stop)
        if risk_ps <= 0 or fx <= 0 or equity <= 0:
            return 0, 0.0, ["invalid sizing inputs"]
        cost_ps = round_trip_cost_per_share(self.costs, currency, cand.entry,
                                            half_spread_bps=spread_bps / 2 if spread_bps else None)
        budget = equity * self.effective_risk_pct() / 100.0
        qty = math.floor(budget / ((risk_ps + cost_ps) * fx))
        cap_notional = math.floor(equity * self.cfg.max_position_notional_pct / 100 / (cand.entry * fx))
        cap_bp = math.floor(0.95 * buying_power / (cand.entry * fx))
        if qty > cap_notional:
            notes.append(f"capped by notional limit {cap_notional}")
            qty = cap_notional
        if qty > cap_bp:
            notes.append(f"capped by buying power {cap_bp}")
            qty = cap_bp
        heat_cap = equity * self.cfg.max_portfolio_risk_pct / 100 - sum(self.state.open_risk.values())
        if qty * (risk_ps + cost_ps) * fx > heat_cap:
            qty = max(0, math.floor(heat_cap / ((risk_ps + cost_ps) * fx)))
            notes.append(f"capped by portfolio heat -> {qty}")
        return max(qty, 0), qty * (risk_ps + cost_ps) * fx, notes

    # ----------------------------------------------------------- gate
    def evaluate(self, cand: TradeCandidate, ctx: PreTradeContext) -> RiskDecision:
        d = RiskDecision(True)
        if ctx.kill_switch_engaged:
            return d.reject("kill switch engaged")
        if self.state.halted:
            return d.reject(f"trading halted: {self.state.halt_reason}")
        try:
            self.guard.check(ctx.instrument)
        except InstrumentNotAllowed as e:
            return d.reject(f"instrument blocked: {e}")
        if ctx.instrument.symbol != cand.symbol:
            return d.reject("instrument/candidate symbol mismatch")
        if not ctx.broker_connected:
            return d.reject("broker disconnected")
        if not ctx.data_fresh or ctx.quote is None:
            return d.reject("market data missing or stale")
        venue = ctx.instrument.venue
        if not self.allow_extended_hours and not mcal.is_open(venue, ctx.now):
            return d.reject(f"{venue} regular session closed")
        mtc = mcal.minutes_to_close(venue, ctx.now)
        if mtc is not None and mtc <= self.cfg.no_new_entries_minutes_before_close:
            return d.reject(f"{mtc:.0f} min to close: no new entries")
        mso = mcal.minutes_since_open(venue, ctx.now)
        if mso is not None and mso < self.cfg.no_entries_minutes_after_open:
            return d.reject("opening auction window")
        if ctx.quote.spread_bps > self.cfg.max_spread_bps:
            return d.reject(f"spread {ctx.quote.spread_bps:.1f}bps > {self.cfg.max_spread_bps}")
        if ctx.quote.last < self.cfg.min_price:
            return d.reject(f"price below {self.cfg.min_price}")
        reason = self.check_loss_limits(ctx.now, ctx.account.equity)
        if reason or self.state.halted:
            return d.reject(f"loss limit: {reason or self.state.halt_reason}")
        open_pos = [p for p in ctx.positions if p.quantity != 0]
        if any(p.symbol == cand.symbol for p in open_pos):
            return d.reject("position already open in symbol")
        if any(o.symbol == cand.symbol and not o.status.is_terminal for o in ctx.open_orders):
            return d.reject("open order already exists for symbol")
        if any(o.client_order_id == cand.client_order_id for o in ctx.open_orders):
            return d.reject("duplicate client order id")
        pending_entries = {o.symbol for o in ctx.open_orders if o.role.value == "entry" and not o.status.is_terminal}
        if len(open_pos) + len(pending_entries) >= self.cfg.max_open_positions:
            return d.reject(f"max open positions {self.cfg.max_open_positions} reached")
        if cand.direction is Direction.LONG and not (cand.stop < cand.entry < cand.target):
            return d.reject("invalid long geometry")
        if cand.direction is Direction.SHORT and not (cand.target < cand.entry < cand.stop):
            return d.reject("invalid short geometry")
        if cand.reward_risk + 1e-9 < self.cfg.min_reward_risk:
            return d.reject(f"R:R {cand.reward_risk:.2f} < {self.cfg.min_reward_risk}")
        # re-anchor entry to the executable side of the book; stop distance must survive
        exec_px = ctx.quote.ask if cand.direction is Direction.LONG else ctx.quote.bid
        if (exec_px - cand.stop) * cand.direction.sign <= 0:
            return d.reject("price already through stop")
        if abs(cand.target - exec_px) / abs(exec_px - cand.stop) < self.cfg.min_reward_risk:
            return d.reject("R:R below minimum at executable price")
        qty, risk_amt, notes = self.size(cand, ctx.account.equity, ctx.account.buying_power, ctx.fx_to_base,
                                         ctx.instrument.currency, ctx.quote.spread_bps)
        d.reasons.extend(notes)
        if qty < 1:
            return d.reject("position size < 1 share after limits")
        d.quantity, d.risk_amount = qty, risk_amt
        d.reasons.append(f"approved qty={qty} risk={risk_amt:.2f} ({self.effective_risk_pct():.2f}%)")
        return d
