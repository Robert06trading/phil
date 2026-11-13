"""Trading engine — one loop for PAPER, REPLAY and LIVE.

Each ``step(now)`` runs, in this order (safety first):
 1. heartbeat
 2. kill switch           -> cancel all + flatten + stop
 3. broker connection     -> reconnect with backoff; no trading while down
 4. account + loss limits -> halt + flatten + notify on breach
 5. live gate (live only) -> stop if approval/evidence no longer valid
 6. sync + reconcile + ensure broker-side stop on every position
 7. end-of-day handling   -> cancel entries / flatten before close
 8. scanner (periodic)    -> ranked candidates
 9. signals -> risk gate -> order manager
10. equity snapshot
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import pandas as pd

from app.domain import Instrument
from broker.base import Broker, BrokerDisconnected, BrokerError
from config import Settings, TradingMode
from execution.order_manager import OrderManager
from market_data import calendar as mcal
from market_data.feeds import StaleDataError, assert_bars_fresh, assert_realtime
from portfolio.store import Database, EquityRow, ScanRow, SignalRow
from risk.manager import PreTradeContext, RiskManager
from scanner.screener import ScanResult, rank, score_symbol
from security import live_gate
from security.instrument_guard import InstrumentGuard, InstrumentNotAllowed
from security.kill_switch import KillSwitch
from strategies.context import build_context
from strategies.signal_engine import SignalEngine
from monitoring.health import Heartbeat

log = logging.getLogger(__name__)


@dataclass
class EngineStatus:
    running: bool = True
    state: str = "init"
    last_scan: datetime | None = None
    top: list[ScanResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class TradingEngine:
    def __init__(self, settings: Settings, broker: Broker, db: Database, guard: InstrumentGuard,
                 risk: RiskManager, signals: SignalEngine, notify, clock=None):
        self.s = settings
        self.broker = broker
        self.db = db
        self.guard = guard
        self.risk = risk
        self.signals = signals
        self.notify = notify
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.kill = KillSwitch(db, settings.kill_switch_file, db.audit)
        self.instruments: dict[str, Instrument] = {s: guard.universe[s] for s in guard.symbols()}
        self.om = OrderManager(broker, db, risk, settings.mode.value, self.instruments, settings.base_currency,
                               notify, self.clock)
        self.hb = Heartbeat(db)
        self.status = EngineStatus()
        self._killed_handled = False
        self._broker_alerted = False
        self._verified: set[str] = set()
        self.replay = settings.mode == TradingMode.REPLAY

    # ------------------------------------------------------------ startup
    def startup(self) -> None:
        self.db.audit("system", "engine.start", {"mode": self.s.mode.value, "broker": self.s.broker.value})
        self.notify("restart", f"Engine starting in {self.s.mode.value.upper()} mode")
        self.broker.connect()
        if self.s.mode == TradingMode.LIVE:
            acct = self.broker.account().account_id
            g = live_gate.evaluate(self.s, self.db, acct, self.kill.engaged(), self.risk.state.halted)
            if not g.allowed:
                self.broker.disconnect()
                raise PermissionError("LIVE refused: " + g.summary())
        # Restart safety: reconcile before anything else
        rep = self.om.sync()
        self.om.ensure_protection(rep)
        if rep.problems:
            self.notify("error", "Startup reconciliation: " + "; ".join(rep.problems))
        self.db.audit("system", "engine.reconciled", {"problems": rep.problems, "flattened": rep.flattened})

    def verify_instrument(self, inst: Instrument) -> Instrument:
        """Broker-side verification (secType/stockType/venue) once per symbol."""
        if inst.symbol in self._verified:
            return inst
        self.guard.check(self.broker.qualify(inst))   # broker-reported fields; unknown stockType fails closed
        self._verified.add(inst.symbol)
        return inst

    # ------------------------------------------------------------ loop
    def step(self) -> str:
        now = self.clock()
        try:
            st = self._step(now)
        except BrokerDisconnected as e:
            st = "broker_down"
            self._alert_broker(str(e))
        except Exception as e:  # noqa: BLE001 - the loop must survive; record & alert
            log.exception("engine step failed")
            st = "error"
            self.status.errors = (self.status.errors + [f"{now.isoformat()} {e}"])[-50:]
            self.db.audit("system", "engine.error", {"error": str(e)}, "ERROR")
            self.notify("error", f"Engine error: {e}")
        self.status.state = st
        self.hb.beat(state=st, mode=self.s.mode.value, broker_connected=self.broker.is_connected(),
                     kill_switch=self.kill.engaged(), halted=self.risk.state.halted,
                     equity=getattr(self, "_last_equity", None))
        return st

    def _alert_broker(self, msg: str) -> None:
        if not self._broker_alerted:
            self.notify("broker_error", msg)
            self.db.audit("system", "broker.disconnected", {"error": msg}, "ERROR")
            self._broker_alerted = True

    def _step(self, now: datetime) -> str:
        self.risk.refresh()
        if self.kill.engaged():
            if not self._killed_handled:
                self._handle_kill()
            return "killed"
        self._killed_handled = False

        if not self.broker.is_connected():
            self.broker.connect()   # raises BrokerDisconnected with backoff
            self.notify("info", "Broker reconnected")
            self.db.audit("system", "broker.reconnected", {})
        self._broker_alerted = False
        if hasattr(self.broker, "process"):
            self.broker.process()

        acct = self.broker.account()
        newly = self.risk.check_loss_limits(now, acct.equity)
        if newly:
            self.notify("loss_limit", f"{newly}. New entries stopped, positions flattened. Manual reset required.")
            self.db.audit("system", "risk.halt", {"reason": newly}, "CRITICAL")
            self.om.flatten_all("loss_limit")

        if self.s.mode == TradingMode.LIVE:
            g = live_gate.evaluate(self.s, self.db, acct.account_id, False, self.risk.state.halted)
            if not g.allowed and not self.risk.state.halted:
                self.kill.engage("system", "live gate failed: " + g.summary())
                return "killed"

        rep = self.om.sync()
        self.om.ensure_protection(rep)
        for p in rep.problems:
            self.status.errors = (self.status.errors + [p])[-50:]

        eod_state = self._end_of_day(now)
        self._snapshot(acct)
        if self.risk.state.halted:
            return "halted"
        if eod_state == "flattening":
            return "eod"

        if not (self.db.get("engine_control") or {"enabled": True}).get("enabled", True):
            return "paused"   # operator stop: positions stay protected (broker stops + EOD), no new entries
        open_venues = [v for v in self.s.venues if mcal.is_open(v, now)]
        if not open_venues:
            return "market_closed"
        if not self.status.last_scan or now - self.status.last_scan >= timedelta(minutes=self.s.scanner.rescan_minutes):
            self._scan(now, open_venues)
        self._trade(now, acct)
        return "running"

    # ------------------------------------------------------------ kill switch
    def _handle_kill(self) -> None:
        reason = self.kill.reason()
        log.critical("KILL SWITCH: %s", reason)
        try:
            if not self.broker.is_connected():
                self.broker.connect()
            closed = self.om.flatten_all("kill_switch")
            if hasattr(self.broker, "process"):
                self.broker.process()
            self.notify("kill_switch", f"Kill switch engaged ({reason}). Flattened: {closed or 'none'}")
            self._killed_handled = True
        except Exception as e:  # noqa: BLE001 - retried next step
            self.notify("kill_switch", f"Kill switch engaged but flatten FAILED, retrying: {e}")

    # ------------------------------------------------------------ EOD
    def _end_of_day(self, now: datetime) -> str:
        state = "normal"
        for venue in self.s.venues:
            mtc = mcal.minutes_to_close(venue, now)
            syms = {s for s, i in self.instruments.items() if i.venue == venue}
            if mtc is None:
                # market closed: an open position here means an unintended overnight hold
                if any(p.symbol in syms for p in self.broker.positions()):
                    self.notify("error", f"{venue} closed with open positions — will flatten at next open")
                continue
            if mtc <= self.s.risk.no_new_entries_minutes_before_close:
                for o in self.broker.open_orders():
                    if o.symbol in syms and o.role.value == "entry" and o.filled == 0:
                        self.broker.cancel_order(o.client_order_id)
            if mtc <= self.s.risk.flatten_minutes_before_close:
                pos = [p for p in self.broker.positions() if p.symbol in syms]
                if pos:
                    for o in self.broker.open_orders():
                        if o.symbol in syms:
                            self.broker.cancel_order(o.client_order_id)
                    for p in pos:
                        self.om._exit_position(p, "EOD")
                    self.db.audit("system", "eod.flatten", {"venue": venue, "symbols": [p.symbol for p in pos]})
                state = "flattening"
            elif mcal.minutes_since_open(venue, now) is not None:
                # first minutes after open: flatten anything held overnight (should never happen)
                held = [p for p in self.broker.positions() if p.symbol in syms]
                orphan = [p for p in held if not any(t.symbol == p.symbol for t in self.db.open_trades(self.s.mode.value))]
                for p in orphan:
                    self.om._exit_position(p, "OVERNIGHT")
        return state

    # ------------------------------------------------------------ scan
    def _bars(self, inst: Instrument, tf: str, days: int, now: datetime) -> pd.DataFrame:
        df = self.broker.bars(inst, tf, days)
        if tf in ("1m", "5m"):
            assert_bars_fresh(df, now, 1 if tf == "1m" else 5, allow_synthetic=self.replay)
        return df

    def _scan(self, now: datetime, venues: list[str]) -> None:
        results = []
        for sym, inst in self.instruments.items():
            if inst.venue not in venues:
                continue
            try:
                intraday = self._bars(inst, "5m", 10, now)
                daily = self.broker.bars(inst, "1d", 40)
                q = self.broker.quote(inst)
                results.append(score_symbol(sym, daily, intraday, tz=mcal.VENUE_TZ[inst.venue],
                                            spread_bps=q.spread_bps if q else None, min_price=self.s.risk.min_price,
                                            min_avg_dollar_volume=self.s.scanner.min_avg_dollar_volume,
                                            max_spread_bps=self.s.risk.max_spread_bps))
            except (StaleDataError, BrokerError, KeyError, ValueError) as e:
                log.info("scan skip %s: %s", sym, e)
        self.status.top = rank(results, self.s.scanner.top_n)
        self.status.last_scan = now
        with self.db.Session.begin() as s:
            for r in self.status.top:
                s.add(ScanRow(ts=now, symbol=r.symbol, score=r.score, bias=r.direction_bias,
                              components=r.components, metrics={k: (None if pd.isna(v) else v) for k, v in r.metrics.items()}))

    # ------------------------------------------------------------ trade
    def _trade(self, now: datetime, acct) -> None:
        for r in self.status.top:
            inst = self.instruments[r.symbol]
            if not mcal.is_open(inst.venue, now):
                continue
            try:
                self.guard.check(inst)
                self.verify_instrument(inst)
                raw = {"5m": self._bars(inst, "5m", 10, now), "15m": self.broker.bars(inst, "15m", 15),
                       "1h": self.broker.bars(inst, "1h", 30)}
                try:
                    raw["1m"] = self._bars(inst, "1m", 2, now)
                except StaleDataError:
                    pass
                q = self.broker.quote(inst)
                ctx = build_context(r.symbol, inst.venue, raw, "5m", q, r.score, r.levels)
                if ctx is None:
                    continue
                res = self.signals.evaluate(ctx)
                cand = res.candidate
                if cand is None:
                    continue
                fresh = True
                try:
                    assert_realtime(q, now, self.s.risk.max_data_age_seconds, allow_replay=self.replay)
                except StaleDataError:
                    fresh = False
                fx = self.broker.fx_rate(inst.currency, self.s.base_currency)
                pctx = PreTradeContext(now, acct, self.broker.positions(), self.broker.open_orders(), q, inst,
                                       self.broker.is_connected(), fresh, fx, self.kill.engaged())
                dec = self.risk.evaluate(cand, pctx)
                with self.db.Session.begin() as s:
                    s.add(SignalRow(ts=now, symbol=cand.symbol, direction=cand.direction.value, entry=cand.entry,
                                    stop=cand.stop, target=cand.target, quality=cand.quality_score,
                                    strategies=",".join(cand.strategies),
                                    decision="approved" if dec.approved else "rejected",
                                    reasons=cand.reasons + dec.reasons))
                if dec.approved:
                    ref = q.ask if cand.direction.value == "LONG" else q.bid
                    self.om.submit(cand, dec, inst, ref)
                    acct = self.broker.account()
            except (StaleDataError, InstrumentNotAllowed, BrokerError, KeyError) as e:
                log.info("trade skip %s: %s", r.symbol, e)

    def _snapshot(self, acct) -> None:
        self._last_equity = acct.equity
        with self.db.Session.begin() as s:
            s.add(EquityRow(ts=self.clock(), mode=self.s.mode.value, equity=acct.equity, cash=acct.cash,
                            unrealized=acct.unrealized_pnl, realized_today=acct.realized_pnl_today))

    def shutdown(self) -> None:
        self.db.audit("system", "engine.stop", {})
        self.broker.disconnect()
