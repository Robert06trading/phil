"""Event-driven intraday backtester.

Re-uses the production components — strategies, SignalEngine, RiskManager
(incl. sizing, loss limits, R:R, max positions, session/EOD windows) and the
shared cost model — so the backtest trades by the same rulebook as paper and
live.

Realism / bias controls
* Look-ahead: indicators are causal (tested); higher timeframes only expose
  bars whose CLOSE <= now; signals at bar t are filled at bar t+1's OPEN.
* Costs: commission (shared model), slippage + half spread on every fill.
* Liquidity / partial fills: an entry fills at most
  ``max_participation_rate * volume`` of the fill bar; the rest is dropped.
* Stops: if a bar gaps through the stop, the exit is at the (worse) open.
  If stop and target are both inside one bar, the STOP is assumed first.
* Sessions: exchange calendars (holidays, early closes, DST); positions are
  flattened ``flatten_minutes_before_close`` before the close.
* Corporate actions: feed split-adjusted data (``apply_splits``).
* Survivorship: pass ``universe_history`` (point-in-time membership) and
  include delisted symbols; a symbol whose data ends is closed at its last
  price (delisting).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pandas as pd

from app.domain import AccountSnapshot, Instrument, Position, Quote, TradeCandidate
from config import Settings
from indicators.technical import enrich, resample_bars
from market_data import calendar as mcal
from market_data.feeds import UniverseHistory
from risk.costs import commission
from risk.manager import MemoryStateStore, PreTradeContext, RiskManager
from scanner.screener import daily_from_intraday, score_symbol
from security.instrument_guard import InstrumentGuard
from strategies.base import MarketContext
from strategies.library import build
from strategies.signal_engine import SignalEngine


@dataclass
class BTTrade:
    symbol: str
    direction: str
    strategies: str
    entry_ts: datetime
    entry_px: float
    qty: int
    stop: float
    target: float
    risk_amount: float
    fx: float
    currency: str
    exit_ts: datetime | None = None
    exit_px: float | None = None
    exit_reason: str = ""
    pnl: float = 0.0
    costs: float = 0.0
    quality: float = 0.0
    entry_hour: int = 0

    @property
    def r_multiple(self) -> float:
        return self.pnl / self.risk_amount if self.risk_amount else 0.0


@dataclass
class BTResult:
    trades: list[BTTrade]
    equity: pd.Series
    start_equity: float
    params: dict
    halts: list[tuple[datetime, str]] = field(default_factory=list)
    synthetic_data: bool = False

    def trades_frame(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        rows = [{**t.__dict__, "r_multiple": t.r_multiple} for t in self.trades]
        return pd.DataFrame(rows)


class _Prepared:
    def __init__(self, symbol: str, inst: Instrument, df: pd.DataFrame):
        tz = mcal.VENUE_TZ[inst.venue]
        self.symbol, self.inst, self.tz = symbol, inst, tz
        self.raw = df
        self.base = enrich(df, tz)
        self.f15 = enrich(resample_bars(df, "15min"), tz)
        self.f1h = enrich(resample_bars(df, "1h"), tz)
        self.idx = self.base.index
        self.pos = {ts: i for i, ts in enumerate(self.idx)}
        self.last_ts = self.idx[-1]

    def frames_at(self, i: int, tail: int = 260) -> dict[str, pd.DataFrame]:
        now = self.idx[i]
        out = {"5m": self.base.iloc[max(0, i - tail + 1): i + 1]}
        for name, f in (("15m", self.f15), ("1h", self.f1h)):
            j = f.index.searchsorted(now, side="right")       # bars with close <= now
            out[name] = f.iloc[max(0, j - 80): j]
        return out


class Backtester:
    def __init__(self, settings: Settings, instruments: dict[str, Instrument], strategy_params: dict | None = None,
                 fx: dict[tuple[str, str], float] | None = None, universe_history: UniverseHistory | None = None,
                 use_scanner: bool = True, scan_every_minutes: int = 60, auto_reset_halt_next_day: bool = True):
        self.s = settings
        self.instruments = instruments
        self.params = strategy_params or {}
        self.fx = fx or {}
        self.uh = universe_history
        self.use_scanner = use_scanner
        self.scan_every = timedelta(minutes=scan_every_minutes)
        self.auto_reset = auto_reset_halt_next_day

    def _fx(self, ccy: str) -> float:
        base = self.s.base_currency
        if ccy == base:
            return 1.0
        if (ccy, base) in self.fx:
            return self.fx[(ccy, base)]
        if (base, ccy) in self.fx:
            return 1 / self.fx[(base, ccy)]
        if self.s.fx_eurusd_fallback and {ccy, base} == {"EUR", "USD"}:
            r = self.s.fx_eurusd_fallback
            return r if ccy == "EUR" else 1 / r
        raise KeyError(f"no FX rate {ccy}->{base}")

    def run(self, frames: dict[str, pd.DataFrame], start_equity: float, start: datetime | None = None,
            end: datetime | None = None) -> BTResult:
        s = self.s
        guard = InstrumentGuard(self.instruments, s.venues, s.allowed_stock_types)
        store = MemoryStateStore()
        risk = RiskManager(s.risk, s.costs, guard, store)
        engine = SignalEngine(build(s.signals.enabled_strategies, self.params), s.signals, s.risk.min_reward_risk)
        prep = {sym: _Prepared(sym, self.instruments[sym], df) for sym, df in frames.items() if sym in self.instruments}
        synthetic = any(df.attrs.get("synthetic") for df in frames.values())
        timeline = sorted(set().union(*[set(p.idx) for p in prep.values()]))
        if start is not None:
            timeline = [t for t in timeline if t >= pd.Timestamp(start)]
        if end is not None:
            timeline = [t for t in timeline if t <= pd.Timestamp(end)]

        cash = start_equity
        open_tr: dict[str, BTTrade] = {}
        pending: dict[str, tuple[TradeCandidate, int]] = {}
        trades: list[BTTrade] = []
        equity_pts: list[tuple[pd.Timestamp, float]] = []
        halts: list[tuple[datetime, str]] = []
        scan_scores: dict[str, float] = {}
        last_scan: pd.Timestamp | None = None
        last_px: dict[str, float] = {}
        cur_day = None
        slip = s.costs.slippage_bps / 1e4
        hs = s.costs.default_half_spread_bps / 1e4

        def close(tr: BTTrade, ts, px_raw: float, reason: str) -> None:
            nonlocal cash
            d = 1 if tr.direction == "LONG" else -1
            px = px_raw * (1 - d * (slip + hs))
            comm = commission(s.costs, tr.currency, tr.qty, px)
            gross = (px - tr.entry_px) * tr.qty * d
            pnl = (gross - comm) * tr.fx - tr.costs
            cash += (gross - comm) * tr.fx
            tr.exit_ts, tr.exit_px, tr.exit_reason = ts.to_pydatetime(), px, reason
            tr.costs += comm * tr.fx
            tr.pnl = pnl
            trades.append(tr)
            risk.release_risk(tr.symbol)
            risk.record_trade_result(pnl)
            open_tr.pop(tr.symbol, None)

        for ts in timeline:
            now = ts.to_pydatetime()
            if cur_day != now.date():
                cur_day = now.date()
                if self.auto_reset and risk.state.halted:
                    risk.manual_reset("backtest-auto-next-day", cash, now, rebase_equity=True)
            # 1) fills of pending entries at this bar's open
            for sym, (cand, qty) in list(pending.items()):
                p = prep[sym]
                i = p.pos.get(ts)
                if i is None:
                    continue
                pending.pop(sym)
                bar = p.base.iloc[i]
                d = cand.direction.sign
                fill = bar["open"] * (1 + d * (slip + hs))
                if (fill - cand.stop) * d <= 0 or (cand.target - fill) * d <= 0:
                    continue   # gapped through stop/target -> entry cancelled
                cap = int(bar["volume"] * s.costs.max_participation_rate)
                q = min(qty, cap)
                if q < 1:
                    continue
                fx = self._fx(p.inst.currency)
                comm = commission(s.costs, p.inst.currency, q, fill)
                cash -= comm * fx
                local = ts.tz_convert(p.tz)
                tr = BTTrade(sym, cand.direction.value, ",".join(cand.strategies), now, fill, q, cand.stop,
                             cand.target, abs(fill - cand.stop) * q * fx, fx, p.inst.currency, costs=comm * fx,
                             quality=cand.quality_score, entry_hour=local.hour)
                open_tr[sym] = tr
                risk.record_entry(sym, tr.risk_amount)
            # 2) exits on this bar
            for sym, tr in list(open_tr.items()):
                p = prep[sym]
                i = p.pos.get(ts)
                if i is None:
                    continue
                bar = p.base.iloc[i]
                last_px[sym] = bar["close"]
                long = tr.direction == "LONG"
                hit_stop = bar["low"] <= tr.stop if long else bar["high"] >= tr.stop
                hit_tgt = bar["high"] >= tr.target if long else bar["low"] <= tr.target
                if hit_stop:
                    gap = bar["open"] <= tr.stop if long else bar["open"] >= tr.stop
                    close(tr, ts, bar["open"] if gap else tr.stop, "stop_loss")
                elif hit_tgt:
                    close(tr, ts, tr.target, "take_profit")
                else:
                    mtc = mcal.minutes_to_close(p.inst.venue, now - timedelta(seconds=1))
                    if mtc is not None and mtc <= s.risk.flatten_minutes_before_close:
                        close(tr, ts, bar["close"], "eod")
                    elif ts >= p.last_ts:
                        close(tr, ts, bar["close"], "delisted_or_data_end")
            for sym in prep:
                i = prep[sym].pos.get(ts)
                if i is not None:
                    last_px[sym] = prep[sym].base["close"].iat[i]
            # 3) equity & loss limits
            mtm = sum((last_px.get(t.symbol, t.entry_px) - t.entry_px) * t.qty * (1 if t.direction == "LONG" else -1) * t.fx
                      for t in open_tr.values())
            equity = cash + mtm
            equity_pts.append((ts, equity))
            reason = risk.check_loss_limits(now, equity)
            if reason:
                halts.append((now, reason))
                for tr in list(open_tr.values()):
                    close(tr, ts, last_px.get(tr.symbol, tr.entry_px), "loss_limit")
                pending.clear()
                continue
            if risk.state.halted:
                continue
            # 4) scanner
            if self.use_scanner and (last_scan is None or ts - last_scan >= self.scan_every):
                scan_scores = self._scan(prep, ts)
                last_scan = ts
            # 5) signals at this bar's close -> pending entry at next bar open
            for sym, p in prep.items():
                i = p.pos.get(ts)
                if i is None or i < 60 or sym in open_tr or sym in pending:
                    continue
                if self.uh is not None and sym not in self.uh.members(now.date()):
                    continue
                if self.use_scanner and scan_scores.get(sym, 0) <= 0:
                    continue
                ctx = MarketContext(sym, p.inst.venue, p.tz, p.frames_at(i), "5m", None, {}, scan_scores.get(sym))
                res = engine.evaluate(ctx)
                cand = res.candidate
                if cand is None:
                    continue
                px = float(p.base["close"].iat[i])
                q = Quote(sym, px * (1 - hs), px * (1 + hs), px, now, is_realtime=False, source="replay")
                gross = sum(abs(t.entry_px * t.qty * t.fx) for t in open_tr.values())
                acct = AccountSnapshot("BT", equity, cash, max(equity * 2 - gross, 0), s.base_currency, now)
                positions = [Position(t.symbol, t.qty, t.entry_px) for t in open_tr.values()]
                positions += [Position(sy, 1, 0.0) for sy in pending]   # pending entries count as open
                pctx = PreTradeContext(now - timedelta(seconds=1), acct, positions, [], q, p.inst, True, True,
                                       self._fx(p.inst.currency))
                dec = risk.evaluate(cand, pctx)
                if dec.approved:
                    pending[sym] = (cand, dec.quantity)
        for tr in list(open_tr.values()):
            close(tr, timeline[-1], last_px.get(tr.symbol, tr.entry_px), "end_of_test")
        eq = pd.Series([e for _, e in equity_pts], index=pd.DatetimeIndex([t for t, _ in equity_pts]), dtype=float)
        return BTResult(trades, eq, start_equity, {"strategies": s.signals.enabled_strategies, **self.params}, halts,
                        synthetic)

    def _scan(self, prep: dict[str, _Prepared], ts: pd.Timestamp) -> dict[str, float]:
        out = {}
        for sym, p in prep.items():
            i = p.pos.get(ts)
            if i is None:
                continue
            hist = p.raw.iloc[: i + 1]
            hist = hist.loc[hist.index[-1] - pd.Timedelta(days=45):]
            daily = daily_from_intraday(hist, p.tz)
            r = score_symbol(sym, daily, hist.tail(800), tz=p.tz, spread_bps=2 * self.s.costs.default_half_spread_bps,
                             min_price=self.s.risk.min_price, min_avg_dollar_volume=self.s.scanner.min_avg_dollar_volume,
                             max_spread_bps=self.s.risk.max_spread_bps)
            out[sym] = r.score if r.passed_filters else 0.0
        top = sorted(out, key=out.get, reverse=True)[: self.s.scanner.top_n]
        return {k: (v if k in top else 0.0) for k, v in out.items()}
