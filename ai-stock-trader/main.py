"""Command-line entry point / composition root.

  python main.py run                 # engine in TRADER_MODE (paper default)
  python main.py replay --data DIR   # paper engine on recorded bars (labelled REPLAY)
  python main.py api                 # control API (uvicorn)
  python main.py backtest --data DIR [--synthetic]
  python main.py validate --data DIR # walk-forward + OOS + Monte Carlo -> readiness/validation.json
  python main.py compare --data DIR
  python main.py train-ml --data DIR
  python main.py gen-token | gen-totp
"""
from __future__ import annotations

import argparse
import base64
import logging
import secrets
import signal
import sys
import time
from pathlib import Path

from config import BrokerKind, Settings, TradingMode, load_settings

log = logging.getLogger("main")


def build_runtime(s: Settings, market=None, clock=None):
    from app.engine import TradingEngine
    from broker.ibkr import IBKRBroker
    from broker.paper import PaperBroker
    from config import RiskConfig
    from notifications.telegram import Notifier
    from portfolio.store import Database
    from risk.manager import RiskManager
    from security.instrument_guard import InstrumentGuard
    from strategies.library import build
    from strategies.signal_engine import SignalEngine

    db = Database(s.database_url)
    overrides = db.get("risk_overrides")
    if overrides:
        s.risk = RiskConfig.model_validate(overrides)       # ceilings re-validated
    guard = InstrumentGuard.from_file(s.universe_file, s.venues, s.allowed_stock_types)
    notify = Notifier(s.telegram_bot_token, s.telegram_chat_id)
    if s.broker == BrokerKind.IBKR:
        broker = IBKRBroker(s.ibkr_host, s.ibkr_port, s.ibkr_client_id, s.ibkr_account,
                            expect_paper=s.mode != TradingMode.LIVE, market_data_type=s.ibkr_market_data_type)
    else:
        if market is None:
            if s.mode == TradingMode.REPLAY:
                raise SystemExit("replay needs recorded data: use `main.py replay --data DIR`")
            # local paper broker with REAL-TIME data from IBKR (read-only use of the connection)
            market = IBKRBroker(s.ibkr_host, s.ibkr_port, s.ibkr_client_id + 1, s.ibkr_account,
                                expect_paper=True, market_data_type=s.ibkr_market_data_type)
            market.connect()
        broker = PaperBroker(market, s.costs, db, s.paper_starting_capital, s.base_currency, clock)
    risk = RiskManager(s.risk, s.costs, guard, db, s.allow_extended_hours)
    ml = None
    if s.signals.use_ml_filter:
        from ml.models import SignalModel
        model_path = s.data_dir / "models" / "signal_model.joblib"
        if model_path.exists():
            ml = SignalModel.load(model_path).scorer()
    signals = SignalEngine(build(s.signals.enabled_strategies), s.signals, s.risk.min_reward_risk, ml)
    return TradingEngine(s, broker, db, guard, risk, signals, notify, clock)


def cmd_run(s: Settings) -> None:
    if s.mode == TradingMode.LIVE and not s.live_trading_enabled:
        raise SystemExit("LIVE mode requested but TRADER_LIVE_TRADING_ENABLED is not true")
    eng = build_runtime(s)
    stop = {"flag": False}

    def _sig(*_):
        stop["flag"] = True

    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)
    eng.startup()
    try:
        while not stop["flag"]:
            t0 = time.monotonic()
            state = eng.step()
            log.info("step -> %s", state)
            time.sleep(max(1.0, s.loop_interval_seconds - (time.monotonic() - t0)))
    finally:
        # positions keep their broker-side stops; EOD logic flattens on the next run if needed
        eng.notify("bot_down", "Engine stopping (signal or error). Broker-side stops remain active.")
        eng.shutdown()


def _load_frames(data: Path, synthetic: bool, symbols: list[str] | None, days: int, s: Settings):
    from market_data.feeds import HistoricalCSVSource, SyntheticSource
    from security.instrument_guard import load_universe

    uni = load_universe(s.universe_file)
    if synthetic:
        syms = symbols or ["AAPL", "MSFT", "NVDA", "JPM", "SAP", "SIE"]
        src = SyntheticSource()
        frames = {x: src.intraday(x, days=days, venue=uni[x].venue, base_price=50 + 25 * i, vol=0.003, regime_strength=0.0)
                  for i, x in enumerate(syms)}
        desc = f"SYNTHETIC random walk ({days} sessions) — engine test only"
    else:
        src = HistoricalCSVSource(data)
        syms = symbols or [x for x in src.symbols() if x in uni]
        frames = {x: src.load(x, "5m") for x in syms}
        desc = f"CSV {data} ({len(syms)} symbols)"
    return frames, {x: uni[x] for x in frames}, desc


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command")
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--symbols", default="")
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--capital", type=float, default=None, help="backtest starting capital (required for backtests)")
    ap.add_argument("--eurusd", type=float, default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args(argv)
    s = load_settings()
    from monitoring.health import setup_logging
    setup_logging(s.log_dir, secrets=[x for x in (s.telegram_bot_token, s.approval_hmac_secret, s.api_totp_secret) if x])
    syms = [x.strip().upper() for x in a.symbols.split(",") if x.strip()] or None
    fx = {("EUR", "USD"): a.eurusd} if a.eurusd else None

    if a.command == "run":
        cmd_run(s)
    elif a.command == "api":
        import uvicorn
        from app.api import create_app
        uvicorn.run(create_app(s), host=a.host, port=a.port)
    elif a.command == "gen-token":
        from security.auth import new_token
        t, h = new_token()
        print(f"token (give to client, store nowhere else): {t}\nsha256 (put in .env): {h}")
    elif a.command == "gen-totp":
        print("TRADER_API_TOTP_SECRET=" + base64.b32encode(secrets.token_bytes(20)).decode().rstrip("="))
    elif a.command in ("backtest", "validate", "compare", "train-ml", "replay"):
        if not a.synthetic and a.data is None:
            raise SystemExit("--data DIR (recorded bars) or --synthetic required")
        frames, insts, desc = _load_frames(a.data, a.synthetic, syms, a.days, s)
        if a.command == "train-ml":
            from market_data.calendar import VENUE_TZ
            from ml.models import SignalModel
            m = SignalModel()
            print(m.fit(m.build_dataset(frames, {k: VENUE_TZ[v.venue] for k, v in insts.items()})))
            m.save(s.data_dir / "models" / "signal_model.joblib")
            return
        if a.capital is None:
            raise SystemExit("--capital is required (starting capital is never hard-coded)")
        if a.command == "replay":
            run_replay(s, frames, insts, a.capital, fx)
            return
        from backtesting import report
        from backtesting.engine import Backtester
        from backtesting.walkforward import compare_strategies, walk_forward, write_validation
        if a.command == "backtest":
            r = Backtester(s, insts, None, fx).run(frames, a.capital)
            paths = report.write(r, s.report_dir, "backtest")
            print(report.build(r, "backtest")["metrics"], paths)
        elif a.command == "compare":
            for k, v in compare_strategies(s, insts, frames, a.capital, fx).items():
                print(k, {m: round(x, 4) if isinstance(x, float) else x for m, x in v.items()})
        else:
            wf = walk_forward(s, insts, frames, a.capital, fx=fx, holdout_days=10)
            p = write_validation(s.report_dir, wf, desc)
            report.write(wf.oos, s.report_dir, "walkforward_oos")
            print(wf.verdict, *wf.notes, f"-> {p}", sep="\n")


def run_replay(s: Settings, frames, insts, capital: float, fx) -> None:
    from market_data.replay import ReplayMarket
    s.mode = TradingMode.REPLAY
    s.broker = BrokerKind.PAPER
    s.paper_starting_capital = capital
    s.database_url = f"sqlite:///{s.data_dir / 'replay.db'}"   # never mix replay into the paper record
    m = ReplayMarket(frames, fx=fx)
    eng = build_runtime(s, market=m, clock=lambda: m.now)
    tl = m.timeline()
    m.set_time(tl[0].to_pydatetime())
    eng.startup()
    for t in tl:
        m.set_time(t.to_pydatetime())
        eng.step()
    print(eng.broker.account())


if __name__ == "__main__":
    main(sys.argv[1:])
