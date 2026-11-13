# AI Stock Trader

A modular, risk-first intraday trading system for **single stocks on NASDAQ, NYSE and XETRA**. It covers scanner → strategies → central signal engine → risk manager → broker bracket orders, with backtesting, walk-forward validation, paper trading, a dashboard and Docker deployment.

> **Live trading is disabled by default** and can only be enabled through a multi-step, authenticated and audit-logged gate.
> **No profits are promised.** The system has so far only been verified mechanically (tests, replay, synthetic data). It has **not** been shown to have a statistical edge on real market data. Backtest results do not predict future results.

## Documentation
1. [Architecture and diagram](docs/01_architecture.md)
2. [Project plan and phase status reports](docs/02_project_plan.md)
3. [Broker and data source comparison](docs/03_broker_and_data.md)
4. [Risk management concept](docs/04_risk_management.md)
5. [Security concept](docs/05_security.md)
6. [Deployment and operations](docs/06_deployment.md)

## Quick start (local)

```bash
cd ai-stock-trader
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
python scripts/run_tests.py                       # 116 tests -> reports/readiness/tests.json

# Engine mechanics on synthetic data (clearly labelled, proves nothing about markets)
python main.py backtest --synthetic --days 30 --capital 50000 --eurusd 1.08
python main.py replay   --synthetic --days 8 --symbols AAPL,JPM,SAP --capital 50000 --eurusd 1.08

# Real research: put SYMBOL_5m.csv files (timestamp,open,high,low,close,volume) in a folder
python main.py compare  --data /path/to/bars --capital 50000 --eurusd 1.08
python main.py validate --data /path/to/bars --capital 50000 --eurusd 1.08   # writes readiness/validation.json
python main.py train-ml --data /path/to/bars                                # optional, advisory only

# Paper trading (IB Gateway paper account on port 4002)
cp .env.example .env    # set TRADER_BROKER=ibkr, TRADER_IBKR_PORT=4002, tokens …
python scripts/check_ibkr_connection.py       # read-only connectivity + whitelist check
python main.py run                            # engine
python main.py api                            # control API on :8000
streamlit run dashboard/app.py                # dashboard on :8501
```

Starting capital is never hard-coded. IBKR accounts report their own equity. The local paper simulator needs `TRADER_PAPER_STARTING_CAPITAL`, and backtests need `--capital`.

## Operating safeguards (summary)
* **Kill switch:** dashboard / `POST /control/kill`, or `touch data/KILL_SWITCH`. It cancels everything, flattens, and stays stopped until an admin resets it with TOTP.
* **Loss limits:** a 2 % daily or 5 % weekly loss triggers a persistent halt, flatten and alert. Resuming requires a manual reset.
* **Every order:** whitelist, session, freshness, spread, limits, duplicates, R:R ≥ 2 and stop-distance sizing, then a broker-side bracket.
* **Day trading:** no entries in the last 30 minutes, flatten 10 minutes before the close, and holidays, early closes and DST come from exchange calendars.

## Project layout
See [docs/01_architecture.md](docs/01_architecture.md#directory-layout).

## What you need to provide
| Item | Why |
|---|---|
| IBKR account (paper first), IB Gateway, API enabled | broker connection: cannot be faked, and is not verified in this repo's build environment |
| Real-time data subscriptions (US + XETRA) | the engine refuses delayed or synthetic data for trading |
| ≥ 1 year of historical 5-minute bars incl. delisted symbols | honest backtests and walk-forward validation |
| Telegram bot token + chat id (optional) | notifications |
| Server with Docker + domain (optional) | cloud deployment with TLS |
