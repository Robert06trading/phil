# 1. Architecture

## Component diagram

```mermaid
flowchart LR
  subgraph External
    IBKR[(IB Gateway / TWS<br/>IBKR API)]
    TG[Telegram]
  end

  subgraph Engine["Trading engine (main.py run) — one loop for PAPER / REPLAY / LIVE"]
    direction TB
    KS[Kill switch<br/>file + DB flag] --> LOOP
    LOOP[TradingEngine.step] --> SCAN[Scanner<br/>0-100 score]
    SCAN --> CTX[MarketContext<br/>1m/5m/15m/1h, causal indicators]
    CTX --> STRAT[Strategies A-E]
    STRAT --> SE[SignalEngine<br/>unified rulebook]
    ML[ML model<br/>advisory: may only REJECT] -.-> SE
    SE --> RM[RiskManager<br/>final authority]
    GUARD[InstrumentGuard<br/>whitelist] --> RM
    CAL[Exchange calendars<br/>holidays / DST / early close] --> RM
    RM --> OM[OrderManager<br/>write-ahead, idempotent ids]
    OM --> MON[Execution monitor<br/>ensure stops, reconcile, EOD]
  end

  subgraph Brokers["Broker adapters (same interface)"]
    IB[IBKRBroker<br/>ib_async]
    PB[PaperBroker<br/>simulated fills]
  end

  OM --> IB & PB
  IB <--> IBKR
  PB -- real-time quotes --> IB
  PB -- recorded bars --> RP[ReplayMarket<br/>labelled REPLAY]

  DB[(PostgreSQL / SQLite<br/>orders, trades, equity,<br/>audit, risk & kill state)]
  OM & RM & KS & LOOP --> DB
  API[FastAPI control API<br/>tokens + TOTP + audit] --> DB
  DASH[Streamlit dashboard] -- read --> DB
  DASH -- control --> API
  WD[Watchdog] -- heartbeat --> DB
  LOOP & WD --> TG
  BT[Backtester / walk-forward /<br/>Monte Carlo] -. reuses .-> STRAT & SE & RM
```

## Design principles

| Principle | How it is implemented |
|---|---|
| One code path for every mode | `TradingEngine`, `SignalEngine`, `RiskManager` and `OrderManager` are identical in paper, replay and live. Only the `Broker` adapter differs. The backtester reuses the strategies, signal engine, risk manager and cost model. |
| Risk is the final authority | Every order passes `RiskManager.evaluate` (16 checks). Limits have hard ceilings in `config.py`; ML has no handle on risk, sizing or the live gate. |
| Broker is the source of truth | Every loop reconciles internal state against broker orders, positions and executions. Unknown positions are flattened and reported. |
| Fail closed | Unknown instrument type → blocked. Stale/absent quote → no trade. Broker down → no trade. Gate evidence missing → no live trading. |
| Restart safety | Write-ahead order intents, deterministic client order ids (`orderRef`), persisted risk/kill/paper-broker state, startup reconciliation before the first new decision. |
| Protective orders at the broker | Entries are brackets with a broker-side stop and take-profit (OCA). The monitor checks every position for a live stop that covers its full size. |

## Directory layout

```
ai-stock-trader/
├── app/            domain types, trading engine, FastAPI control API
├── broker/         Broker interface, IBKR adapter (ib_async), paper broker
├── market_data/    exchange calendars, CSV/synthetic sources, replay market
├── scanner/        transparent 0-100 stock screener
├── indicators/     causal technical indicators (EMA, ATR, VWAP, RVOL, ADX, S/R …)
├── strategies/     strategy modules A-E, context builder, central signal engine
├── risk/           risk manager (sizing, limits, gate), cost model
├── execution/      order manager, position protection, reconciliation, EOD
├── portfolio/      persistence (SQLAlchemy models, state store, audit log)
├── ml/             optional advisory ML (regimes, triple-barrier signal model)
├── backtesting/    event-driven backtester, metrics, walk-forward, Monte Carlo, reports
├── dashboard/      Streamlit trading terminal
├── notifications/  Telegram notifier (informational only)
├── monitoring/     logging (JSON + secret redaction), heartbeat, health
├── security/       instrument whitelist, kill switch, auth/TOTP/HMAC, live gate
├── tests/          pytest suite (116 tests)
├── scripts/        run_tests (readiness evidence), IBKR check, backup, watchdog, healthcheck
├── deploy/         Caddyfile (TLS), systemd unit
├── data/           universe.csv (approved symbols); DB / models at runtime
├── logs/ reports/  runtime output
├── main.py config.py requirements.txt Dockerfile docker-compose.yml .env.example
```

## Deliberate technology decisions

* **Polars and Redis are not used.** Pandas covers the data volumes involved: a few dozen symbols with intraday bars.
  The database already persists all shared state (kill switch, risk halt, heartbeat), and in this system durability matters more than latency.
  Redis would add a second source of truth without a clear benefit, which conflicts with "avoid unnecessary dependencies". Both can be added if the universe grows to thousands of symbols.
* **`ib_async`** is the maintained successor of `ib_insync`. It wraps IBKR's official TWS API socket protocol.
* **`exchange_calendars`** provides holidays, early closes and DST for XNYS/XNAS/XETR.
* **SQLite** is the default for development and **PostgreSQL** is used in Docker. Both run through the same SQLAlchemy models.
