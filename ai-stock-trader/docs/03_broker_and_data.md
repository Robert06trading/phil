# 3. Broker and data source comparison

Requirements: US stocks (NASDAQ, NYSE) **and** German stocks (XETRA) in one account, an official and documented API,
broker-side stop orders and brackets, a paper account, real-time data, and API keys that cannot withdraw funds.

| Criterion | **Interactive Brokers** | Alpaca | Tradier | Saxo Bank | Lynx (IBKR introducing broker) | German neo-brokers (TR, Scalable) |
|---|---|---|---|---|---|---|
| NASDAQ / NYSE | ✅ | ✅ | ✅ | ✅ | ✅ (IBKR infra) | ✅ (via LS/gettex, not primary exchange) |
| XETRA | ✅ (IBIS) | ❌ | ❌ | ✅ | ✅ | partly, no XETRA API |
| Official trading API | ✅ TWS API / Client Portal / FIX | ✅ REST | ✅ REST | ✅ OpenAPI | via IBKR TWS API | ❌ none |
| Bracket / OCA / stop-limit at broker | ✅ native | ✅ (US only) | partial | ✅ | ✅ | – |
| Paper account with the same API | ✅ (DU… accounts) | ✅ | ✅ sandbox | ✅ SIM | ✅ | – |
| Shorting intraday | ✅ (borrow-dependent) | ✅ | ✅ | ✅ | ✅ | ❌ |
| Commissions (indicative) | US $0.0035/sh tiered; XETRA ~0.05% | $0 US | $0 / subscription | higher | higher than IBKR | low, but no API |
| Withdrawals via trading API | ❌ not possible via TWS API | ❌ | ❌ | ❌ | ❌ | – |

**Decision: Interactive Brokers.** It is the only broker that meets all hard requirements (US + XETRA, official API, broker-side brackets, paper account with identical behaviour).
Alpaca is excellent for US-only trading but has no XETRA access.

### Notes on IBKR
* Run **IB Gateway** (or TWS) with *Enable ActiveX and Socket Clients* switched on and *Read-Only API* off.
  Ports: 4002 for the paper account and 4001 for live.
  IBKR requires a daily or weekly re-login. Use IBC or a maintained gateway container image for auto-restart and 2FA handling.
* The TWS API **cannot withdraw funds**. As an additional measure, use a dedicated IBKR user for the API with trading permissions only, and no transfer or withdrawal rights.
* Pacing: historical data is limited to about 60 requests per 10 minutes. The engine keeps the universe small (top-N re-scans every 15 minutes).
  For a larger universe, switch to streaming bars (`reqRealTimeBars` with `keepUpToDate`).
* XETRA tick sizes depend on price and liquidity band. The adapter rounds to 0.01; IBKR rejects invalid ticks, so this fails safe.

## Market data

| Source | Use | Real-time | Cost | Notes |
|---|---|---|---|---|
| **IBKR market data subscriptions** | paper + live | ✅ | US Securities Snapshot & Futures Value Bundle (~$10/mo, waived above commission threshold); XETRA L1 (~€10-20/mo) | Same feed for paper and live. The system **refuses delayed data** for trading (`is_realtime` check). |
| IBKR historical bars | warm-up, backtest | – | included | Limited depth and pacing; split-adjusted; no delisted symbols. |
| Polygon.io / Databento / Tiingo | research backtests | ✅ | paid | Point-in-time data including delisted symbols is needed to avoid survivorship bias. Export to CSV in the format `SYMBOL_5m.csv` (`timestamp,open,high,low,close,volume`). |
| Synthetic generator | **tests only** | – | – | Labelled `synthetic`. It can never validate a strategy or drive paper or live trading. |

### Data the operator must provide (not available in this environment)
1. **IBKR account** (paper "DU…" account first) plus a running IB Gateway, for live and paper connectivity.
2. **Real-time market data subscriptions** for NASDAQ/NYSE and XETRA on that account.
3. **Historical intraday data** (≥ 1 year, 5-minute bars, split-adjusted, including delisted symbols, plus `universe_history.csv`) for backtesting and walk-forward validation.

Until this data exists, the system can only be verified mechanically (tests, replay, synthetic backtests). It has **not** been shown to have any trading edge.
