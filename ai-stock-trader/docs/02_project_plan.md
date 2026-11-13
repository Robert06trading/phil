# 2. Project plan and phase status reports

Status as of 2026-11-13. "Verified" means covered by automated tests or run end to end in the build environment.
"Open" means it requires external access (IBKR account, market data, real historical data, Docker registry) that is not available in this environment.

| Phase | Content | Status | Evidence |
|---|---|---|---|
| 1 | Architecture, plan, broker/data comparison, security and risk concepts | ✅ done | `docs/01…05` |
| 2 | Market data and broker connection | ✅ code done, ⚠️ live connection **open** | IBKR adapter tested against a fake gateway with the real `ib_async` order/contract classes (`test_ibkr_adapter.py`, 8 tests). A real gateway check needs your account: `scripts/check_ibkr_connection.py` |
| 3 | Stock scanner | ✅ verified | transparent 0–100 score, hard filters (`test_strategies_scanner.py`) |
| 4 | Indicators and strategies A–E, central signal engine | ✅ verified | causality/look-ahead tripwires, crafted breakout and false-breakout cases, conflict/MTF/ML-reject rules |
| 5 | Risk management and position sizing | ✅ verified | 18 tests: sizing, FX, every rejection path, daily/weekly halt persistence, no martingale, heat/notional caps |
| 6 | Backtesting engine | ✅ verified (mechanics) | costs, partial fills, gap stops, EOD, delisting, point-in-time universe, future-perturbation look-ahead test, walk-forward, Monte Carlo, reports |
| 7 | Paper trading | ✅ verified in REPLAY | Full engine + paper broker + DB on recorded bars. Restart without duplicates, kill switch, loss-limit halt, broker outage, no entries after cutoff (`test_engine.py`) |
| 8 | Dashboard and notifications | ✅ verified | Streamlit started against the replay DB; all tabs rendered without exceptions. Telegram notifier is failure-safe |
| 9 | Cloud deployment | ✅ files done, ⚠️ container build **not verified here** | `docker compose config` valid. The image build was blocked by the sandbox (Docker Hub HTTP 429 / no egress for apt). Run `docker compose up -d --build` on the server |
| 10 | Integration, security tests, docs | ✅ 116/116 tests pass | `python scripts/run_tests.py` → `reports/readiness/tests.json` |
| 11 | Live readiness with manual approval | ✅ gate implemented, 🔒 **live stays disabled** | requires real-data validation PASS + ≥ 20 paper days / 50 trades + signed approval |

## Phase reports (short)

**Phase 2: Data and broker.** I chose IBKR (see the comparison in doc 03).
The adapter covers connection with backoff and reconnect, paper/live account-type enforcement, contract qualification with the broker-reported stockType for the whitelist, real-time quotes (delayed data is refused), historical bars re-stamped to close time, atomic brackets with `orderRef`, cancel/modify, executions and EUR/USD.
*Open:* proof against a real gateway. I cannot fake it, so it has to run on your machine.

**Phase 3: Scanner.** Nine weighted, documented components plus hard filters. The score is the exact weighted sum, which a test asserts.

**Phase 4: Strategies.** Momentum, Breakout (with false-breakout filters), VWAP reclaim/rejection, EMA 9/20/50/200 stack pullback, and multi-timeframe alignment.
They are combined through one rulebook (conflict → no trade, MTF gate, widest stop, R:R ≥ 2, quality ≥ threshold).

**Phase 5: Risk.** See doc 04. Everything there is enforced and tested.

**Phase 6: Backtesting.** A sanity check found that synthetic data with built-in daily trends produced strong but meaningless profits.
Over six pure random-walk seeds the system averaged −0.08 R/trade (costs only), which shows no leakage.
A dedicated test perturbs all data after time T and asserts identical results before T.
Synthetic data can never produce a PASS verdict.

**Phase 7: Paper trading.** Paper uses exactly the same engine, risk checks and order manager as live.
The recommended setup is an IBKR **paper account** (`TRADER_BROKER=ibkr`, port 4002), so even order handling happens at the broker.

**Phase 8: Dashboard.** Six areas (Portfolio, Live, Performance, Scanner, System, Control). Controls go through the authenticated API only.

**Phase 9: Deployment.** Postgres, engine, API, dashboard, watchdog, nightly backup, Caddy TLS, healthchecks, `restart: unless-stopped`, and a systemd unit for boot start.

**Phase 10–11.** Readiness evidence files and the signed approval flow are implemented and tested. Live remains disabled.

## Assumptions
* Account base currency is USD (configurable). EUR stocks are converted at the IBKR EUR/USD rate.
* Only regular trading hours are used (extended hours can be configured but are off by default).
* Commission model approximates IBKR Pro tiered pricing. Adjust `CostConfig` to your plan.
* ADRs (e.g. TSM) count as single stocks. ETFs, ETNs, funds, preferreds, warrants and units are blocked.
* Shorting is allowed. IBKR can reject shorts when no borrow is available, which fails safe (no position).

## Next steps for the operator
1. Open an IBKR paper account, start IB Gateway, subscribe to market data, and run `scripts/check_ibkr_connection.py`.
2. Obtain ≥ 1 year of 5-minute bars (including delisted symbols), then run `main.py compare` and `main.py validate`.
   Expect most strategy variants to **fail** honestly; only keep what passes out of sample.
3. Run paper trading for at least 20 trading days. Review `reports/` and the dashboard.
4. Only then consider the live approval flow, starting with a small account.
