# 4. Risk management concept

Priority order: **capital preservation > technical reliability > security > statistical edge > performance**.

## Limits (defaults → hard ceilings in code)

| Parameter | Default | Ceiling | Where enforced |
|---|---|---|---|
| Risk per trade | 0.5 % of equity | 1.0 % | `RiskManager.size` |
| Max daily loss | 2 % of day-start equity | 3 % | `check_loss_limits` → persistent halt |
| Max weekly loss | 5 % of week-start equity | 8 % | `check_loss_limits` → persistent halt |
| Max open positions | 3 (pending entries count) | 10 | `evaluate` |
| Min reward:risk | 1 : 2 (checked again at the executable bid/ask) | – | `evaluate` |
| Portfolio heat (sum of open risk) | 1.5 % | 3 % | `size` |
| Max position notional | 25 % of equity | 50 % | `size` |
| Max spread | 15 bps | – | `evaluate` |
| Max quote age | 15 s, real-time only | – | `evaluate` + `assert_realtime` |
| No entries after open / before close | 5 min / 30 min | – | `evaluate` |
| Flatten before close | 10 min | – | engine `_end_of_day` |

The ceilings are validated by Pydantic. A config file, the API (`PUT /settings/risk`) or the dashboard cannot exceed them.

## Position sizing

```
budget      = equity × effective_risk_pct
per_share   = |entry − stop| + round-trip costs (2×(slippage + half-spread) + 2×commission/share)
qty         = floor(budget / (per_share × FX(instrument ccy → base ccy)))
qty         = min(qty, notional cap, 95 % of buying power, remaining portfolio heat)
```

* The stop distance is the *actual* stop of the trade, not a fixed percentage.
* **No martingale:** after `loss_streak_reduce_after` (3) consecutive losses, risk is multiplied by 0.5.
  It returns to normal after a win and is never increased. A test verifies this.
* No averaging down: a symbol with an open position or working order is rejected.

## Pre-trade checklist (every order, paper and live)

1. Kill switch clear
2. Risk not halted
3. Instrument passes the whitelist (static universe **and** broker-reported secType/stockType/venue/currency/conId)
4. Broker connected
5. Real-time, two-sided, fresh quote
6. Regular session open (exchange calendar, holidays, DST, early close)
7. Not in the opening window or the no-new-entries window
8. Spread ≤ max
9. Price ≥ minimum
10. Daily and weekly loss limits not breached
11. No existing position in the symbol
12. No working order in the symbol, and no duplicate client id
13. Open positions + pending entries < max
14. Stop and target geometry valid for the direction
15. R:R ≥ 2 from the signal entry **and** from the executable bid/ask
16. Size ≥ 1 share after all caps

## When a loss limit is hit

1. `RiskManager.halt()` persists `halted=true` with the reason. The halt survives restarts.
2. The engine cancels all orders and flattens all positions (`flatten_all("loss_limit")`).
3. No new entries are accepted. Each step returns `halted`.
4. Telegram alert and an audit-log entry at CRITICAL level.
5. **Restart requires a manual reset**: `POST /risk/reset` with the admin token and a TOTP code, which is audit-logged.
   By default the day anchor is kept, so a reset on the same day re-halts while the loss persists. `rebase_equity=true` is an explicit operator decision to allow trading again.

## Kill switch

Two independent triggers: the file `data/KILL_SWITCH` (works even when the API is down: `docker compose exec engine touch data/KILL_SWITCH`) and a database flag (API or dashboard).
The engine cancels every order, closes every position with market orders, notifies, and stays stopped. A reset needs the admin token and TOTP.
The live gate also engages the kill switch automatically if approval or evidence becomes invalid during live trading.

## Execution safety

* Entries are marketable **limit** orders, at most 5 bps through the book, never naked market orders.
* Each entry is a **bracket** with a broker-side stop-loss and take-profit (OCA), transmitted atomically.
* The execution monitor checks every loop that each broker position has a working stop covering its full size. If not, it places one immediately, or exits if price is already through the stop.
* Partial entry fills resize the protective legs to the filled quantity. Unfilled entries are cancelled after 10 minutes.
* Unknown broker positions (for example, after manual trading or a crash) are flattened and reported.
* End of day: entries are cancelled 30 minutes before the close, positions are flattened 10 minutes before the close, and any overnight holding found at the next open is closed.

## Costs, slippage, FX

The same `risk/costs.py` model (IBKR-like tiered US commission, XETRA percentage commission, slippage and half spread) is used for sizing, the paper broker and backtests.
EUR positions are converted at the live EUR/USD rate (IBKR) or a configured rate (backtest).
