# 5. Security concept

| Threat | Control |
|---|---|
| Leaked credentials in source | Configuration is read from environment and `.env` only (`.env` is git- and docker-ignored). No secrets in code. Logs pass through a `SecretFilter` that redacts configured secrets. |
| Stolen API token | Only SHA-256 hashes of tokens are configured. Tokens are compared in constant time. Viewer and admin roles are separate. Sensitive actions also need **TOTP** (RFC 6238, verified against the RFC test vector). |
| Brute force / abuse | Per-client rate limit (60/min by default) and a 15-minute lockout after 5 failed authentications. Failed attempts are audit-logged. |
| Unauthorised live trading | Six independent conditions (see below). Approval is an **HMAC-signed** record bound to one IBKR account, with an expiry. Editing the database by hand does not work, because the signature is checked at startup and in every loop. |
| Paper/live mix-up | Paper mode refuses to connect to a non-paper ("DU…") IBKR account. Live mode refuses a paper account. Replay uses a separate database. |
| Fund withdrawal | The TWS API cannot withdraw. Use a dedicated IBKR API user without transfer or withdrawal permission. |
| Eavesdropping | Caddy terminates TLS (automatic certificates, HSTS). The API, dashboard and database are only on the internal Docker network. The IB Gateway should stay on localhost or a private network. |
| Tampering / repudiation | An append-only `audit_log` table records every control action: kill, reset, risk changes, approvals, auth failures, orders and halts. |
| Unsafe deserialisation | ML models are loaded with joblib (pickle). Load only model files this system created (`data/models/`). The loader also rejects a changed feature set. |
| Container breakout | The process runs as non-root user `trader` (uid 10001) on a slim base image, with `tini` as init. |

## Live-trading activation (all required)

1. `TRADER_MODE=live` **and** `TRADER_LIVE_TRADING_ENABLED=true`.
2. `reports/readiness/tests.json`: full test suite passed, less than 7 days old (`python scripts/run_tests.py`).
3. `reports/readiness/validation.json`: walk-forward / out-of-sample / Monte Carlo verdict **PASS** on **real** data, less than 30 days old (`python main.py validate --data DIR --capital X`). Synthetic data always yields `INVALID_SYNTHETIC_DATA`.
4. Paper experience: ≥ 20 trading days and ≥ 50 closed paper trades recorded in the database.
5. Signed approval via `POST /live/approve` with the admin token, a TOTP code, the confirmation phrase `I APPROVE LIVE TRADING WITH REAL MONEY` and the live account id. Valid for at most 7 days (default 24 h).
6. Kill switch clear and risk not halted.

The ML model cannot influence any of these inputs.

## Operational recommendations
* Store `.env` with `chmod 600`, or use the cloud provider's secret manager and inject the variables into the containers.
* Rotate the API tokens and the HMAC secret regularly. Rotating the HMAC secret invalidates existing approvals, which is intended.
* Add `basic_auth` or an IP allow-list in `deploy/Caddyfile` as a second layer in front of the dashboard.
