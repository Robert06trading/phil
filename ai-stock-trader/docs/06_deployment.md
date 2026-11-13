# 6. Cloud deployment and operations

## Server
A small VM (2 vCPU, 4 GB RAM) in a region close to the exchanges is enough, e.g. Frankfurt for XETRA or US-East for NYSE/NASDAQ. Use Ubuntu 24.04 with Docker Engine and the Compose plugin.

```bash
git clone <repo> /opt/ai-stock-trader && cd /opt/ai-stock-trader/ai-stock-trader
cp .env.example .env && chmod 600 .env        # fill in tokens, TOTP, HMAC, POSTGRES_PASSWORD, DOMAIN
python3 main.py gen-token   # admin; repeat for viewer
python3 main.py gen-totp    # add to your authenticator app
docker compose up -d --build
docker compose ps           # engine/api should become "healthy"
sudo cp deploy/ai-stock-trader.service /etc/systemd/system/ && sudo systemctl enable ai-stock-trader
```

### IB Gateway
Run IBKR's IB Gateway on the same host or a private network, either with IBC for unattended daily restarts or with a maintained container image.
Point `TRADER_IBKR_HOST` to it (`host.docker.internal` when it runs on the host). Never expose ports 4001/4002 publicly.

## What runs

| Service | Purpose | Health |
|---|---|---|
| `db` | PostgreSQL 16 (volume `pg-data`) | `pg_isready` |
| `engine` | trading loop | `scripts/healthcheck.py` (heartbeat ≤ 3 loop intervals) |
| `api` | control API | `GET /health` |
| `dashboard` | Streamlit terminal | – |
| `watchdog` | Telegram alert if the heartbeat goes stale | – |
| `backup` | nightly `pg_dump`, 14-day retention in `./backups` | – |
| `caddy` | HTTPS (Let's Encrypt), security headers | – |

## Restart and failure behaviour

* **Server or container restart:** `engine.startup()` reconciles against the broker *before* any new decision.
  Open trades come from the DB, orders are matched by `orderRef`, missing stops are re-placed, and unknown positions are flattened.
  Deterministic client order ids prevent re-sending the same signal (tested in `test_restart_keeps_positions_and_creates_no_duplicates`).
* **Engine crash during the session:** positions keep their **broker-side** stop and take-profit. The watchdog alerts.
  The `DAY` time-in-force means the brackets expire at the close. The EOD routine or the next start flattens anything left.
* **Broker disconnect:** no new trades, reconnect with exponential backoff, a single alert, and broker-side protection stays active.
* **Database loss:** restore the latest `backups/trader_*.sql.gz` (`gunzip -c … | psql $DATABASE_URL`).
  The broker remains the source of truth, so reconciliation rebuilds the live picture.

## Logs and monitoring
* JSON logs in the `trader-logs` volume (`trader.jsonl`, rotated at 20 MB × 10), with secrets redacted.
* `audit_log` table: every control action, order, halt and auth failure.
* `GET /status` (viewer token) for external uptime monitors.

## Updating
```bash
git pull && python3 scripts/run_tests.py && docker compose up -d --build engine api dashboard
```
Deploy updates outside market hours. The engine reconciles on start.
