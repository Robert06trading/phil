#!/usr/bin/env sh
# Nightly PostgreSQL backup with 14-day retention (runs in the `backup` service).
set -eu
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT=/backups/trader_${STAMP}.sql.gz
pg_dump --no-owner --clean "$DATABASE_URL" | gzip > "$OUT"
find /backups -name 'trader_*.sql.gz' -mtime +14 -delete
echo "backup written: $OUT"
