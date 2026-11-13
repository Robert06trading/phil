"""READ-ONLY end-to-end check against your IB Gateway / TWS.

Places NO orders. Verifies: connection, account type (paper 'DU…' vs live),
account summary, contract qualification + whitelist for every universe
symbol, a real-time quote, and historical bars.

    TRADER_IBKR_PORT=4002 python scripts/check_ibkr_connection.py [--expect live]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from broker.ibkr import IBKRBroker  # noqa: E402
from config import load_settings  # noqa: E402
from security.instrument_guard import InstrumentGuard, InstrumentNotAllowed  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expect", choices=["paper", "live"], default="paper")
    ap.add_argument("--limit", type=int, default=0, help="check only the first N universe symbols")
    a = ap.parse_args()
    s = load_settings()
    b = IBKRBroker(s.ibkr_host, s.ibkr_port, s.ibkr_client_id + 50, s.ibkr_account, expect_paper=a.expect == "paper",
                   market_data_type=s.ibkr_market_data_type)
    b.connect()
    acct = b.account()
    print(f"[ok] connected: account {acct.account_id} equity {acct.equity:,.2f} {acct.currency}")
    guard = InstrumentGuard.from_file(s.universe_file, s.venues, s.allowed_stock_types)
    bad = 0
    syms = list(guard.universe.items())[: a.limit or None]
    for sym, inst in syms:
        try:
            q = b.qualify(inst)
            guard.check(q)
            print(f"[ok] {sym:6s} {q.venue:7s} {q.stock_type:7s} conId={q.con_id}")
        except (InstrumentNotAllowed, Exception) as e:  # noqa: BLE001
            bad += 1
            print(f"[FAIL] {sym}: {e}")
    inst = syms[0][1]
    q = b.quote(inst)
    print(f"[{'ok' if q and q.is_realtime else 'WARN'}] quote {inst.symbol}: {q}")
    df = b.bars(inst, "5m", 2)
    print(f"[{'ok' if len(df) else 'FAIL'}] {len(df)} 5m bars, last close time {df.index[-1] if len(df) else None}")
    print(f"[ok] EURUSD {b.fx_rate('EUR', 'USD'):.4f}")
    b.disconnect()
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
