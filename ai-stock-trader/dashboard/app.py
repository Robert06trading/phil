"""Streamlit trading terminal.

Read-only views query the database directly; every control action goes
through the authenticated API (token kept only in this browser session), so
the dashboard has no privileged path of its own.

    streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from sqlalchemy import select  # noqa: E402

from backtesting import metrics as M  # noqa: E402
from config import load_settings  # noqa: E402
from monitoring.health import health_report  # noqa: E402
from portfolio.store import AuditRow, Database, EquityRow, OrderRow, ScanRow, SignalRow, TradeRow  # noqa: E402

st.set_page_config(page_title="AI Stock Trader", page_icon="📈", layout="wide")
S = load_settings()
API = os.environ.get("TRADER_API_URL", "http://127.0.0.1:8000")


@st.cache_resource
def get_db() -> Database:
    return Database(S.database_url)


db = get_db()


def frame(model, order_col, limit=1000, **filt) -> pd.DataFrame:
    with db.session() as s:
        q = select(model)
        for k, v in filt.items():
            q = q.where(getattr(model, k) == v)
        rows = s.scalars(q.order_by(order_col.desc()).limit(limit)).all()
        return pd.DataFrame([{c.name: getattr(r, c.name) for c in model.__table__.columns} for r in rows])


def api(method: str, path: str, body: dict | None = None):
    tok = st.session_state.get("token")
    if not tok:
        st.error("Enter the admin token in the sidebar first.")
        return None
    req = urllib.request.Request(API + path, method=method, data=json.dumps(body or {}).encode(),
                                 headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        st.error(f"{e.code}: {e.read().decode()[:400]}")
    except OSError as e:
        st.error(f"API unreachable: {e}")
    return None


# ------------------------------------------------------------------ header
hr = health_report(db)
mode = S.mode.value.upper()
eng_state = (hr["engine"] or {}).get("state", "unknown")
connected = (hr["engine"] or {}).get("broker_connected")


def chip(label: str, value: str, ok: bool) -> str:
    color = "#3fb68b" if ok else "#ff6b6b"
    return (f"<span style='border:1px solid {color};border-radius:6px;padding:4px 10px;margin-right:8px;"
            f"display:inline-block;margin-bottom:6px'><span style='color:#888'>{label}</span> "
            f"<b style='color:{color}'>{value}</b></span>")


st.markdown("### 📈 AI Stock Trader")
st.markdown("".join([
    chip("Mode", mode, mode != "LIVE"),
    chip("Engine", eng_state, eng_state not in ("error", "killed", "halted", "broker_down")),
    chip("Broker", "connected" if connected else "DISCONNECTED", bool(connected)),
    chip("Kill switch", "ENGAGED" if hr["kill_switch"] else "clear", not hr["kill_switch"]),
    chip("Risk", "HALTED" if hr["risk_halted"] else "ok", not hr["risk_halted"]),
]), unsafe_allow_html=True)
if mode == "LIVE":
    st.error("LIVE TRADING — real money at risk")
with st.sidebar:
    st.text_input("Admin API token", type="password", key="token")
    st.caption(f"API: {API}")
    st.caption("Heartbeat age: " + (f"{hr['heartbeat_age_s']:.0f}s" if hr["heartbeat_age_s"] is not None else "n/a"))

tabs = st.tabs(["Portfolio", "Live trading", "Performance", "Scanner", "System", "Control"])
eq = frame(EquityRow, EquityRow.ts, 20000, mode=S.mode.value)
trades = frame(TradeRow, TradeRow.opened_at, 5000, mode=S.mode.value)

with tabs[0]:
    if eq.empty:
        st.info("No equity snapshots yet.")
    else:
        last = eq.iloc[0]
        cols = st.columns(4)
        cols[0].metric(f"Equity ({S.base_currency})", f"{last.equity:,.0f}")
        cols[1].metric("Cash", f"{last.cash:,.2f}")
        cols[2].metric("Realized today", f"{last.realized_today:+,.2f}")
        cols[3].metric("Unrealized", f"{last.unrealized:+,.2f}")
        first = eq.iloc[-1]
        st.metric("Total performance", f"{(last.equity / first.equity - 1):+.2%}")
    open_tr = trades[trades.status == "open"] if not trades.empty else trades
    st.subheader("Open positions")
    if open_tr.empty:
        st.info("Flat — no open positions.")
    else:
        st.dataframe(open_tr[["symbol", "direction", "quantity", "entry_price", "stop_price", "target_price",
                              "strategies", "opened_at"]], use_container_width=True)

with tabs[1]:
    st.subheader("Latest signals")
    sig = frame(SignalRow, SignalRow.ts, 200)
    st.dataframe(sig, use_container_width=True)
    st.subheader("Working orders")
    orders = frame(OrderRow, OrderRow.created_at, 300, mode=S.mode.value)
    if not orders.empty:
        orders = orders[~orders.status.isin(["Filled", "Cancelled", "Rejected", "Inactive"])]
    st.dataframe(orders, use_container_width=True)

with tabs[2]:
    closed = trades[trades.status == "closed"] if not trades.empty else trades
    if closed.empty or eq.empty:
        st.info("No closed trades yet.")
    else:
        e = eq.set_index(pd.to_datetime(eq.ts, utc=True))["equity"].sort_index()
        t = closed.rename(columns={"closed_at": "exit_ts"}).copy()
        t["r_multiple"] = t.pnl / t.risk_amount.replace(0, float("nan"))
        t["costs"] = t.commission
        m = M.compute(t, e, float(e.iloc[0]))
        cols = st.columns(4)
        cols[0].metric("Win rate", f"{m['win_rate']:.1%}")
        cols[1].metric("Profit factor", f"{m['profit_factor']:.2f}")
        cols[2].metric("Sharpe", f"{m['sharpe']:.2f}")
        cols[3].metric("Max drawdown", f"{m['max_drawdown']:.2%}")
        import altair as alt
        ed = e.reset_index()
        ed.columns = ["ts", "equity"]
        st.altair_chart(alt.Chart(ed).mark_line(color="#3fb68b").encode(
            x=alt.X("ts:T", title=None), y=alt.Y("equity:Q", scale=alt.Scale(zero=False), title="Equity")
        ).properties(height=260), use_container_width=True)
        t["month"] = pd.to_datetime(t.exit_ts, utc=True).dt.strftime("%Y-%m")
        st.bar_chart(t.groupby("month")["pnl"].sum())
        st.caption("Past performance does not predict future results.")

with tabs[3]:
    sc = frame(ScanRow, ScanRow.ts, 200)
    if sc.empty:
        st.info("No scan results yet.")
    else:
        latest = sc[sc.ts == sc.ts.max()].sort_values("score", ascending=False)
        met = pd.json_normalize(latest.metrics.tolist())
        view = pd.concat([latest[["symbol", "score", "bias"]].reset_index(drop=True), met], axis=1)
        st.dataframe(view, use_container_width=True)
        st.bar_chart(latest.set_index("symbol")["score"])

with tabs[4]:
    st.json(hr)
    st.subheader("Audit / error log")
    st.dataframe(frame(AuditRow, AuditRow.ts, 300), use_container_width=True)

with tabs[5]:
    st.warning("All actions are authenticated and audit-logged.")
    a, b, k = st.columns(3)
    if a.button("Start bot"):
        st.write(api("POST", "/control/start"))
    if b.button("Stop bot (no new entries)"):
        st.write(api("POST", "/control/stop"))
    reason = k.text_input("Kill reason", "manual")
    if k.button("🛑 KILL SWITCH", type="primary"):
        st.write(api("POST", "/control/kill", {"reason": reason}))
    totp = st.text_input("TOTP code (for resets / live approval)", max_chars=6)
    r1, r2 = st.columns(2)
    if r1.button("Reset kill switch"):
        st.write(api("POST", "/control/kill/reset", {"totp": totp}))
    if r2.button("Reset loss-limit halt"):
        st.write(api("POST", "/risk/reset", {"totp": totp}))
    st.subheader("Risk parameters (applied at next engine start; hard ceilings enforced)")
    with st.form("risk"):
        vals = {
            "risk_per_trade_pct": st.number_input("Risk per trade %", 0.05, 1.0, float(S.risk.risk_per_trade_pct), 0.05),
            "max_daily_loss_pct": st.number_input("Max daily loss %", 0.5, 3.0, float(S.risk.max_daily_loss_pct), 0.1),
            "max_weekly_loss_pct": st.number_input("Max weekly loss %", 1.0, 8.0, float(S.risk.max_weekly_loss_pct), 0.5),
            "max_open_positions": int(st.number_input("Max open positions", 1, 10, int(S.risk.max_open_positions))),
            "min_reward_risk": st.number_input("Min reward:risk", 1.0, 5.0, float(S.risk.min_reward_risk), 0.1),
        }
        if st.form_submit_button("Save risk parameters"):
            st.write(api("PUT", "/settings/risk", vals))
    st.subheader("Live-trading approval")
    if st.button("Check readiness"):
        st.write(api("GET", "/live/readiness"))
    acct = st.text_input("IBKR live account id")
    phrase = st.text_input("Type: I APPROVE LIVE TRADING WITH REAL MONEY")
    if st.button("Approve live trading"):
        st.write(api("POST", "/live/approve", {"totp": totp, "confirm_phrase": phrase, "account": acct}))
    if st.button("Revoke live approval"):
        st.write(api("POST", "/live/revoke"))
