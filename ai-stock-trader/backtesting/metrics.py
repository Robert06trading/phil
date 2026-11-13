"""Performance metrics, Monte Carlo and the overfitting-aware validation verdict."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def daily_returns(equity: pd.Series) -> pd.Series:
    if equity.empty:
        return pd.Series(dtype=float)
    d = equity.groupby(equity.index.date).last()
    return d.pct_change().dropna()


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    return float(((equity - peak) / peak).min())


def compute(trades: pd.DataFrame, equity: pd.Series, start_equity: float) -> dict:
    out: dict = {"trades": int(len(trades))}
    end = float(equity.iloc[-1]) if len(equity) else start_equity
    out["total_return"] = end / start_equity - 1
    out["max_drawdown"] = max_drawdown(equity)
    dr = daily_returns(equity)
    out["sharpe"] = float(dr.mean() / dr.std() * math.sqrt(252)) if len(dr) > 1 and dr.std() > 0 else 0.0
    downside = dr[dr < 0]
    dd = math.sqrt((downside ** 2).mean()) if len(downside) else 0.0
    out["sortino"] = float(dr.mean() / dd * math.sqrt(252)) if dd > 0 else 0.0
    if trades.empty:
        out.update(win_rate=0.0, profit_factor=0.0, avg_win=0.0, avg_loss=0.0, expectancy=0.0, expectancy_r=0.0)
        return out
    pnl = trades["pnl"]
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    out["win_rate"] = float(len(wins) / len(pnl))
    out["profit_factor"] = float(wins.sum() / -losses.sum()) if losses.sum() < 0 else float("inf") if len(wins) else 0.0
    out["avg_win"] = float(wins.mean()) if len(wins) else 0.0
    out["avg_loss"] = float(losses.mean()) if len(losses) else 0.0
    out["expectancy"] = float(pnl.mean())
    out["expectancy_r"] = float(trades["r_multiple"].mean())
    out["total_costs"] = float(trades["costs"].sum())
    return out


def breakdown(trades: pd.DataFrame) -> dict[str, dict]:
    if trades.empty:
        return {}
    res = {}
    t = trades.copy()
    t["primary_strategy"] = t["strategies"].str.split(",").str[0]
    for key, col in (("by_symbol", "symbol"), ("by_strategy", "primary_strategy"), ("by_hour", "entry_hour"),
                     ("by_exit_reason", "exit_reason"), ("by_direction", "direction")):
        g = t.groupby(col)["pnl"]
        res[key] = {str(k): {"trades": int(v.count()), "pnl": round(float(v.sum()), 2),
                             "win_rate": round(float((v > 0).mean()), 3)} for k, v in g}
    t["month"] = pd.to_datetime(t["exit_ts"], utc=True).dt.strftime("%Y-%m")
    res["by_month"] = {k: round(float(v), 2) for k, v in t.groupby("month")["pnl"].sum().items()}
    return res


def monte_carlo(pnls: np.ndarray, start_equity: float, n: int = 2000, seed: int = 1) -> dict:
    """Bootstrap the trade sequence: distribution of final return and max DD."""
    if len(pnls) == 0:
        return {"runs": 0}
    rng = np.random.default_rng(seed)
    finals, dds = np.empty(n), np.empty(n)
    for k in range(n):
        seq = rng.choice(pnls, size=len(pnls), replace=True)
        eq = start_equity + np.cumsum(seq)
        peak = np.maximum.accumulate(np.concatenate([[start_equity], eq]))[1:]
        finals[k] = eq[-1] / start_equity - 1
        dds[k] = ((eq - peak) / peak).min()
    return {"runs": n, "return_p05": float(np.percentile(finals, 5)), "return_p50": float(np.percentile(finals, 50)),
            "return_p95": float(np.percentile(finals, 95)), "max_dd_p05": float(np.percentile(dds, 5)),
            "max_dd_p50": float(np.percentile(dds, 50)), "prob_loss": float((finals < 0).mean())}


# Criteria a strategy must meet OUT-OF-SAMPLE to be considered validated.
VALIDATION_CRITERIA = {
    "min_oos_trades": 30,
    "min_oos_profit_factor": 1.15,
    "min_oos_expectancy_r": 0.05,
    "min_mc_return_p05": -0.05,
    "min_oos_to_is_pf_ratio": 0.6,     # degradation guard against overfitting
    "max_oos_drawdown": -0.10,
}


def verdict(is_metrics: dict, oos_metrics: dict, mc: dict, synthetic: bool) -> tuple[str, list[str]]:
    c = VALIDATION_CRITERIA
    notes = []
    if synthetic:
        return "INVALID_SYNTHETIC_DATA", ["synthetic data can never validate a strategy"]
    o = oos_metrics
    trades, pf, er = o.get("trades", 0), o.get("profit_factor", 0), o.get("expectancy_r", 0)
    p05, dd = mc.get("return_p05", -1), o.get("max_drawdown", -1)
    checks = [
        (trades >= c["min_oos_trades"], f"OOS trades {trades} >= {c['min_oos_trades']}"),
        (pf >= c["min_oos_profit_factor"], f"OOS PF {pf:.2f} >= {c['min_oos_profit_factor']}"),
        (er >= c["min_oos_expectancy_r"], f"OOS E[R] {er:.3f} >= {c['min_oos_expectancy_r']}"),
        (p05 >= c["min_mc_return_p05"], f"MC p05 return {p05:.3f} >= {c['min_mc_return_p05']}"),
        (dd >= c["max_oos_drawdown"], f"OOS maxDD {dd:.3f} >= {c['max_oos_drawdown']}"),
    ]
    is_pf = is_metrics.get("profit_factor", 0)
    if is_pf and is_pf != float("inf"):
        ratio = oos_metrics.get("profit_factor", 0) / is_pf
        checks.append((ratio >= c["min_oos_to_is_pf_ratio"], f"OOS/IS PF ratio {ratio:.2f} >= {c['min_oos_to_is_pf_ratio']}"))
    for ok, msg in checks:
        notes.append(("PASS " if ok else "FAIL ") + msg)
    return ("PASS" if all(ok for ok, _ in checks) else "FAIL"), notes
