"""Walk-forward optimisation, out-of-sample testing, strategy comparison and
the readiness ``validation.json`` evidence file.

Procedure (anchored on trading sessions, never shuffled):
  [train_days][test_days] -> optimise params on train, run once on test
  slide forward by test_days, repeat. Only TEST results are reported as OOS.
A final hold-out tail (``holdout_days``) is never seen by any optimisation.
"""
from __future__ import annotations

import itertools
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from backtesting import metrics as M
from backtesting.engine import Backtester, BTResult
from config import Settings

# Small, deliberately coarse grids: fewer combinations -> less overfitting.
PARAM_GRID = {
    "momentum": {"rvol_min": [1.5, 2.0], "rr": [2.0, 2.5]},
    "breakout": {"vol_mult": [1.3, 1.8], "rr": [2.0, 3.0]},
    "vwap": {"stop_atr": [0.5, 0.8]},
    "ema_trend": {"rr": [2.0, 2.5]},
    "mtf": {"rr": [2.0, 2.5]},
}


def grid(strategies: list[str]) -> list[dict]:
    axes = []
    for s in strategies:
        for k, vals in PARAM_GRID.get(s, {}).items():
            axes.append([(s, k, v) for v in vals])
    combos = []
    for prod in itertools.product(*axes) if axes else [()]:
        d: dict[str, dict] = {}
        for s, k, v in prod:
            d.setdefault(s, {})[k] = v
        combos.append(d)
    return combos


def objective(res: BTResult, min_trades: int = 10) -> float:
    """Expectancy in R, shrunk toward 0 for small samples; penalise drawdown."""
    t = res.trades_frame()
    if len(t) < min_trades:
        return -1e9
    e = float(t["r_multiple"].mean())
    shrink = len(t) / (len(t) + 30)
    dd = abs(M.max_drawdown(res.equity))
    return e * shrink - 2.0 * dd


_G: dict = {}


def _init(settings, instruments, frames, fx, bt_kwargs) -> None:
    _G.update(settings=settings, instruments=instruments, frames=frames, fx=fx, kw=bt_kwargs)


def _run_one(args):
    params, equity, start, end = args
    r = Backtester(_G["settings"], _G["instruments"], params, _G["fx"], **_G["kw"]).run(_G["frames"], equity, start, end)
    return params, r


def sessions(frames: dict[str, pd.DataFrame]) -> list:
    return sorted(set().union(*[set(df.index.date) for df in frames.values()]))


@dataclass
class WFResult:
    folds: list[dict]
    oos: BTResult
    is_metrics: dict
    oos_metrics: dict
    holdout_metrics: dict | None
    monte_carlo: dict
    verdict: str
    notes: list[str]


def walk_forward(settings: Settings, instruments, frames: dict[str, pd.DataFrame], start_equity: float,
                 train_days: int = 40, test_days: int = 10, holdout_days: int = 0, fx=None, max_combos: int = 16,
                 n_jobs: int | None = None, **bt_kwargs) -> WFResult:
    days = sessions(frames)
    usable = days[: len(days) - holdout_days] if holdout_days else days
    combos = grid(settings.signals.enabled_strategies)[:max_combos]
    folds, oos_trades, oos_eq, is_scores = [], [], [], []
    equity = start_equity
    n_jobs = n_jobs or min(len(combos), os.cpu_count() or 1)
    pool = None
    if n_jobs > 1 and len(combos) > 1:
        pool = ProcessPoolExecutor(n_jobs, mp.get_context("fork"), initializer=_init,
                                   initargs=(settings, instruments, frames, fx, bt_kwargs))
    else:
        _init(settings, instruments, frames, fx, bt_kwargs)
    i = train_days
    while i + 1 <= len(usable) - 1:
        tr_start, tr_end = usable[i - train_days], usable[i - 1]
        te_start, te_end = usable[i], usable[min(i + test_days, len(usable)) - 1]
        best, best_score, best_res = {}, -np.inf, None
        jobs = [(p, start_equity, _ts(tr_start), _ts(tr_end, end=True)) for p in combos]
        results = pool.map(_run_one, jobs) if pool else map(_run_one, jobs)
        for params, r in results:      # deterministic order -> ties resolved identically
            sc = objective(r)
            if sc > best_score:
                best, best_score, best_res = params, sc, r
        test = Backtester(settings, instruments, best, fx, **bt_kwargs).run(
            frames, equity, _ts(te_start), _ts(te_end, end=True))
        equity = float(test.equity.iloc[-1]) if len(test.equity) else equity
        oos_trades += test.trades
        oos_eq.append(test.equity)
        if best_res is not None:
            is_scores.append(M.compute(best_res.trades_frame(), best_res.equity, start_equity))
        folds.append({"train": [str(tr_start), str(tr_end)], "test": [str(te_start), str(te_end)], "params": best,
                      "train_objective": best_score, "test_trades": len(test.trades),
                      "test_pnl": round(sum(t.pnl for t in test.trades), 2)})
        i += test_days
    if pool:
        pool.shutdown()
    eq = pd.concat(oos_eq) if oos_eq else pd.Series(dtype=float)
    synthetic = any(df.attrs.get("synthetic") for df in frames.values())
    oos = BTResult(oos_trades, eq, start_equity, {"walk_forward": True}, [], synthetic)
    oos_m = M.compute(oos.trades_frame(), eq, start_equity)
    is_m = _mean_metrics(is_scores)
    hold_m = None
    if holdout_days:
        last_params = folds[-1]["params"] if folds else {}
        h = Backtester(settings, instruments, last_params, fx, **bt_kwargs).run(
            frames, start_equity, _ts(days[-holdout_days]), _ts(days[-1], end=True))
        hold_m = M.compute(h.trades_frame(), h.equity, start_equity)
    mc = M.monte_carlo(np.array([t.pnl for t in oos_trades]), start_equity)
    v, notes = M.verdict(is_m, oos_m, mc, synthetic)
    return WFResult(folds, oos, is_m, oos_m, hold_m, mc, v, notes)


def compare_strategies(settings: Settings, instruments, frames, start_equity: float, fx=None, **bt_kwargs) -> dict:
    out = {}
    for name in settings.signals.enabled_strategies:
        s = settings.model_copy(deep=True)
        s.signals.enabled_strategies = [name]
        s.signals.require_mtf_alignment = name != "mtf" and settings.signals.require_mtf_alignment
        r = Backtester(s, instruments, None, fx, **bt_kwargs).run(frames, start_equity)
        out[name] = M.compute(r.trades_frame(), r.equity, start_equity)
    return out


def _mean_metrics(ms: list[dict]) -> dict:
    if not ms:
        return {}
    keys = [k for k in ms[0] if isinstance(ms[0][k], (int, float))]
    return {k: float(np.mean([m[k] for m in ms if np.isfinite(m.get(k, np.nan))] or [0])) for k in keys}


def _ts(d, end: bool = False) -> datetime:
    t = pd.Timestamp(d, tz="UTC")
    return (t + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)).to_pydatetime() if end else t.to_pydatetime()


def write_validation(report_dir: Path, wf: WFResult, data_desc: str) -> Path:
    rd = report_dir / "readiness"
    rd.mkdir(parents=True, exist_ok=True)
    p = rd / "validation.json"
    p.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(), "verdict": wf.verdict, "notes": wf.notes,
        "criteria": M.VALIDATION_CRITERIA, "oos_metrics": wf.oos_metrics, "is_metrics": wf.is_metrics,
        "holdout_metrics": wf.holdout_metrics, "monte_carlo": wf.monte_carlo, "folds": wf.folds, "data": data_desc,
    }, indent=2, default=str))
    return p
