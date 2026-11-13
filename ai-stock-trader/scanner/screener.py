"""Transparent stock screener: each symbol gets a 0-100 score made of named,
weighted components. Every component is a deterministic function of
measured data — no opaque predictions.

Component (weight)          | full score when
----------------------------|------------------------------------------------
liquidity (15)              | avg daily dollar volume >= 10x the minimum
rvol (20)                   | relative volume >= 3
volatility (10)             | intraday ATR% in the 1.5%..5% sweet spot
momentum (15)               | |ROC| of the session >= 3%
gap (10)                    | |gap| >= 3% (US pre-market move counts here)
trend (10)                  | ADX >= 35
relative_strength (10)      | |excess return vs benchmark| >= 3%
vwap (5)                    | price on the trend side of VWAP, 0.2..2% away
spread (5)                  | spread <= 2 bps (linear to the max allowed)

Hard filters (score forced to 0, reason recorded): price < min_price,
avg dollar volume < minimum, spread > max_spread_bps, missing data.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from indicators.technical import adx, atr, swing_levels, vwap

WEIGHTS = {
    "liquidity": 15, "rvol": 20, "volatility": 10, "momentum": 15, "gap": 10,
    "trend": 10, "relative_strength": 10, "vwap": 5, "spread": 5,
}
assert sum(WEIGHTS.values()) == 100


@dataclass
class ScanResult:
    symbol: str
    score: float
    components: dict[str, float]           # 0..1 per component
    metrics: dict[str, float]
    direction_bias: str                    # LONG / SHORT / NEUTRAL
    passed_filters: bool
    reasons: list[str] = field(default_factory=list)
    levels: dict[str, list[float]] = field(default_factory=dict)


def _lin(x: float, lo: float, hi: float) -> float:
    if np.isnan(x):
        return 0.0
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def score_symbol(symbol: str, daily: pd.DataFrame, intraday: pd.DataFrame, *, tz: str,
                 benchmark_intraday: pd.DataFrame | None = None, spread_bps: float | None = None,
                 min_price: float = 5.0, min_avg_dollar_volume: float = 20e6,
                 max_spread_bps: float = 15.0, premarket_change: float | None = None) -> ScanResult:
    reasons: list[str] = []
    metrics: dict[str, float] = {}
    comps = {k: 0.0 for k in WEIGHTS}
    if daily is None or len(daily) < 6 or intraday is None or len(intraday) < 15:
        return ScanResult(symbol, 0.0, comps, metrics, "NEUTRAL", False, ["insufficient data"])

    d = daily.iloc[-21:-1] if _is_partial_today(daily, intraday) else daily.iloc[-20:]   # up to 20 completed days
    price = float(intraday["close"].iloc[-1])
    adv = float((d["close"] * d["volume"]).mean())
    metrics.update(price=price, avg_dollar_volume=adv)

    today = intraday[intraday.index.tz_convert(tz).date == intraday.index[-1].tz_convert(tz).date()]
    sess_open = float(today["open"].iloc[0])
    prev_close = float(d["close"].iloc[-1])
    gap = sess_open / prev_close - 1 if premarket_change is None else premarket_change
    sess_ret = price / sess_open - 1
    cum_vol = float(today["volume"].sum())
    # compare against the same fraction of an average day
    frac = len(today) / max(_bars_per_session(intraday, tz), 1)
    rv = cum_vol / (float(d["volume"].mean()) * max(frac, 1e-3))
    atr_pct = float(atr(intraday, 14).iloc[-1] / price * np.sqrt(_bars_per_session(intraday, tz)))
    adx_v = float(adx(intraday, 14).iloc[-1])
    vw = float(vwap(intraday, tz).iloc[-1])
    vwap_dev = price / vw - 1 if vw else np.nan
    rs = np.nan
    if benchmark_intraday is not None and len(benchmark_intraday):
        b = benchmark_intraday["close"].reindex(today.index).ffill().dropna()
        if len(b) > 1:
            rs = sess_ret - (float(b.iloc[-1]) / float(b.iloc[0]) - 1)
    metrics.update(gap=gap, session_return=sess_ret, rvol=rv, atr_pct_day=atr_pct, adx=adx_v,
                   vwap=vw, vwap_dev=vwap_dev, relative_strength=float(rs) if not np.isnan(rs) else np.nan,
                   spread_bps=spread_bps if spread_bps is not None else np.nan)

    passed = True
    if price < min_price:
        passed = False
        reasons.append(f"price {price:.2f} < {min_price}")
    if adv < min_avg_dollar_volume:
        passed = False
        reasons.append(f"avg $vol {adv/1e6:.1f}M < {min_avg_dollar_volume/1e6:.0f}M")
    if spread_bps is not None and spread_bps > max_spread_bps:
        passed = False
        reasons.append(f"spread {spread_bps:.1f}bps > {max_spread_bps}")

    comps["liquidity"] = _lin(np.log10(adv), np.log10(min_avg_dollar_volume), np.log10(min_avg_dollar_volume * 10))
    comps["rvol"] = _lin(rv, 1.0, 3.0)
    if 0.015 <= atr_pct <= 0.05:
        comps["volatility"] = 1.0
    elif atr_pct < 0.015:
        comps["volatility"] = _lin(atr_pct, 0.005, 0.015)
    else:
        comps["volatility"] = _lin(-atr_pct, -0.10, -0.05)   # too volatile -> fades to 0 at 10%
    comps["momentum"] = _lin(abs(sess_ret), 0.0, 0.03)
    comps["gap"] = _lin(abs(gap), 0.0, 0.03)
    comps["trend"] = _lin(adx_v, 15, 35)
    comps["relative_strength"] = _lin(abs(rs), 0.0, 0.03) if not np.isnan(rs) else 0.0
    bias = "LONG" if sess_ret > 0 and price > vw else "SHORT" if sess_ret < 0 and price < vw else "NEUTRAL"
    if bias != "NEUTRAL" and not np.isnan(vwap_dev):
        comps["vwap"] = 1.0 if 0.002 <= abs(vwap_dev) <= 0.02 else 0.5
    comps["spread"] = 1.0 - _lin(spread_bps, 2.0, max_spread_bps) if spread_bps is not None else 0.0

    score = sum(WEIGHTS[k] * v for k, v in comps.items()) if passed else 0.0
    if passed:
        top = sorted(comps, key=lambda k: WEIGHTS[k] * comps[k], reverse=True)[:3]
        reasons.append("top drivers: " + ", ".join(f"{k}={comps[k]:.2f}" for k in top))
    return ScanResult(symbol, round(score, 2), comps, metrics, bias, passed, reasons,
                      swing_levels(intraday.tail(300)))


def rank(results: list[ScanResult], top_n: int) -> list[ScanResult]:
    return sorted([r for r in results if r.passed_filters], key=lambda r: r.score, reverse=True)[:top_n]


def _bars_per_session(df: pd.DataFrame, tz: str) -> int:
    counts = pd.Series(1, index=df.index).groupby(df.index.tz_convert(tz).date).sum()
    return int(counts.iloc[:-1].median()) if len(counts) > 1 else int(counts.iloc[-1])


def _is_partial_today(daily: pd.DataFrame, intraday: pd.DataFrame) -> bool:
    return daily.index[-1].date() >= intraday.index[-1].date()


def daily_from_intraday(df: pd.DataFrame, tz: str) -> pd.DataFrame:
    key = df.index.tz_convert(tz).date
    g = df.groupby(key)
    out = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                        "close": g["close"].last(), "volume": g["volume"].sum()})
    out.index = pd.DatetimeIndex([pd.Timestamp(d, tz="UTC") for d in out.index])
    return out
