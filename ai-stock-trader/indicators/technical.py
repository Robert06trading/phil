"""Technical indicators. Every function is CAUSAL: the value at row t depends
only on rows <= t. ``tests/test_indicators.py`` verifies this by comparing
values computed on truncated vs. full history (look-ahead tripwire).

Bar frames: DatetimeIndex (tz-aware, bar CLOSE time) and columns
open, high, low, close, volume.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    prev = df["close"].shift(1)
    return pd.concat([df["high"] - df["low"], (df["high"] - prev).abs(), (df["low"] - prev).abs()], axis=1).max(axis=1)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Wilder's ATR."""
    return true_range(df).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100.0).where(up.notna())


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = true_range(df).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    pdi = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean() / tr
    mdi = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean() / tr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def session_key(index: pd.DatetimeIndex, tz: str) -> pd.Index:
    return index.tz_convert(tz).date


def vwap(df: pd.DataFrame, tz: str = "America/New_York") -> pd.Series:
    """Session-anchored VWAP (resets each trading day in exchange time)."""
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    key = session_key(df.index, tz)
    pv = (tp * df["volume"]).groupby(key).cumsum()
    vv = df["volume"].groupby(key).cumsum()
    return pv / vv.replace(0, np.nan)


def rvol(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """Relative volume of each bar vs. the mean of the previous n bars."""
    base = df["volume"].shift(1).rolling(n, min_periods=n).mean()
    return df["volume"] / base.replace(0, np.nan)


def rvol_time_of_day(df: pd.DataFrame, tz: str, days: int = 10) -> pd.Series:
    """Cumulative session volume up to this bar vs. the average cumulative
    volume at the same time-of-day over the previous ``days`` sessions."""
    local = df.index.tz_convert(tz)
    key = pd.Index(local.date)
    tod = pd.Index(local.time)
    cum = df["volume"].groupby(key).cumsum()
    frame = pd.DataFrame({"cum": cum.values, "day": key, "tod": tod}, index=df.index)
    hist = frame.pivot_table(index="day", columns="tod", values="cum", aggfunc="last").sort_index()
    avg_prev = hist.shift(1).rolling(days, min_periods=max(3, days // 2)).mean()
    lookup = avg_prev.stack(future_stack=True)
    ref = pd.Series([lookup.get((d, t), np.nan) for d, t in zip(key, tod, strict=True)], index=df.index)
    return cum / ref.replace(0, np.nan)


def roc(s: pd.Series, n: int) -> pd.Series:
    return s.pct_change(n)


def relative_strength(s: pd.Series, bench: pd.Series, n: int) -> pd.Series:
    """Excess return over the benchmark across n bars (aligned on index)."""
    b = bench.reindex(s.index).ffill()
    return s.pct_change(n) - b.pct_change(n)


def donchian(df: pd.DataFrame, n: int) -> tuple[pd.Series, pd.Series]:
    """Highest high / lowest low of the PREVIOUS n bars (excludes current)."""
    return df["high"].shift(1).rolling(n, min_periods=n).max(), df["low"].shift(1).rolling(n, min_periods=n).min()


def swing_levels(df: pd.DataFrame, left: int = 3, right: int = 3, max_levels: int = 6,
                 tolerance: float = 0.003) -> dict[str, list[float]]:
    """Support/resistance zones from confirmed pivot highs/lows.

    A pivot at bar i is only *confirmed* after ``right`` later bars, so the
    most recent ``right`` bars are never treated as pivots (no look-ahead).
    Nearby pivots (within ``tolerance``) are clustered into one zone.
    """
    h, lo = df["high"].to_numpy(), df["low"].to_numpy()
    highs, lows = [], []
    for i in range(left, len(df) - right):
        if h[i] == max(h[i - left:i + right + 1]):
            highs.append(h[i])
        if lo[i] == min(lo[i - left:i + right + 1]):
            lows.append(lo[i])

    def cluster(vals: list[float]) -> list[float]:
        zones: list[list[float]] = []
        for v in sorted(vals):
            if zones and abs(v - np.mean(zones[-1])) / v <= tolerance:
                zones[-1].append(v)
            else:
                zones.append([v])
        zones.sort(key=len, reverse=True)
        return sorted(float(np.mean(z)) for z in zones[:max_levels])

    return {"resistance": cluster(highs), "support": cluster(lows)}


def gap_pct(daily: pd.DataFrame) -> pd.Series:
    """Opening gap vs previous close (daily bars)."""
    return daily["open"] / daily["close"].shift(1) - 1.0


def enrich(df: pd.DataFrame, tz: str = "America/New_York") -> pd.DataFrame:
    """Add the standard indicator set used by strategies and the scanner."""
    out = df.copy()
    c = out["close"]
    for n in (9, 20, 50, 200):
        out[f"ema{n}"] = ema(c, n)
    out["atr"] = atr(out, 14)
    out["rsi"] = rsi(c, 14)
    out["adx"] = adx(out, 14)
    out["vwap"] = vwap(out, tz)
    out["rvol"] = rvol(out, 20)
    out["roc"] = roc(c, 12)
    out["vol_avg"] = out["volume"].shift(1).rolling(20, min_periods=20).mean()
    out["hh20"], out["ll20"] = donchian(out, 20)
    return out


def resample_bars(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Aggregate bars indexed by CLOSE time into a higher timeframe.
    ``closed='right', label='right'`` keeps the close-time convention: the 15m
    bar labelled 10:00 contains the 1m bars closing 09:46..10:00. The last
    group may be incomplete — callers must drop groups whose label > now
    (see ``completed``)."""
    agg = df.resample(rule, closed="right", label="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    return agg.dropna(subset=["open"])


def completed(df: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    """Only bars whose close time <= now (prevents using a forming bar)."""
    return df.loc[: now]
