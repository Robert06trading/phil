"""Market data sources.

* ``HistoricalCSVSource`` — loads recorded bars for backtests and replay.
  Expected CSV columns: timestamp (bar close, ISO8601 with tz), open, high,
  low, close, volume. Optional ``splits.csv`` (symbol,date,ratio) is applied
  to produce split-adjusted prices; ``universe_history.csv``
  (symbol,start,end) provides point-in-time membership to avoid
  survivorship bias.
* ``SyntheticSource`` — deterministic random-walk bars for TESTS ONLY. Every
  frame it produces carries ``df.attrs['synthetic'] = True`` and the engine
  refuses to treat such data as real-time (see ``assert_realtime``).
* Live/paper real-time data comes from the broker adapter
  (``broker.ibkr.IBKRBroker``), never from this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from app.domain import Quote

REQUIRED = ["open", "high", "low", "close", "volume"]


class StaleDataError(Exception):
    pass


def validate_bars(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns {missing}")
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None:
        raise ValueError("index must be a tz-aware DatetimeIndex (bar close time)")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    bad = (df["high"] < df[["open", "close", "low"]].max(axis=1)) | (df["low"] > df[["open", "close", "high"]].min(axis=1))
    if bad.any():
        raise ValueError(f"{int(bad.sum())} bars violate OHLC consistency")
    if (df[REQUIRED] < 0).any().any():
        raise ValueError("negative prices/volume")
    return df


def apply_splits(df: pd.DataFrame, splits: list[tuple[pd.Timestamp, float]]) -> pd.DataFrame:
    """Back-adjust prices/volume for splits. ratio=4 means 4-for-1. Bars
    strictly before the split's effective date are divided by the ratio."""
    out = df.copy()
    for eff, ratio in splits:
        mask = out.index < eff
        out.loc[mask, ["open", "high", "low", "close"]] /= ratio
        out.loc[mask, "volume"] *= ratio
    return out


@dataclass
class UniverseHistory:
    """Point-in-time listing: symbol -> (start, end or None)."""
    spans: dict[str, tuple[date, date | None]]

    def members(self, day: date) -> list[str]:
        return [s for s, (a, b) in self.spans.items() if a <= day and (b is None or day <= b)]

    @classmethod
    def from_csv(cls, path: Path) -> "UniverseHistory":
        df = pd.read_csv(path)
        spans = {}
        for r in df.itertuples():
            end = None if pd.isna(r.end) else pd.Timestamp(r.end).date()
            spans[r.symbol] = (pd.Timestamp(r.start).date(), end)
        return cls(spans)


class HistoricalCSVSource:
    def __init__(self, directory: Path):
        self.dir = Path(directory)
        self._splits = self._load_splits()

    def _load_splits(self) -> dict[str, list[tuple[pd.Timestamp, float]]]:
        p = self.dir / "splits.csv"
        out: dict[str, list] = {}
        if p.exists():
            for r in pd.read_csv(p).itertuples():
                out.setdefault(r.symbol, []).append((pd.Timestamp(r.date, tz="UTC"), float(r.ratio)))
        return out

    def symbols(self) -> list[str]:
        return sorted(p.stem.split("_")[0] for p in self.dir.glob("*_*.csv"))

    def load(self, symbol: str, timeframe: str = "5m") -> pd.DataFrame:
        path = self.dir / f"{symbol}_{timeframe}.csv"
        df = pd.read_csv(path)
        df.index = pd.to_datetime(df.pop("timestamp"), utc=True)
        df = validate_bars(df[REQUIRED].astype(float))
        if symbol in self._splits:
            df = apply_splits(df, self._splits[symbol])
        df.attrs["synthetic"] = False
        df.attrs["symbol"] = symbol
        return df


class SyntheticSource:
    """Deterministic test data — NOT market data. Never used for live/paper."""

    def __init__(self, seed: int = 7):
        self.seed = seed

    def intraday(self, symbol: str, days: int = 30, start: date = date(2025, 3, 3), venue: str = "NYSE",
                 bar_minutes: int = 5, drift: float = 0.0, vol: float = 0.0015, base_price: float = 100.0,
                 base_volume: float = 50_000, regime_strength: float = 0.4) -> pd.DataFrame:
        from market_data.calendar import session_for

        rng = np.random.default_rng(abs(hash((symbol, self.seed))) % (2**32))
        rows, idx = [], []
        price = base_price
        d = start
        n_days = 0
        while n_days < days:
            s = session_for(venue, d)
            d += timedelta(days=1)
            if s is None:
                continue
            n_days += 1
            regime = rng.choice([-1.0, 0.0, 1.0]) * vol * regime_strength + drift
            t = s.open + timedelta(minutes=bar_minutes)
            price *= 1 + rng.normal(0, vol * 3)  # overnight gap
            while t <= s.close:
                o = price
                ret = rng.normal(regime, vol)
                c = max(0.5, o * (1 + ret))
                hi = max(o, c) * (1 + abs(rng.normal(0, vol / 2)))
                lo = min(o, c) * (1 - abs(rng.normal(0, vol / 2)))
                v = base_volume * rng.lognormal(0, 0.4)
                rows.append((o, hi, lo, c, v))
                idx.append(t)
                price = c
                t += timedelta(minutes=bar_minutes)
        df = pd.DataFrame(rows, columns=REQUIRED, index=pd.DatetimeIndex(idx, tz=timezone.utc))
        df.attrs["synthetic"] = True
        df.attrs["symbol"] = symbol
        return df


def assert_realtime(quote: Quote | None, now: datetime, max_age_s: float, allow_replay: bool = False) -> None:
    """Raise unless the quote is real-time (or, only in REPLAY mode, an
    explicitly labelled replay quote), two-sided and fresh."""
    if quote is None:
        raise StaleDataError("no quote")
    if not quote.is_realtime and not (allow_replay and quote.source == "replay"):
        raise StaleDataError(f"{quote.symbol}: quote is not real-time")
    if quote.bid <= 0 or quote.ask <= 0 or quote.ask < quote.bid:
        raise StaleDataError(f"{quote.symbol}: invalid book bid={quote.bid} ask={quote.ask}")
    age = (now - quote.ts).total_seconds()
    if age > max_age_s or age < -5:
        raise StaleDataError(f"{quote.symbol}: quote age {age:.1f}s exceeds {max_age_s}s")


def assert_bars_fresh(df: pd.DataFrame, now: datetime, bar_minutes: int, tolerance_bars: float = 2.0,
                      allow_synthetic: bool = False) -> None:
    if df is None or df.empty:
        raise StaleDataError("no bars")
    if df.attrs.get("synthetic") and not allow_synthetic:
        raise StaleDataError("synthetic data cannot drive paper/live trading")
    age_min = (now - df.index[-1].to_pydatetime()).total_seconds() / 60.0
    if age_min > bar_minutes * tolerance_bars:
        raise StaleDataError(f"last bar {age_min:.1f} min old")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)
