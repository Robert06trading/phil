"""Strategy interface and the shared market context.

A strategy is a pure function of a ``MarketContext`` (completed bars only)
that returns at most one ``Signal`` with entry, stop and target. Strategies
never size positions, never talk to the broker and never bypass risk.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.domain import Direction, Quote, Signal

TIMEFRAMES = ["1m", "5m", "15m", "1h"]


@dataclass
class MarketContext:
    symbol: str
    venue: str
    tz: str
    frames: dict[str, pd.DataFrame]          # enriched, completed bars only
    base_tf: str = "5m"
    quote: Quote | None = None
    levels: dict[str, list[float]] = field(default_factory=dict)
    scan_score: float | None = None

    @property
    def base(self) -> pd.DataFrame:
        return self.frames[self.base_tf]

    @property
    def now(self) -> pd.Timestamp:
        return self.base.index[-1]


def trend_direction(df: pd.DataFrame) -> float:
    """-1..+1 trend vote of one timeframe from close/EMA9/EMA20 positions."""
    if df is None or len(df) == 0:
        return 0.0
    r = df.iloc[-1]
    if np.isnan(r.get("ema20", np.nan)) or np.isnan(r.get("ema9", np.nan)):
        return 0.0
    v = np.sign(r["close"] - r["ema20"]) + np.sign(r["ema9"] - r["ema20"])
    return float(v / 2.0)


def mtf_alignment(ctx: MarketContext, direction: Direction) -> float:
    """Fraction-weighted agreement of all available timeframes (-1..1)."""
    votes = [trend_direction(ctx.frames[tf]) for tf in TIMEFRAMES if tf in ctx.frames]
    if not votes:
        return 0.0
    return float(np.mean(votes)) * direction.sign


class Strategy(ABC):
    name: str = "base"
    min_bars: int = 60

    def __init__(self, params: dict | None = None):
        self.params = {**self.default_params(), **(params or {})}

    @classmethod
    def default_params(cls) -> dict:
        return {}

    def generate(self, ctx: MarketContext) -> Signal | None:
        df = ctx.base
        if len(df) < self.min_bars:
            return None
        last = df.iloc[-1]
        if np.isnan(last.get("atr", np.nan)) or last["atr"] <= 0:
            return None
        sig = self._generate(ctx, df, last)
        if sig is None or not sig.is_geometrically_valid():
            return None
        return sig

    @abstractmethod
    def _generate(self, ctx: MarketContext, df: pd.DataFrame, last: pd.Series) -> Signal | None: ...

    def _signal(self, ctx: MarketContext, direction: Direction, entry: float, stop: float, rr: float,
                confidence: float, reasons: list[str]) -> Signal:
        risk = abs(entry - stop)
        target = entry + direction.sign * risk * rr
        return Signal(self.name, ctx.symbol, direction, float(entry), float(stop), float(target),
                      float(np.clip(confidence, 0, 1)), ctx.now.to_pydatetime(), reasons)
