"""Builds MarketContext objects from raw bar frames (shared by the live/paper
engine and the backtester so both see identical inputs)."""
from __future__ import annotations

import pandas as pd

from app.domain import Quote
from indicators.technical import enrich
from market_data.calendar import VENUE_TZ
from strategies.base import MarketContext


def build_context(symbol: str, venue: str, raw: dict[str, pd.DataFrame], base_tf: str = "5m",
                  quote: Quote | None = None, scan_score: float | None = None,
                  levels: dict | None = None) -> MarketContext | None:
    tz = VENUE_TZ[venue]
    frames = {tf: enrich(df, tz) for tf, df in raw.items() if df is not None and len(df) >= 2}
    if base_tf not in frames:
        return None
    return MarketContext(symbol, venue, tz, frames, base_tf, quote, levels or {}, scan_score)
