"""Replay market: drives the full paper engine with RECORDED bars.

Quotes produced here are labelled ``is_realtime=False, source='replay'`` and
are accepted only when the engine runs in REPLAY mode. They are derived from
the last *completed* bar's close with a configured half spread; nothing from
the future is visible (``now`` is advanced by the caller).
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from app.domain import Instrument, Quote
from indicators.technical import resample_bars

TF_RULE = {"5m": "5min", "15m": "15min", "1h": "1h"}


class ReplayMarket:
    def __init__(self, frames: dict[str, pd.DataFrame], base_tf: str = "5m", half_spread_bps: float = 2.0,
                 fx: dict[tuple[str, str], float] | None = None):
        self.frames = frames
        self.base_tf = base_tf
        self.hs = half_spread_bps / 1e4
        self.now: datetime | None = None
        self.fx = fx or {}
        self.synthetic = any(df.attrs.get("synthetic") for df in frames.values())

    def timeline(self) -> list[pd.Timestamp]:
        idx = sorted(set().union(*[set(df.index) for df in self.frames.values()]))
        return idx

    def set_time(self, now: datetime) -> None:
        self.now = now

    def _upto(self, symbol: str) -> pd.DataFrame | None:
        df = self.frames.get(symbol)
        if df is None or self.now is None:
            return None
        out = df.loc[: pd.Timestamp(self.now)]
        out.attrs.update(df.attrs)
        return out

    def quote(self, inst: Instrument) -> Quote | None:
        df = self._upto(inst.symbol)
        if df is None or df.empty:
            return None
        last_ts = df.index[-1]
        if last_ts != pd.Timestamp(self.now):
            return None   # symbol not trading at this instant (holiday / halted / delisted)
        px = float(df["close"].iloc[-1])
        return Quote(inst.symbol, px * (1 - self.hs), px * (1 + self.hs), px, last_ts.to_pydatetime(),
                     is_realtime=False, source="replay")

    def bars(self, inst: Instrument, timeframe: str, lookback_days: int) -> pd.DataFrame:
        df = self._upto(inst.symbol)
        if df is None or df.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = df.loc[df.index[-1] - pd.Timedelta(days=lookback_days * 1.6 + 3):]
        if timeframe == self.base_tf:
            out = df
        elif timeframe in TF_RULE and timeframe != "1m":
            out = resample_bars(df, TF_RULE[timeframe]).loc[: pd.Timestamp(self.now)]
            # drop the forming higher-timeframe bar
            out = out[out.index <= pd.Timestamp(self.now)]
            if len(out) and out.index[-1] > df.index[-1]:
                out = out.iloc[:-1]
        elif timeframe == "1d":
            from scanner.screener import daily_from_intraday
            out = daily_from_intraday(df, "UTC")
        else:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        out = out.copy()
        out.attrs["synthetic"] = self.synthetic
        return out

    def fx_rate(self, from_ccy: str, to_ccy: str) -> float:
        if from_ccy == to_ccy:
            return 1.0
        if (from_ccy, to_ccy) in self.fx:
            return self.fx[(from_ccy, to_ccy)]
        if (to_ccy, from_ccy) in self.fx:
            return 1.0 / self.fx[(to_ccy, from_ccy)]
        raise KeyError(f"no FX {from_ccy}->{to_ccy} configured for replay")
