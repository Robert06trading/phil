"""Strategy modules A-E. Each rule is explicit; confidence is a documented,
bounded function of how strongly the rule conditions are met."""
from __future__ import annotations

import numpy as np

from app.domain import Direction
from strategies.base import TIMEFRAMES, Strategy, trend_direction


class MomentumStrategy(Strategy):
    """A: strong move + relative volume + momentum + trend confirmation.
    LONG when: close > VWAP, EMA9 > EMA20, ROC(12) > roc_min, RVOL >= rvol_min,
    last bar closes in the upper half of its range. SHORT mirrored.
    Stop: beyond the 6-bar swing extreme, at least stop_atr*ATR away."""
    name = "momentum"

    @classmethod
    def default_params(cls) -> dict:
        return {"roc_min": 0.006, "rvol_min": 1.8, "stop_atr": 1.2, "rr": 2.2}

    def _generate(self, ctx, df, last):
        p = self.params
        rng = last["high"] - last["low"]
        pos = (last["close"] - last["low"]) / rng if rng > 0 else 0.5
        rv = last["rvol"]
        if np.isnan(rv) or rv < p["rvol_min"]:
            return None
        for d in (Direction.LONG, Direction.SHORT):
            s = d.sign
            if (s * (last["close"] - last["vwap"]) > 0 and s * (last["ema9"] - last["ema20"]) > 0
                    and s * last["roc"] > p["roc_min"] and (pos > 0.5 if s > 0 else pos < 0.5)):
                swing = df["low"].iloc[-6:].min() if s > 0 else df["high"].iloc[-6:].max()
                stop = min(swing, last["close"] - p["stop_atr"] * last["atr"]) if s > 0 \
                    else max(swing, last["close"] + p["stop_atr"] * last["atr"])
                conf = 0.4 + 0.3 * min(rv / (2 * p["rvol_min"]), 1) + 0.3 * min(abs(last["roc"]) / (3 * p["roc_min"]), 1)
                return self._signal(ctx, d, last["close"], stop, p["rr"], conf,
                                    [f"roc={last['roc']:.3%}", f"rvol={rv:.1f}", "vwap+ema aligned"])
        return None


class BreakoutStrategy(Strategy):
    """B: volume-confirmed break of the prior N-bar range.
    False-breakout filters: close must clear the level by buffer_atr*ATR,
    close in the outer 30% of the bar, volume >= vol_mult * average, and the
    previous bar must still have been inside the range (fresh break).
    Stop: back inside the range (level -/+ stop_atr*ATR)."""
    name = "breakout"

    @classmethod
    def default_params(cls) -> dict:
        return {"buffer_atr": 0.1, "vol_mult": 1.5, "stop_atr": 0.8, "rr": 2.5}

    def _generate(self, ctx, df, last):
        p = self.params
        prev = df.iloc[-2]
        hh, ll, a = last["hh20"], last["ll20"], last["atr"]
        if np.isnan(hh) or np.isnan(last["vol_avg"]) or last["volume"] < p["vol_mult"] * last["vol_avg"]:
            return None
        rng = last["high"] - last["low"]
        pos = (last["close"] - last["low"]) / rng if rng > 0 else 0.5
        if last["close"] > hh + p["buffer_atr"] * a and prev["close"] <= hh and pos >= 0.7:
            stop = hh - p["stop_atr"] * a
            conf = 0.5 + 0.5 * min(last["volume"] / (3 * last["vol_avg"]), 1)
            return self._signal(ctx, Direction.LONG, last["close"], stop, p["rr"], conf,
                                [f"break of {hh:.2f}", f"vol x{last['volume']/last['vol_avg']:.1f}"])
        if last["close"] < ll - p["buffer_atr"] * a and prev["close"] >= ll and pos <= 0.3:
            stop = ll + p["stop_atr"] * a
            conf = 0.5 + 0.5 * min(last["volume"] / (3 * last["vol_avg"]), 1)
            return self._signal(ctx, Direction.SHORT, last["close"], stop, p["rr"], conf,
                                [f"breakdown of {ll:.2f}", f"vol x{last['volume']/last['vol_avg']:.1f}"])
        return None


class VWAPStrategy(Strategy):
    """C: VWAP reclaim (long) / rejection (short) in the direction of the
    15m trend. Reclaim: previous close below VWAP, current close above,
    EMA20 rising. Rejection: high tagged VWAP from below, close back below,
    EMA20 falling. Stop: beyond VWAP by stop_atr*ATR or the bar extreme."""
    name = "vwap"

    @classmethod
    def default_params(cls) -> dict:
        return {"stop_atr": 0.6, "rr": 2.0, "slope_bars": 5}

    def _generate(self, ctx, df, last):
        p = self.params
        prev = df.iloc[-2]
        slope = last["ema20"] - df["ema20"].iloc[-1 - p["slope_bars"]]
        htf = trend_direction(ctx.frames.get("15m"))
        vw, a = last["vwap"], last["atr"]
        if np.isnan(vw):
            return None
        if prev["close"] < prev["vwap"] and last["close"] > vw and slope > 0 and htf >= 0:
            stop = min(last["low"], vw - p["stop_atr"] * a)
            return self._signal(ctx, Direction.LONG, last["close"], stop, p["rr"], 0.55 + 0.2 * (htf > 0),
                                ["VWAP reclaim", f"15m trend={htf:+.1f}"])
        if last["high"] >= vw > last["close"] and prev["close"] < prev["vwap"] and slope < 0 and htf <= 0:
            stop = max(last["high"], vw + p["stop_atr"] * a)
            return self._signal(ctx, Direction.SHORT, last["close"], stop, p["rr"], 0.55 + 0.2 * (htf < 0),
                                ["VWAP rejection", f"15m trend={htf:+.1f}"])
        return None


class EMATrendStrategy(Strategy):
    """D: EMA 9/20/50/200 stack + pullback continuation.
    LONG: EMA9 > EMA20 > EMA50 > EMA200, the last 3 bars touched EMA20, and
    the current bar closes back above EMA9. Stop below EMA50 or the pullback
    low (whichever is further, min stop_atr*ATR)."""
    name = "ema_trend"
    min_bars = 210

    @classmethod
    def default_params(cls) -> dict:
        return {"stop_atr": 1.0, "rr": 2.0}

    def _generate(self, ctx, df, last):
        p = self.params
        e = [last[f"ema{n}"] for n in (9, 20, 50, 200)]
        if any(np.isnan(e)):
            return None
        recent = df.iloc[-4:-1]
        a = last["atr"]
        if e[0] > e[1] > e[2] > e[3] and (recent["low"] <= recent["ema20"]).any() and last["close"] > e[0]:
            stop = min(e[2], recent["low"].min(), last["close"] - p["stop_atr"] * a)
            return self._signal(ctx, Direction.LONG, last["close"], stop, p["rr"], 0.6,
                                ["EMA stack bullish", "pullback to EMA20 resolved"])
        if e[0] < e[1] < e[2] < e[3] and (recent["high"] >= recent["ema20"]).any() and last["close"] < e[0]:
            stop = max(e[2], recent["high"].max(), last["close"] + p["stop_atr"] * a)
            return self._signal(ctx, Direction.SHORT, last["close"], stop, p["rr"], 0.6,
                                ["EMA stack bearish", "pullback to EMA20 resolved"])
        return None


class MultiTimeframeStrategy(Strategy):
    """E: all available timeframes (1m/5m/15m/1h) agree on trend direction
    and the base timeframe crosses back over EMA9 in that direction.
    Alignment is also exported as a gate used by the signal engine."""
    name = "mtf"

    @classmethod
    def default_params(cls) -> dict:
        return {"min_alignment": 1.0, "stop_atr": 1.0, "rr": 2.0, "min_frames": 3}

    def _generate(self, ctx, df, last):
        p = self.params
        votes = [trend_direction(ctx.frames[tf]) for tf in TIMEFRAMES if tf in ctx.frames]
        if len(votes) < p["min_frames"]:
            return None
        align = float(np.mean(votes))
        prev = df.iloc[-2]
        if align >= p["min_alignment"] and prev["close"] <= prev["ema9"] and last["close"] > last["ema9"]:
            stop = min(df["low"].iloc[-4:].min(), last["close"] - p["stop_atr"] * last["atr"])
            return self._signal(ctx, Direction.LONG, last["close"], stop, p["rr"], 0.5 + 0.1 * len(votes),
                                [f"{len(votes)} timeframes aligned up"])
        if align <= -p["min_alignment"] and prev["close"] >= prev["ema9"] and last["close"] < last["ema9"]:
            stop = max(df["high"].iloc[-4:].max(), last["close"] + p["stop_atr"] * last["atr"])
            return self._signal(ctx, Direction.SHORT, last["close"], stop, p["rr"], 0.5 + 0.1 * len(votes),
                                [f"{len(votes)} timeframes aligned down"])
        return None


REGISTRY: dict[str, type[Strategy]] = {
    s.name: s for s in (MomentumStrategy, BreakoutStrategy, VWAPStrategy, EMATrendStrategy, MultiTimeframeStrategy)
}


def build(names: list[str], params: dict[str, dict] | None = None) -> list[Strategy]:
    params = params or {}
    unknown = set(names) - set(REGISTRY)
    if unknown:
        raise ValueError(f"unknown strategies {sorted(unknown)}")
    return [REGISTRY[n](params.get(n)) for n in names]
