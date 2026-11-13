from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from app.domain import Direction, Signal
from config import SignalConfig
from indicators.technical import resample_bars
from market_data.feeds import SyntheticSource
from scanner.screener import WEIGHTS, daily_from_intraday, rank, score_symbol
from strategies.base import Strategy
from strategies.context import build_context
from strategies.library import REGISTRY, BreakoutStrategy, build
from strategies.signal_engine import SignalEngine


@pytest.fixture(scope="module")
def series():
    return SyntheticSource(seed=5).intraday("AAPL", days=12, vol=0.003, regime_strength=0.4)


def ctx_at(df, i, score=60.0):
    raw = {"5m": df.iloc[: i + 1]}
    raw["15m"] = resample_bars(raw["5m"], "15min")
    raw["1h"] = resample_bars(raw["5m"], "1h")
    return build_context("AAPL", "NASDAQ", raw, "5m", None, score)


def test_every_strategy_emits_valid_signals(series):
    counts = {n: 0 for n in REGISTRY}
    for i in range(250, len(series), 3):
        ctx = ctx_at(series, i)
        for st in build(list(REGISTRY)):
            s = st.generate(ctx)
            if s:
                counts[s.strategy] += 1
                assert s.is_geometrically_valid() and s.reward_risk >= 1.9
                assert 0 <= s.confidence <= 1
                assert s.bar_ts == series.index[i].to_pydatetime()
    assert all(c > 0 for c in counts.values()), counts


def test_signals_do_not_depend_on_future_bars(series):
    """Same signal whether or not later bars exist."""
    full_i = len(series) - 1
    for i in range(300, full_i, 37):
        a = [s.generate(ctx_at(series, i)) for s in build(list(REGISTRY))]
        b = [s.generate(ctx_at(series.iloc[: i + 1], i)) for s in build(list(REGISTRY))]
        assert [(x.direction, round(x.stop, 6)) if x else None for x in a] == \
               [(x.direction, round(x.stop, 6)) if x else None for x in b]


def _breakout_frame():
    idx = pd.date_range("2026-11-09 14:35", periods=120, freq="5min", tz="UTC")
    rng = np.random.default_rng(0)
    close = 100 + rng.normal(0, 0.05, len(idx))
    df = pd.DataFrame({"open": close, "high": close + 0.1, "low": close - 0.1, "close": close,
                       "volume": rng.uniform(9e3, 11e3, len(idx))}, index=idx)
    df.iloc[-1] = [100.1, 101.2, 100.05, 101.15, 40_000]     # strong close at the high, 4x volume
    return df


def test_breakout_rule_and_false_breakout_filter():
    df = _breakout_frame()
    ctx = build_context("AAPL", "NASDAQ", {"5m": df})
    s = BreakoutStrategy().generate(ctx)
    assert s and s.direction is Direction.LONG and s.stop < df["high"].iloc[-21:-1].max()
    weak = df.copy()
    weak.iloc[-1] = [100.1, 101.2, 100.05, 100.3, 40_000]   # long wick, close back near range -> rejected
    assert BreakoutStrategy().generate(build_context("AAPL", "NASDAQ", {"5m": weak})) is None
    quiet = df.copy()
    quiet.iloc[-1, quiet.columns.get_loc("volume")] = 9_000  # no volume confirmation -> rejected
    assert BreakoutStrategy().generate(build_context("AAPL", "NASDAQ", {"5m": quiet})) is None


class Fixed(Strategy):
    def __init__(self, name, direction, conf=0.8):
        super().__init__()
        self.name, self.d, self.conf = name, direction, conf

    def _generate(self, ctx, df, last):
        px = last["close"]
        stop = px - self.d.sign * 1.0
        return self._signal(ctx, self.d, px, stop, 2.5, self.conf, ["fixed"])


def test_engine_conflict_rule(series):
    eng = SignalEngine([Fixed("a", Direction.LONG), Fixed("b", Direction.SHORT)],
                       SignalConfig(require_mtf_alignment=False, min_quality_score=0), 2.0)
    r = eng.evaluate(ctx_at(series, 400))
    assert r.candidate is None and "conflicting" in r.rejections[0]


def test_engine_combines_and_enforces_min_rr(series):
    eng = SignalEngine([Fixed("a", Direction.LONG), Fixed("b", Direction.LONG, 0.6)],
                       SignalConfig(require_mtf_alignment=False, min_quality_score=0), 2.0)
    c = eng.evaluate(ctx_at(series, 400)).candidate
    assert c and c.reward_risk >= 2.0 and set(c.strategies) == {"a", "b"} and 0 <= c.quality_score <= 100


def test_engine_mtf_gate(series):
    eng = SignalEngine([Fixed("a", Direction.LONG)], SignalConfig(require_mtf_alignment=True, min_mtf_alignment=1.01,
                                                                  min_quality_score=0), 2.0)
    r = eng.evaluate(ctx_at(series, 400))
    assert r.candidate is None and "MTF" in r.rejections[0]


def test_ml_can_only_reject(series):
    cfg = SignalConfig(require_mtf_alignment=False, min_quality_score=0, use_ml_filter=True, ml_min_probability=0.6)
    base = SignalEngine([Fixed("a", Direction.LONG)], SignalConfig(require_mtf_alignment=False, min_quality_score=0), 2.0)
    ref = base.evaluate(ctx_at(series, 400)).candidate
    hi = SignalEngine([Fixed("a", Direction.LONG)], cfg, 2.0, ml_scorer=lambda c, d: 0.99).evaluate(ctx_at(series, 400)).candidate
    lo = SignalEngine([Fixed("a", Direction.LONG)], cfg, 2.0, ml_scorer=lambda c, d: 0.1).evaluate(ctx_at(series, 400))
    none = SignalEngine([Fixed("a", Direction.LONG)], cfg, 2.0, ml_scorer=lambda c, d: None).evaluate(ctx_at(series, 400))
    assert hi and (hi.entry, hi.stop, hi.target, hi.quality_score) == (ref.entry, ref.stop, ref.target, ref.quality_score)
    assert lo.candidate is None and none.candidate is None


def test_scanner_score_transparent_and_bounded(series):
    daily = daily_from_intraday(series, "America/New_York")
    r = score_symbol("AAPL", daily, series, tz="America/New_York", spread_bps=3, min_avg_dollar_volume=1e6)
    assert r.passed_filters and 0 <= r.score <= 100
    assert set(r.components) == set(WEIGHTS) and all(0 <= v <= 1 for v in r.components.values())
    assert r.score == pytest.approx(sum(WEIGHTS[k] * v for k, v in r.components.items()), abs=0.01)


def test_scanner_hard_filters(series):
    daily = daily_from_intraday(series, "America/New_York")
    r = score_symbol("AAPL", daily, series, tz="America/New_York", spread_bps=50, max_spread_bps=15)
    assert not r.passed_filters and r.score == 0 and any("spread" in x for x in r.reasons)
    cheap = series.copy()
    cheap[["open", "high", "low", "close"]] /= 50
    r2 = score_symbol("X", daily_from_intraday(cheap, "America/New_York"), cheap, tz="America/New_York")
    assert not r2.passed_filters
    assert rank([r, r2], 5) == []


def test_signal_dataclass_geometry():
    t = series_ts = pd.Timestamp("2026-11-11 15:30", tz="UTC").to_pydatetime()
    assert not Signal("x", "A", Direction.SHORT, 100, 99, 98, 0.5, t).is_geometrically_valid()
    assert Signal("x", "A", Direction.SHORT, 100, 101, 98, 0.5, series_ts + timedelta(0)).is_geometrically_valid()
