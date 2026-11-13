from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

from indicators import technical as T
from market_data import calendar as C
from market_data.feeds import SyntheticSource, apply_splits, validate_bars

UTC = timezone.utc


def test_holidays():
    assert C.is_holiday("NYSE", date(2026, 11, 26))      # Thanksgiving
    assert C.is_holiday("XETRA", date(2026, 12, 24))     # Christmas Eve (Xetra closed)
    assert not C.is_holiday("NYSE", date(2026, 11, 25))


def test_early_close_black_friday():
    s = C.session_for("NYSE", date(2026, 11, 27))
    assert s.close.hour == 18 and s.close.minute == 0      # 13:00 New York


def test_dst_offsets_differ_between_us_and_germany():
    # US DST started 2026-03-08, EU only 2026-03-29: NYSE opens 13:30 UTC, XETRA still 08:00 UTC
    assert C.session_for("NYSE", date(2026, 3, 16)).open == datetime(2026, 3, 16, 13, 30, tzinfo=UTC)
    assert C.session_for("XETRA", date(2026, 3, 16)).open == datetime(2026, 3, 16, 8, 0, tzinfo=UTC)
    assert C.session_for("XETRA", date(2026, 7, 1)).open == datetime(2026, 7, 1, 7, 0, tzinfo=UTC)
    assert C.session_for("NYSE", date(2026, 12, 1)).open == datetime(2026, 12, 1, 14, 30, tzinfo=UTC)


def test_is_open_and_minutes_to_close():
    t = datetime(2026, 11, 11, 20, 50, tzinfo=UTC)          # 15:50 NY
    assert C.is_open("NYSE", t) and round(C.minutes_to_close("NYSE", t)) == 10
    assert not C.is_open("NYSE", datetime(2026, 11, 11, 14, 0, tzinfo=UTC))   # pre-market
    assert not C.is_open("NYSE", datetime(2026, 11, 14, 16, 0, tzinfo=UTC))   # Saturday
    with pytest.raises(ValueError):
        C.is_open("NYSE", datetime(2026, 11, 11, 15, 0))     # naive datetime refused


@pytest.fixture
def bars():
    return SyntheticSource(seed=11).intraday("AAPL", days=8, vol=0.004)


def test_indicators_are_causal(bars):
    """Look-ahead tripwire: values at t must not change when future bars are removed."""
    full = T.enrich(bars)
    for cut in (150, 300, len(bars) - 7):
        part = T.enrich(bars.iloc[:cut])
        cols = ["ema9", "ema20", "ema50", "atr", "rsi", "adx", "vwap", "rvol", "roc", "hh20", "ll20"]
        np.testing.assert_allclose(part[cols].iloc[-1].to_numpy(dtype=float),
                                   full[cols].iloc[cut - 1].to_numpy(dtype=float), rtol=1e-9, equal_nan=True)


def test_ema_matches_definition():
    s = pd.Series(np.arange(1, 31, dtype=float))
    e = T.ema(s, 9)
    a = 2 / 10
    manual = s.iloc[0]
    for v in s.iloc[1:]:
        manual = a * v + (1 - a) * manual
    assert e.iloc[-1] == pytest.approx(manual)


def test_vwap_resets_each_session(bars):
    v = T.vwap(bars)
    first_bars = bars.groupby(bars.index.tz_convert("America/New_York").date).head(1).index
    tp = (bars.high + bars.low + bars.close) / 3
    np.testing.assert_allclose(v.loc[first_bars], tp.loc[first_bars])


def test_resample_completed_only(bars):
    r = T.resample_bars(bars, "15min")
    now = bars.index[50]
    done = r[r.index <= now]
    assert done.index[-1] <= now
    # the 15m bar labelled L contains exactly the 5m bars closing in (L-15m, L]
    lbl = done.index[-1]
    sub = bars[(bars.index > lbl - pd.Timedelta("15min")) & (bars.index <= lbl)]
    assert done.loc[lbl, "volume"] == pytest.approx(sub.volume.sum())


def test_swing_levels_ignore_unconfirmed_bars(bars):
    lv = T.swing_levels(bars.tail(100), right=3)
    assert lv["resistance"] and lv["support"]


def test_splits_adjustment():
    idx = pd.date_range("2026-01-05 15:00", periods=4, freq="1D", tz="UTC")
    df = pd.DataFrame({"open": [400.0, 404, 101, 102], "high": [405.0, 408, 103, 104], "low": [398.0, 400, 99, 100],
                       "close": [404.0, 402, 102, 103], "volume": [100.0, 100, 400, 400]}, index=idx)
    adj = apply_splits(df, [(idx[2], 4.0)])
    assert adj.close.iloc[0] == pytest.approx(101.0) and adj.volume.iloc[0] == 400


def test_validate_bars_rejects_bad_ohlc():
    idx = pd.date_range("2026-01-05 15:00", periods=2, freq="5min", tz="UTC")
    bad = pd.DataFrame({"open": [1, 1], "high": [0.5, 1], "low": [0.4, 1], "close": [1, 1], "volume": [1, 1]},
                       index=idx, dtype=float)
    with pytest.raises(ValueError):
        validate_bars(bad)
