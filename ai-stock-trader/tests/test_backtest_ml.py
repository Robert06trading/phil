import numpy as np
import pandas as pd
import pytest

from backtesting import metrics as M
from backtesting import report
from backtesting.engine import Backtester
from backtesting.walkforward import grid, walk_forward
from market_data.feeds import SyntheticSource, UniverseHistory
from ml.models import FEATURES, SignalModel, classify_regime, make_features, time_split, triple_barrier


@pytest.fixture(scope="module")
def bt_frames():
    src = SyntheticSource(seed=21)
    return {"AAPL": src.intraday("AAPL", days=14, venue="NASDAQ", vol=0.003, base_price=120),
            "SAP": src.intraday("SAP", days=14, venue="XETRA", vol=0.003, base_price=180)}


@pytest.fixture
def bt(settings, universe):
    def make(**kw):
        return Backtester(settings, {k: universe[k] for k in ("AAPL", "SAP")}, fx={("EUR", "USD"): 1.08}, **kw)
    return make


def test_backtest_runs_with_costs_and_rules(bt, bt_frames, settings):
    r = bt().run(bt_frames, 100_000)
    t = r.trades_frame()
    assert len(t) > 5 and r.synthetic_data
    assert (t["costs"] > 0).all()
    assert set(t["exit_reason"]) <= {"stop_loss", "take_profit", "eod", "loss_limit", "end_of_test",
                                     "delisted_or_data_end"}
    assert (pd.to_datetime(t.entry_ts).dt.date == pd.to_datetime(t.exit_ts).dt.date).all()   # no overnight
    assert (t["risk_amount"] <= 100_000 * settings.risk.risk_per_trade_pct / 100 * 1.6).all()
    # never more than max_open_positions concurrently
    ev = sorted([(a, 1) for a in t.entry_ts] + [(b, -1) for b in t.exit_ts], key=lambda x: (x[0], x[1]))
    assert max(np.cumsum([e for _, e in ev])) <= settings.risk.max_open_positions


def test_no_lookahead_by_future_perturbation(bt, bt_frames):
    """Trades that completed before T must be identical when all data after T is replaced."""
    r1 = bt(use_scanner=False).run(bt_frames, 100_000)
    cut = bt_frames["AAPL"].index[int(len(bt_frames["AAPL"]) * 0.6)]
    noisy = {}
    for k, df in bt_frames.items():
        d = df.copy()
        rng = np.random.default_rng(99)
        m = d.index > cut
        scale = rng.uniform(0.8, 1.2, m.sum())
        for c in ("open", "high", "low", "close"):
            d.loc[m, c] = d.loc[m, c] * scale
        d.loc[m, "high"] = d.loc[m, ["open", "high", "low", "close"]].max(axis=1)
        d.loc[m, "low"] = d.loc[m, ["open", "high", "low", "close"]].min(axis=1)
        d.attrs.update(df.attrs)
        noisy[k] = d
    r2 = bt(use_scanner=False).run(noisy, 100_000)
    before = lambda r: [(t.symbol, t.entry_ts, round(t.entry_px, 6), round(t.pnl, 6)) for t in r.trades  # noqa: E731
                        if pd.Timestamp(t.exit_ts) < cut]
    assert before(r1) and before(r1) == before(r2)


def test_partial_fill_capped_by_volume(bt, bt_frames, settings):
    thin = {k: v.assign(volume=v.volume / 400) for k, v in bt_frames.items()}
    for k in thin:
        thin[k].attrs.update(bt_frames[k].attrs)
    r = bt(use_scanner=False).run(thin, 100_000)
    for t in r.trades:
        bar = thin[t.symbol].loc[pd.Timestamp(t.entry_ts)]
        assert t.qty <= bar.volume * settings.costs.max_participation_rate + 1


def test_delisting_and_universe_history(bt, bt_frames, universe):
    short = dict(bt_frames)
    short["SAP"] = bt_frames["SAP"].iloc[: len(bt_frames["SAP"]) // 2]
    short["SAP"].attrs.update(bt_frames["SAP"].attrs)
    r = bt(use_scanner=False).run(short, 100_000)
    last = short["SAP"].index[-1]
    assert all(pd.Timestamp(t.exit_ts) <= last for t in r.trades if t.symbol == "SAP")
    first_day = bt_frames["AAPL"].index[0].date()
    uh = UniverseHistory({"AAPL": (first_day, None), "SAP": (first_day, first_day)})
    r2 = bt(use_scanner=False, universe_history=uh).run(bt_frames, 100_000)
    assert all(t.symbol == "AAPL" for t in r2.trades)


def test_metrics_on_known_trades():
    t = pd.DataFrame({"pnl": [100.0, -50.0, 200.0, -50.0], "r_multiple": [2, -1, 4, -1], "costs": [1, 1, 1, 1],
                      "symbol": "A", "strategies": "x", "entry_hour": 10, "exit_reason": "tp", "direction": "LONG",
                      "exit_ts": pd.date_range("2026-01-05", periods=4, tz="UTC")})
    eq = pd.Series([1000, 1100, 1050, 1250, 1200.0], index=pd.date_range("2026-01-04", periods=5, tz="UTC"))
    m = M.compute(t, eq, 1000)
    assert m["win_rate"] == 0.5 and m["profit_factor"] == pytest.approx(3.0) and m["expectancy"] == 50
    assert m["max_drawdown"] == pytest.approx(-50 / 1100) and m["total_return"] == pytest.approx(0.2)
    mc = M.monte_carlo(t.pnl.to_numpy(), 1000, n=500)
    assert mc["return_p05"] <= mc["return_p50"] <= mc["return_p95"]
    assert M.breakdown(t)["by_symbol"]["A"]["trades"] == 4


def test_verdict_never_passes_synthetic_and_detects_overfit():
    good = {"trades": 100, "profit_factor": 1.5, "expectancy_r": 0.2, "max_drawdown": -0.05}
    mc = {"return_p05": 0.01}
    assert M.verdict({"profit_factor": 1.6}, good, mc, synthetic=True)[0] == "INVALID_SYNTHETIC_DATA"
    assert M.verdict({"profit_factor": 1.6}, good, mc, synthetic=False)[0] == "PASS"
    assert M.verdict({"profit_factor": 4.0}, good, mc, synthetic=False)[0] == "FAIL"    # IS>>OOS: overfit
    assert M.verdict({"profit_factor": 1.6}, {**good, "trades": 5}, mc, synthetic=False)[0] == "FAIL"


def test_walk_forward_out_of_sample(settings, universe, bt_frames, tmp_path):
    settings.signals.enabled_strategies = ["momentum", "breakout"]
    wf = walk_forward(settings, {k: universe[k] for k in bt_frames}, bt_frames, 100_000, train_days=6,
                      test_days=3, fx={("EUR", "USD"): 1.08}, max_combos=2, use_scanner=False)
    assert len(wf.folds) >= 2
    for f in wf.folds:
        assert f["train"][1] < f["test"][0]                      # strictly time-ordered, no overlap
    assert wf.verdict == "INVALID_SYNTHETIC_DATA"
    paths = report.write(wf.oos, tmp_path, "wf test")
    assert "SYNTHETIC" in paths["html"].read_text()
    assert len(grid(["momentum", "breakout"])) == 16


# ----------------------------------------------------------------- ML
def test_time_split_has_embargo():
    tr, va, te = time_split(1000, 24)
    assert tr.stop + 24 <= va.start and va.stop + 24 <= te.start


def test_triple_barrier_labels_use_only_future_window(bt_frames):
    df = bt_frames["AAPL"]
    y = triple_barrier(df, horizon=6)
    assert set(y.dropna().unique()) <= {0.0, 1.0}
    y2 = triple_barrier(df.iloc[:-50], horizon=6)
    pd.testing.assert_series_equal(y.iloc[:-60], y2.iloc[:-10], check_names=False)


def test_features_are_causal(bt_frames):
    df = bt_frames["AAPL"]
    f_full = make_features(df, "America/New_York")
    f_cut = make_features(df.iloc[:400], "America/New_York")
    np.testing.assert_allclose(f_full.iloc[399].to_numpy(dtype=float), f_cut.iloc[-1].to_numpy(dtype=float), rtol=1e-9)
    assert list(f_full.columns) == FEATURES


def test_model_on_noise_is_not_accepted(bt_frames, tmp_path):
    src = SyntheticSource(seed=4)
    noise = {s: src.intraday(s, days=20, vol=0.003, regime_strength=0.0) for s in ("AAPL", "MSFT")}
    m = SignalModel(kind="random_forest")
    res = m.fit(m.build_dataset(noise, {s: "America/New_York" for s in noise}), min_auc=0.56)
    assert res["test"]["n"] > 100 and not m.accepted
    assert m.predict_up(make_features(noise["AAPL"], "America/New_York").tail(1)) is None   # rejected model is inert
    m.save(tmp_path / "m.joblib")
    assert SignalModel.load(tmp_path / "m.joblib").accepted is False


def test_regime_labels(bt_frames):
    r = classify_regime(bt_frames["AAPL"])
    assert set(r.unique()) <= {f"{a}{b}" for a in ("trend_up", "trend_down", "range") for b in ("", "_hv", "_lv")}
