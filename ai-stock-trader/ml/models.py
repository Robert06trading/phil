"""Optional machine-learning layer (advisory only).

What it does
* ``regime_features`` / ``classify_regime``: transparent rule-based regime
  labels (trend / range, volatility expansion / contraction).
* ``SignalModel``: estimates P(+k·ATR is reached before -k·ATR within H bars)
  — a *defined* price move (triple-barrier label) — with XGBoost,
  RandomForest or GradientBoosting.

What it can NOT do (by construction)
* It only returns a probability. ``SignalEngine`` may use it to *reject* a
  candidate; it cannot create trades, change size, stops or limits, and has
  no reference to the risk manager, broker or live gate.

Leakage controls
* Features use only bars <= t (causal indicators, ``shift`` where needed).
* Labels use bars t+1..t+H; the time-ordered split leaves an EMBARGO of H
  bars between train/validation/test so no label window overlaps the next
  set. No shuffling anywhere.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score

from app.domain import Direction
from indicators.technical import enrich

FEATURES = ["ret1", "ret3", "ret6", "ret12", "atr_pct", "rsi", "adx", "vwap_dist", "ema9_20", "ema20_50",
            "ema20_slope", "rvol", "tod", "vol_ratio", "efficiency"]


def efficiency_ratio(close: pd.Series, n: int = 20) -> pd.Series:
    change = (close - close.shift(n)).abs()
    path = close.diff().abs().rolling(n).sum()
    return change / path.replace(0, np.nan)


def make_features(df: pd.DataFrame, tz: str) -> pd.DataFrame:
    e = df if "ema9" in df.columns else enrich(df, tz)
    c = e["close"]
    f = pd.DataFrame(index=e.index)
    for n in (1, 3, 6, 12):
        f[f"ret{n}"] = c.pct_change(n)
    f["atr_pct"] = e["atr"] / c
    f["rsi"] = e["rsi"] / 100
    f["adx"] = e["adx"] / 100
    f["vwap_dist"] = (c - e["vwap"]) / e["atr"]
    f["ema9_20"] = (e["ema9"] - e["ema20"]) / e["atr"]
    f["ema20_50"] = (e["ema20"] - e["ema50"]) / e["atr"]
    f["ema20_slope"] = e["ema20"].diff(5) / e["atr"]
    f["rvol"] = np.log1p(e["rvol"])
    local = e.index.tz_convert(tz)
    f["tod"] = (local.hour * 60 + local.minute) / (24 * 60)
    f["vol_ratio"] = e["atr"] / e["atr"].rolling(100, min_periods=20).mean()
    f["efficiency"] = efficiency_ratio(c, 20)
    return f[FEATURES]


def triple_barrier(df: pd.DataFrame, horizon: int = 12, k: float = 1.0) -> pd.Series:
    """1 if the upper barrier (close + k*ATR) is touched first within the next
    ``horizon`` bars, 0 if the lower barrier first, NaN if neither / tie."""
    e = df if "atr" in df.columns else enrich(df)
    hi, lo, c, a = e["high"].to_numpy(), e["low"].to_numpy(), e["close"].to_numpy(), e["atr"].to_numpy()
    y = np.full(len(e), np.nan)
    for t in range(len(e) - 1):
        if np.isnan(a[t]):
            continue
        up, dn = c[t] + k * a[t], c[t] - k * a[t]
        for j in range(t + 1, min(t + 1 + horizon, len(e))):
            u, d = hi[j] >= up, lo[j] <= dn
            if u and d:
                break          # ambiguous inside one bar -> leave NaN
            if u:
                y[t] = 1
                break
            if d:
                y[t] = 0
                break
    return pd.Series(y, index=e.index)


def regime_features(df: pd.DataFrame) -> pd.DataFrame:
    e = df if "adx" in df.columns else enrich(df)
    out = pd.DataFrame(index=e.index)
    out["adx"] = e["adx"]
    out["efficiency"] = efficiency_ratio(e["close"], 20)
    atr_pct = e["atr"] / e["close"]
    out["vol_pct_rank"] = atr_pct.rolling(200, min_periods=50).rank(pct=True)
    return out


def classify_regime(df: pd.DataFrame) -> pd.Series:
    """trend_up / trend_down / range, suffixed with _hv / _lv for volatility."""
    r = regime_features(df)
    e = df if "ema20" in df.columns else enrich(df)
    trend = (r["adx"] >= 25) & (r["efficiency"] >= 0.35)
    base = np.where(trend & (e["close"] > e["ema50"]), "trend_up", np.where(trend, "trend_down", "range"))
    vol = np.where(r["vol_pct_rank"] >= 0.8, "_hv", np.where(r["vol_pct_rank"] <= 0.2, "_lv", ""))
    return pd.Series([a + b for a, b in zip(base, vol, strict=True)], index=df.index)


def time_split(n: int, embargo: int, frac=(0.6, 0.2, 0.2)) -> tuple[slice, slice, slice]:
    a = int(n * frac[0])
    b = int(n * (frac[0] + frac[1]))
    return slice(0, a - embargo), slice(a, b - embargo), slice(b, n)


@dataclass
class SignalModel:
    kind: str = "xgboost"
    horizon: int = 12
    barrier_k: float = 1.0
    model: object = None
    metrics: dict = field(default_factory=dict)
    trained_at: str = ""
    accepted: bool = False

    def _new(self):
        if self.kind == "xgboost":
            from xgboost import XGBClassifier
            return XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
                                 colsample_bytree=0.8, eval_metric="logloss", n_jobs=2, random_state=0)
        if self.kind == "random_forest":
            return RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=50, n_jobs=2, random_state=0)
        if self.kind == "gradient_boosting":
            return GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=0)
        raise ValueError(self.kind)

    def build_dataset(self, frames: dict[str, pd.DataFrame], tz_of: dict[str, str]) -> pd.DataFrame:
        parts = []
        for sym, df in frames.items():
            e = enrich(df, tz_of[sym])
            X = make_features(e, tz_of[sym])
            X["y"] = triple_barrier(e, self.horizon, self.barrier_k)
            X["symbol"] = sym
            parts.append(X)
        data = pd.concat(parts).dropna()
        return data.sort_index(kind="stable")

    def fit(self, data: pd.DataFrame, min_auc: float = 0.53) -> dict:
        n = len(data)
        if n < 500:
            raise ValueError(f"not enough samples ({n})")
        # embargo in rows ~ horizon * number of symbols (rows are time-sorted across symbols)
        emb = self.horizon * max(data["symbol"].nunique(), 1)
        tr, va, te = time_split(n, emb)
        X, y = data[FEATURES].to_numpy(), data["y"].to_numpy().astype(int)
        self.model = self._new()
        self.model.fit(X[tr], y[tr])
        out = {}
        for name, sl in (("train", tr), ("validation", va), ("test", te)):
            p = self.model.predict_proba(X[sl])[:, 1]
            out[name] = {"n": int(len(p)), "auc": float(roc_auc_score(y[sl], p)) if len(set(y[sl])) > 1 else 0.5,
                         "brier": float(brier_score_loss(y[sl], p)), "base_rate": float(y[sl].mean())}
        out["periods"] = {k: [str(data.index[s][0]), str(data.index[s][-1])] for k, s in
                          (("train", tr), ("validation", va), ("test", te))}
        # accepted only if it beats chance on BOTH validation and the untouched test set
        self.accepted = out["validation"]["auc"] >= min_auc and out["test"]["auc"] >= min_auc
        out["accepted"] = self.accepted
        self.metrics = out
        self.trained_at = datetime.now(timezone.utc).isoformat()
        return out

    def predict_up(self, features_row: pd.DataFrame) -> float | None:
        if self.model is None or not self.accepted:
            return None
        x = features_row[FEATURES].to_numpy()
        if np.isnan(x).any():
            return None
        return float(self.model.predict_proba(x)[:, 1][-1])

    def scorer(self):
        """Callable for SignalEngine: (ctx, direction) -> probability or None."""
        def score(ctx, direction: Direction) -> float | None:
            f = make_features(ctx.base, ctx.tz).tail(1)
            p = self.predict_up(f)
            if p is None:
                return None
            return p if direction is Direction.LONG else 1 - p
        return score

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"kind": self.kind, "horizon": self.horizon, "k": self.barrier_k, "model": self.model,
                     "metrics": self.metrics, "trained_at": self.trained_at, "accepted": self.accepted,
                     "features": FEATURES}, path)
        path.with_suffix(".json").write_text(json.dumps({k: v for k, v in self.metrics.items()}, indent=2))

    @classmethod
    def load(cls, path: Path) -> "SignalModel":
        d = joblib.load(path)
        if d["features"] != FEATURES:
            raise ValueError("feature set changed since training — retrain")
        return cls(d["kind"], d["horizon"], d["k"], d["model"], d["metrics"], d["trained_at"], d["accepted"])
