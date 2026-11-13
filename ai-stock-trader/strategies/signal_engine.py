"""Central signal system: one rulebook for every trade idea.

Rulebook (applied in order, every rejection is recorded):
 1. Collect signals from all enabled strategies for the symbol.
 2. Conflict rule: if LONG and SHORT signals coexist on the same bar -> no trade.
 3. Geometry: stop on the correct side, target beyond entry.
 4. MTF gate: alignment of available timeframes in the trade direction must
    be >= min_mtf_alignment (if enabled).
 5. Combine agreeing signals: entry = latest close; stop = the most
    conservative (closest-to-entry stops are NOT chosen; we take the widest
    stop proposed so no strategy's invalidation point is violated); target
    re-derived so that reward:risk == max(min_reward_risk, median strategy RR).
 6. Quality score (0..100) = 45*mean(confidence) + 20*confluence
    (min(n_strategies,3)/3) + 20*mtf_alignment(0..1) + 15*scanner_score/100.
    Optional ML filter: if enabled and the model probability < threshold ->
    reject. ML can only REMOVE trades; it never adds or resizes them.
 7. Reject if quality < min_quality_score.
The risk manager then applies its own independent checks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from app.domain import Direction, Signal, TradeCandidate
from config import SignalConfig
from strategies.base import MarketContext, Strategy, mtf_alignment


@dataclass
class EngineResult:
    candidate: TradeCandidate | None
    signals: list[Signal] = field(default_factory=list)
    rejections: list[str] = field(default_factory=list)


class SignalEngine:
    def __init__(self, strategies: list[Strategy], cfg: SignalConfig, min_reward_risk: float,
                 ml_scorer: Callable[[MarketContext, Direction], float | None] | None = None):
        self.strategies = strategies
        self.cfg = cfg
        self.min_rr = min_reward_risk
        self.ml_scorer = ml_scorer

    def evaluate(self, ctx: MarketContext) -> EngineResult:
        signals = [s for s in (st.generate(ctx) for st in self.strategies) if s is not None]
        res = EngineResult(None, signals)
        if not signals:
            return res
        dirs = {s.direction for s in signals}
        if len(dirs) > 1:
            res.rejections.append("conflicting LONG/SHORT signals")
            return res
        direction = dirs.pop()
        valid = [s for s in signals if s.is_geometrically_valid()]
        if not valid:
            res.rejections.append("no geometrically valid signal")
            return res
        align = mtf_alignment(ctx, direction)
        if self.cfg.require_mtf_alignment and align < self.cfg.min_mtf_alignment:
            res.rejections.append(f"MTF alignment {align:+.2f} < {self.cfg.min_mtf_alignment}")
            return res

        entry = float(ctx.base["close"].iloc[-1])
        stop = min(s.stop for s in valid) if direction is Direction.LONG else max(s.stop for s in valid)
        risk = abs(entry - stop)
        if risk <= 0:
            res.rejections.append("zero risk distance")
            return res
        rr = max(self.min_rr, float(np.median([s.reward_risk for s in valid])))
        target = entry + direction.sign * rr * risk

        conf = float(np.mean([s.confidence for s in valid]))
        confluence = min(len(valid), 3) / 3.0
        scan = (ctx.scan_score or 0.0) / 100.0
        quality = 45 * conf + 20 * confluence + 20 * max(align, 0.0) + 15 * scan

        ml_p = None
        if self.cfg.use_ml_filter and self.ml_scorer is not None:
            ml_p = self.ml_scorer(ctx, direction)
            if ml_p is None or ml_p < self.cfg.ml_min_probability:
                res.rejections.append(f"ML filter: p={ml_p}")
                return res
        if quality < self.cfg.min_quality_score:
            res.rejections.append(f"quality {quality:.1f} < {self.cfg.min_quality_score}")
            return res

        reasons = [f"{s.strategy}: {'; '.join(s.reasons)}" for s in valid]
        reasons.append(f"mtf={align:+.2f} quality={quality:.1f}")
        res.candidate = TradeCandidate(ctx.symbol, direction, entry, float(stop), float(target), round(quality, 2),
                                       [s.strategy for s in valid], ctx.now.to_pydatetime(), reasons, ml_p)
        return res
