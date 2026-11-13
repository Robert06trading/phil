"""Transaction cost model shared by sizing, the paper broker and backtests,
so that all three agree on what a trade costs."""
from __future__ import annotations

from config import CostConfig


def commission(cfg: CostConfig, currency: str, qty: int, price: float) -> float:
    """Commission in the instrument currency (IBKR-like tiered approximation;
    adjust CostConfig to your actual plan)."""
    if qty <= 0:
        return 0.0
    notional = qty * price
    if currency == "USD":
        c = max(cfg.us_commission_min, qty * cfg.us_commission_per_share)
        return min(c, notional * cfg.us_commission_max_pct / 100)
    return max(cfg.de_commission_min, notional * cfg.de_commission_pct / 100)


def slippage_per_share(cfg: CostConfig, price: float, half_spread_bps: float | None = None) -> float:
    hs = cfg.default_half_spread_bps if half_spread_bps is None else half_spread_bps
    return price * (cfg.slippage_bps + hs) / 1e4


def round_trip_cost_per_share(cfg: CostConfig, currency: str, price: float, qty_hint: int = 100,
                              half_spread_bps: float | None = None) -> float:
    return 2 * slippage_per_share(cfg, price, half_spread_bps) + 2 * commission(cfg, currency, qty_hint, price) / qty_hint
