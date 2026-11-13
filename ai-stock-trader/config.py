"""Central configuration.

All settings come from environment variables (prefix ``TRADER_``) or an
optional ``.env`` file. Nothing secret is hard-coded. Risk parameters carry
HARD CEILINGS enforced by validators: no config file, API call or ML model can
raise them beyond these limits.
"""
from __future__ import annotations

import os
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator

ROOT = Path(__file__).resolve().parent


class TradingMode(str, Enum):
    BACKTEST = "backtest"
    REPLAY = "replay"   # paper engine fed by recorded historical data, clearly labelled
    PAPER = "paper"
    LIVE = "live"


class BrokerKind(str, Enum):
    PAPER = "paper"     # local simulated broker (uses real market data feed)
    IBKR = "ibkr"       # Interactive Brokers via ib_async (paper OR live account)


# Absolute ceilings. Changing these is a code change + review, never config.
HARD_LIMITS = {
    "risk_per_trade_pct": 1.0,
    "max_daily_loss_pct": 3.0,
    "max_weekly_loss_pct": 8.0,
    "max_open_positions": 10,
    "max_portfolio_risk_pct": 3.0,
    "max_position_notional_pct": 50.0,
}


class RiskConfig(BaseModel):
    risk_per_trade_pct: float = Field(0.5, gt=0)
    max_daily_loss_pct: float = Field(2.0, gt=0)
    max_weekly_loss_pct: float = Field(5.0, gt=0)
    max_open_positions: int = Field(3, ge=1)
    min_reward_risk: float = Field(2.0, ge=1.0)
    max_portfolio_risk_pct: float = Field(1.5, gt=0)
    max_position_notional_pct: float = Field(25.0, gt=0)
    max_spread_bps: float = Field(15.0, gt=0)
    max_data_age_seconds: float = Field(15.0, gt=0)
    min_price: float = Field(5.0, gt=0)
    no_new_entries_minutes_before_close: int = Field(30, ge=0)
    flatten_minutes_before_close: int = Field(10, ge=1)
    no_entries_minutes_after_open: int = Field(5, ge=0)
    # risk may only be REDUCED after losing streaks, never increased
    loss_streak_reduce_after: int = Field(3, ge=1)
    loss_streak_risk_factor: float = Field(0.5, gt=0, le=1.0)

    @model_validator(mode="after")
    def _ceilings(self) -> "RiskConfig":
        for key, ceiling in HARD_LIMITS.items():
            if getattr(self, key) > ceiling:
                raise ValueError(f"{key}={getattr(self, key)} exceeds hard ceiling {ceiling}")
        if self.flatten_minutes_before_close > self.no_new_entries_minutes_before_close:
            raise ValueError("flatten window must lie inside the no-new-entries window")
        return self


class CostConfig(BaseModel):
    """Transaction cost model (used by backtest, paper broker and sizing)."""
    us_commission_per_share: float = 0.0035
    us_commission_min: float = 0.35
    us_commission_max_pct: float = 1.0
    de_commission_pct: float = 0.05
    de_commission_min: float = 1.25
    slippage_bps: float = 2.0
    default_half_spread_bps: float = 2.0
    max_participation_rate: float = Field(0.1, gt=0, le=1.0)  # share of bar volume we may fill


class ScannerConfig(BaseModel):
    min_avg_dollar_volume: float = 20_000_000
    min_rvol: float = 1.2
    top_n: int = 10
    rescan_minutes: int = 15


class SignalConfig(BaseModel):
    enabled_strategies: list[str] = ["momentum", "breakout", "vwap", "ema_trend", "mtf"]
    min_quality_score: float = 60.0
    require_mtf_alignment: bool = True
    min_mtf_alignment: float = 0.5
    use_ml_filter: bool = False
    ml_min_probability: float = 0.5


class Settings(BaseModel):
    mode: TradingMode = TradingMode.PAPER
    broker: BrokerKind = BrokerKind.PAPER
    live_trading_enabled: bool = False      # second, independent switch for live

    # Capital: never hard-coded. Read from broker; paper fallback via config.
    paper_starting_capital: float | None = None
    base_currency: str = "USD"
    fx_eurusd_fallback: float | None = None  # only for backtests without FX data

    venues: list[str] = ["NASDAQ", "NYSE", "XETRA"]
    universe_file: Path = ROOT / "data" / "universe.csv"
    allowed_stock_types: list[str] = ["COMMON", "ADR"]
    allow_extended_hours: bool = False

    ibkr_host: str = "127.0.0.1"
    ibkr_port: int = 4002          # 4002 = IB Gateway paper, 4001 = live
    ibkr_client_id: int = 17
    ibkr_account: str | None = None
    ibkr_market_data_type: int = 1  # 1 realtime, 3 delayed (never used for trading)

    database_url: str = f"sqlite:///{ROOT / 'data' / 'trader.db'}"
    data_dir: Path = ROOT / "data"
    log_dir: Path = ROOT / "logs"
    report_dir: Path = ROOT / "reports"
    kill_switch_file: Path = ROOT / "data" / "KILL_SWITCH"

    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None

    api_admin_token_sha256: str | None = None
    api_viewer_token_sha256: str | None = None
    api_totp_secret: str | None = None       # base32, required for live approval
    approval_hmac_secret: str | None = None  # signs live approvals
    api_rate_limit_per_minute: int = 60

    loop_interval_seconds: int = 60
    min_paper_days_for_live: int = 20
    min_paper_trades_for_live: int = 50

    risk: RiskConfig = RiskConfig()
    costs: CostConfig = CostConfig()
    scanner: ScannerConfig = ScannerConfig()
    signals: SignalConfig = SignalConfig()

    @field_validator("venues")
    @classmethod
    def _venues(cls, v: list[str]) -> list[str]:
        bad = set(v) - {"NASDAQ", "NYSE", "XETRA"}
        if bad:
            raise ValueError(f"venues not permitted: {sorted(bad)}")
        return v

    @field_validator("allowed_stock_types")
    @classmethod
    def _stock_types(cls, v: list[str]) -> list[str]:
        bad = set(v) - {"COMMON", "ADR"}
        if bad:
            raise ValueError(f"stock types not permitted: {sorted(bad)} (only COMMON/ADR)")
        return v

    @model_validator(mode="after")
    def _mode_consistency(self) -> "Settings":
        if self.mode == TradingMode.LIVE and self.broker != BrokerKind.IBKR:
            raise ValueError("live mode requires broker=ibkr")
        return self


def _load_dotenv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def load_settings(env: dict[str, str] | None = None, dotenv: Path | None = ROOT / ".env") -> Settings:
    """Build Settings from TRADER_* variables. Nested keys use '__', e.g.
    TRADER_RISK__MAX_OPEN_POSITIONS=2. Lists are comma separated."""
    merged: dict[str, str] = {}
    if dotenv is not None:
        merged.update(_load_dotenv(dotenv))
    merged.update(os.environ if env is None else env)
    data: dict = {}
    for key, raw in merged.items():
        if not key.startswith("TRADER_"):
            continue
        parts = key[len("TRADER_"):].lower().split("__")
        target = data
        for p in parts[:-1]:
            target = target.setdefault(p, {})
        target[parts[-1]] = raw
    _coerce_lists(data, Settings)
    return Settings.model_validate(data)


def _coerce_lists(data: dict, model: type[BaseModel]) -> None:
    for name, field in model.model_fields.items():
        if name not in data:
            continue
        ann = field.annotation
        if isinstance(data[name], dict) and isinstance(ann, type) and issubclass(ann, BaseModel):
            _coerce_lists(data[name], ann)
        elif isinstance(data[name], str) and getattr(ann, "__origin__", None) is list:
            data[name] = [s.strip() for s in data[name].split(",") if s.strip()]
        elif data[name] == "" and field.default is None:
            data[name] = None
