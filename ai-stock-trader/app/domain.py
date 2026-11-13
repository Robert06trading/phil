"""Shared domain types used by every module (broker, risk, execution, backtest)."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class Direction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"

    @property
    def sign(self) -> int:
        return 1 if self is Direction.LONG else -1


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MKT"
    LIMIT = "LMT"
    STOP = "STP"
    STOP_LIMIT = "STP LMT"


class OrderStatus(str, Enum):
    PENDING_SUBMIT = "PendingSubmit"   # written to DB before it is sent (write-ahead)
    SUBMITTED = "Submitted"
    PARTIALLY_FILLED = "PartiallyFilled"
    FILLED = "Filled"
    CANCELLED = "Cancelled"
    REJECTED = "Rejected"
    INACTIVE = "Inactive"

    @property
    def is_terminal(self) -> bool:
        return self in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.INACTIVE)


class OrderRole(str, Enum):
    ENTRY = "entry"
    STOP = "stop"
    TAKE_PROFIT = "take_profit"
    EXIT = "exit"           # discretionary / EOD / kill-switch close


@dataclass(frozen=True)
class Instrument:
    symbol: str
    venue: str                 # NASDAQ | NYSE | XETRA
    currency: str              # USD | EUR
    sec_type: str = "STK"
    stock_type: str | None = None   # COMMON / ADR / ETF ... as reported by broker
    con_id: int | None = None
    name: str = ""


@dataclass
class Quote:
    symbol: str
    bid: float
    ask: float
    last: float
    ts: datetime
    bid_size: float = 0.0
    ask_size: float = 0.0
    is_realtime: bool = True
    source: str = "broker"      # "broker" (real-time feed) | "replay" (recorded data, labelled)

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread_bps(self) -> float:
        if self.bid <= 0 or self.ask <= 0 or self.ask < self.bid:
            return float("inf")
        return (self.ask - self.bid) / self.mid * 1e4


@dataclass
class Signal:
    strategy: str
    symbol: str
    direction: Direction
    entry: float
    stop: float
    target: float
    confidence: float           # 0..1, rule-based, documented per strategy
    bar_ts: datetime            # close time of the bar that produced the signal
    reasons: list[str] = field(default_factory=list)

    @property
    def risk_per_share(self) -> float:
        return abs(self.entry - self.stop)

    @property
    def reward_risk(self) -> float:
        r = self.risk_per_share
        return abs(self.target - self.entry) / r if r > 0 else 0.0

    def is_geometrically_valid(self) -> bool:
        if min(self.entry, self.stop, self.target) <= 0:
            return False
        if self.direction is Direction.LONG:
            return self.stop < self.entry < self.target
        return self.target < self.entry < self.stop


@dataclass
class TradeCandidate:
    """Output of the signal engine: a single, rule-checked trade idea."""
    symbol: str
    direction: Direction
    entry: float
    stop: float
    target: float
    quality_score: float               # 0..100
    strategies: list[str]
    bar_ts: datetime
    reasons: list[str] = field(default_factory=list)
    ml_probability: float | None = None

    @property
    def reward_risk(self) -> float:
        r = abs(self.entry - self.stop)
        return abs(self.target - self.entry) / r if r > 0 else 0.0

    @property
    def client_order_id(self) -> str:
        """Deterministic id -> the same signal can never create two orders,
        even across restarts."""
        key = f"{self.symbol}|{self.direction.value}|{self.bar_ts.isoformat()}"
        return "AST-" + hashlib.sha1(key.encode()).hexdigest()[:16]


@dataclass
class OrderRequest:
    client_order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType
    role: OrderRole
    limit_price: float | None = None
    stop_price: float | None = None
    tif: str = "DAY"
    parent_id: str | None = None
    oca_group: str | None = None

    def validate(self) -> None:
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if self.order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT) and not self.limit_price:
            raise ValueError("limit price required")
        if self.order_type in (OrderType.STOP, OrderType.STOP_LIMIT) and not self.stop_price:
            raise ValueError("stop price required")


@dataclass
class BracketRequest:
    entry: OrderRequest
    stop: OrderRequest
    take_profit: OrderRequest

    def validate(self) -> None:
        for o in (self.entry, self.stop, self.take_profit):
            o.validate()
        if not (self.entry.quantity == self.stop.quantity == self.take_profit.quantity):
            raise ValueError("bracket legs must have equal quantity")
        if self.stop.side == self.entry.side or self.take_profit.side == self.entry.side:
            raise ValueError("protective legs must be on the opposite side")


@dataclass
class OrderState:
    client_order_id: str
    broker_order_id: str | None
    symbol: str
    side: OrderSide
    quantity: int
    filled: int
    avg_fill_price: float | None
    status: OrderStatus
    order_type: OrderType
    role: OrderRole
    limit_price: float | None = None
    stop_price: float | None = None
    parent_id: str | None = None

    @property
    def remaining(self) -> int:
        return max(self.quantity - self.filled, 0)


@dataclass
class Fill:
    client_order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    price: float
    commission: float
    ts: datetime
    exec_id: str = ""


@dataclass
class Position:
    symbol: str
    quantity: int               # signed: + long, - short
    avg_price: float
    currency: str = "USD"
    market_price: float | None = None

    @property
    def direction(self) -> Direction | None:
        if self.quantity > 0:
            return Direction.LONG
        if self.quantity < 0:
            return Direction.SHORT
        return None

    @property
    def unrealized_pnl(self) -> float:
        if self.market_price is None:
            return 0.0
        return (self.market_price - self.avg_price) * self.quantity


@dataclass
class AccountSnapshot:
    account_id: str
    equity: float               # net liquidation in base currency
    cash: float
    buying_power: float
    currency: str
    ts: datetime
    realized_pnl_today: float = 0.0
    unrealized_pnl: float = 0.0
