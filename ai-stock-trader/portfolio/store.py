"""Persistence (SQLAlchemy 2.x). SQLite by default, PostgreSQL in Docker
(``TRADER_DATABASE_URL=postgresql+psycopg://...``).

The database is the system's memory across restarts: order intents are
written BEFORE they are sent (write-ahead), so a crash between "decide" and
"send" can be reconciled against the broker by client order id.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, Integer, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class KV(Base):
    __tablename__ = "kv_state"
    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[str] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)


class OrderRow(Base):
    __tablename__ = "orders"
    client_order_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    broker_order_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    parent_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    mode: Mapped[str] = mapped_column(String(16))
    symbol: Mapped[str] = mapped_column(String(16), index=True)
    side: Mapped[str] = mapped_column(String(4))
    role: Mapped[str] = mapped_column(String(16))
    order_type: Mapped[str] = mapped_column(String(8))
    quantity: Mapped[int] = mapped_column(Integer)
    filled: Mapped[int] = mapped_column(Integer, default=0)
    limit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    stop_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_fill_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(20), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)


class TradeRow(Base):
    __tablename__ = "trades"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trade_key: Mapped[str] = mapped_column(String(64), unique=True)   # entry client_order_id
    mode: Mapped[str] = mapped_column(String(16), index=True)
    symbol: Mapped[str] = mapped_column(String(16), index=True)
    direction: Mapped[str] = mapped_column(String(5))
    strategies: Mapped[str] = mapped_column(String(128))
    quantity: Mapped[int] = mapped_column(Integer)
    entry_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    stop_price: Mapped[float] = mapped_column(Float)
    target_price: Mapped[float] = mapped_column(Float)
    exit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    exit_reason: Mapped[str | None] = mapped_column(String(32), nullable=True)
    pnl: Mapped[float | None] = mapped_column(Float, nullable=True)        # base currency, net of costs
    commission: Mapped[float] = mapped_column(Float, default=0.0)
    quality_score: Mapped[float] = mapped_column(Float, default=0.0)
    risk_amount: Mapped[float] = mapped_column(Float, default=0.0)
    opened_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(String(12), default="open", index=True)
    meta: Mapped[dict] = mapped_column(JSON, default=dict)


class EquityRow(Base):
    __tablename__ = "equity_snapshots"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    mode: Mapped[str] = mapped_column(String(16))
    equity: Mapped[float] = mapped_column(Float)
    cash: Mapped[float] = mapped_column(Float)
    unrealized: Mapped[float] = mapped_column(Float, default=0.0)
    realized_today: Mapped[float] = mapped_column(Float, default=0.0)


class AuditRow(Base):
    __tablename__ = "audit_log"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    actor: Mapped[str] = mapped_column(String(64))
    action: Mapped[str] = mapped_column(String(64), index=True)
    detail: Mapped[dict] = mapped_column(JSON, default=dict)
    level: Mapped[str] = mapped_column(String(8), default="INFO")


class SignalRow(Base):
    __tablename__ = "signals"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    symbol: Mapped[str] = mapped_column(String(16))
    direction: Mapped[str] = mapped_column(String(5))
    entry: Mapped[float] = mapped_column(Float)
    stop: Mapped[float] = mapped_column(Float)
    target: Mapped[float] = mapped_column(Float)
    quality: Mapped[float] = mapped_column(Float)
    strategies: Mapped[str] = mapped_column(String(128))
    decision: Mapped[str] = mapped_column(String(16))   # approved / rejected
    reasons: Mapped[list] = mapped_column(JSON, default=list)


class ScanRow(Base):
    __tablename__ = "scan_results"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    symbol: Mapped[str] = mapped_column(String(16))
    score: Mapped[float] = mapped_column(Float)
    bias: Mapped[str] = mapped_column(String(8))
    components: Mapped[dict] = mapped_column(JSON, default=dict)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)


class Database:
    def __init__(self, url: str):
        kw = {"connect_args": {"check_same_thread": False}} if url.startswith("sqlite") else {"pool_pre_ping": True}
        self.engine = create_engine(url, future=True, **kw)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(self.engine, expire_on_commit=False)

    # ---- StateStore protocol (used by risk manager, paper broker, kill switch)
    def get(self, key: str) -> dict | None:
        with self.Session() as s:
            row = s.get(KV, key)
            return json.loads(row.value) if row else None

    def set(self, key: str, value: dict) -> None:
        with self.Session.begin() as s:
            row = s.get(KV, key)
            payload = json.dumps(value, default=str)
            if row:
                row.value = payload
            else:
                s.add(KV(key=key, value=payload))

    # ---- audit
    def audit(self, actor: str, action: str, detail: dict | None = None, level: str = "INFO") -> None:
        with self.Session.begin() as s:
            s.add(AuditRow(actor=actor, action=action, detail=detail or {}, level=level))

    def session(self) -> Session:
        return self.Session()

    def upsert_order(self, mode: str, **fields) -> None:
        with self.Session.begin() as s:
            row = s.get(OrderRow, fields["client_order_id"])
            if row is None:
                s.add(OrderRow(mode=mode, **fields))
            else:
                for k, v in fields.items():
                    setattr(row, k, v)

    def open_order_rows(self, mode: str) -> list[OrderRow]:
        terminal = ("Filled", "Cancelled", "Rejected", "Inactive")
        with self.Session() as s:
            return list(s.scalars(select(OrderRow).where(OrderRow.mode == mode, OrderRow.status.not_in(terminal))))

    def open_trades(self, mode: str) -> list[TradeRow]:
        with self.Session() as s:
            return list(s.scalars(select(TradeRow).where(TradeRow.mode == mode, TradeRow.status == "open")))

    def closed_trades(self, mode: str | None = None) -> list[TradeRow]:
        with self.Session() as s:
            q = select(TradeRow).where(TradeRow.status == "closed")
            if mode:
                q = q.where(TradeRow.mode == mode)
            return list(s.scalars(q.order_by(TradeRow.closed_at)))
