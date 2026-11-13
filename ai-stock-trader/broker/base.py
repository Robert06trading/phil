"""Broker adapter interface. Live (IBKR) and paper (simulated) brokers
implement the same contract, so the engine, risk checks and order manager are
identical in every mode."""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd

from app.domain import AccountSnapshot, BracketRequest, Fill, Instrument, OrderRequest, OrderState, Position, Quote


class BrokerError(Exception):
    pass


class BrokerDisconnected(BrokerError):
    pass


class Broker(ABC):
    name: str = "base"
    is_paper: bool = True

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def is_connected(self) -> bool: ...

    @abstractmethod
    def account(self) -> AccountSnapshot: ...

    @abstractmethod
    def positions(self) -> list[Position]: ...

    @abstractmethod
    def open_orders(self) -> list[OrderState]: ...

    @abstractmethod
    def qualify(self, inst: Instrument) -> Instrument:
        """Return the instrument as the broker knows it (secType, stockType,
        primary exchange, conId) so the whitelist can verify it."""

    @abstractmethod
    def quote(self, inst: Instrument) -> Quote | None: ...

    @abstractmethod
    def bars(self, inst: Instrument, timeframe: str, lookback_days: int) -> pd.DataFrame: ...

    @abstractmethod
    def place_bracket(self, br: BracketRequest, inst: Instrument) -> list[OrderState]: ...

    @abstractmethod
    def place_order(self, req: OrderRequest, inst: Instrument) -> OrderState: ...

    @abstractmethod
    def cancel_order(self, client_order_id: str) -> None: ...

    @abstractmethod
    def modify_quantity(self, client_order_id: str, quantity: int) -> None: ...

    @abstractmethod
    def fills_since(self, since: datetime) -> list[Fill]: ...

    def fx_rate(self, from_ccy: str, to_ccy: str) -> float:
        if from_ccy == to_ccy:
            return 1.0
        raise BrokerError(f"no FX rate {from_ccy}->{to_ccy}")
