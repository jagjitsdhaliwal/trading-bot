"""Broker adapter interface. Every broker (Tradovate, IBKR, paper-sim) implements this
so strategy/risk/backtest code never depends on a specific broker's API shape."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


@dataclass
class Bar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class Quote:
    timestamp: datetime
    bid: float
    ask: float
    last: float


@dataclass
class OrderResult:
    order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType
    status: str
    filled_price: float | None = None
    filled_at: datetime | None = None


@dataclass
class Position:
    symbol: str
    quantity: int
    avg_price: float
    unrealized_pnl: float


class BrokerAdapter(ABC):
    """Common interface for market data + execution, regardless of backend."""

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def get_bars(
        self, symbol: str, timeframe: str, start: datetime, end: datetime
    ) -> list[Bar]:
        """Historical OHLCV bars. timeframe e.g. '1m', '5m', '1h', '1d'."""
        ...

    @abstractmethod
    def get_quote(self, symbol: str) -> Quote: ...

    @abstractmethod
    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
        stop_price: float | None = None,
    ) -> OrderResult: ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool: ...

    @abstractmethod
    def get_positions(self) -> list[Position]: ...

    @abstractmethod
    def get_account_balance(self) -> float: ...
