"""Interactive Brokers adapter stub.

Requires TWS or IB Gateway running locally and logged in (desktop app, not
pure headless-friendly). Install `ib_insync` or `ibapi` if you go this route:

    uv add ib-insync

This is left as a stub -- Tradovate is the recommended starting point since
it's REST/WS based with no desktop app dependency and maps directly onto the
prop-firm evaluation stack. Fill this in only if you specifically need IBKR's
broader asset-class coverage.
"""

from __future__ import annotations

from datetime import datetime

from engine.broker.base import (
    BrokerAdapter,
    Bar,
    OrderResult,
    OrderSide,
    OrderType,
    Position,
    Quote,
)


class IBKRAdapter(BrokerAdapter):
    def __init__(self, host: str = "127.0.0.1", port: int = 7497, client_id: int = 1) -> None:
        self.host = host
        self.port = port
        self.client_id = client_id

    def connect(self) -> None:
        raise NotImplementedError(
            "Install ib-insync (`uv add ib-insync`) and implement connection "
            "to a running TWS/IB Gateway instance on this host:port."
        )

    def disconnect(self) -> None:
        raise NotImplementedError

    def get_bars(
        self, symbol: str, timeframe: str, start: datetime, end: datetime
    ) -> list[Bar]:
        raise NotImplementedError

    def get_quote(self, symbol: str) -> Quote:
        raise NotImplementedError

    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
        stop_price: float | None = None,
    ) -> OrderResult:
        raise NotImplementedError

    def cancel_order(self, order_id: str) -> bool:
        raise NotImplementedError

    def get_positions(self) -> list[Position]:
        raise NotImplementedError

    def get_account_balance(self) -> float:
        raise NotImplementedError
