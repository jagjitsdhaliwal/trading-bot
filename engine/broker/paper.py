"""Simulated broker: fills orders instantly at the last known quote, tracks
positions/balance in memory. Wraps any BrokerAdapter's market data (e.g.
Tradovate demo feed or yfinance) so you can paper-trade without an account.

This is distinct from a broker's own "demo account" -- it's a local simulator
useful for dry-running the full pipeline (strategy -> risk -> execution ->
db) before you even have broker credentials.
"""

from __future__ import annotations

import uuid
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


class PaperBrokerAdapter(BrokerAdapter):
    def __init__(self, starting_balance: float = 50_000.0) -> None:
        self.balance = starting_balance
        self.starting_balance = starting_balance
        self._positions: dict[str, Position] = {}
        self._last_quotes: dict[str, Quote] = {}

    def connect(self) -> None:
        pass

    def disconnect(self) -> None:
        pass

    def get_bars(
        self, symbol: str, timeframe: str, start: datetime, end: datetime
    ) -> list[Bar]:
        raise NotImplementedError(
            "PaperBrokerAdapter has no market data of its own -- use "
            "engine.data.yfinance_source or engine.data.tradovate_source "
            "for bars, and feed quotes in via set_quote()."
        )

    def set_quote(self, symbol: str, quote: Quote) -> None:
        """Feed the simulator the latest price so it can fill orders."""
        self._last_quotes[symbol] = quote

    def get_quote(self, symbol: str) -> Quote:
        if symbol not in self._last_quotes:
            raise ValueError(f"No quote fed for {symbol} yet -- call set_quote() first")
        return self._last_quotes[symbol]

    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
        stop_price: float | None = None,
    ) -> OrderResult:
        quote = self.get_quote(symbol)
        fill_price = quote.ask if side == OrderSide.BUY else quote.bid

        pos = self._positions.get(symbol)
        signed_qty = quantity if side == OrderSide.BUY else -quantity

        if pos is None:
            self._positions[symbol] = Position(
                symbol=symbol,
                quantity=signed_qty,
                avg_price=fill_price,
                unrealized_pnl=0.0,
            )
        else:
            new_qty = pos.quantity + signed_qty
            if new_qty == 0:
                del self._positions[symbol]
            else:
                pos.quantity = new_qty
                pos.avg_price = fill_price
                self._positions[symbol] = pos

        return OrderResult(
            order_id=str(uuid.uuid4()),
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            status="FILLED",
            filled_price=fill_price,
            filled_at=quote.timestamp,
        )

    def cancel_order(self, order_id: str) -> bool:
        return False  # market orders fill instantly; nothing to cancel

    def get_positions(self) -> list[Position]:
        return list(self._positions.values())

    def get_account_balance(self) -> float:
        return self.balance
