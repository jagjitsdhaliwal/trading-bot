"""Tradovate REST/WebSocket adapter (demo/simulated account by default).

Tradovate is the API most futures prop firms (Topstep, Apex, MyFundedFutures)
run on under the hood, so code written against this adapter carries over to a
funded account later with just a config/credential change.

Docs: https://api.tradovate.com/
Auth: https://api.tradovate.com/#operation/api-v1-auth-accessTokenRequest-post

This is a stub with the real auth/order flow wired up but not battle-tested
against a live account yet -- verify against the demo environment before
relying on it.
"""

from __future__ import annotations

from datetime import datetime

import httpx

from engine.broker.base import (
    BrokerAdapter,
    Bar,
    OrderResult,
    OrderSide,
    OrderType,
    Position,
    Quote,
)

DEMO_URL = "https://demo.tradovateapi.com/v1"
LIVE_URL = "https://live.tradovateapi.com/v1"
MD_DEMO_URL = "https://md.tradovateapi.com/v1"


class TradovateAdapter(BrokerAdapter):
    def __init__(
        self,
        username: str,
        password: str,
        app_id: str,
        app_version: str,
        cid: str,
        sec: str,
        demo: bool = True,
    ) -> None:
        self.username = username
        self.password = password
        self.app_id = app_id
        self.app_version = app_version
        self.cid = cid
        self.sec = sec
        self.base_url = DEMO_URL if demo else LIVE_URL
        self._access_token: str | None = None
        self._client: httpx.Client | None = None

    def connect(self) -> None:
        self._client = httpx.Client(base_url=self.base_url, timeout=10.0)
        resp = self._client.post(
            "/auth/accesstokenrequest",
            json={
                "name": self.username,
                "password": self.password,
                "appId": self.app_id,
                "appVersion": self.app_version,
                "cid": self.cid,
                "sec": self.sec,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        self._access_token = data["accessToken"]
        self._client.headers["Authorization"] = f"Bearer {self._access_token}"

    def disconnect(self) -> None:
        if self._client:
            self._client.close()
            self._client = None

    def get_bars(
        self, symbol: str, timeframe: str, start: datetime, end: datetime
    ) -> list[Bar]:
        raise NotImplementedError(
            "Tradovate historical bars require the chart/history endpoint "
            "(md.tradovateapi.com) -- wire up once you've confirmed your "
            "market data subscription tier. Use the yfinance data source "
            "for backtesting in the meantime."
        )

    def get_quote(self, symbol: str) -> Quote:
        raise NotImplementedError("Wire up md.tradovateapi.com/v1/md/getQuote")

    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
        stop_price: float | None = None,
    ) -> OrderResult:
        assert self._client is not None, "call connect() first"
        payload = {
            "accountSpec": self.username,
            "symbol": symbol,
            "action": side.value,
            "orderQty": quantity,
            "orderType": order_type.value.capitalize(),
        }
        if limit_price is not None:
            payload["price"] = limit_price
        if stop_price is not None:
            payload["stopPrice"] = stop_price

        resp = self._client.post("/order/placeorder", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return OrderResult(
            order_id=str(data.get("orderId", "")),
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            status=data.get("status", "unknown"),
        )

    def cancel_order(self, order_id: str) -> bool:
        assert self._client is not None, "call connect() first"
        resp = self._client.post("/order/cancelorder", json={"orderId": order_id})
        return resp.status_code == 200

    def get_positions(self) -> list[Position]:
        assert self._client is not None, "call connect() first"
        resp = self._client.get("/position/list")
        resp.raise_for_status()
        return [
            Position(
                symbol=str(p.get("contractId", "")),
                quantity=p.get("netPos", 0),
                avg_price=p.get("netPrice", 0.0),
                unrealized_pnl=0.0,
            )
            for p in resp.json()
        ]

    def get_account_balance(self) -> float:
        assert self._client is not None, "call connect() first"
        resp = self._client.get("/cashBalance/list")
        resp.raise_for_status()
        balances = resp.json()
        return sum(b.get("amount", 0.0) for b in balances)
