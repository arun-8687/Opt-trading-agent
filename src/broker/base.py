"""Abstract broker interface for trading operations."""

from abc import ABC, abstractmethod
from datetime import date, datetime
from typing import Callable, Optional

import pandas as pd

from src.broker.models import Candle, OptionChainRow, Order, Position, Quote


class BaseBroker(ABC):
    """Abstract base class for broker implementations.

    All broker integrations must implement this interface to ensure
    the trading system remains broker-agnostic at the strategy level.
    """

    @abstractmethod
    def login(self) -> bool:
        """Authenticate with the broker. Returns True on success."""
        ...

    @abstractmethod
    def logout(self) -> None:
        """Logout and clean up resources."""
        ...

    @abstractmethod
    def get_ltp(self, exchange: str, trading_symbol: str, token: str) -> float:
        """Get last traded price for a symbol."""
        ...

    @abstractmethod
    def get_quote(self, exchange: str, trading_symbol: str, token: str) -> Quote:
        """Get full quote with bid/ask, volume, OI."""
        ...

    @abstractmethod
    def get_option_chain(
        self,
        symbol: str,
        expiry: date,
        exchange: str = "NFO",
    ) -> list[OptionChainRow]:
        """Get full options chain for a symbol and expiry."""
        ...

    @abstractmethod
    def get_historical_candles(
        self,
        exchange: str,
        trading_symbol: str,
        token: str,
        interval: str,
        from_date: datetime,
        to_date: datetime,
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data.

        Returns DataFrame with columns: timestamp, open, high, low, close, volume
        """
        ...

    @abstractmethod
    def place_order(self, order: Order) -> str:
        """Place an order. Returns order_id on success."""
        ...

    @abstractmethod
    def modify_order(
        self,
        order_id: str,
        price: Optional[float] = None,
        quantity: Optional[int] = None,
        trigger_price: Optional[float] = None,
        order_type: Optional[str] = None,
    ) -> bool:
        """Modify an existing order. Returns True on success."""
        ...

    @abstractmethod
    def cancel_order(self, order_id: str, variety: str = "NORMAL") -> bool:
        """Cancel an order. Returns True on success."""
        ...

    @abstractmethod
    def get_order_book(self) -> list[dict]:
        """Get all orders for the day."""
        ...

    @abstractmethod
    def get_positions(self) -> list[dict]:
        """Get all open positions."""
        ...

    @abstractmethod
    def get_holdings(self) -> list[dict]:
        """Get portfolio holdings."""
        ...

    @abstractmethod
    def get_margins(self) -> dict:
        """Get account margin/balance details."""
        ...

    @abstractmethod
    def lookup_token(self, exchange: str, trading_symbol: str) -> str:
        """Look up the broker token for a trading symbol."""
        ...

    @abstractmethod
    def start_websocket(
        self,
        tokens: list[str],
        on_tick: Callable,
        on_connect: Optional[Callable] = None,
        on_disconnect: Optional[Callable] = None,
    ) -> None:
        """Start WebSocket connection for real-time data."""
        ...

    @abstractmethod
    def stop_websocket(self) -> None:
        """Stop WebSocket connection."""
        ...

    @abstractmethod
    def get_instrument_list(self) -> pd.DataFrame:
        """Download and return the instrument master list."""
        ...
