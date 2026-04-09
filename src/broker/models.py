"""Data models for the trading system."""

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Optional


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOPLOSS_LIMIT = "STOPLOSS_LIMIT"
    STOPLOSS_MARKET = "STOPLOSS_MARKET"


class OrderStatus(Enum):
    PENDING = "pending"
    OPEN = "open"
    COMPLETE = "complete"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


class ProductType(Enum):
    INTRADAY = "INTRADAY"
    DELIVERY = "DELIVERY"
    CARRYFORWARD = "CARRYFORWARD"


class OptionType(Enum):
    CE = "CE"
    PE = "PE"


class Direction(Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


@dataclass
class Candle:
    """OHLCV candle data."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    oi: int = 0  # Open Interest (for options/futures)

    @property
    def is_green(self) -> bool:
        return self.close >= self.open

    @property
    def body_size(self) -> float:
        return abs(self.close - self.open)

    @property
    def range_size(self) -> float:
        return self.high - self.low


@dataclass
class Quote:
    """Real-time quote data."""

    symbol: str
    token: str
    ltp: float
    open: float
    high: float
    low: float
    close: float  # Previous close
    volume: int
    oi: int = 0
    bid: float = 0.0
    ask: float = 0.0
    bid_qty: int = 0
    ask_qty: int = 0
    timestamp: Optional[datetime] = None

    @property
    def change(self) -> float:
        if self.close == 0:
            return 0.0
        return self.ltp - self.close

    @property
    def change_pct(self) -> float:
        if self.close == 0:
            return 0.0
        return ((self.ltp - self.close) / self.close) * 100

    @property
    def bid_ask_spread(self) -> float:
        if self.bid == 0:
            return 0.0
        return self.ask - self.bid

    @property
    def bid_ask_spread_pct(self) -> float:
        if self.ltp == 0:
            return 0.0
        return (self.bid_ask_spread / self.ltp) * 100


@dataclass
class OptionContract:
    """Represents a specific option contract."""

    symbol: str           # Underlying symbol (e.g., NIFTY)
    token: str            # Broker-specific token
    trading_symbol: str   # Full trading symbol
    strike: int
    option_type: OptionType
    expiry: date
    lot_size: int
    exchange: str = "NFO"
    ltp: float = 0.0
    oi: int = 0
    volume: int = 0
    iv: float = 0.0
    bid: float = 0.0
    ask: float = 0.0

    @property
    def display_name(self) -> str:
        exp_str = self.expiry.strftime("%d%b").upper()
        return f"{self.symbol} {exp_str} {self.strike} {self.option_type.value}"


@dataclass
class OptionChainRow:
    """Single row of an options chain (one strike)."""

    strike: int
    ce_ltp: float = 0.0
    ce_oi: int = 0
    ce_oi_change: int = 0
    ce_volume: int = 0
    ce_iv: float = 0.0
    ce_bid: float = 0.0
    ce_ask: float = 0.0
    pe_ltp: float = 0.0
    pe_oi: int = 0
    pe_oi_change: int = 0
    pe_volume: int = 0
    pe_iv: float = 0.0
    pe_bid: float = 0.0
    pe_ask: float = 0.0


@dataclass
class Order:
    """Trade order to be placed with the broker."""

    symbol: str
    trading_symbol: str
    token: str
    exchange: str
    side: OrderSide
    order_type: OrderType
    product: ProductType
    quantity: int
    price: float = 0.0
    trigger_price: float = 0.0
    order_id: str = ""
    status: OrderStatus = OrderStatus.PENDING
    filled_price: float = 0.0
    filled_quantity: int = 0
    timestamp: Optional[datetime] = None
    tag: str = ""  # Strategy tag for identification
    parent_order_id: str = ""  # For bracket/cover orders

    @property
    def is_complete(self) -> bool:
        return self.status == OrderStatus.COMPLETE

    @property
    def is_pending(self) -> bool:
        return self.status in (OrderStatus.PENDING, OrderStatus.OPEN)


@dataclass
class Position:
    """An open trading position."""

    symbol: str
    trading_symbol: str
    token: str
    exchange: str
    option_type: OptionType
    strike: int
    expiry: date
    side: OrderSide
    quantity: int
    lot_size: int
    entry_price: float
    entry_time: datetime
    current_price: float = 0.0
    high_price: float = 0.0  # Highest price since entry (for trailing SL)
    stop_loss: float = 0.0
    target: float = 0.0
    strategy: str = ""
    signal_score: float = 0.0
    order_id: str = ""
    sl_order_id: str = ""
    exit_price: float = 0.0
    exit_time: Optional[datetime] = None
    exit_reason: str = ""

    @property
    def is_open(self) -> bool:
        return self.exit_time is None

    @property
    def pnl(self) -> float:
        price = self.exit_price if self.exit_price > 0 else self.current_price
        if self.side == OrderSide.BUY:
            return (price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - price) * self.quantity

    @property
    def pnl_pct(self) -> float:
        if self.entry_price == 0:
            return 0.0
        price = self.exit_price if self.exit_price > 0 else self.current_price
        if self.side == OrderSide.BUY:
            return ((price - self.entry_price) / self.entry_price) * 100
        else:
            return ((self.entry_price - price) / self.entry_price) * 100

    @property
    def lots(self) -> int:
        if self.lot_size == 0:
            return 0
        return self.quantity // self.lot_size

    @property
    def holding_duration_minutes(self) -> float:
        end = self.exit_time or datetime.now()
        return (end - self.entry_time).total_seconds() / 60


@dataclass
class DailyPnL:
    """Daily profit and loss summary."""

    date: date
    total_pnl: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    max_drawdown: float = 0.0
    capital: float = 0.0

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return (self.winning_trades / self.total_trades) * 100

    @property
    def pnl_pct(self) -> float:
        if self.capital == 0:
            return 0.0
        return (self.total_pnl / self.capital) * 100
