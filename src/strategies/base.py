"""Abstract base strategy interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

from src.broker.models import OptionContract, Position
from src.signals.models import Signal


@dataclass
class TradeSetup:
    """A complete trade setup ready for execution."""

    symbol: str
    signal: Signal
    contract: OptionContract
    entry_price: float
    stop_loss: float
    target: float
    quantity: int
    strategy_name: str
    reason: str

    @property
    def risk_per_lot(self) -> float:
        return (self.entry_price - self.stop_loss) * self.contract.lot_size

    @property
    def reward_per_lot(self) -> float:
        return (self.target - self.entry_price) * self.contract.lot_size

    @property
    def risk_reward_ratio(self) -> float:
        risk = self.entry_price - self.stop_loss
        if risk <= 0:
            return 0
        return (self.target - self.entry_price) / risk


@dataclass
class ExitSignal:
    """Signal to exit a position."""

    should_exit: bool
    reason: str = ""
    exit_price: float = 0.0


class BaseStrategy(ABC):
    """Abstract base for all trading strategies."""

    name: str = "base"
    enabled: bool = True

    @abstractmethod
    def should_enter(
        self,
        signal: Signal,
        spot_price: float,
        current_time: datetime,
    ) -> bool:
        """Determine if the strategy should enter based on the signal.

        Args:
            signal: Combined trading signal
            spot_price: Current underlying price
            current_time: Current timestamp
        """
        ...

    @abstractmethod
    def create_setup(
        self,
        signal: Signal,
        contract: OptionContract,
        spot_price: float,
        capital: float,
    ) -> Optional[TradeSetup]:
        """Create a complete trade setup with entry, SL, and target.

        Args:
            signal: Trading signal that triggered entry
            contract: Selected option contract
            spot_price: Current underlying price
            capital: Available trading capital
        """
        ...

    @abstractmethod
    def should_exit(
        self,
        position: Position,
        current_price: float,
        spot_price: float,
        current_time: datetime,
    ) -> ExitSignal:
        """Check if an existing position should be exited.

        Called on every tick/candle for open positions.
        """
        ...

    def select_expiry(self, symbol: str, current_date: date) -> date:
        """Select appropriate expiry for the strategy."""
        from src.utils.helpers import get_monthly_expiry, get_next_expiry
        return get_next_expiry(symbol, current_date)
