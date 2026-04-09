"""Position manager — tracks open positions, monitors exits, and records P&L."""

from datetime import datetime
from typing import Optional

from src.broker.models import (
    OptionType,
    OrderSide,
    Position,
)
from src.data.store import DataStore
from src.execution.order_manager import OrderManager
from src.risk.stop_loss import StopLossManager
from src.strategies.base import BaseStrategy, TradeSetup
from src.utils.logger import get_logger

logger = get_logger("position_manager")


class PositionManager:
    """Manages open positions, monitors SL/target, and handles exits."""

    def __init__(
        self,
        order_manager: OrderManager,
        sl_manager: StopLossManager,
        store: DataStore,
    ):
        self.order_manager = order_manager
        self.sl_manager = sl_manager
        self.store = store
        self._positions: list[Position] = []
        self._strategies: dict[str, BaseStrategy] = {}

    def register_strategy(self, strategy: BaseStrategy):
        """Register a strategy for exit management."""
        self._strategies[strategy.name] = strategy

    def open_position(self, setup: TradeSetup) -> Optional[Position]:
        """Open a new position by placing an entry order.

        Args:
            setup: Complete trade setup with entry/SL/target

        Returns:
            Position object if order was filled, None otherwise
        """
        order = self.order_manager.place_entry_order(setup)
        if not order or not order.is_complete:
            logger.error(f"Failed to open position for {setup.contract.display_name}")
            return None

        position = Position(
            symbol=setup.symbol,
            trading_symbol=setup.contract.trading_symbol,
            token=setup.contract.token,
            exchange=setup.contract.exchange,
            option_type=setup.contract.option_type,
            strike=setup.contract.strike,
            expiry=setup.contract.expiry,
            side=OrderSide.BUY,
            quantity=setup.quantity,
            lot_size=setup.contract.lot_size,
            entry_price=order.filled_price,
            entry_time=datetime.now(),
            current_price=order.filled_price,
            high_price=order.filled_price,
            stop_loss=setup.stop_loss,
            target=setup.target,
            strategy=setup.strategy_name,
            signal_score=setup.signal.score,
            order_id=order.order_id,
        )

        self._positions.append(position)

        # Save trade to database
        self.store.save_trade({
            "symbol": position.symbol,
            "trading_symbol": position.trading_symbol,
            "exchange": position.exchange,
            "option_type": position.option_type.value,
            "strike": position.strike,
            "expiry": position.expiry.isoformat(),
            "side": position.side.value,
            "quantity": position.quantity,
            "lot_size": position.lot_size,
            "entry_price": position.entry_price,
            "entry_time": position.entry_time.isoformat(),
            "stop_loss": position.stop_loss,
            "target": position.target,
            "strategy": position.strategy,
            "signal_score": position.signal_score,
            "order_id": position.order_id,
        })

        logger.info(
            f"POSITION OPENED: BUY {position.trading_symbol} "
            f"@ {position.entry_price} qty={position.quantity} "
            f"SL={position.stop_loss} Target={position.target} "
            f"Strategy={position.strategy}"
        )

        return position

    def update_positions(
        self,
        price_map: dict[str, float],
        spot_prices: dict[str, float],
        entry_spot_prices: dict[str, float],
    ):
        """Update all positions with current prices and check exit conditions.

        Called on every tick or at regular intervals.

        Args:
            price_map: {token: current_ltp} for all position tokens
            spot_prices: {symbol: current_spot_price} for underlyings
            entry_spot_prices: {symbol: entry_spot_price} at time of entry
        """
        now = datetime.now()

        for position in self._positions:
            if not position.is_open:
                continue

            # Update current price
            current_price = price_map.get(position.token, position.current_price)
            position.current_price = current_price

            spot = spot_prices.get(position.symbol, 0)
            entry_spot = entry_spot_prices.get(position.symbol, 0)

            # Check strategy-specific exit
            strategy = self._strategies.get(position.strategy)
            if strategy:
                exit_signal = strategy.should_exit(
                    position, current_price, spot, now
                )
                if exit_signal.should_exit:
                    self._close_position(position, current_price, exit_signal.reason)
                    continue

            # Check stop loss manager
            sl_update = self.sl_manager.evaluate(
                position, current_price, spot, entry_spot, now
            )

            if sl_update.should_exit:
                self._close_position(position, current_price, sl_update.reason)
            elif sl_update.new_sl > position.stop_loss:
                position.stop_loss = sl_update.new_sl

    def _close_position(self, position: Position, exit_price: float, reason: str):
        """Close a position by placing an exit order."""
        order = self.order_manager.place_exit_order(
            trading_symbol=position.trading_symbol,
            token=position.token,
            exchange=position.exchange,
            quantity=position.quantity,
            price=exit_price,
            reason=reason,
        )

        actual_exit_price = exit_price
        if order and order.is_complete:
            actual_exit_price = order.filled_price

        position.exit_price = actual_exit_price
        position.exit_time = datetime.now()
        position.exit_reason = reason

        # Update trade in database
        trade_id = self._find_trade_id(position.order_id)
        if trade_id:
            self.store.update_trade_exit(
                trade_id=trade_id,
                exit_price=position.exit_price,
                exit_time=position.exit_time.isoformat(),
                exit_reason=reason,
                pnl=position.pnl,
                pnl_pct=position.pnl_pct,
            )

        logger.info(
            f"POSITION CLOSED: {position.trading_symbol} "
            f"@ {actual_exit_price} reason={reason} "
            f"P&L={position.pnl:.0f} ({position.pnl_pct:.1f}%)"
        )

    def _find_trade_id(self, order_id: str) -> Optional[int]:
        """Find trade ID by order ID from database."""
        trades = self.store.get_todays_trades()
        if trades.empty:
            return None
        match = trades[trades["order_id"] == order_id]
        if not match.empty:
            return int(match.iloc[0]["id"])
        return None

    def get_open_positions(self) -> list[Position]:
        """Get all currently open positions."""
        return [p for p in self._positions if p.is_open]

    def get_closed_positions(self) -> list[Position]:
        """Get all closed positions for the session."""
        return [p for p in self._positions if not p.is_open]

    def get_total_pnl(self) -> float:
        """Get total P&L (realized + unrealized)."""
        return sum(p.pnl for p in self._positions)

    def get_realized_pnl(self) -> float:
        """Get realized P&L from closed positions."""
        return sum(p.pnl for p in self._positions if not p.is_open)

    def get_unrealized_pnl(self) -> float:
        """Get unrealized P&L from open positions."""
        return sum(p.pnl for p in self._positions if p.is_open)

    def close_all_positions(self, reason: str = "MANUAL_CLOSE"):
        """Emergency: close all open positions."""
        open_positions = self.get_open_positions()
        for position in open_positions:
            self._close_position(
                position, position.current_price, reason
            )
        logger.warning(f"ALL POSITIONS CLOSED: {len(open_positions)} positions, reason={reason}")

    def get_position_count_by_symbol(self, symbol: str) -> int:
        """Count open positions for a specific symbol."""
        return sum(
            1 for p in self._positions
            if p.is_open and p.symbol == symbol
        )
