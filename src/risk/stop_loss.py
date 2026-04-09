"""Stop loss management — initial, trailing, time-based, and underlying-based."""

from dataclasses import dataclass
from datetime import datetime, time

from src.broker.models import Position
from src.utils.logger import get_logger

logger = get_logger("stop_loss")


@dataclass
class StopLossUpdate:
    """Result of stop loss evaluation."""

    should_exit: bool
    new_sl: float
    reason: str = ""


class StopLossManager:
    """Manages all stop loss logic for open positions."""

    def __init__(
        self,
        initial_sl_pct: float = 30.0,
        trailing_activation_pct: float = 20.0,
        trailing_lock_pct: float = 50.0,
        time_exit_minutes: int = 60,
        hard_cutoff: time = time(15, 0),
        underlying_sl_pct: float = 1.0,
    ):
        self.initial_sl_pct = initial_sl_pct
        self.trailing_activation_pct = trailing_activation_pct
        self.trailing_lock_pct = trailing_lock_pct
        self.time_exit_minutes = time_exit_minutes
        self.hard_cutoff = hard_cutoff
        self.underlying_sl_pct = underlying_sl_pct

    def calculate_initial_sl(self, entry_price: float) -> float:
        """Calculate initial stop loss price."""
        sl = entry_price * (1 - self.initial_sl_pct / 100)
        return round(max(sl, 0.05), 2)

    def evaluate(
        self,
        position: Position,
        current_price: float,
        spot_price: float,
        entry_spot_price: float,
        current_time: datetime,
    ) -> StopLossUpdate:
        """Evaluate all stop loss conditions for a position.

        Returns StopLossUpdate with exit decision and updated SL level.
        """
        # 1. Hard cutoff time
        if current_time.time() >= self.hard_cutoff:
            return StopLossUpdate(
                should_exit=True,
                new_sl=position.stop_loss,
                reason="HARD_CUTOFF",
            )

        # 2. Premium-based SL hit
        if current_price <= position.stop_loss:
            return StopLossUpdate(
                should_exit=True,
                new_sl=position.stop_loss,
                reason=f"SL_HIT at {current_price:.2f} (SL={position.stop_loss:.2f})",
            )

        # 3. Underlying-based SL
        if entry_spot_price > 0 and self.underlying_sl_pct > 0:
            spot_change_pct = abs(
                (spot_price - entry_spot_price) / entry_spot_price
            ) * 100

            # Check if underlying moved against the position
            from src.broker.models import OptionType

            if position.option_type == OptionType.CE and spot_price < entry_spot_price:
                if spot_change_pct >= self.underlying_sl_pct:
                    return StopLossUpdate(
                        should_exit=True,
                        new_sl=position.stop_loss,
                        reason=f"UNDERLYING_SL: spot down {spot_change_pct:.1f}%",
                    )
            elif position.option_type == OptionType.PE and spot_price > entry_spot_price:
                if spot_change_pct >= self.underlying_sl_pct:
                    return StopLossUpdate(
                        should_exit=True,
                        new_sl=position.stop_loss,
                        reason=f"UNDERLYING_SL: spot up {spot_change_pct:.1f}%",
                    )

        # 4. Time-based exit (losing position held too long)
        holding_mins = position.holding_duration_minutes
        if holding_mins > self.time_exit_minutes:
            pnl_pct = ((current_price - position.entry_price) / position.entry_price) * 100
            if pnl_pct < 0:
                return StopLossUpdate(
                    should_exit=True,
                    new_sl=position.stop_loss,
                    reason=f"TIME_EXIT: {holding_mins:.0f}min, P&L={pnl_pct:.1f}%",
                )

        # 5. Update trailing stop loss
        new_sl = position.stop_loss

        # Track highest price since entry
        if current_price > position.high_price:
            position.high_price = current_price

        profit_from_entry_pct = (
            (position.high_price - position.entry_price) / position.entry_price
        ) * 100

        if profit_from_entry_pct >= self.trailing_activation_pct:
            trail_sl = position.entry_price + (
                (position.high_price - position.entry_price)
                * (self.trailing_lock_pct / 100)
            )
            trail_sl = round(trail_sl, 2)

            if trail_sl > new_sl:
                new_sl = trail_sl
                logger.debug(
                    f"Trailing SL: {position.trading_symbol} "
                    f"SL={new_sl:.2f} (peak={position.high_price:.2f}, "
                    f"profit={profit_from_entry_pct:.1f}%)"
                )

        return StopLossUpdate(
            should_exit=False,
            new_sl=new_sl,
        )
