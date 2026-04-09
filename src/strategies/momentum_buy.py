"""Momentum-based option buying strategy.

Entry: Multi-factor score > threshold with volume confirmation.
Options: 1-2 strikes OTM, current/next week expiry for index, monthly for stocks.
Target: 30-50% of premium. Stop Loss: 30% of premium.
Trailing SL: Activated at 20% profit, locks 50% of peak.
"""

from datetime import date, datetime, time, timedelta
from typing import Optional

from src.broker.models import OptionContract, Position
from src.signals.models import Signal, SignalStrength
from src.strategies.base import BaseStrategy, ExitSignal, TradeSetup
from src.utils.constants import INDEX_LOT_SIZES
from src.utils.helpers import (
    days_to_expiry,
    get_monthly_expiry,
    get_next_expiry,
    is_expiry_day,
)
from src.utils.logger import get_logger

logger = get_logger("momentum_buy")


class MomentumBuyStrategy(BaseStrategy):
    """Buy CE/PE based on momentum signals with strict risk management."""

    name = "momentum_buy"

    def __init__(
        self,
        min_score: float = 75.0,
        target_pct: float = 40.0,
        stop_loss_pct: float = 30.0,
        trailing_activation_pct: float = 20.0,
        trailing_lock_pct: float = 50.0,
        time_exit_minutes: int = 60,
        hard_cutoff: time = time(15, 0),
        min_premium: float = 5.0,
        max_premium: float = 500.0,
    ):
        self.min_score = min_score
        self.target_pct = target_pct
        self.stop_loss_pct = stop_loss_pct
        self.trailing_activation_pct = trailing_activation_pct
        self.trailing_lock_pct = trailing_lock_pct
        self.time_exit_minutes = time_exit_minutes
        self.hard_cutoff = hard_cutoff
        self.min_premium = min_premium
        self.max_premium = max_premium

    def should_enter(
        self,
        signal: Signal,
        spot_price: float,
        current_time: datetime,
    ) -> bool:
        """Check if signal meets momentum entry criteria."""
        if not signal.is_actionable:
            return False

        if signal.score < self.min_score:
            return False

        if signal.agreement_count < 3:
            return False

        # Don't enter too close to market close
        if current_time.time() >= self.hard_cutoff:
            return False

        # Require technical alignment
        if signal.technical_direction != signal.direction:
            return False

        return True

    def create_setup(
        self,
        signal: Signal,
        contract: OptionContract,
        spot_price: float,
        capital: float,
    ) -> Optional[TradeSetup]:
        """Create trade setup with entry, SL, and target."""
        entry_price = contract.ltp

        if entry_price <= 0:
            return None

        # Validate premium range
        if entry_price < self.min_premium or entry_price > self.max_premium:
            logger.info(
                f"Skipping {contract.display_name}: premium {entry_price} "
                f"outside range [{self.min_premium}, {self.max_premium}]"
            )
            return None

        # Calculate SL and target
        sl_points = entry_price * (self.stop_loss_pct / 100)
        stop_loss = round(entry_price - sl_points, 2)
        stop_loss = max(stop_loss, 0.05)  # Never go below 0.05

        target_points = entry_price * (self.target_pct / 100)
        target = round(entry_price + target_points, 2)

        # Calculate quantity based on risk
        risk_per_lot = sl_points * contract.lot_size
        if risk_per_lot <= 0:
            return None

        risk_amount = capital * 0.02  # 2% risk per trade
        lots = max(1, int(risk_amount / risk_per_lot))
        quantity = lots * contract.lot_size

        setup = TradeSetup(
            symbol=signal.symbol,
            signal=signal,
            contract=contract,
            entry_price=entry_price,
            stop_loss=stop_loss,
            target=target,
            quantity=quantity,
            strategy_name=self.name,
            reason=(
                f"Momentum {signal.direction.value} score={signal.score:.0f} "
                f"agreement={signal.agreement_count}/4"
            ),
        )

        logger.info(
            f"SETUP: BUY {contract.display_name} @ {entry_price} "
            f"SL={stop_loss} Target={target} Qty={quantity} "
            f"R:R={setup.risk_reward_ratio:.1f}"
        )

        return setup

    def should_exit(
        self,
        position: Position,
        current_price: float,
        spot_price: float,
        current_time: datetime,
    ) -> ExitSignal:
        """Check exit conditions for an open position."""
        # 1. Hard cutoff time
        if current_time.time() >= self.hard_cutoff:
            return ExitSignal(
                should_exit=True,
                reason="HARD_CUTOFF",
                exit_price=current_price,
            )

        # 2. Stop loss hit
        if current_price <= position.stop_loss:
            return ExitSignal(
                should_exit=True,
                reason="SL_HIT",
                exit_price=current_price,
            )

        # 3. Target hit
        if current_price >= position.target:
            return ExitSignal(
                should_exit=True,
                reason="TARGET_HIT",
                exit_price=current_price,
            )

        # 4. Time-based exit (holding too long in a losing position)
        holding_mins = position.holding_duration_minutes
        if holding_mins > self.time_exit_minutes:
            pnl_pct = ((current_price - position.entry_price) / position.entry_price) * 100
            if pnl_pct < 0:
                return ExitSignal(
                    should_exit=True,
                    reason=f"TIME_EXIT ({holding_mins:.0f}min, P&L={pnl_pct:.1f}%)",
                    exit_price=current_price,
                )

        # 5. Update trailing stop loss
        if current_price > position.high_price:
            position.high_price = current_price

        profit_pct = ((position.high_price - position.entry_price) / position.entry_price) * 100

        if profit_pct >= self.trailing_activation_pct:
            # Trail SL at 50% of peak profit
            trail_sl = position.entry_price + (
                (position.high_price - position.entry_price) * (self.trailing_lock_pct / 100)
            )
            if trail_sl > position.stop_loss:
                position.stop_loss = round(trail_sl, 2)
                logger.debug(
                    f"Trailing SL updated: {position.trading_symbol} "
                    f"SL={position.stop_loss:.2f} (peak={position.high_price:.2f})"
                )

        return ExitSignal(should_exit=False)

    def select_expiry(self, symbol: str, current_date: date) -> date:
        """Select expiry: current week if >2 DTE for index, monthly for stocks."""
        if symbol in INDEX_LOT_SIZES:
            next_exp = get_next_expiry(symbol, current_date)
            dte = days_to_expiry(next_exp, current_date)

            if dte >= 2:
                return next_exp
            else:
                # Too close to expiry, take next week
                return get_next_expiry(symbol, next_exp + timedelta(days=1))
        else:
            # Stock options: use monthly expiry
            return get_monthly_expiry(current_date)
