"""Multi-timeframe confluence option buying strategy.

Enters trades only when multiple timeframes (5min, 15min, and daily)
all agree on direction. This is the highest conviction strategy —
fewer trades but significantly better win rate.

Best for: Patient traders who want fewer but higher-quality trades.
"""

from datetime import date, datetime, time, timedelta
from typing import Optional

import pandas as pd

from src.broker.models import OptionContract, Position
from src.signals.models import Direction, Signal
from src.signals.technical import (
    TechnicalConfig,
    calculate_ema,
    calculate_rsi,
    calculate_supertrend,
)
from src.strategies.base import BaseStrategy, ExitSignal, TradeSetup
from src.utils.constants import INDEX_LOT_SIZES
from src.utils.helpers import get_monthly_expiry, get_next_expiry, days_to_expiry
from src.utils.logger import get_logger

logger = get_logger("multi_timeframe")


def get_timeframe_direction(df: pd.DataFrame) -> Direction:
    """Determine direction from a single timeframe's candle data.

    Uses EMA crossover + SuperTrend + RSI for direction assessment.
    """
    if df.empty or len(df) < 30:
        return Direction.NEUTRAL

    close = df["close"]
    bullish = 0
    bearish = 0

    # EMA 9/21 crossover
    ema_fast = calculate_ema(close, 9)
    ema_slow = calculate_ema(close, 21)
    if ema_fast.iloc[-1] > ema_slow.iloc[-1]:
        bullish += 1
    elif ema_fast.iloc[-1] < ema_slow.iloc[-1]:
        bearish += 1

    # RSI
    rsi = calculate_rsi(close, 14)
    if rsi.iloc[-1] > 55:
        bullish += 1
    elif rsi.iloc[-1] < 45:
        bearish += 1

    # SuperTrend
    st_dir = calculate_supertrend(df, 10, 3)
    if not st_dir.empty and pd.notna(st_dir.iloc[-1]):
        if st_dir.iloc[-1] == 1:
            bullish += 1
        elif st_dir.iloc[-1] == -1:
            bearish += 1

    if bullish >= 2:
        return Direction.BULLISH
    elif bearish >= 2:
        return Direction.BEARISH
    return Direction.NEUTRAL


class MultiTimeframeStrategy(BaseStrategy):
    """Buy options when 5min, 15min, and daily all agree on direction."""

    name = "multi_timeframe"

    def __init__(
        self,
        min_score: float = 70.0,
        target_pct: float = 50.0,
        stop_loss_pct: float = 30.0,
        trailing_activation_pct: float = 20.0,
        trailing_lock_pct: float = 50.0,
        time_exit_minutes: int = 120,
        hard_cutoff: time = time(15, 0),
    ):
        self.min_score = min_score
        self.target_pct = target_pct
        self.stop_loss_pct = stop_loss_pct
        self.trailing_activation_pct = trailing_activation_pct
        self.trailing_lock_pct = trailing_lock_pct
        self.time_exit_minutes = time_exit_minutes
        self.hard_cutoff = hard_cutoff

        # Cache MTF analysis results
        self._mtf_directions: dict[str, dict[str, Direction]] = {}

    def analyze_timeframes(
        self,
        symbol: str,
        df_5min: pd.DataFrame,
        df_15min: pd.DataFrame,
        df_daily: pd.DataFrame,
    ) -> dict[str, Direction]:
        """Analyze all three timeframes and cache the result."""
        directions = {
            "5min": get_timeframe_direction(df_5min),
            "15min": get_timeframe_direction(df_15min),
            "daily": get_timeframe_direction(df_daily),
        }
        self._mtf_directions[symbol] = directions
        return directions

    def should_enter(
        self,
        signal: Signal,
        spot_price: float,
        current_time: datetime,
    ) -> bool:
        """Enter only when all timeframes agree with signal direction.

        This is the strictest entry criterion — all three timeframes
        must be aligned. This dramatically reduces false signals.
        """
        if signal.score < self.min_score:
            return False

        if current_time.time() >= self.hard_cutoff:
            return False

        # Check MTF alignment
        mtf = self._mtf_directions.get(signal.symbol)
        if not mtf:
            return False

        target_dir = signal.direction
        if target_dir == Direction.NEUTRAL:
            return False

        all_agree = all(d == target_dir for d in mtf.values())
        if not all_agree:
            non_agree = [
                f"{tf}={d.value}" for tf, d in mtf.items() if d != target_dir
            ]
            logger.debug(
                f"MTF not aligned for {signal.symbol}: "
                f"signal={target_dir.value}, misaligned={non_agree}"
            )
            return False

        logger.info(
            f"MTF CONFLUENCE: {signal.symbol} ALL timeframes {target_dir.value} "
            f"| score={signal.score:.0f}"
        )
        return True

    def create_setup(
        self,
        signal: Signal,
        contract: OptionContract,
        spot_price: float,
        capital: float,
    ) -> Optional[TradeSetup]:
        """Create high-conviction MTF trade setup.

        Higher target since MTF confluence trades tend to have
        larger moves. Standard risk per trade.
        """
        entry_price = contract.ltp
        if entry_price <= 0:
            return None

        if entry_price < 5.0 or entry_price > 500.0:
            return None

        stop_loss = round(entry_price * (1 - self.stop_loss_pct / 100), 2)
        stop_loss = max(stop_loss, 0.05)
        target = round(entry_price * (1 + self.target_pct / 100), 2)

        risk_per_lot = (entry_price - stop_loss) * contract.lot_size
        if risk_per_lot <= 0:
            return None

        # Higher conviction = full 2% risk
        risk_amount = capital * 0.02
        lots = max(1, int(risk_amount / risk_per_lot))
        quantity = lots * contract.lot_size

        mtf = self._mtf_directions.get(signal.symbol, {})
        mtf_str = " ".join(f"{tf}={d.value}" for tf, d in mtf.items())

        return TradeSetup(
            symbol=signal.symbol,
            signal=signal,
            contract=contract,
            entry_price=entry_price,
            stop_loss=stop_loss,
            target=target,
            quantity=quantity,
            strategy_name=self.name,
            reason=f"MTF Confluence: {mtf_str} score={signal.score:.0f}",
        )

    def should_exit(
        self,
        position: Position,
        current_price: float,
        spot_price: float,
        current_time: datetime,
    ) -> ExitSignal:
        """Exit management — more patient with MTF trades."""
        if current_time.time() >= self.hard_cutoff:
            return ExitSignal(True, "HARD_CUTOFF", current_price)

        if current_price <= position.stop_loss:
            return ExitSignal(True, "SL_HIT", current_price)

        if current_price >= position.target:
            return ExitSignal(True, "TARGET_HIT", current_price)

        # Longer time allowance (MTF trades need time)
        holding_mins = position.holding_duration_minutes
        if holding_mins > self.time_exit_minutes:
            pnl_pct = ((current_price - position.entry_price) / position.entry_price) * 100
            if pnl_pct < -10:
                return ExitSignal(
                    True,
                    f"TIME_EXIT ({holding_mins:.0f}min, P&L={pnl_pct:.1f}%)",
                    current_price,
                )

        # Trailing SL
        if current_price > position.high_price:
            position.high_price = current_price

        profit_pct = ((position.high_price - position.entry_price) / position.entry_price) * 100
        if profit_pct >= self.trailing_activation_pct:
            trail_sl = position.entry_price + (
                (position.high_price - position.entry_price) * (self.trailing_lock_pct / 100)
            )
            if trail_sl > position.stop_loss:
                position.stop_loss = round(trail_sl, 2)

        return ExitSignal(False)

    def select_expiry(self, symbol: str, current_date: date) -> date:
        """Next week expiry — MTF trades may hold for hours."""
        if symbol in INDEX_LOT_SIZES:
            next_exp = get_next_expiry(symbol, current_date)
            dte = days_to_expiry(next_exp, current_date)
            if dte >= 3:
                return next_exp
            return get_next_expiry(symbol, next_exp + timedelta(days=1))
        return get_monthly_expiry(current_date)
