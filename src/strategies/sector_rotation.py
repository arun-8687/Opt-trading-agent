"""Sector Rotation option buying strategy.

Buy options on stocks in the strongest (or weakest) sector relative to
NIFTY. When a sector outperforms the broad market, its constituent F&O
stocks get institutional flow. Buy CE on leaders in strong sectors,
PE on laggards in weak sectors.

Best for: Trending market days with clear sector leadership.
"""

from datetime import date, datetime, time, timedelta
from typing import Optional

from src.broker.models import OptionContract, Position
from src.signals.models import Direction, Signal
from src.strategies.base import BaseStrategy, ExitSignal, TradeSetup
from src.utils.constants import INDEX_LOT_SIZES
from src.utils.helpers import get_monthly_expiry, get_next_expiry, days_to_expiry
from src.utils.logger import get_logger

logger = get_logger("sector_rotation")

# NSE Sector Indices
SECTOR_INDICES = [
    "NIFTY IT", "NIFTY BANK", "NIFTY PHARMA", "NIFTY AUTO",
    "NIFTY METAL", "NIFTY FMCG", "NIFTY REALTY", "NIFTY ENERGY",
    "NIFTY INFRA", "NIFTY PSE", "NIFTY MEDIA",
]


class SectorRotationStrategy(BaseStrategy):
    """Buy options on stocks with strong sector relative strength."""

    name = "sector_rotation"

    def __init__(
        self,
        min_score: float = 72.0,
        target_pct: float = 40.0,
        stop_loss_pct: float = 28.0,
        min_sector_rs: float = 1.05,
        min_stock_rs: float = 1.03,
        trailing_activation_pct: float = 20.0,
        trailing_lock_pct: float = 50.0,
        time_exit_minutes: int = 90,
        hard_cutoff: time = time(15, 0),
        entry_start: time = time(10, 0),
        entry_end: time = time(14, 0),
    ):
        self.min_score = min_score
        self.target_pct = target_pct
        self.stop_loss_pct = stop_loss_pct
        self.min_sector_rs = min_sector_rs
        self.min_stock_rs = min_stock_rs
        self.trailing_activation_pct = trailing_activation_pct
        self.trailing_lock_pct = trailing_lock_pct
        self.time_exit_minutes = time_exit_minutes
        self.hard_cutoff = hard_cutoff
        self.entry_start = entry_start
        self.entry_end = entry_end

        # Sector data — keyed by symbol, set externally
        self._sector_data: dict[str, dict] = {}

    def set_sector_data(
        self,
        symbol: str,
        sector: str,
        sector_rs: float,
        stock_rs: float,
    ):
        """Set relative strength data for a stock.

        Args:
            symbol: Stock symbol (e.g., "TCS")
            sector: Sector name (e.g., "NIFTY IT")
            sector_rs: Sector RS vs NIFTY (>1.0 = outperforming)
            stock_rs: Stock RS vs sector (>1.0 = outperforming sector)
        """
        self._sector_data[symbol] = {
            "sector": sector,
            "sector_rs": sector_rs,
            "stock_rs": stock_rs,
        }

        if sector_rs >= self.min_sector_rs and stock_rs >= self.min_stock_rs:
            logger.info(
                f"Sector Leader: {symbol} ({sector}) "
                f"sector_rs={sector_rs:.2f} stock_rs={stock_rs:.2f}"
            )

    def should_enter(
        self,
        signal: Signal,
        spot_price: float,
        current_time: datetime,
    ) -> bool:
        """Enter on stocks with strong sector relative strength.

        Conditions:
        1. Stocks only (not indices)
        2. Sector data available for the symbol
        3. BULLISH: sector outperforming NIFTY + stock outperforming sector
        4. BEARISH: sector underperforming NIFTY + stock underperforming sector
        5. Signal direction matches RS direction
        """
        # Stocks only
        if signal.symbol in INDEX_LOT_SIZES:
            return False

        if signal.score < self.min_score:
            return False

        if current_time.time() < self.entry_start:
            return False

        if current_time.time() >= self.entry_end:
            return False

        data = self._sector_data.get(signal.symbol)
        if not data:
            return False

        sector_rs = data["sector_rs"]
        stock_rs = data["stock_rs"]

        # Bullish: buy leaders in strong sectors
        if signal.direction == Direction.BULLISH:
            if sector_rs < self.min_sector_rs:
                return False
            if stock_rs < self.min_stock_rs:
                return False

            logger.info(
                f"SECTOR ROTATION BUY: {signal.symbol} ({data['sector']}) "
                f"sector_rs={sector_rs:.2f} stock_rs={stock_rs:.2f} "
                f"score={signal.score:.0f}"
            )
            return True

        # Bearish: short laggards in weak sectors
        if signal.direction == Direction.BEARISH:
            bearish_sector_threshold = 1.0 / self.min_sector_rs
            bearish_stock_threshold = 1.0 / self.min_stock_rs
            if sector_rs > bearish_sector_threshold:
                return False
            if stock_rs > bearish_stock_threshold:
                return False

            logger.info(
                f"SECTOR ROTATION SHORT: {signal.symbol} ({data['sector']}) "
                f"sector_rs={sector_rs:.2f} stock_rs={stock_rs:.2f} "
                f"score={signal.score:.0f}"
            )
            return True

        return False

    def create_setup(
        self,
        signal: Signal,
        contract: OptionContract,
        spot_price: float,
        capital: float,
    ) -> Optional[TradeSetup]:
        """Create sector rotation trade setup."""
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

        risk_amount = capital * 0.02
        lots = max(1, int(risk_amount / risk_per_lot))
        quantity = lots * contract.lot_size

        data = self._sector_data.get(signal.symbol, {})

        return TradeSetup(
            symbol=signal.symbol,
            signal=signal,
            contract=contract,
            entry_price=entry_price,
            stop_loss=stop_loss,
            target=target,
            quantity=quantity,
            strategy_name=self.name,
            reason=(
                f"Sector Rotation {signal.direction.value} "
                f"{data.get('sector', '')} "
                f"sec_rs={data.get('sector_rs', 0):.2f} "
                f"stk_rs={data.get('stock_rs', 0):.2f} "
                f"score={signal.score:.0f}"
            ),
        )

    def should_exit(
        self,
        position: Position,
        current_price: float,
        spot_price: float,
        current_time: datetime,
    ) -> ExitSignal:
        """Standard exit — SL, target, trailing, time, hard cutoff."""
        if current_time.time() >= self.hard_cutoff:
            return ExitSignal(True, "HARD_CUTOFF", current_price)

        if current_price <= position.stop_loss:
            return ExitSignal(True, "SL_HIT", current_price)

        if current_price >= position.target:
            return ExitSignal(True, "TARGET_HIT", current_price)

        # Time-based exit
        holding_mins = position.holding_duration_minutes
        if holding_mins > self.time_exit_minutes:
            pnl_pct = ((current_price - position.entry_price) / position.entry_price) * 100
            if pnl_pct < 5:
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
        """Monthly expiry for stocks (sector rotation is swing-friendly)."""
        return get_monthly_expiry(current_date)
