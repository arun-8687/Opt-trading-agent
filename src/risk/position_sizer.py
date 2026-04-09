"""Position sizing calculator.

Determines the number of lots to trade based on risk per trade,
stop loss distance, account capital, and current VIX level.
"""

from src.utils.logger import get_logger

logger = get_logger("position_sizer")


class PositionSizer:
    """Calculates position size based on risk parameters."""

    def __init__(
        self,
        capital: float = 100000,
        risk_per_trade_pct: float = 2.0,
        max_lots_per_trade: int = 10,
        vix_threshold: float = 25.0,
        vix_reduction_factor: float = 0.5,
    ):
        self.capital = capital
        self.risk_per_trade_pct = risk_per_trade_pct
        self.max_lots_per_trade = max_lots_per_trade
        self.vix_threshold = vix_threshold
        self.vix_reduction_factor = vix_reduction_factor

    def calculate(
        self,
        entry_price: float,
        stop_loss: float,
        lot_size: int,
        current_vix: float = 0.0,
    ) -> int:
        """Calculate number of shares/units to trade.

        Formula:
            risk_amount = capital * risk_per_trade_pct
            sl_distance = entry_price - stop_loss
            lots = risk_amount / (sl_distance * lot_size)

        Args:
            entry_price: Expected entry price
            stop_loss: Stop loss price
            lot_size: Contract lot size
            current_vix: Current India VIX value

        Returns:
            Total quantity (lots * lot_size)
        """
        if entry_price <= 0 or stop_loss <= 0 or lot_size <= 0:
            return lot_size  # Minimum 1 lot

        sl_distance = entry_price - stop_loss
        if sl_distance <= 0:
            logger.warning("SL distance is zero or negative")
            return lot_size

        risk_amount = self.capital * (self.risk_per_trade_pct / 100)

        # VIX adjustment
        if current_vix > self.vix_threshold:
            risk_amount *= self.vix_reduction_factor
            logger.info(
                f"VIX={current_vix:.1f} > {self.vix_threshold}: "
                f"risk reduced to {risk_amount:.0f}"
            )

        risk_per_lot = sl_distance * lot_size
        lots = int(risk_amount / risk_per_lot)

        # Enforce limits
        lots = max(1, min(lots, self.max_lots_per_trade))
        quantity = lots * lot_size

        logger.info(
            f"Position size: {lots} lots ({quantity} qty) | "
            f"Risk/lot={risk_per_lot:.0f} | Total risk={lots * risk_per_lot:.0f}"
        )

        return quantity

    def update_capital(self, new_capital: float):
        """Update capital (e.g., after daily P&L settlement)."""
        self.capital = new_capital
