"""Pre-trade and portfolio-level risk management.

Enforces position limits, daily loss caps, liquidity checks,
market hour constraints, and VIX-based adjustments.
"""

from dataclasses import dataclass
from datetime import datetime, time
from typing import Optional

from src.broker.models import Position
from src.strategies.base import TradeSetup
from src.utils.helpers import is_trading_window
from src.utils.logger import get_logger

logger = get_logger("risk_manager")


@dataclass
class RiskCheckResult:
    """Result of a risk check."""

    passed: bool
    reason: str = ""


class RiskManager:
    """Manages all pre-trade and portfolio-level risk checks."""

    def __init__(
        self,
        capital: float = 100000,
        max_risk_per_trade_pct: float = 2.0,
        max_daily_loss_pct: float = 5.0,
        max_open_positions: int = 5,
        max_per_instrument: int = 2,
        trading_start: time = time(9, 20),
        trading_end: time = time(15, 0),
        min_option_oi: int = 5000,
        min_option_volume: int = 1000,
        max_bid_ask_spread_pct: float = 2.0,
        vix_threshold: float = 25.0,
        vix_size_reduction: float = 0.5,
    ):
        self.capital = capital
        self.max_risk_per_trade_pct = max_risk_per_trade_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_open_positions = max_open_positions
        self.max_per_instrument = max_per_instrument
        self.trading_start = trading_start
        self.trading_end = trading_end
        self.min_option_oi = min_option_oi
        self.min_option_volume = min_option_volume
        self.max_bid_ask_spread_pct = max_bid_ask_spread_pct
        self.vix_threshold = vix_threshold
        self.vix_size_reduction = vix_size_reduction

        self._daily_pnl: float = 0.0
        self._daily_trades: int = 0
        self._trading_halted: bool = False

    def check_all(
        self,
        setup: TradeSetup,
        open_positions: list[Position],
        current_vix: float = 0.0,
    ) -> RiskCheckResult:
        """Run all pre-trade risk checks.

        Returns RiskCheckResult indicating pass/fail with reason.
        """
        checks = [
            self._check_trading_halted(),
            self._check_market_hours(),
            self._check_daily_loss_limit(),
            self._check_risk_per_trade(setup),
            self._check_position_limits(open_positions),
            self._check_instrument_limit(setup.symbol, open_positions),
            self._check_liquidity(setup),
            self._check_vix(current_vix, setup),
        ]

        for check in checks:
            if not check.passed:
                logger.warning(
                    f"RISK REJECTED: {setup.contract.display_name} - {check.reason}"
                )
                return check

        logger.info(f"RISK APPROVED: {setup.contract.display_name}")
        return RiskCheckResult(passed=True)

    def _check_trading_halted(self) -> RiskCheckResult:
        """Check if trading has been halted for the day."""
        if self._trading_halted:
            return RiskCheckResult(False, "Trading halted for the day (daily loss limit hit)")
        return RiskCheckResult(True)

    def _check_market_hours(self) -> RiskCheckResult:
        """Check if within trading window."""
        if not is_trading_window(start=self.trading_start, end=self.trading_end):
            return RiskCheckResult(False, "Outside trading window")
        return RiskCheckResult(True)

    def _check_daily_loss_limit(self) -> RiskCheckResult:
        """Check if daily loss limit has been breached."""
        max_loss = self.capital * (self.max_daily_loss_pct / 100)
        if self._daily_pnl <= -max_loss:
            self._trading_halted = True
            return RiskCheckResult(
                False,
                f"Daily loss limit hit: {self._daily_pnl:.0f} "
                f"(max: -{max_loss:.0f})",
            )
        return RiskCheckResult(True)

    def _check_risk_per_trade(self, setup: TradeSetup) -> RiskCheckResult:
        """Check if risk per trade is within limits."""
        max_risk = self.capital * (self.max_risk_per_trade_pct / 100)
        trade_risk = (setup.entry_price - setup.stop_loss) * setup.quantity

        if trade_risk > max_risk:
            return RiskCheckResult(
                False,
                f"Risk per trade too high: {trade_risk:.0f} (max: {max_risk:.0f})",
            )
        return RiskCheckResult(True)

    def _check_position_limits(self, open_positions: list[Position]) -> RiskCheckResult:
        """Check if max open positions limit is reached."""
        open_count = sum(1 for p in open_positions if p.is_open)
        if open_count >= self.max_open_positions:
            return RiskCheckResult(
                False,
                f"Max positions reached: {open_count}/{self.max_open_positions}",
            )
        return RiskCheckResult(True)

    def _check_instrument_limit(
        self,
        symbol: str,
        open_positions: list[Position],
    ) -> RiskCheckResult:
        """Check positions per instrument limit."""
        instrument_count = sum(
            1 for p in open_positions
            if p.is_open and p.symbol == symbol
        )
        if instrument_count >= self.max_per_instrument:
            return RiskCheckResult(
                False,
                f"Max positions for {symbol}: {instrument_count}/{self.max_per_instrument}",
            )
        return RiskCheckResult(True)

    def _check_liquidity(self, setup: TradeSetup) -> RiskCheckResult:
        """Check option liquidity (OI, volume, bid-ask spread)."""
        contract = setup.contract

        if contract.oi > 0 and contract.oi < self.min_option_oi:
            return RiskCheckResult(
                False,
                f"Low OI: {contract.oi} (min: {self.min_option_oi})",
            )

        if contract.volume > 0 and contract.volume < self.min_option_volume:
            return RiskCheckResult(
                False,
                f"Low volume: {contract.volume} (min: {self.min_option_volume})",
            )

        if contract.bid > 0 and contract.ask > 0:
            spread_pct = ((contract.ask - contract.bid) / contract.ltp) * 100
            if spread_pct > self.max_bid_ask_spread_pct:
                return RiskCheckResult(
                    False,
                    f"Wide spread: {spread_pct:.1f}% (max: {self.max_bid_ask_spread_pct}%)",
                )

        return RiskCheckResult(True)

    def _check_vix(self, current_vix: float, setup: TradeSetup) -> RiskCheckResult:
        """Adjust for high VIX (volatility) environment."""
        if current_vix > 0 and current_vix > self.vix_threshold:
            # Don't reject, but flag for reduced sizing
            logger.warning(
                f"High VIX ({current_vix:.1f}): Position size should be reduced "
                f"by {self.vix_size_reduction * 100:.0f}%"
            )
        return RiskCheckResult(True)

    def update_daily_pnl(self, pnl_change: float):
        """Update running daily P&L."""
        self._daily_pnl += pnl_change
        logger.info(f"Daily P&L: {self._daily_pnl:.0f}")

    def reset_daily(self):
        """Reset daily counters (call at start of each trading day)."""
        self._daily_pnl = 0.0
        self._daily_trades = 0
        self._trading_halted = False
        logger.info("Daily risk counters reset")

    def get_adjusted_quantity(
        self,
        quantity: int,
        lot_size: int,
        current_vix: float = 0.0,
    ) -> int:
        """Adjust quantity based on VIX and remaining daily risk budget."""
        adjusted = quantity

        # VIX adjustment
        if current_vix > self.vix_threshold:
            adjusted = int(adjusted * self.vix_size_reduction)

        # Ensure at least 1 lot
        adjusted = max(adjusted, lot_size)

        # Round to lot size
        adjusted = (adjusted // lot_size) * lot_size

        return adjusted
