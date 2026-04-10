"""NIFTY VIX Regime-Based option buying strategy.

Times option purchases based on India VIX level. Low VIX means cheap
options and imminent volatility spike. High VIX means expensive but
trending markets. Adjusts position size, targets, and expiry based
on the current VIX regime.

NIFTY-specific: India VIX is derived from NIFTY options. VIX below 12
historically precedes sharp NIFTY moves within 5-10 sessions.

Best for: All market conditions — adapts to VIX regime.
"""

from datetime import date, datetime, time, timedelta
from enum import Enum
from typing import Optional

from src.broker.models import OptionContract, Position
from src.signals.models import Direction, Signal
from src.strategies.base import BaseStrategy, ExitSignal, TradeSetup
from src.utils.helpers import get_monthly_expiry, get_next_expiry, days_to_expiry
from src.utils.logger import get_logger

logger = get_logger("nifty_vix_regime")


class VIXRegime(Enum):
    CHEAP = "CHEAP"           # VIX < 13 — options are cheap, buy aggressively
    NORMAL = "NORMAL"         # VIX 13-17 — standard buying
    EXPENSIVE = "EXPENSIVE"   # VIX 18-22 — only with catalyst
    CRUSH_ZONE = "CRUSH_ZONE" # VIX > 22 and falling — avoid buying


def classify_vix_regime(vix: float, vix_change: float = 0.0) -> VIXRegime:
    """Classify the current VIX level into a trading regime."""
    if vix > 22 and vix_change < 0:
        return VIXRegime.CRUSH_ZONE
    if vix >= 18:
        return VIXRegime.EXPENSIVE
    if vix >= 13:
        return VIXRegime.NORMAL
    return VIXRegime.CHEAP


class NiftyVIXRegimeStrategy(BaseStrategy):
    """Buy NIFTY options with sizing and targets adapted to VIX regime."""

    name = "nifty_vix_regime"

    def __init__(
        self,
        min_score: float = 72.0,
        hard_cutoff: time = time(15, 0),
        entry_start: time = time(9, 30),
        entry_end: time = time(14, 0),
    ):
        self.min_score = min_score
        self.hard_cutoff = hard_cutoff
        self.entry_start = entry_start
        self.entry_end = entry_end

        # VIX state — set externally
        self.current_vix: float = 0.0
        self.vix_change: float = 0.0  # Day-over-day change

    def set_vix(self, vix: float, vix_change: float = 0.0):
        """Update current VIX reading."""
        self.current_vix = vix
        self.vix_change = vix_change

    @property
    def regime(self) -> VIXRegime:
        return classify_vix_regime(self.current_vix, self.vix_change)

    def _get_regime_params(self) -> dict:
        """Get target, SL, and risk parameters for current VIX regime."""
        regime = self.regime
        if regime == VIXRegime.CHEAP:
            return {
                "target_pct": 50.0,
                "stop_loss_pct": 30.0,
                "risk_pct": 0.03,  # 3% — options are cheap
                "trailing_activation": 20.0,
                "trailing_lock": 50.0,
                "time_exit_min": 90,
            }
        elif regime == VIXRegime.NORMAL:
            return {
                "target_pct": 40.0,
                "stop_loss_pct": 30.0,
                "risk_pct": 0.02,
                "trailing_activation": 20.0,
                "trailing_lock": 50.0,
                "time_exit_min": 60,
            }
        elif regime == VIXRegime.EXPENSIVE:
            return {
                "target_pct": 25.0,  # Lower target — premiums inflated
                "stop_loss_pct": 25.0,
                "risk_pct": 0.01,  # 1% — premiums expensive
                "trailing_activation": 15.0,
                "trailing_lock": 60.0,
                "time_exit_min": 45,
            }
        else:  # CRUSH_ZONE
            return {
                "target_pct": 0.0,
                "stop_loss_pct": 0.0,
                "risk_pct": 0.0,
                "trailing_activation": 0.0,
                "trailing_lock": 0.0,
                "time_exit_min": 0,
            }

    def should_enter(
        self,
        signal: Signal,
        spot_price: float,
        current_time: datetime,
    ) -> bool:
        """Enter based on VIX regime and signal quality.

        Conditions:
        1. NIFTY only
        2. VIX data must be available and not in CRUSH_ZONE
        3. Score >= threshold (adjusted by regime)
        4. Standard time window
        """
        if signal.symbol != "NIFTY":
            return False

        if self.current_vix <= 0:
            return False

        if self.regime == VIXRegime.CRUSH_ZONE:
            logger.debug(
                f"VIX CRUSH_ZONE ({self.current_vix:.1f}, change={self.vix_change:+.1f}) "
                "— skipping option buying"
            )
            return False

        if signal.score < self.min_score:
            return False

        if current_time.time() < self.entry_start:
            return False

        if current_time.time() >= self.entry_end:
            return False

        if signal.direction == Direction.NEUTRAL:
            return False

        # In EXPENSIVE regime, require stronger signal
        if self.regime == VIXRegime.EXPENSIVE and signal.score < 80:
            return False

        # Technical must confirm direction
        if signal.technical_direction != signal.direction:
            return False

        logger.info(
            f"VIX Regime {self.regime.value}: VIX={self.current_vix:.1f} "
            f"signal={signal.direction.value} score={signal.score:.0f}"
        )
        return True

    def create_setup(
        self,
        signal: Signal,
        contract: OptionContract,
        spot_price: float,
        capital: float,
    ) -> Optional[TradeSetup]:
        """Create trade setup with regime-adjusted parameters."""
        entry_price = contract.ltp
        if entry_price <= 0:
            return None

        if entry_price < 5.0 or entry_price > 500.0:
            return None

        params = self._get_regime_params()
        if params["risk_pct"] <= 0:
            return None

        stop_loss = round(entry_price * (1 - params["stop_loss_pct"] / 100), 2)
        stop_loss = max(stop_loss, 0.05)
        target = round(entry_price * (1 + params["target_pct"] / 100), 2)

        risk_per_lot = (entry_price - stop_loss) * contract.lot_size
        if risk_per_lot <= 0:
            return None

        risk_amount = capital * params["risk_pct"]
        lots = max(1, int(risk_amount / risk_per_lot))
        quantity = lots * contract.lot_size

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
                f"VIX Regime={self.regime.value} VIX={self.current_vix:.1f} "
                f"{signal.direction.value} score={signal.score:.0f}"
            ),
        )

    def should_exit(
        self,
        position: Position,
        current_price: float,
        spot_price: float,
        current_time: datetime,
    ) -> ExitSignal:
        """Exit with regime-adapted trailing and time exits."""
        if current_time.time() >= self.hard_cutoff:
            return ExitSignal(True, "HARD_CUTOFF", current_price)

        if current_price <= position.stop_loss:
            return ExitSignal(True, "SL_HIT", current_price)

        if current_price >= position.target:
            return ExitSignal(True, "TARGET_HIT", current_price)

        params = self._get_regime_params()

        # Time exit
        holding_mins = position.holding_duration_minutes
        time_limit = params.get("time_exit_min", 60)
        if time_limit > 0 and holding_mins > time_limit:
            pnl_pct = ((current_price - position.entry_price) / position.entry_price) * 100
            if pnl_pct < 0:
                return ExitSignal(
                    True,
                    f"TIME_EXIT ({holding_mins:.0f}min, P&L={pnl_pct:.1f}%)",
                    current_price,
                )

        # Trailing SL
        if current_price > position.high_price:
            position.high_price = current_price

        activation = params.get("trailing_activation", 20.0)
        lock = params.get("trailing_lock", 50.0)
        profit_pct = ((position.high_price - position.entry_price) / position.entry_price) * 100
        if profit_pct >= activation:
            trail_sl = position.entry_price + (
                (position.high_price - position.entry_price) * (lock / 100)
            )
            if trail_sl > position.stop_loss:
                position.stop_loss = round(trail_sl, 2)

        return ExitSignal(False)

    def select_expiry(self, symbol: str, current_date: date) -> date:
        """CHEAP regime uses next week; others use current week."""
        next_exp = get_next_expiry(symbol, current_date)
        dte = days_to_expiry(next_exp, current_date)

        if self.regime == VIXRegime.CHEAP:
            # Next week — need time for IV expansion
            if dte >= 5:
                return next_exp
            return get_next_expiry(symbol, next_exp + timedelta(days=1))
        else:
            if dte >= 1:
                return next_exp
            return get_next_expiry(symbol, next_exp + timedelta(days=1))
