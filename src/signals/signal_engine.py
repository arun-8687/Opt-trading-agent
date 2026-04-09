"""Signal engine — combines all signal components into a final trading signal.

Uses a multi-factor scoring approach:
- Technical (40%) + OI Analysis (30%) + IV Analysis (15%) + Price Action (15%)
- Requires minimum 3 of 4 categories to agree on direction
- Configurable score threshold for signal generation
"""

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from src.data.options_chain import OptionsChainAnalysis
from src.signals.iv_analysis import analyze_iv
from src.signals.models import ComponentScore, Direction, Signal, SignalStrength
from src.signals.oi_analysis import analyze_oi
from src.signals.price_action import analyze_price_action
from src.signals.technical import TechnicalConfig, analyze_technical
from src.utils.logger import get_logger

logger = get_logger("signal_engine")


class SignalEngine:
    """Combines multiple signal components into a final trading signal."""

    def __init__(
        self,
        min_score: float = 75.0,
        min_categories_agree: int = 3,
        cooldown_minutes: int = 15,
        weights: Optional[dict[str, float]] = None,
    ):
        self.min_score = min_score
        self.min_categories_agree = min_categories_agree
        self.cooldown_minutes = cooldown_minutes
        self.weights = weights or {
            "technical": 0.40,
            "oi_analysis": 0.30,
            "iv_analysis": 0.15,
            "price_action": 0.15,
        }
        self._last_signal_time: dict[str, datetime] = {}
        self.technical_config = TechnicalConfig()

    def generate_signal(
        self,
        symbol: str,
        intraday_df: pd.DataFrame,
        daily_df: pd.DataFrame,
        chain_analysis: Optional[OptionsChainAnalysis],
        spot_price: float,
        iv_history: pd.Series = pd.Series(dtype=float),
    ) -> Signal:
        """Generate a combined trading signal for a symbol.

        Args:
            symbol: Trading symbol (e.g., NIFTY, RELIANCE)
            intraday_df: Intraday candles (5min recommended)
            daily_df: Daily candles (last 20+ days)
            chain_analysis: Options chain analysis (can be None)
            spot_price: Current spot/underlying price
            iv_history: Historical ATM IV series

        Returns:
            Signal with combined score, direction, and component details
        """
        now = datetime.now()

        # Check cooldown
        if symbol in self._last_signal_time:
            elapsed = (now - self._last_signal_time[symbol]).total_seconds()
            if elapsed < self.cooldown_minutes * 60:
                return self._neutral_signal(
                    symbol, now, f"Cooldown active ({self.cooldown_minutes - elapsed/60:.0f}min left)"
                )

        components: list[ComponentScore] = []

        # 1. Technical Analysis
        tech_score = analyze_technical(intraday_df, self.technical_config)
        components.append(tech_score)

        # 2. OI Analysis
        if chain_analysis and chain_analysis.chain:
            oi_score = analyze_oi(chain_analysis, spot_price)
        else:
            oi_score = ComponentScore(
                name="oi_analysis",
                direction=Direction.NEUTRAL,
                score=50,
                weight=self.weights["oi_analysis"],
                details="No OI data",
            )
        components.append(oi_score)

        # 3. IV Analysis
        if chain_analysis:
            iv_score_comp = analyze_iv(chain_analysis, iv_history)
        else:
            iv_score_comp = ComponentScore(
                name="iv_analysis",
                direction=Direction.NEUTRAL,
                score=50,
                weight=self.weights["iv_analysis"],
                details="No IV data",
            )
        components.append(iv_score_comp)

        # 4. Price Action
        pa_score = analyze_price_action(intraday_df, daily_df, spot_price)
        components.append(pa_score)

        # Count directional agreement
        bullish_count = sum(
            1 for c in components if c.direction == Direction.BULLISH
        )
        bearish_count = sum(
            1 for c in components if c.direction == Direction.BEARISH
        )

        # Determine majority direction
        if bullish_count >= self.min_categories_agree:
            final_direction = Direction.BULLISH
            agreement = bullish_count
        elif bearish_count >= self.min_categories_agree:
            final_direction = Direction.BEARISH
            agreement = bearish_count
        else:
            final_direction = Direction.NEUTRAL
            agreement = max(bullish_count, bearish_count)

        # Calculate weighted combined score
        # Only count scores from components that agree with the majority direction
        combined_score = 0.0
        for comp in components:
            weight = self.weights.get(comp.name, 0.15)
            if comp.direction == final_direction:
                combined_score += comp.score * weight
            elif comp.direction == Direction.NEUTRAL:
                combined_score += comp.score * weight * 0.5  # Half weight for neutral
            # Opposing direction gets 0 contribution

        # Normalize to 0-100
        combined_score = min(100, combined_score)

        # Determine signal strength
        if combined_score >= 80:
            strength = SignalStrength.STRONG
        elif combined_score >= 70:
            strength = SignalStrength.MODERATE
        elif combined_score >= 60:
            strength = SignalStrength.WEAK
        else:
            strength = SignalStrength.NONE

        # Determine option type
        option_type = ""
        if final_direction == Direction.BULLISH and combined_score >= self.min_score:
            option_type = "CE"
        elif final_direction == Direction.BEARISH and combined_score >= self.min_score:
            option_type = "PE"

        # Build details string
        detail_parts = [f"{c.name}: {c.direction.value} ({c.score:.0f})" for c in components]
        details = f"Agreement: {agreement}/4 | " + " | ".join(detail_parts)

        signal = Signal(
            timestamp=now,
            symbol=symbol,
            direction=final_direction,
            score=round(combined_score, 1),
            strength=strength,
            technical_score=tech_score.score,
            oi_score=oi_score.score,
            iv_score=iv_score_comp.score,
            price_action_score=pa_score.score,
            technical_direction=tech_score.direction,
            oi_direction=oi_score.direction,
            iv_direction=iv_score_comp.direction,
            price_action_direction=pa_score.direction,
            agreement_count=agreement,
            option_type=option_type,
            details=details,
            components=components,
        )

        # Update cooldown timer if signal is actionable
        if signal.is_actionable and option_type:
            self._last_signal_time[symbol] = now
            logger.info(
                f"SIGNAL: {symbol} {final_direction.value} score={combined_score:.0f} "
                f"strength={strength.value} → BUY {option_type}"
            )
        else:
            logger.debug(
                f"No signal: {symbol} {final_direction.value} score={combined_score:.0f} "
                f"agreement={agreement}/4"
            )

        return signal

    def _neutral_signal(self, symbol: str, now: datetime, reason: str) -> Signal:
        """Create a neutral (no-trade) signal."""
        return Signal(
            timestamp=now,
            symbol=symbol,
            direction=Direction.NEUTRAL,
            score=0,
            strength=SignalStrength.NONE,
            details=reason,
        )

    def reset_cooldown(self, symbol: str):
        """Reset cooldown for a symbol (used after position exit)."""
        self._last_signal_time.pop(symbol, None)
