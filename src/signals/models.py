"""Signal data models."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class Direction(Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class SignalStrength(Enum):
    STRONG = "STRONG"      # Score >= 80
    MODERATE = "MODERATE"  # Score 70-79
    WEAK = "WEAK"          # Score 60-69
    NONE = "NONE"          # Score < 60


@dataclass
class ComponentScore:
    """Score from a single signal component."""

    name: str
    direction: Direction
    score: float  # 0-100
    weight: float  # 0-1
    details: str = ""


@dataclass
class Signal:
    """Combined trading signal from all analysis components."""

    timestamp: datetime
    symbol: str
    direction: Direction
    score: float  # 0-100 combined score
    strength: SignalStrength

    # Component scores
    technical_score: float = 0.0
    oi_score: float = 0.0
    iv_score: float = 0.0
    price_action_score: float = 0.0

    # Component directions
    technical_direction: Direction = Direction.NEUTRAL
    oi_direction: Direction = Direction.NEUTRAL
    iv_direction: Direction = Direction.NEUTRAL
    price_action_direction: Direction = Direction.NEUTRAL

    # How many components agree on direction
    agreement_count: int = 0

    # Action recommendation
    option_type: str = ""  # CE or PE
    strategy: str = ""
    details: str = ""
    components: list[ComponentScore] = field(default_factory=list)

    @property
    def is_actionable(self) -> bool:
        """Check if signal is strong enough to trade."""
        return self.strength in (SignalStrength.STRONG, SignalStrength.MODERATE)

    @property
    def is_bullish(self) -> bool:
        return self.direction == Direction.BULLISH

    @property
    def is_bearish(self) -> bool:
        return self.direction == Direction.BEARISH

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "direction": self.direction.value,
            "score": self.score,
            "technical_score": self.technical_score,
            "oi_score": self.oi_score,
            "iv_score": self.iv_score,
            "price_action_score": self.price_action_score,
            "strategy": self.strategy,
            "details": self.details,
        }
