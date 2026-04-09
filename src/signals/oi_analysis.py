"""Open Interest-based signal generation.

Analyzes PCR, OI buildup/unwinding, max pain positioning, and
OI change patterns to determine market direction.
"""

from src.data.options_chain import OptionsChainAnalysis
from src.signals.models import ComponentScore, Direction
from src.utils.logger import get_logger

logger = get_logger("oi_analysis")


def analyze_oi(
    chain_analysis: OptionsChainAnalysis,
    spot_price: float,
) -> ComponentScore:
    """Analyze Open Interest data for directional bias.

    Args:
        chain_analysis: Options chain analysis with OI data
        spot_price: Current spot price of underlying

    Returns:
        ComponentScore with OI-based direction and score (0-100)
    """
    if not chain_analysis.chain:
        return ComponentScore(
            name="oi_analysis",
            direction=Direction.NEUTRAL,
            score=50,
            weight=0.30,
            details="No OI data available",
        )

    bullish_points = 0
    bearish_points = 0
    details = []

    # 1. PCR Analysis (weight: 15)
    # PCR > 1.2: More puts written = support = bullish
    # PCR < 0.7: More calls written = resistance = bearish
    pcr = chain_analysis.pcr_oi
    if pcr >= 1.5:
        bullish_points += 15
        details.append(f"PCR very bullish ({pcr:.2f})")
    elif pcr >= 1.2:
        bullish_points += 12
        details.append(f"PCR bullish ({pcr:.2f})")
    elif pcr >= 0.9:
        bullish_points += 5
        details.append(f"PCR neutral ({pcr:.2f})")
    elif pcr >= 0.7:
        bearish_points += 12
        details.append(f"PCR bearish ({pcr:.2f})")
    else:
        bearish_points += 15
        details.append(f"PCR very bearish ({pcr:.2f})")

    # 2. OI Change Analysis (weight: 10)
    # Put OI increasing = writers confident about support = bullish
    # Call OI increasing = writers confident about resistance = bearish
    ce_oi_change = chain_analysis.total_ce_oi_change
    pe_oi_change = chain_analysis.total_pe_oi_change

    if pe_oi_change > 0 and ce_oi_change < 0:
        # Put writing + Call unwinding = strongly bullish
        bullish_points += 10
        details.append("Put writing + Call unwinding")
    elif pe_oi_change > 0 and ce_oi_change > 0:
        # Both increasing - check which is dominant
        if pe_oi_change > ce_oi_change * 1.5:
            bullish_points += 7
            details.append("Dominant put writing")
        elif ce_oi_change > pe_oi_change * 1.5:
            bearish_points += 7
            details.append("Dominant call writing")
        else:
            details.append("OI change mixed")
    elif ce_oi_change > 0 and pe_oi_change < 0:
        # Call writing + Put unwinding = strongly bearish
        bearish_points += 10
        details.append("Call writing + Put unwinding")
    elif ce_oi_change < 0 and pe_oi_change < 0:
        # Both unwinding - volatile / directional move
        details.append("OI unwinding both sides")
        bullish_points += 3
        bearish_points += 3

    # 3. Max Pain Analysis (weight: 5)
    # Price tends to gravitate towards max pain by expiry
    max_pain = chain_analysis.max_pain
    if max_pain > 0:
        diff_pct = ((spot_price - max_pain) / max_pain) * 100

        if diff_pct < -1.0:
            # Price below max pain → tendency to rise
            bullish_points += 5
            details.append(f"Price below max pain ({max_pain})")
        elif diff_pct > 1.0:
            # Price above max pain → tendency to fall
            bearish_points += 5
            details.append(f"Price above max pain ({max_pain})")
        else:
            details.append(f"Price near max pain ({max_pain})")

    # 4. OI Concentration / Support-Resistance (weight: 10)
    max_pe_strike = chain_analysis.max_pe_oi_strike  # Support
    max_ce_strike = chain_analysis.max_ce_oi_strike  # Resistance

    if max_pe_strike > 0 and max_ce_strike > 0:
        # Check where spot is relative to S/R walls
        range_size = max_ce_strike - max_pe_strike
        if range_size > 0:
            position_in_range = (spot_price - max_pe_strike) / range_size

            if position_in_range < 0.3:
                # Near support wall → bullish
                bullish_points += 10
                details.append(f"Near PE support wall ({max_pe_strike})")
            elif position_in_range > 0.7:
                # Near resistance wall → bearish
                bearish_points += 10
                details.append(f"Near CE resistance wall ({max_ce_strike})")
            elif position_in_range < 0.5:
                bullish_points += 5
                details.append(f"Below OI midpoint")
            else:
                bearish_points += 5
                details.append(f"Above OI midpoint")

    # Calculate final score
    total_possible = 40
    if bullish_points > bearish_points:
        direction = Direction.BULLISH
        score = min(100, (bullish_points / total_possible) * 100)
    elif bearish_points > bullish_points:
        direction = Direction.BEARISH
        score = min(100, (bearish_points / total_possible) * 100)
    else:
        direction = Direction.NEUTRAL
        score = 50

    logger.info(
        f"OI Analysis {chain_analysis.symbol}: {direction.value} "
        f"score={score:.0f} PCR={pcr:.2f}"
    )

    return ComponentScore(
        name="oi_analysis",
        direction=direction,
        score=round(score, 1),
        weight=0.30,
        details=" | ".join(details),
    )
