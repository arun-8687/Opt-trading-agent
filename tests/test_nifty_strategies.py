"""Tests for NIFTY-specific trading strategies."""

from datetime import date, datetime, time, timedelta

import pytest

from src.broker.models import OptionContract, OptionType, OrderSide, Position
from src.signals.models import ComponentScore, Direction, Signal, SignalStrength
from src.strategies.nifty_event_day import EventPhase, NiftyEventDayStrategy
from src.strategies.nifty_gamma_blast import NiftyGammaBlastStrategy
from src.strategies.nifty_gift_gap import NiftyGIFTGapStrategy
from src.strategies.nifty_oi_wall import NiftyOIWallStrategy, OIWallAction
from src.strategies.nifty_orb import NiftyORBStrategy
from src.strategies.nifty_pcr_reversal import NiftyPCRReversalStrategy
from src.strategies.nifty_vix_regime import (
    NiftyVIXRegimeStrategy,
    VIXRegime,
    classify_vix_regime,
)


# ----- Helpers -----

def make_signal(
    direction=Direction.BULLISH,
    score=80.0,
    strength=SignalStrength.STRONG,
    symbol="NIFTY",
    technical_direction=None,
    technical_score=75.0,
    oi_direction=Direction.NEUTRAL,
    iv_score=50.0,
    agreement_count=3,
    components=None,
):
    if technical_direction is None:
        technical_direction = direction
    return Signal(
        timestamp=datetime.now(),
        symbol=symbol,
        direction=direction,
        score=score,
        strength=strength,
        technical_score=technical_score,
        technical_direction=technical_direction,
        oi_direction=oi_direction,
        iv_score=iv_score,
        agreement_count=agreement_count,
        option_type="CE" if direction == Direction.BULLISH else "PE",
        components=components or [],
    )


def make_contract(ltp=100.0, lot_size=65, strike=24500):
    return OptionContract(
        symbol="NIFTY",
        token="12345",
        trading_symbol="NIFTY10APR24500CE",
        strike=strike,
        option_type=OptionType.CE,
        expiry=date(2026, 4, 16),
        lot_size=lot_size,
        ltp=ltp,
        oi=50000,
        volume=10000,
    )


def make_position(
    entry_price=100.0,
    high_price=100.0,
    stop_loss=70.0,
    target=140.0,
    strategy="nifty_orb",
    entry_minutes_ago=10,
):
    entry_time = datetime.now() - timedelta(minutes=entry_minutes_ago)
    return Position(
        symbol="NIFTY",
        trading_symbol="NIFTY10APR24500CE",
        token="12345",
        exchange="NFO",
        option_type=OptionType.CE,
        strike=24500,
        expiry=date(2026, 4, 16),
        side=OrderSide.BUY,
        quantity=65,
        lot_size=65,
        entry_price=entry_price,
        entry_time=entry_time,
        current_price=entry_price,
        high_price=high_price,
        stop_loss=stop_loss,
        target=target,
        strategy=strategy,
    )


def trading_time(hour=10, minute=30):
    return datetime(2026, 4, 9, hour, minute)  # Thursday


def expiry_thursday(hour=14, minute=0):
    """Create a time on a Thursday (NIFTY expiry day)."""
    return datetime(2026, 4, 9, hour, minute)  # Thursday


# ===== NIFTY ORB Tests =====

class TestNiftyORB:
    def setup_method(self):
        self.strategy = NiftyORBStrategy()

    def test_rejects_non_nifty(self):
        sig = make_signal(symbol="BANKNIFTY")
        self.strategy.set_orb_range(24500, 24400)
        assert not self.strategy.should_enter(sig, 24550, trading_time())

    def test_rejects_no_orb_range(self):
        sig = make_signal(score=80.0)
        assert not self.strategy.should_enter(sig, 24550, trading_time())

    def test_rejects_no_breakout(self):
        sig = make_signal(score=80.0)
        self.strategy.set_orb_range(24500, 24400)
        # Price inside range
        assert not self.strategy.should_enter(sig, 24450, trading_time())

    def test_rejects_direction_mismatch(self):
        sig = make_signal(direction=Direction.BEARISH, technical_direction=Direction.BEARISH)
        self.strategy.set_orb_range(24500, 24400)
        # Price broke above ORB (bullish) but signal is bearish
        assert not self.strategy.should_enter(sig, 24550, trading_time())

    def test_rejects_no_volume(self):
        sig = make_signal(
            score=80.0,
            components=[
                ComponentScore("technical", Direction.BULLISH, 80, 0.4, "EMA bullish"),
            ],
        )
        self.strategy.set_orb_range(24500, 24400)
        assert not self.strategy.should_enter(sig, 24550, trading_time())

    def test_enters_bullish_breakout(self):
        sig = make_signal(
            score=80.0,
            components=[
                ComponentScore("technical", Direction.BULLISH, 80, 0.4, "EMA bullish, volume surge"),
            ],
        )
        self.strategy.set_orb_range(24500, 24400)
        assert self.strategy.should_enter(sig, 24550, trading_time())

    def test_enters_bearish_breakdown(self):
        sig = make_signal(
            score=80.0,
            direction=Direction.BEARISH,
            technical_direction=Direction.BEARISH,
            components=[
                ComponentScore("technical", Direction.BEARISH, 80, 0.4, "EMA bearish, volume ok"),
            ],
        )
        self.strategy.set_orb_range(24500, 24400)
        assert self.strategy.should_enter(sig, 24350, trading_time())

    def test_one_trade_per_day(self):
        sig = make_signal(
            score=80.0,
            components=[
                ComponentScore("technical", Direction.BULLISH, 80, 0.4, "EMA bullish, volume surge"),
            ],
        )
        self.strategy.set_orb_range(24500, 24400)
        assert self.strategy.should_enter(sig, 24550, trading_time())
        # Second entry should be rejected
        assert not self.strategy.should_enter(sig, 24600, trading_time(11, 0))

    def test_rejects_before_entry(self):
        sig = make_signal(score=80.0)
        self.strategy.set_orb_range(24500, 24400)
        assert not self.strategy.should_enter(sig, 24550, trading_time(9, 25))

    def test_create_setup(self):
        sig = make_signal()
        contract = make_contract(ltp=150.0)
        self.strategy.set_orb_range(24500, 24400)
        setup = self.strategy.create_setup(sig, contract, 24500, 200000)
        assert setup is not None
        assert setup.strategy_name == "nifty_orb"
        assert "ORB" in setup.reason

    def test_exit_time_based_losing(self):
        pos = make_position(entry_price=100, stop_loss=70, target=140, entry_minutes_ago=50)
        exit_sig = self.strategy.should_exit(pos, 95, 24500, trading_time())
        assert exit_sig.should_exit
        assert "TIME_EXIT" in exit_sig.reason


# ===== NIFTY Gamma Blast Tests =====

class TestNiftyGammaBlast:
    def setup_method(self):
        self.strategy = NiftyGammaBlastStrategy()

    def test_rejects_non_nifty(self):
        sig = make_signal(symbol="BANKNIFTY")
        assert not self.strategy.should_enter(sig, 24500, expiry_thursday())

    def test_rejects_non_expiry_day(self):
        sig = make_signal()
        self.strategy.set_day_range(24500, 24470)
        # Wednesday is not NIFTY expiry
        assert not self.strategy.should_enter(sig, 24510, datetime(2026, 4, 8, 14, 0))

    def test_rejects_wide_range(self):
        sig = make_signal()
        self.strategy.set_day_range(24550, 24450)  # 100pt range > 50 limit
        assert not self.strategy.should_enter(sig, 24560, expiry_thursday())

    def test_rejects_before_entry_time(self):
        sig = make_signal()
        self.strategy.set_day_range(24500, 24470)
        assert not self.strategy.should_enter(sig, 24510, expiry_thursday(11, 0))

    def test_rejects_no_breakout(self):
        sig = make_signal()
        self.strategy.set_day_range(24500, 24470)
        # Spot inside range
        assert not self.strategy.should_enter(sig, 24490, expiry_thursday())

    def test_enters_bullish_breakout(self):
        sig = make_signal(score=70.0, direction=Direction.BULLISH)
        self.strategy.set_day_range(24500, 24470)
        assert self.strategy.should_enter(sig, 24510, expiry_thursday())

    def test_enters_bearish_breakdown(self):
        sig = make_signal(score=70.0, direction=Direction.BEARISH)
        self.strategy.set_day_range(24500, 24470)
        assert self.strategy.should_enter(sig, 24460, expiry_thursday())

    def test_rejects_expensive_premium(self):
        sig = make_signal()
        contract = make_contract(ltp=100.0)  # > 60 max premium
        setup = self.strategy.create_setup(sig, contract, 24500, 200000)
        assert setup is None

    def test_accepts_cheap_premium(self):
        sig = make_signal()
        contract = make_contract(ltp=25.0)
        self.strategy.set_day_range(24500, 24470)
        setup = self.strategy.create_setup(sig, contract, 24500, 200000)
        assert setup is not None
        assert setup.strategy_name == "nifty_gamma_blast"

    def test_exit_near_zero(self):
        pos = make_position(entry_price=20, stop_loss=0.05, target=40, strategy="nifty_gamma_blast")
        exit_sig = self.strategy.should_exit(pos, 0.5, 24500, expiry_thursday())
        assert exit_sig.should_exit
        assert "NEAR_ZERO" in exit_sig.reason

    def test_exit_expiry_cutoff(self):
        pos = make_position(strategy="nifty_gamma_blast")
        exit_sig = self.strategy.should_exit(pos, 30, 24500, expiry_thursday(15, 20))
        assert exit_sig.should_exit
        assert "EXPIRY_CUTOFF" in exit_sig.reason


# ===== NIFTY VIX Regime Tests =====

class TestNiftyVIXRegime:
    def setup_method(self):
        self.strategy = NiftyVIXRegimeStrategy()

    def test_classify_cheap(self):
        assert classify_vix_regime(11.0) == VIXRegime.CHEAP

    def test_classify_normal(self):
        assert classify_vix_regime(15.0) == VIXRegime.NORMAL

    def test_classify_expensive(self):
        assert classify_vix_regime(20.0) == VIXRegime.EXPENSIVE

    def test_classify_crush_zone(self):
        assert classify_vix_regime(25.0, vix_change=-2.0) == VIXRegime.CRUSH_ZONE

    def test_high_vix_rising_is_expensive_not_crush(self):
        assert classify_vix_regime(25.0, vix_change=1.0) == VIXRegime.EXPENSIVE

    def test_rejects_non_nifty(self):
        self.strategy.set_vix(15.0)
        sig = make_signal(symbol="BANKNIFTY")
        assert not self.strategy.should_enter(sig, 24500, trading_time())

    def test_rejects_no_vix(self):
        sig = make_signal()
        assert not self.strategy.should_enter(sig, 24500, trading_time())

    def test_rejects_crush_zone(self):
        self.strategy.set_vix(25.0, vix_change=-3.0)
        sig = make_signal(score=85.0)
        assert not self.strategy.should_enter(sig, 24500, trading_time())

    def test_enters_cheap_vix(self):
        self.strategy.set_vix(11.0)
        sig = make_signal(score=80.0)
        assert self.strategy.should_enter(sig, 24500, trading_time())

    def test_enters_normal_vix(self):
        self.strategy.set_vix(15.0)
        sig = make_signal(score=80.0)
        assert self.strategy.should_enter(sig, 24500, trading_time())

    def test_expensive_requires_high_score(self):
        self.strategy.set_vix(20.0)
        sig_low = make_signal(score=75.0)
        sig_high = make_signal(score=85.0)
        assert not self.strategy.should_enter(sig_low, 24500, trading_time())
        assert self.strategy.should_enter(sig_high, 24500, trading_time())

    def test_regime_params_cheap_higher_risk(self):
        self.strategy.set_vix(11.0)
        params = self.strategy._get_regime_params()
        assert params["risk_pct"] == 0.03  # 3% for cheap

    def test_regime_params_expensive_lower_risk(self):
        self.strategy.set_vix(20.0)
        params = self.strategy._get_regime_params()
        assert params["risk_pct"] == 0.01  # 1% for expensive

    def test_create_setup(self):
        self.strategy.set_vix(15.0)
        sig = make_signal()
        contract = make_contract(ltp=150.0)
        setup = self.strategy.create_setup(sig, contract, 24500, 200000)
        assert setup is not None
        assert "VIX Regime=NORMAL" in setup.reason


# ===== NIFTY PCR Reversal Tests =====

class TestNiftyPCRReversal:
    def setup_method(self):
        self.strategy = NiftyPCRReversalStrategy()

    def test_rejects_non_nifty(self):
        self.strategy.set_pcr_data(1.5, 24000, 25000)
        sig = make_signal(symbol="BANKNIFTY")
        assert not self.strategy.should_enter(sig, 24100, trading_time())

    def test_rejects_no_pcr(self):
        sig = make_signal()
        assert not self.strategy.should_enter(sig, 24100, trading_time())

    def test_rejects_neutral_pcr(self):
        self.strategy.set_pcr_data(1.0, 24000, 25000)  # PCR in normal range
        sig = make_signal()
        assert not self.strategy.should_enter(sig, 24100, trading_time())

    def test_enters_bullish_pcr(self):
        self.strategy.set_pcr_data(1.5, 24000, 25000)
        sig = make_signal(direction=Direction.BULLISH, score=75.0)
        assert self.strategy.should_enter(sig, 24050, trading_time())

    def test_enters_bearish_pcr(self):
        self.strategy.set_pcr_data(0.6, 23000, 24500)
        sig = make_signal(direction=Direction.BEARISH, score=75.0, technical_direction=Direction.BEARISH)
        assert self.strategy.should_enter(sig, 24450, trading_time())

    def test_rejects_too_far_from_wall(self):
        self.strategy.set_pcr_data(1.5, 24000, 25000)
        sig = make_signal(direction=Direction.BULLISH, score=75.0)
        # 24300 is 300 pts above 24000 put wall — too far
        assert not self.strategy.should_enter(sig, 24300, trading_time())

    def test_rejects_opposing_oi(self):
        self.strategy.set_pcr_data(1.5, 24000, 25000)
        sig = make_signal(direction=Direction.BULLISH, score=75.0, oi_direction=Direction.BEARISH)
        assert not self.strategy.should_enter(sig, 24050, trading_time())

    def test_create_setup_contrarian_sizing(self):
        self.strategy.set_pcr_data(1.5, 24000, 25000)
        sig = make_signal()
        contract = make_contract(ltp=100.0)
        setup = self.strategy.create_setup(sig, contract, 24500, 200000)
        assert setup is not None
        assert "PCR Reversal" in setup.reason

    def test_exit_time_limit(self):
        pos = make_position(
            entry_price=100, stop_loss=75, target=140,
            strategy="nifty_pcr_reversal", entry_minutes_ago=100,
        )
        exit_sig = self.strategy.should_exit(pos, 100, 24500, trading_time())
        assert exit_sig.should_exit
        assert "TIME_EXIT" in exit_sig.reason


# ===== NIFTY GIFT Gap Tests =====

class TestNiftyGIFTGap:
    def setup_method(self):
        self.strategy = NiftyGIFTGapStrategy()

    def test_rejects_non_nifty(self):
        self.strategy.set_gap_data(0.8, 24000, True)
        sig = make_signal(symbol="BANKNIFTY")
        assert not self.strategy.should_enter(sig, 24200, trading_time())

    def test_rejects_small_gap(self):
        self.strategy.set_gap_data(0.3, 24000, True)  # < 0.5% threshold
        sig = make_signal()
        assert not self.strategy.should_enter(sig, 24100, trading_time())

    def test_rejects_no_fii_confirm(self):
        # Gap up but FII were sellers (not confirming)
        self.strategy.set_gap_data(0.8, 24000, False)
        sig = make_signal(score=80.0, direction=Direction.BULLISH)
        assert not self.strategy.should_enter(sig, 24200, trading_time())

    def test_rejects_gap_filled(self):
        # Gap up but price below prev close = gap filled
        self.strategy.set_gap_data(0.8, 24000, True)
        sig = make_signal(score=80.0, direction=Direction.BULLISH)
        assert not self.strategy.should_enter(sig, 23900, trading_time())

    def test_enters_bullish_gap_confirmed(self):
        self.strategy.set_gap_data(0.8, 24000, True)
        sig = make_signal(score=80.0, direction=Direction.BULLISH)
        # Use Wednesday (non-expiry day) to avoid expiry rejection
        wed = datetime(2026, 4, 8, 10, 30)
        assert self.strategy.should_enter(sig, 24200, wed)

    def test_enters_bearish_gap_confirmed(self):
        # Gap down, FII were sellers (confirming bearish)
        self.strategy.set_gap_data(-0.8, 24000, False)
        sig = make_signal(
            score=80.0, direction=Direction.BEARISH,
            technical_direction=Direction.BEARISH,
        )
        wed = datetime(2026, 4, 8, 10, 30)
        assert self.strategy.should_enter(sig, 23800, wed)

    def test_rejects_direction_mismatch(self):
        self.strategy.set_gap_data(0.8, 24000, True)
        sig = make_signal(score=80.0, direction=Direction.BEARISH, technical_direction=Direction.BEARISH)
        assert not self.strategy.should_enter(sig, 24200, trading_time())

    def test_rejects_after_entry_end(self):
        self.strategy.set_gap_data(0.8, 24000, True)
        sig = make_signal(score=80.0)
        assert not self.strategy.should_enter(sig, 24200, trading_time(12, 0))

    def test_create_setup(self):
        self.strategy.set_gap_data(0.8, 24000, True)
        sig = make_signal()
        contract = make_contract(ltp=200.0)
        setup = self.strategy.create_setup(sig, contract, 24200, 200000)
        assert setup is not None
        assert "GIFT Gap" in setup.reason


# ===== NIFTY Event Day Tests =====

class TestNiftyEventDay:
    def setup_method(self):
        self.strategy = NiftyEventDayStrategy()

    def test_rejects_non_nifty(self):
        self.strategy.set_event("RBI Policy", date(2026, 4, 9), date(2026, 4, 7))
        sig = make_signal(symbol="BANKNIFTY")
        assert not self.strategy.should_enter(sig, 24500, trading_time())

    def test_rejects_no_event(self):
        sig = make_signal()
        assert not self.strategy.should_enter(sig, 24500, trading_time())

    def test_pre_event_phase(self):
        self.strategy.set_event("Budget", date(2026, 4, 11), date(2026, 4, 9))
        assert self.strategy.event_phase == EventPhase.PRE_EVENT
        assert self.strategy.days_to_event == 2

    def test_post_event_phase(self):
        self.strategy.set_event("RBI Policy", date(2026, 4, 9), date(2026, 4, 9))
        assert self.strategy.event_phase == EventPhase.POST_EVENT

    def test_no_event_far_away(self):
        self.strategy.set_event("Budget", date(2026, 4, 20), date(2026, 4, 9))
        assert self.strategy.event_phase == EventPhase.NO_EVENT

    def test_enters_pre_event(self):
        self.strategy.set_event("Budget", date(2026, 4, 11), date(2026, 4, 9))
        sig = make_signal(score=70.0)
        assert self.strategy.should_enter(sig, 24500, trading_time())

    def test_post_event_rejects_early(self):
        self.strategy.set_event("RBI Policy", date(2026, 4, 9), date(2026, 4, 9))
        sig = make_signal(score=80.0)
        # Before 10:00 AM
        assert not self.strategy.should_enter(sig, 24500, trading_time(9, 30))

    def test_post_event_enters_after_wait(self):
        self.strategy.set_event("RBI Policy", date(2026, 4, 9), date(2026, 4, 9))
        sig = make_signal(score=80.0)
        assert self.strategy.should_enter(sig, 24500, trading_time(10, 30))

    def test_post_event_rejects_low_score(self):
        self.strategy.set_event("RBI Policy", date(2026, 4, 9), date(2026, 4, 9))
        sig = make_signal(score=65.0)
        assert not self.strategy.should_enter(sig, 24500, trading_time(10, 30))

    def test_create_setup_pre_event(self):
        self.strategy.set_event("Budget", date(2026, 4, 11), date(2026, 4, 9))
        sig = make_signal()
        contract = make_contract(ltp=150.0)
        setup = self.strategy.create_setup(sig, contract, 24500, 200000)
        assert setup is not None
        assert "PRE_EVENT" in setup.reason

    def test_create_setup_post_event(self):
        self.strategy.set_event("RBI Policy", date(2026, 4, 9), date(2026, 4, 9))
        sig = make_signal()
        contract = make_contract(ltp=150.0)
        setup = self.strategy.create_setup(sig, contract, 24500, 200000)
        assert setup is not None
        assert "POST_EVENT" in setup.reason
        # Post-event has higher target (60% vs 30%)
        assert setup.target == round(150 * 1.60, 2)

    def test_clear_event(self):
        self.strategy.set_event("Budget", date(2026, 4, 11), date(2026, 4, 9))
        self.strategy.clear_event()
        assert self.strategy.event_phase == EventPhase.NO_EVENT


# ===== NIFTY OI Wall Tests =====

class TestNiftyOIWall:
    def setup_method(self):
        self.strategy = NiftyOIWallStrategy()

    def test_rejects_non_nifty(self):
        self.strategy.set_oi_walls(24000, 25000)
        sig = make_signal(symbol="BANKNIFTY")
        assert not self.strategy.should_enter(sig, 24050, trading_time())

    def test_rejects_no_oi_data(self):
        sig = make_signal()
        assert not self.strategy.should_enter(sig, 24050, trading_time())

    def test_detects_bounce_at_put_wall(self):
        self.strategy.set_oi_walls(24000, 25000)
        sig = make_signal(score=75.0, direction=Direction.BULLISH)
        # Use Wednesday (non-expiry) to avoid expiry rejection
        wed = datetime(2026, 4, 8, 10, 30)
        assert self.strategy.should_enter(sig, 24030, wed)
        assert self.strategy._last_action == OIWallAction.BOUNCE

    def test_detects_bounce_at_call_wall(self):
        self.strategy.set_oi_walls(24000, 25000)
        sig = make_signal(
            score=75.0, direction=Direction.BEARISH,
            technical_direction=Direction.BEARISH,
        )
        wed = datetime(2026, 4, 8, 10, 30)
        assert self.strategy.should_enter(sig, 24970, wed)
        assert self.strategy._last_action == OIWallAction.BOUNCE

    def test_detects_break_below_put_wall(self):
        self.strategy.set_oi_walls(24000, 25000)
        sig = make_signal(
            score=75.0, direction=Direction.BEARISH,
            technical_direction=Direction.BEARISH,
        )
        wed = datetime(2026, 4, 8, 10, 30)
        assert self.strategy.should_enter(sig, 23950, wed)
        assert self.strategy._last_action == OIWallAction.BREAK

    def test_detects_break_above_call_wall(self):
        self.strategy.set_oi_walls(24000, 25000)
        sig = make_signal(score=75.0, direction=Direction.BULLISH)
        wed = datetime(2026, 4, 8, 10, 30)
        assert self.strategy.should_enter(sig, 25050, wed)
        assert self.strategy._last_action == OIWallAction.BREAK

    def test_rejects_middle_of_range(self):
        self.strategy.set_oi_walls(24000, 25000)
        sig = make_signal(score=75.0, direction=Direction.BULLISH)
        wed = datetime(2026, 4, 8, 10, 30)
        assert not self.strategy.should_enter(sig, 24500, wed)

    def test_create_setup_bounce_vs_break(self):
        self.strategy.set_oi_walls(24000, 25000)
        sig = make_signal()
        contract = make_contract(ltp=100.0)

        # Bounce setup
        self.strategy._last_action = OIWallAction.BOUNCE
        setup = self.strategy.create_setup(sig, contract, 24050, 200000)
        assert setup is not None
        assert setup.target == round(100 * 1.35, 2)  # 35% bounce target

        # Break setup
        self.strategy._last_action = OIWallAction.BREAK
        setup = self.strategy.create_setup(sig, contract, 25050, 200000)
        assert setup is not None
        assert setup.target == round(100 * 1.50, 2)  # 50% break target

    def test_exit_time_based(self):
        pos = make_position(
            entry_price=100, stop_loss=75, target=135,
            strategy="nifty_oi_wall", entry_minutes_ago=70,
        )
        self.strategy._last_action = OIWallAction.BOUNCE
        exit_sig = self.strategy.should_exit(pos, 100, 24500, trading_time())
        assert exit_sig.should_exit
        assert "TIME_EXIT" in exit_sig.reason

    def test_break_time_exit_longer(self):
        """Break trades get more time than bounce trades."""
        pos = make_position(
            entry_price=100, stop_loss=70, target=150,
            strategy="nifty_oi_wall", entry_minutes_ago=70,
        )
        self.strategy._last_action = OIWallAction.BREAK
        # At 70 min, bounce would exit but break should not
        exit_sig = self.strategy.should_exit(pos, 100, 24500, trading_time())
        assert not exit_sig.should_exit  # 90 min limit for breaks


# ===== Cross-cutting NIFTY Tests =====

class TestNiftyConstants:
    def test_nifty_lot_size_updated(self):
        from src.utils.constants import INDEX_LOT_SIZES
        assert INDEX_LOT_SIZES["NIFTY"] == 65

    def test_banknifty_lot_size_updated(self):
        from src.utils.constants import INDEX_LOT_SIZES
        assert INDEX_LOT_SIZES["BANKNIFTY"] == 30

    def test_nifty_strike_interval(self):
        from src.utils.constants import INDEX_STRIKE_INTERVALS
        assert INDEX_STRIKE_INTERVALS["NIFTY"] == 50

    def test_nifty_expiry_thursday(self):
        from src.utils.constants import INDEX_EXPIRY_DAYS
        assert INDEX_EXPIRY_DAYS["NIFTY"] == 3  # Thursday
