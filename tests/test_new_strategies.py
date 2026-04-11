"""Tests for additional trading strategies:
EarningsPlay, VolumeProfile, BollingerSqueeze, FibonacciRetracement, SectorRotation.
"""

from datetime import date, datetime, time, timedelta

import pytest

from src.broker.models import OptionContract, OptionType, OrderSide, Position
from src.signals.models import ComponentScore, Direction, Signal, SignalStrength
from src.strategies.bollinger_squeeze import BollingerSqueezeStrategy
from src.strategies.earnings_play import EarningsPhase, EarningsPlayStrategy
from src.strategies.fibonacci_retracement import FibonacciRetracementStrategy
from src.strategies.sector_rotation import SectorRotationStrategy
from src.strategies.volume_profile import VolumeProfileStrategy, VPAction


# ----- Helpers -----

def make_signal(
    direction=Direction.BULLISH,
    score=80.0,
    strength=SignalStrength.STRONG,
    symbol="RELIANCE",
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


def make_contract(ltp=100.0, lot_size=250, strike=2800, symbol="RELIANCE"):
    return OptionContract(
        symbol=symbol,
        token="12345",
        trading_symbol=f"{symbol}09APR{strike}CE",
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
    strategy="earnings_play",
    entry_minutes_ago=10,
    symbol="RELIANCE",
):
    entry_time = datetime.now() - timedelta(minutes=entry_minutes_ago)
    return Position(
        symbol=symbol,
        trading_symbol=f"{symbol}09APR2800CE",
        token="12345",
        exchange="NFO",
        option_type=OptionType.CE,
        strike=2800,
        expiry=date(2026, 4, 16),
        side=OrderSide.BUY,
        quantity=250,
        lot_size=250,
        entry_price=entry_price,
        entry_time=entry_time,
        current_price=entry_price,
        high_price=high_price,
        stop_loss=stop_loss,
        target=target,
        strategy=strategy,
    )


def trading_time(hour=10, minute=30):
    """Wednesday — not an expiry day for any index."""
    return datetime(2026, 4, 8, hour, minute)


# ===== Earnings Play Tests =====

class TestEarningsPlay:
    def setup_method(self):
        self.strategy = EarningsPlayStrategy()

    def test_rejects_index_symbol(self):
        sig = make_signal(symbol="NIFTY")
        self.strategy.set_earnings("NIFTY", date(2026, 4, 10), date(2026, 4, 8))
        assert not self.strategy.should_enter(sig, 24500, trading_time())

    def test_rejects_no_earnings_set(self):
        sig = make_signal(symbol="RELIANCE")
        assert not self.strategy.should_enter(sig, 2800, trading_time())

    def test_rejects_neutral_signal(self):
        sig = make_signal(direction=Direction.NEUTRAL, symbol="RELIANCE")
        self.strategy.set_earnings("RELIANCE", date(2026, 4, 9), date(2026, 4, 8))
        assert not self.strategy.should_enter(sig, 2800, trading_time())

    def test_rejects_low_score(self):
        sig = make_signal(score=50, symbol="RELIANCE")
        self.strategy.set_earnings("RELIANCE", date(2026, 4, 9), date(2026, 4, 8))
        assert not self.strategy.should_enter(sig, 2800, trading_time())

    def test_pre_earnings_entry(self):
        sig = make_signal(score=75, symbol="RELIANCE")
        # D-1
        self.strategy.set_earnings("RELIANCE", date(2026, 4, 9), date(2026, 4, 8))
        assert self.strategy.should_enter(sig, 2800, trading_time())
        assert self.strategy._current_phase == EarningsPhase.PRE_EARNINGS

    def test_pre_earnings_d3(self):
        sig = make_signal(score=75, symbol="TCS")
        # D-3
        self.strategy.set_earnings("TCS", date(2026, 4, 11), date(2026, 4, 8))
        assert self.strategy.should_enter(sig, 3500, trading_time())

    def test_pre_earnings_rejects_d4(self):
        sig = make_signal(score=75, symbol="RELIANCE")
        # D-4 — too far
        self.strategy.set_earnings("RELIANCE", date(2026, 4, 12), date(2026, 4, 8))
        assert not self.strategy.should_enter(sig, 2800, trading_time())

    def test_post_earnings_entry(self):
        sig = make_signal(
            score=80, symbol="RELIANCE",
            components=[ComponentScore("price_action", Direction.BULLISH, 80, 0.15, "Gap up sustained 3%")],
        )
        # D-day
        self.strategy.set_earnings("RELIANCE", date(2026, 4, 8), date(2026, 4, 8))
        assert self.strategy.should_enter(sig, 2800, trading_time(10, 30))
        assert self.strategy._current_phase == EarningsPhase.POST_EARNINGS

    def test_post_earnings_rejects_before_10am(self):
        sig = make_signal(
            score=80, symbol="RELIANCE",
            components=[ComponentScore("price_action", Direction.BULLISH, 80, 0.15, "Gap up 3%")],
        )
        self.strategy.set_earnings("RELIANCE", date(2026, 4, 8), date(2026, 4, 8))
        assert not self.strategy.should_enter(sig, 2800, trading_time(9, 30))

    def test_post_earnings_rejects_no_gap(self):
        sig = make_signal(score=80, symbol="RELIANCE")
        self.strategy.set_earnings("RELIANCE", date(2026, 4, 8), date(2026, 4, 8))
        assert not self.strategy.should_enter(sig, 2800, trading_time(10, 30))

    def test_post_earnings_rejects_low_score(self):
        sig = make_signal(
            score=70, symbol="RELIANCE",
            components=[ComponentScore("price_action", Direction.BULLISH, 80, 0.15, "Gap up 3%")],
        )
        self.strategy.set_earnings("RELIANCE", date(2026, 4, 8), date(2026, 4, 8))
        assert not self.strategy.should_enter(sig, 2800, trading_time(10, 30))

    def test_post_earnings_rejects_tech_mismatch(self):
        sig = make_signal(
            score=80, symbol="RELIANCE",
            technical_direction=Direction.BEARISH,
            components=[ComponentScore("price_action", Direction.BULLISH, 80, 0.15, "Gap up 3%")],
        )
        self.strategy.set_earnings("RELIANCE", date(2026, 4, 8), date(2026, 4, 8))
        assert not self.strategy.should_enter(sig, 2800, trading_time(10, 30))

    def test_create_setup_pre_earnings(self):
        sig = make_signal(score=75, symbol="RELIANCE")
        self.strategy.set_earnings("RELIANCE", date(2026, 4, 9), date(2026, 4, 8))
        self.strategy.should_enter(sig, 2800, trading_time())

        contract = make_contract(ltp=100)
        setup = self.strategy.create_setup(sig, contract, 2800, 100000)
        assert setup is not None
        assert setup.strategy_name == "earnings_play"
        assert setup.stop_loss == 85.0  # 100 * (1 - 15/100)
        assert setup.target == 125.0    # 100 * (1 + 25/100)

    def test_create_setup_post_earnings(self):
        sig = make_signal(
            score=80, symbol="RELIANCE",
            components=[ComponentScore("price_action", Direction.BULLISH, 80, 0.15, "Gap up 3%")],
        )
        self.strategy.set_earnings("RELIANCE", date(2026, 4, 8), date(2026, 4, 8))
        self.strategy.should_enter(sig, 2800, trading_time(10, 30))

        contract = make_contract(ltp=100)
        setup = self.strategy.create_setup(sig, contract, 2800, 100000)
        assert setup is not None
        assert setup.stop_loss == 65.0   # 100 * (1 - 35/100)
        assert setup.target == 160.0     # 100 * (1 + 60/100)

    def test_create_setup_rejects_zero_ltp(self):
        sig = make_signal(symbol="RELIANCE")
        self.strategy._current_phase = EarningsPhase.PRE_EARNINGS
        contract = make_contract(ltp=0)
        assert self.strategy.create_setup(sig, contract, 2800, 100000) is None

    def test_exit_sl_hit(self):
        pos = make_position(entry_price=100, stop_loss=85, target=125)
        result = self.strategy.should_exit(pos, 80, 2800, trading_time())
        assert result.should_exit
        assert result.reason == "SL_HIT"

    def test_exit_target_hit(self):
        pos = make_position(entry_price=100, stop_loss=85, target=125)
        result = self.strategy.should_exit(pos, 130, 2800, trading_time())
        assert result.should_exit
        assert result.reason == "TARGET_HIT"

    def test_exit_hard_cutoff(self):
        pos = make_position(entry_price=100, stop_loss=85, target=125)
        result = self.strategy.should_exit(pos, 110, 2800, trading_time(15, 1))
        assert result.should_exit
        assert result.reason == "HARD_CUTOFF"

    def test_exit_time_exit(self):
        pos = make_position(entry_price=100, stop_loss=85, target=125, entry_minutes_ago=100)
        result = self.strategy.should_exit(pos, 101, 2800, trading_time())
        assert result.should_exit
        assert "TIME_EXIT" in result.reason

    def test_trailing_sl(self):
        pos = make_position(entry_price=100, high_price=125, stop_loss=85, target=160)
        self.strategy.should_exit(pos, 120, 2800, trading_time())
        # 20% profit activated trailing → lock 50% of (125-100)=12.5 → SL at 112.5
        assert pos.stop_loss == 112.5

    def test_clear_earnings(self):
        self.strategy.set_earnings("RELIANCE", date(2026, 4, 9), date(2026, 4, 8))
        self.strategy.clear_earnings("RELIANCE")
        assert self.strategy._get_phase("RELIANCE") == EarningsPhase.NO_EARNINGS

    def test_select_expiry_pre_earnings(self):
        self.strategy.set_earnings("RELIANCE", date(2026, 4, 9), date(2026, 4, 8))
        self.strategy._current_phase = EarningsPhase.PRE_EARNINGS
        exp = self.strategy.select_expiry("RELIANCE", date(2026, 4, 8))
        # Monthly expiry — last Thursday of April 2026 = Apr 30
        assert exp == date(2026, 4, 30)


# ===== Volume Profile Tests =====

class TestVolumeProfile:
    def setup_method(self):
        self.strategy = VolumeProfileStrategy()

    def test_rejects_no_profile_data(self):
        sig = make_signal(symbol="NIFTY")
        assert not self.strategy.should_enter(sig, 24500, trading_time())

    def test_rejects_low_score(self):
        sig = make_signal(score=50, symbol="NIFTY")
        self.strategy.set_volume_profile(24500, 24600, 24400)
        assert not self.strategy.should_enter(sig, 24500, trading_time())

    def test_rejects_before_entry_start(self):
        sig = make_signal(symbol="NIFTY")
        self.strategy.set_volume_profile(24500, 24600, 24400)
        assert not self.strategy.should_enter(sig, 24500, trading_time(9, 30))

    def test_bounce_at_val_bullish(self):
        sig = make_signal(direction=Direction.BULLISH, symbol="NIFTY")
        self.strategy.set_volume_profile(poc=24500, vah=24600, val=24400)
        # Spot at VAL (support) + bullish = bounce
        assert self.strategy.should_enter(sig, 24400, trading_time())
        assert self.strategy._last_action == VPAction.BOUNCE

    def test_bounce_at_vah_bearish(self):
        sig = make_signal(direction=Direction.BEARISH, symbol="NIFTY")
        self.strategy.set_volume_profile(poc=24500, vah=24600, val=24400)
        # Spot at VAH (resistance) + bearish = bounce
        assert self.strategy.should_enter(sig, 24600, trading_time())
        assert self.strategy._last_action == VPAction.BOUNCE

    def test_bounce_at_poc(self):
        sig = make_signal(direction=Direction.BULLISH, symbol="NIFTY")
        self.strategy.set_volume_profile(poc=24500, vah=24600, val=24400)
        assert self.strategy.should_enter(sig, 24500, trading_time())
        assert self.strategy._last_action == VPAction.BOUNCE

    def test_break_above_vah(self):
        sig = make_signal(direction=Direction.BULLISH, symbol="NIFTY")
        self.strategy.set_volume_profile(poc=24500, vah=24600, val=24400)
        # Spot well above VAH (beyond proximity)
        assert self.strategy.should_enter(sig, 24700, trading_time())
        assert self.strategy._last_action == VPAction.BREAK

    def test_break_below_val(self):
        sig = make_signal(direction=Direction.BEARISH, symbol="NIFTY")
        self.strategy.set_volume_profile(poc=24500, vah=24600, val=24400)
        # Spot well below VAL
        assert self.strategy.should_enter(sig, 24300, trading_time())
        assert self.strategy._last_action == VPAction.BREAK

    def test_no_entry_wrong_direction_at_val(self):
        sig = make_signal(direction=Direction.BEARISH, symbol="NIFTY")
        self.strategy.set_volume_profile(poc=24500, vah=24600, val=24400)
        # Bearish at VAL (support) — not a bounce setup
        # VAL bounce requires BULLISH; the signal is BEARISH
        # However spot >= val and near val, but direction is bearish
        # Check: bounce at VAL only triggers for BULLISH
        assert not self.strategy.should_enter(sig, 24400, trading_time())

    def test_create_setup_bounce(self):
        sig = make_signal(symbol="NIFTY")
        self.strategy.set_volume_profile(poc=24500, vah=24600, val=24400)
        self.strategy.should_enter(sig, 24500, trading_time())

        contract = make_contract(ltp=100, lot_size=65, strike=24500, symbol="NIFTY")
        setup = self.strategy.create_setup(sig, contract, 24500, 100000)
        assert setup is not None
        assert setup.stop_loss == 75.0   # 100 * (1 - 25/100) bounce SL
        assert setup.target == 135.0     # 100 * (1 + 35/100) bounce target

    def test_create_setup_break(self):
        sig = make_signal(direction=Direction.BULLISH, symbol="NIFTY")
        self.strategy.set_volume_profile(poc=24500, vah=24600, val=24400)
        self.strategy.should_enter(sig, 24700, trading_time())

        contract = make_contract(ltp=100, lot_size=65, strike=24700, symbol="NIFTY")
        setup = self.strategy.create_setup(sig, contract, 24700, 100000)
        assert setup is not None
        assert setup.stop_loss == 70.0   # 100 * (1 - 30/100) break SL
        assert setup.target == 145.0     # 100 * (1 + 45/100) break target

    def test_exit_sl_hit(self):
        pos = make_position(entry_price=100, stop_loss=75, target=135, symbol="NIFTY")
        result = self.strategy.should_exit(pos, 70, 24500, trading_time())
        assert result.should_exit
        assert result.reason == "SL_HIT"

    def test_exit_target_hit(self):
        pos = make_position(entry_price=100, stop_loss=75, target=135, symbol="NIFTY")
        result = self.strategy.should_exit(pos, 140, 24500, trading_time())
        assert result.should_exit
        assert result.reason == "TARGET_HIT"

    def test_exit_hard_cutoff(self):
        pos = make_position(entry_price=100, stop_loss=75, target=135, symbol="NIFTY")
        result = self.strategy.should_exit(pos, 110, 24500, trading_time(15, 1))
        assert result.should_exit
        assert result.reason == "HARD_CUTOFF"


# ===== Bollinger Squeeze Tests =====

class TestBollingerSqueeze:
    def setup_method(self):
        self.strategy = BollingerSqueezeStrategy()

    def test_rejects_no_bb_data(self):
        sig = make_signal(symbol="NIFTY")
        assert not self.strategy.should_enter(sig, 24500, trading_time())

    def test_rejects_low_score(self):
        sig = make_signal(score=50, symbol="NIFTY")
        self.strategy.set_bb_data(24600, 24400, 24500, 0.06, 0.03)
        assert not self.strategy.should_enter(sig, 24650, trading_time())

    def test_rejects_no_squeeze(self):
        sig = make_signal(symbol="NIFTY")
        # prev_bandwidth > max_bandwidth (0.04) → no squeeze was present
        self.strategy.set_bb_data(24600, 24400, 24500, 0.08, 0.06)
        assert not self.strategy.should_enter(sig, 24650, trading_time())

    def test_rejects_insufficient_expansion(self):
        sig = make_signal(symbol="NIFTY")
        # prev=0.03, current=0.04, ratio=1.33 < 1.5
        self.strategy.set_bb_data(24600, 24400, 24500, 0.04, 0.03)
        assert not self.strategy.should_enter(sig, 24650, trading_time())

    def test_bullish_breakout(self):
        sig = make_signal(direction=Direction.BULLISH, symbol="NIFTY")
        # Squeeze: prev=0.03, current=0.06 → expansion=2.0x
        self.strategy.set_bb_data(upper=24600, lower=24400, middle=24500,
                                   bandwidth=0.06, prev_bandwidth=0.03)
        # Spot above upper band
        assert self.strategy.should_enter(sig, 24650, trading_time())

    def test_bearish_breakout(self):
        sig = make_signal(direction=Direction.BEARISH, symbol="NIFTY")
        self.strategy.set_bb_data(upper=24600, lower=24400, middle=24500,
                                   bandwidth=0.06, prev_bandwidth=0.03)
        # Spot below lower band
        assert self.strategy.should_enter(sig, 24350, trading_time())

    def test_rejects_bullish_below_upper(self):
        sig = make_signal(direction=Direction.BULLISH, symbol="NIFTY")
        self.strategy.set_bb_data(upper=24600, lower=24400, middle=24500,
                                   bandwidth=0.06, prev_bandwidth=0.03)
        # Spot NOT above upper band
        assert not self.strategy.should_enter(sig, 24550, trading_time())

    def test_rejects_bearish_above_lower(self):
        sig = make_signal(direction=Direction.BEARISH, symbol="NIFTY")
        self.strategy.set_bb_data(upper=24600, lower=24400, middle=24500,
                                   bandwidth=0.06, prev_bandwidth=0.03)
        # Spot NOT below lower band
        assert not self.strategy.should_enter(sig, 24450, trading_time())

    def test_create_setup(self):
        sig = make_signal(direction=Direction.BULLISH, symbol="NIFTY")
        self.strategy.set_bb_data(upper=24600, lower=24400, middle=24500,
                                   bandwidth=0.06, prev_bandwidth=0.03)
        self.strategy.should_enter(sig, 24650, trading_time())

        contract = make_contract(ltp=100, lot_size=65, strike=24700, symbol="NIFTY")
        setup = self.strategy.create_setup(sig, contract, 24650, 100000)
        assert setup is not None
        assert setup.stop_loss == 70.0   # 100 * (1 - 30/100)
        assert setup.target == 145.0     # 100 * (1 + 45/100)
        assert "Squeeze" in setup.reason

    def test_create_setup_rejects_high_premium(self):
        sig = make_signal(symbol="NIFTY")
        self.strategy._last_action = None
        contract = make_contract(ltp=600, lot_size=65, symbol="NIFTY")
        assert self.strategy.create_setup(sig, contract, 24650, 100000) is None

    def test_exit_sl_hit(self):
        pos = make_position(entry_price=100, stop_loss=70, target=145, symbol="NIFTY")
        result = self.strategy.should_exit(pos, 65, 24500, trading_time())
        assert result.should_exit
        assert result.reason == "SL_HIT"

    def test_exit_target_hit(self):
        pos = make_position(entry_price=100, stop_loss=70, target=145, symbol="NIFTY")
        result = self.strategy.should_exit(pos, 150, 24500, trading_time())
        assert result.should_exit
        assert result.reason == "TARGET_HIT"

    def test_exit_time_exit(self):
        pos = make_position(entry_price=100, stop_loss=70, target=145, symbol="NIFTY",
                           entry_minutes_ago=80)
        result = self.strategy.should_exit(pos, 101, 24500, trading_time())
        assert result.should_exit
        assert "TIME_EXIT" in result.reason

    def test_trailing_sl(self):
        pos = make_position(entry_price=100, high_price=125, stop_loss=70, target=145, symbol="NIFTY")
        self.strategy.should_exit(pos, 120, 24500, trading_time())
        # 25% profit → trailing active → lock 50% of (125-100)=12.5 → SL=112.5
        assert pos.stop_loss == 112.5


# ===== Fibonacci Retracement Tests =====

class TestFibonacciRetracement:
    def setup_method(self):
        self.strategy = FibonacciRetracementStrategy()

    def test_rejects_no_swing_points(self):
        sig = make_signal(symbol="NIFTY")
        assert not self.strategy.should_enter(sig, 24500, trading_time())

    def test_rejects_low_score(self):
        sig = make_signal(score=50, symbol="NIFTY")
        self.strategy.set_swing_points(25000, 24000)
        assert not self.strategy.should_enter(sig, 24618, trading_time())

    def test_rejects_invalid_swing(self):
        sig = make_signal(symbol="NIFTY")
        self.strategy.set_swing_points(24000, 25000)  # high < low
        assert not self.strategy.should_enter(sig, 24500, trading_time())

    def test_bullish_at_382_fib(self):
        sig = make_signal(direction=Direction.BULLISH, symbol="NIFTY")
        # Swing high=25000, low=24000 → range=1000
        # 38.2% retracement = 25000 - 1000*0.382 = 24618
        self.strategy.set_swing_points(25000, 24000)
        assert self.strategy.should_enter(sig, 24618, trading_time())

    def test_bullish_at_50_fib(self):
        sig = make_signal(direction=Direction.BULLISH, symbol="NIFTY")
        self.strategy.set_swing_points(25000, 24000)
        # 50% retracement = 24500
        assert self.strategy.should_enter(sig, 24500, trading_time())

    def test_bullish_at_618_fib(self):
        sig = make_signal(direction=Direction.BULLISH, symbol="NIFTY")
        self.strategy.set_swing_points(25000, 24000)
        # 61.8% retracement = 25000 - 1000*0.618 = 24382
        assert self.strategy.should_enter(sig, 24382, trading_time())

    def test_rejects_price_outside_range(self):
        sig = make_signal(direction=Direction.BULLISH, symbol="NIFTY")
        self.strategy.set_swing_points(25000, 24000)
        # Below swing low — not a pullback anymore
        assert not self.strategy.should_enter(sig, 23900, trading_time())

    def test_rejects_price_above_swing_high(self):
        sig = make_signal(direction=Direction.BULLISH, symbol="NIFTY")
        self.strategy.set_swing_points(25000, 24000)
        assert not self.strategy.should_enter(sig, 25100, trading_time())

    def test_rejects_not_near_fib(self):
        sig = make_signal(direction=Direction.BULLISH, symbol="NIFTY")
        self.strategy.set_swing_points(25000, 24000)
        # 24800 is not near any Fib level
        assert not self.strategy.should_enter(sig, 24800, trading_time())

    def test_rejects_tech_mismatch(self):
        sig = make_signal(direction=Direction.BULLISH, symbol="NIFTY",
                         technical_direction=Direction.BEARISH)
        self.strategy.set_swing_points(25000, 24000)
        assert not self.strategy.should_enter(sig, 24618, trading_time())

    def test_bearish_at_fib(self):
        sig = make_signal(direction=Direction.BEARISH, symbol="NIFTY")
        self.strategy.set_swing_points(25000, 24000)
        # Bearish: price rallying up to 38.2% Fib level in downtrend
        assert self.strategy.should_enter(sig, 24618, trading_time())

    def test_create_setup(self):
        sig = make_signal(direction=Direction.BULLISH, symbol="NIFTY")
        self.strategy.set_swing_points(25000, 24000)
        self.strategy.should_enter(sig, 24618, trading_time())

        contract = make_contract(ltp=100, lot_size=65, strike=24600, symbol="NIFTY")
        setup = self.strategy.create_setup(sig, contract, 24618, 100000)
        assert setup is not None
        assert setup.stop_loss == 75.0   # 100 * (1 - 25/100)
        assert setup.target == 140.0     # 100 * (1 + 40/100)
        assert "Fib" in setup.reason

    def test_exit_sl_hit(self):
        pos = make_position(entry_price=100, stop_loss=75, target=140, symbol="NIFTY")
        result = self.strategy.should_exit(pos, 70, 24500, trading_time())
        assert result.should_exit
        assert result.reason == "SL_HIT"

    def test_exit_target_hit(self):
        pos = make_position(entry_price=100, stop_loss=75, target=140, symbol="NIFTY")
        result = self.strategy.should_exit(pos, 145, 24500, trading_time())
        assert result.should_exit
        assert result.reason == "TARGET_HIT"

    def test_trailing_sl(self):
        pos = make_position(entry_price=100, high_price=125, stop_loss=75, target=140, symbol="NIFTY")
        self.strategy.should_exit(pos, 120, 24500, trading_time())
        assert pos.stop_loss == 112.5


# ===== Sector Rotation Tests =====

class TestSectorRotation:
    def setup_method(self):
        self.strategy = SectorRotationStrategy()

    def test_rejects_index(self):
        sig = make_signal(symbol="NIFTY")
        self.strategy.set_sector_data("NIFTY", "NIFTY", 1.1, 1.1)
        assert not self.strategy.should_enter(sig, 24500, trading_time())

    def test_rejects_no_sector_data(self):
        sig = make_signal(symbol="TCS")
        assert not self.strategy.should_enter(sig, 3500, trading_time())

    def test_rejects_low_score(self):
        sig = make_signal(score=50, symbol="TCS")
        self.strategy.set_sector_data("TCS", "NIFTY IT", 1.1, 1.1)
        assert not self.strategy.should_enter(sig, 3500, trading_time())

    def test_rejects_before_entry_start(self):
        sig = make_signal(symbol="TCS")
        self.strategy.set_sector_data("TCS", "NIFTY IT", 1.1, 1.1)
        assert not self.strategy.should_enter(sig, 3500, trading_time(9, 30))

    def test_bullish_strong_sector_strong_stock(self):
        sig = make_signal(direction=Direction.BULLISH, symbol="TCS")
        self.strategy.set_sector_data("TCS", "NIFTY IT", 1.10, 1.05)
        assert self.strategy.should_enter(sig, 3500, trading_time())

    def test_rejects_bullish_weak_sector(self):
        sig = make_signal(direction=Direction.BULLISH, symbol="TCS")
        self.strategy.set_sector_data("TCS", "NIFTY IT", 1.02, 1.10)
        assert not self.strategy.should_enter(sig, 3500, trading_time())

    def test_rejects_bullish_weak_stock(self):
        sig = make_signal(direction=Direction.BULLISH, symbol="TCS")
        self.strategy.set_sector_data("TCS", "NIFTY IT", 1.10, 1.01)
        assert not self.strategy.should_enter(sig, 3500, trading_time())

    def test_bearish_weak_sector_weak_stock(self):
        sig = make_signal(direction=Direction.BEARISH, symbol="TATASTEEL")
        # 1/1.05 ≈ 0.952 → sector must be <= 0.952
        self.strategy.set_sector_data("TATASTEEL", "NIFTY METAL", 0.90, 0.92)
        assert self.strategy.should_enter(sig, 1200, trading_time())

    def test_rejects_bearish_strong_sector(self):
        sig = make_signal(direction=Direction.BEARISH, symbol="TATASTEEL")
        self.strategy.set_sector_data("TATASTEEL", "NIFTY METAL", 0.98, 0.90)
        assert not self.strategy.should_enter(sig, 1200, trading_time())

    def test_create_setup(self):
        sig = make_signal(direction=Direction.BULLISH, symbol="TCS")
        self.strategy.set_sector_data("TCS", "NIFTY IT", 1.10, 1.05)
        self.strategy.should_enter(sig, 3500, trading_time())

        contract = make_contract(ltp=100, lot_size=150, strike=3500, symbol="TCS")
        setup = self.strategy.create_setup(sig, contract, 3500, 100000)
        assert setup is not None
        assert setup.stop_loss == 72.0   # 100 * (1 - 28/100)
        assert setup.target == 140.0     # 100 * (1 + 40/100)
        assert "Sector Rotation" in setup.reason

    def test_exit_sl_hit(self):
        pos = make_position(entry_price=100, stop_loss=72, target=140, symbol="TCS")
        result = self.strategy.should_exit(pos, 70, 3500, trading_time())
        assert result.should_exit
        assert result.reason == "SL_HIT"

    def test_exit_target_hit(self):
        pos = make_position(entry_price=100, stop_loss=72, target=140, symbol="TCS")
        result = self.strategy.should_exit(pos, 145, 3500, trading_time())
        assert result.should_exit
        assert result.reason == "TARGET_HIT"

    def test_exit_hard_cutoff(self):
        pos = make_position(entry_price=100, stop_loss=72, target=140, symbol="TCS")
        result = self.strategy.should_exit(pos, 110, 3500, trading_time(15, 1))
        assert result.should_exit
        assert result.reason == "HARD_CUTOFF"

    def test_exit_time_exit(self):
        pos = make_position(entry_price=100, stop_loss=72, target=140, symbol="TCS",
                           entry_minutes_ago=100)
        result = self.strategy.should_exit(pos, 101, 3500, trading_time())
        assert result.should_exit
        assert "TIME_EXIT" in result.reason

    def test_select_expiry_monthly(self):
        exp = self.strategy.select_expiry("TCS", date(2026, 4, 8))
        # Monthly expiry — last Thursday of April 2026 = Apr 30
        assert exp == date(2026, 4, 30)
