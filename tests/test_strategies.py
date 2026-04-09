"""Tests for all trading strategies."""

from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd
import pytest

from src.broker.models import OptionContract, OptionType, OrderSide, Position
from src.signals.models import ComponentScore, Direction, Signal, SignalStrength
from src.strategies.base import ExitSignal, TradeSetup
from src.strategies.gap_and_go import GapAndGoStrategy
from src.strategies.momentum_buy import MomentumBuyStrategy
from src.strategies.multi_timeframe import MultiTimeframeStrategy, get_timeframe_direction
from src.strategies.rsi_divergence import RSIDivergenceStrategy, detect_rsi_divergence
from src.strategies.scalping import ScalpingStrategy
from src.strategies.straddle_breakout import StraddleBreakoutStrategy
from src.strategies.vwap_pullback import VWAPPullbackStrategy


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


def make_contract(ltp=100.0, lot_size=50, strike=22500):
    return OptionContract(
        symbol="NIFTY",
        token="12345",
        trading_symbol="NIFTY09APR22500CE",
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
    current_price=100.0,
    high_price=100.0,
    stop_loss=70.0,
    target=140.0,
    strategy="momentum_buy",
    entry_minutes_ago=10,
):
    entry_time = datetime.now() - timedelta(minutes=entry_minutes_ago)
    return Position(
        symbol="NIFTY",
        trading_symbol="NIFTY09APR22500CE",
        token="12345",
        exchange="NFO",
        option_type=OptionType.CE,
        strike=22500,
        expiry=date(2026, 4, 16),
        side=OrderSide.BUY,
        quantity=50,
        lot_size=50,
        entry_price=entry_price,
        entry_time=entry_time,
        current_price=current_price,
        high_price=high_price,
        stop_loss=stop_loss,
        target=target,
        strategy=strategy,
    )


def trading_time(hour=10, minute=30):
    return datetime(2026, 4, 9, hour, minute)


# ===== VWAP Pullback Tests =====

class TestVWAPPullback:
    def setup_method(self):
        self.strategy = VWAPPullbackStrategy()

    def test_rejects_low_score(self):
        sig = make_signal(score=60.0)
        assert not self.strategy.should_enter(sig, 22000, trading_time())

    def test_rejects_before_entry_start(self):
        sig = make_signal(score=80.0)
        assert not self.strategy.should_enter(sig, 22000, trading_time(9, 30))

    def test_rejects_after_entry_end(self):
        sig = make_signal(score=80.0)
        assert not self.strategy.should_enter(sig, 22000, trading_time(14, 30))

    def test_rejects_no_tech_alignment(self):
        sig = make_signal(
            direction=Direction.BULLISH,
            technical_direction=Direction.BEARISH,
        )
        assert not self.strategy.should_enter(sig, 22000, trading_time())

    def test_rejects_no_vwap_signal(self):
        sig = make_signal(
            components=[
                ComponentScore("technical", Direction.BULLISH, 80, 0.4, "EMA bullish supertrend bullish"),
            ],
        )
        assert not self.strategy.should_enter(sig, 22000, trading_time())

    def test_enters_with_vwap_and_supertrend(self):
        sig = make_signal(
            score=80.0,
            direction=Direction.BULLISH,
            components=[
                ComponentScore("price_action", Direction.BULLISH, 75, 0.15, "Price above VWAP support"),
                ComponentScore("technical", Direction.BULLISH, 80, 0.4, "EMA bullish SuperTrend bullish"),
            ],
        )
        assert self.strategy.should_enter(sig, 22000, trading_time())

    def test_create_setup(self):
        sig = make_signal()
        contract = make_contract(ltp=100.0, lot_size=50)
        setup = self.strategy.create_setup(sig, contract, 22000, 100000)
        assert setup is not None
        assert setup.strategy_name == "vwap_pullback"
        assert setup.stop_loss == round(100 * 0.75, 2)
        assert setup.target == round(100 * 1.35, 2)

    def test_create_setup_rejects_zero_ltp(self):
        sig = make_signal()
        contract = make_contract(ltp=0.0)
        assert self.strategy.create_setup(sig, contract, 22000, 100000) is None

    def test_exit_sl_hit(self):
        pos = make_position(entry_price=100, stop_loss=75, target=135, strategy="vwap_pullback")
        exit_sig = self.strategy.should_exit(pos, 74.0, 22000, trading_time())
        assert exit_sig.should_exit
        assert "SL_HIT" in exit_sig.reason

    def test_exit_target_hit(self):
        pos = make_position(entry_price=100, stop_loss=75, target=135, strategy="vwap_pullback")
        exit_sig = self.strategy.should_exit(pos, 136.0, 22000, trading_time())
        assert exit_sig.should_exit
        assert "TARGET_HIT" in exit_sig.reason

    def test_exit_time_based_losing(self):
        pos = make_position(
            entry_price=100, stop_loss=75, target=135,
            strategy="vwap_pullback", entry_minutes_ago=60,
        )
        exit_sig = self.strategy.should_exit(pos, 95.0, 22000, trading_time())
        assert exit_sig.should_exit
        assert "TIME_EXIT" in exit_sig.reason

    def test_no_exit_time_based_profitable(self):
        pos = make_position(
            entry_price=100, stop_loss=75, target=135,
            strategy="vwap_pullback", entry_minutes_ago=60,
        )
        exit_sig = self.strategy.should_exit(pos, 110.0, 22000, trading_time())
        assert not exit_sig.should_exit


# ===== Gap and Go Tests =====

class TestGapAndGo:
    def setup_method(self):
        self.strategy = GapAndGoStrategy()

    def test_rejects_low_score(self):
        sig = make_signal(score=60.0)
        assert not self.strategy.should_enter(sig, 22000, trading_time())

    def test_rejects_before_entry_start(self):
        sig = make_signal(score=80.0)
        assert not self.strategy.should_enter(sig, 22000, trading_time(9, 20))

    def test_rejects_after_entry_end(self):
        sig = make_signal(score=80.0)
        assert not self.strategy.should_enter(sig, 22000, trading_time(12, 0))

    def test_rejects_no_gap_sustained(self):
        sig = make_signal(
            score=80.0,
            direction=Direction.BULLISH,
            components=[
                ComponentScore("price_action", Direction.BULLISH, 75, 0.15, "Price breakout above PDH"),
            ],
        )
        assert not self.strategy.should_enter(sig, 22000, trading_time())

    def test_rejects_tech_mismatch(self):
        sig = make_signal(
            score=80.0,
            direction=Direction.BULLISH,
            technical_direction=Direction.BEARISH,
            components=[
                ComponentScore("price_action", Direction.BULLISH, 75, 0.15, "Gap up sustained above VWAP"),
            ],
        )
        assert not self.strategy.should_enter(sig, 22000, trading_time())

    def test_enters_bullish_gap(self):
        sig = make_signal(
            score=80.0,
            direction=Direction.BULLISH,
            components=[
                ComponentScore("price_action", Direction.BULLISH, 75, 0.15, "Gap up sustained above open"),
            ],
        )
        assert self.strategy.should_enter(sig, 22000, trading_time())

    def test_enters_bearish_gap(self):
        sig = make_signal(
            score=80.0,
            direction=Direction.BEARISH,
            technical_direction=Direction.BEARISH,
            components=[
                ComponentScore("price_action", Direction.BEARISH, 75, 0.15, "Gap down sustained below open"),
            ],
        )
        assert self.strategy.should_enter(sig, 22000, trading_time())

    def test_create_setup(self):
        sig = make_signal()
        contract = make_contract(ltp=150.0)
        setup = self.strategy.create_setup(sig, contract, 22000, 100000)
        assert setup is not None
        assert setup.strategy_name == "gap_and_go"
        assert setup.target == round(150 * 1.45, 2)

    def test_exit_hard_cutoff(self):
        pos = make_position(strategy="gap_and_go")
        exit_sig = self.strategy.should_exit(pos, 110.0, 22000, trading_time(15, 5))
        assert exit_sig.should_exit
        assert "HARD_CUTOFF" in exit_sig.reason


# ===== RSI Divergence Tests =====

class TestRSIDivergence:
    def setup_method(self):
        self.strategy = RSIDivergenceStrategy()

    def test_detect_divergence_insufficient_data(self):
        df = pd.DataFrame({"close": [100.0] * 10})
        direction, desc = detect_rsi_divergence(df)
        assert direction == Direction.NEUTRAL
        assert "Insufficient" in desc

    def test_detect_divergence_no_divergence(self):
        # Steady uptrend — no divergence
        prices = list(range(100, 160))
        df = pd.DataFrame({"close": prices})
        direction, desc = detect_rsi_divergence(df)
        assert direction == Direction.NEUTRAL

    def test_rejects_low_score(self):
        sig = make_signal(score=50.0)
        assert not self.strategy.should_enter(sig, 22000, trading_time())

    def test_rejects_no_divergence_cached(self):
        sig = make_signal(score=75.0)
        # No divergence cached
        assert not self.strategy.should_enter(sig, 22000, trading_time())

    def test_rejects_direction_mismatch(self):
        sig = make_signal(score=75.0, direction=Direction.BULLISH)
        self.strategy._last_divergence["NIFTY"] = (Direction.BEARISH, "Bearish div")
        assert not self.strategy.should_enter(sig, 22000, trading_time())

    def test_enters_with_matching_divergence(self):
        sig = make_signal(score=75.0, direction=Direction.BULLISH)
        self.strategy._last_divergence["NIFTY"] = (Direction.BULLISH, "Bullish RSI divergence")
        assert self.strategy.should_enter(sig, 22000, trading_time())

    def test_rejects_opposing_oi(self):
        sig = make_signal(
            score=75.0,
            direction=Direction.BULLISH,
            oi_direction=Direction.BEARISH,
        )
        self.strategy._last_divergence["NIFTY"] = (Direction.BULLISH, "Bullish RSI divergence")
        assert not self.strategy.should_enter(sig, 22000, trading_time())

    def test_create_setup_contrarian_sizing(self):
        sig = make_signal()
        contract = make_contract(ltp=100.0, lot_size=50)
        self.strategy._last_divergence["NIFTY"] = (Direction.BULLISH, "Bullish divergence")
        setup = self.strategy.create_setup(sig, contract, 22000, 100000)
        assert setup is not None
        assert setup.strategy_name == "rsi_divergence"
        # 1.5% risk for contrarian play
        # risk_amount = 100000 * 0.015 = 1500
        # SL = 75, risk_per_lot = 25 * 50 = 1250
        # lots = 1500 / 1250 = 1
        assert setup.quantity == 50  # 1 lot

    def test_create_setup_rejects_high_premium(self):
        sig = make_signal()
        contract = make_contract(ltp=500.0)
        assert self.strategy.create_setup(sig, contract, 22000, 100000) is None

    def test_exit_time_based_losing(self):
        pos = make_position(
            entry_price=100, stop_loss=75, target=140,
            strategy="rsi_divergence", entry_minutes_ago=100,
        )
        exit_sig = self.strategy.should_exit(pos, 90.0, 22000, trading_time())
        assert exit_sig.should_exit
        assert "TIME_EXIT" in exit_sig.reason


# ===== Scalping Tests =====

class TestScalping:
    def setup_method(self):
        self.strategy = ScalpingStrategy()

    def test_rejects_low_score(self):
        sig = make_signal(score=70.0)
        assert not self.strategy.should_enter(sig, 22000, trading_time())

    def test_rejects_non_strong(self):
        sig = make_signal(score=85.0, strength=SignalStrength.MODERATE)
        assert not self.strategy.should_enter(sig, 22000, trading_time())

    def test_rejects_no_tech_alignment(self):
        sig = make_signal(
            score=85.0,
            direction=Direction.BULLISH,
            technical_direction=Direction.BEARISH,
        )
        assert not self.strategy.should_enter(sig, 22000, trading_time())

    def test_rejects_low_tech_score(self):
        sig = make_signal(score=85.0, technical_score=60.0)
        assert not self.strategy.should_enter(sig, 22000, trading_time())

    def test_rejects_no_volume(self):
        sig = make_signal(
            score=85.0,
            technical_score=75.0,
            components=[
                ComponentScore("technical", Direction.BULLISH, 80, 0.4, "EMA bullish"),
            ],
        )
        assert not self.strategy.should_enter(sig, 22000, trading_time())

    def test_enters_with_volume_surge(self):
        sig = make_signal(
            score=85.0,
            technical_score=75.0,
            components=[
                ComponentScore("technical", Direction.BULLISH, 80, 0.4, "EMA bullish, volume surge confirmed"),
            ],
        )
        assert self.strategy.should_enter(sig, 22000, trading_time())

    def test_enters_with_volume_ok(self):
        sig = make_signal(
            score=85.0,
            technical_score=75.0,
            components=[
                ComponentScore("technical", Direction.BULLISH, 80, 0.4, "EMA bullish, volume ok"),
            ],
        )
        assert self.strategy.should_enter(sig, 22000, trading_time())

    def test_create_setup_higher_risk(self):
        sig = make_signal(score=85.0)
        contract = make_contract(ltp=100.0, lot_size=50)
        setup = self.strategy.create_setup(sig, contract, 22000, 100000)
        assert setup is not None
        assert setup.strategy_name == "scalping"
        # 2.5% risk for scalping
        # risk_amount = 100000 * 0.025 = 2500
        # SL = 85, risk_per_lot = 15 * 50 = 750
        # lots = 2500 / 750 = 3
        assert setup.quantity == 150  # 3 lots

    def test_exit_scalp_timeout(self):
        pos = make_position(
            entry_price=100, stop_loss=85, target=120,
            strategy="scalping", entry_minutes_ago=25,
        )
        exit_sig = self.strategy.should_exit(pos, 100.0, 22000, trading_time())
        assert exit_sig.should_exit
        assert "SCALP_TIMEOUT" in exit_sig.reason

    def test_no_exit_timeout_profitable(self):
        pos = make_position(
            entry_price=100, stop_loss=85, target=120,
            strategy="scalping", entry_minutes_ago=25,
        )
        exit_sig = self.strategy.should_exit(pos, 108.0, 22000, trading_time())
        assert not exit_sig.should_exit

    def test_rejects_premium_too_low(self):
        sig = make_signal()
        contract = make_contract(ltp=5.0)
        assert self.strategy.create_setup(sig, contract, 22000, 100000) is None

    def test_rejects_premium_too_high(self):
        sig = make_signal()
        contract = make_contract(ltp=350.0)
        assert self.strategy.create_setup(sig, contract, 22000, 100000) is None


# ===== Straddle Breakout Tests =====

class TestStraddleBreakout:
    def setup_method(self):
        self.strategy = StraddleBreakoutStrategy()

    def test_rejects_low_score(self):
        sig = make_signal(score=60.0)
        assert not self.strategy.should_enter(sig, 22000, trading_time())

    def test_rejects_before_entry(self):
        sig = make_signal(score=80.0)
        assert not self.strategy.should_enter(sig, 22000, trading_time(9, 30))

    def test_rejects_after_entry(self):
        sig = make_signal(score=80.0)
        assert not self.strategy.should_enter(sig, 22000, trading_time(14, 0))

    def test_rejects_low_agreement(self):
        sig = make_signal(score=80.0, agreement_count=1)
        assert not self.strategy.should_enter(sig, 22000, trading_time())

    def test_rejects_no_range_break(self):
        sig = make_signal(
            score=80.0,
            components=[
                ComponentScore("price_action", Direction.BULLISH, 75, 0.15, "Price above VWAP"),
            ],
        )
        assert not self.strategy.should_enter(sig, 22000, trading_time())

    def test_rejects_low_iv_score(self):
        sig = make_signal(
            score=80.0,
            iv_score=25.0,
            components=[
                ComponentScore("price_action", Direction.BULLISH, 75, 0.15, "ORB breakout above range"),
            ],
        )
        assert not self.strategy.should_enter(sig, 22000, trading_time())

    def test_enters_orb_breakout(self):
        sig = make_signal(
            score=80.0,
            iv_score=50.0,
            components=[
                ComponentScore("price_action", Direction.BULLISH, 75, 0.15, "ORB breakout above range"),
            ],
        )
        assert self.strategy.should_enter(sig, 22000, trading_time())

    def test_enters_pdh_break(self):
        sig = make_signal(
            score=80.0,
            iv_score=50.0,
            components=[
                ComponentScore("price_action", Direction.BULLISH, 75, 0.15, "Price above PDH"),
            ],
        )
        assert self.strategy.should_enter(sig, 22000, trading_time())

    def test_enters_pivot_break(self):
        sig = make_signal(
            score=80.0,
            iv_score=50.0,
            components=[
                ComponentScore("price_action", Direction.BULLISH, 75, 0.15, "Price above R1 pivot"),
            ],
        )
        assert self.strategy.should_enter(sig, 22000, trading_time())

    def test_create_setup(self):
        sig = make_signal()
        contract = make_contract(ltp=200.0)
        setup = self.strategy.create_setup(sig, contract, 22000, 100000)
        assert setup is not None
        assert setup.strategy_name == "straddle_breakout"
        assert setup.target == round(200 * 1.50, 2)
        assert setup.stop_loss == round(200 * 0.65, 2)

    def test_exit_time_based_no_profit(self):
        pos = make_position(
            entry_price=100, stop_loss=65, target=150,
            strategy="straddle_breakout", entry_minutes_ago=100,
        )
        exit_sig = self.strategy.should_exit(pos, 102.0, 22000, trading_time())
        assert exit_sig.should_exit
        assert "TIME_EXIT" in exit_sig.reason


# ===== Multi Timeframe Tests =====

class TestMultiTimeframe:
    def setup_method(self):
        self.strategy = MultiTimeframeStrategy()

    def test_get_timeframe_direction_insufficient_data(self):
        df = pd.DataFrame({"close": [100.0] * 10, "high": [101.0] * 10, "low": [99.0] * 10})
        assert get_timeframe_direction(df) == Direction.NEUTRAL

    def test_rejects_low_score(self):
        sig = make_signal(score=60.0)
        assert not self.strategy.should_enter(sig, 22000, trading_time())

    def test_rejects_no_mtf_data(self):
        sig = make_signal(score=80.0)
        assert not self.strategy.should_enter(sig, 22000, trading_time())

    def test_rejects_misaligned_timeframes(self):
        sig = make_signal(score=80.0, direction=Direction.BULLISH)
        self.strategy._mtf_directions["NIFTY"] = {
            "5min": Direction.BULLISH,
            "15min": Direction.BEARISH,
            "daily": Direction.BULLISH,
        }
        assert not self.strategy.should_enter(sig, 22000, trading_time())

    def test_enters_all_aligned(self):
        sig = make_signal(score=80.0, direction=Direction.BULLISH)
        self.strategy._mtf_directions["NIFTY"] = {
            "5min": Direction.BULLISH,
            "15min": Direction.BULLISH,
            "daily": Direction.BULLISH,
        }
        assert self.strategy.should_enter(sig, 22000, trading_time())

    def test_enters_bearish_alignment(self):
        sig = make_signal(score=80.0, direction=Direction.BEARISH)
        self.strategy._mtf_directions["NIFTY"] = {
            "5min": Direction.BEARISH,
            "15min": Direction.BEARISH,
            "daily": Direction.BEARISH,
        }
        assert self.strategy.should_enter(sig, 22000, trading_time())

    def test_rejects_neutral_signal(self):
        sig = make_signal(score=80.0, direction=Direction.NEUTRAL)
        self.strategy._mtf_directions["NIFTY"] = {
            "5min": Direction.NEUTRAL,
            "15min": Direction.NEUTRAL,
            "daily": Direction.NEUTRAL,
        }
        assert not self.strategy.should_enter(sig, 22000, trading_time())

    def test_analyze_timeframes_caches(self):
        df = pd.DataFrame({
            "close": list(range(100, 150)),
            "high": list(range(101, 151)),
            "low": list(range(99, 149)),
        })
        result = self.strategy.analyze_timeframes("NIFTY", df, df, df)
        assert "NIFTY" in self.strategy._mtf_directions
        assert set(result.keys()) == {"5min", "15min", "daily"}

    def test_create_setup_full_risk(self):
        sig = make_signal()
        contract = make_contract(ltp=100.0, lot_size=50)
        self.strategy._mtf_directions["NIFTY"] = {
            "5min": Direction.BULLISH,
            "15min": Direction.BULLISH,
            "daily": Direction.BULLISH,
        }
        setup = self.strategy.create_setup(sig, contract, 22000, 100000)
        assert setup is not None
        assert setup.strategy_name == "multi_timeframe"
        # 2% risk for MTF (high conviction)
        # risk_amount = 100000 * 0.02 = 2000
        # SL = 70, risk_per_lot = 30 * 50 = 1500
        # lots = 2000 / 1500 = 1
        assert setup.quantity == 50

    def test_exit_long_time_losing(self):
        pos = make_position(
            entry_price=100, stop_loss=70, target=150,
            strategy="multi_timeframe", entry_minutes_ago=130,
        )
        exit_sig = self.strategy.should_exit(pos, 85.0, 22000, trading_time())
        assert exit_sig.should_exit
        assert "TIME_EXIT" in exit_sig.reason

    def test_no_exit_long_time_slightly_losing(self):
        """MTF allows more time — only exit if losing > 10%."""
        pos = make_position(
            entry_price=100, stop_loss=70, target=150,
            strategy="multi_timeframe", entry_minutes_ago=130,
        )
        exit_sig = self.strategy.should_exit(pos, 95.0, 22000, trading_time())
        assert not exit_sig.should_exit

    def test_select_expiry_prefers_further(self):
        """MTF prefers next week expiry if current too close."""
        today = date(2026, 4, 9)
        expiry = self.strategy.select_expiry("NIFTY", today)
        assert expiry >= today


# ===== Trailing SL Tests (shared behavior across strategies) =====

class TestTrailingSL:
    def test_trailing_sl_activation(self):
        """All strategies should trail SL after profit threshold."""
        strategy = VWAPPullbackStrategy()  # 15% activation, 50% lock
        pos = make_position(
            entry_price=100, high_price=120, stop_loss=75, target=135,
            strategy="vwap_pullback",
        )
        strategy.should_exit(pos, 118.0, 22000, trading_time())
        # high_price = 120, profit = 20%, activation at 15%
        # trail_sl = 100 + (120 - 100) * 0.50 = 110
        assert pos.stop_loss == 110.0

    def test_trailing_sl_updates_high(self):
        """High price should update when current > high."""
        strategy = ScalpingStrategy()
        pos = make_position(
            entry_price=100, high_price=105, stop_loss=85, target=120,
            strategy="scalping",
        )
        strategy.should_exit(pos, 115.0, 22000, trading_time())
        assert pos.high_price == 115.0

    def test_trailing_sl_never_decreases(self):
        """Trailing SL should never move down."""
        strategy = GapAndGoStrategy()
        pos = make_position(
            entry_price=100, high_price=130, stop_loss=115, target=145,
            strategy="gap_and_go",
        )
        # Price drops but high_price stays
        strategy.should_exit(pos, 118.0, 22000, trading_time())
        # trail_sl = 100 + (130 - 100) * 0.50 = 115, same as current SL
        assert pos.stop_loss >= 115.0
