"""Tests for helper utilities."""

from datetime import date, datetime, time

import pytest

from src.utils.helpers import (
    calculate_pnl,
    calculate_pnl_pct,
    days_to_expiry,
    get_atm_strike,
    get_monthly_expiry,
    get_next_expiry,
    get_otm_strike,
    is_expiry_day,
    is_market_open,
    round_to_strike,
)


class TestMarketHours:

    def test_weekday_during_market(self):
        """Should be open during market hours on weekday."""
        # Wednesday 10:30 AM
        dt = datetime(2024, 4, 10, 10, 30)
        assert is_market_open(dt) is True

    def test_weekend_closed(self):
        """Should be closed on weekends."""
        # Saturday
        dt = datetime(2024, 4, 13, 10, 30)
        assert is_market_open(dt) is False

    def test_before_market(self):
        """Should be closed before market opens."""
        dt = datetime(2024, 4, 10, 8, 0)
        assert is_market_open(dt) is False

    def test_after_market(self):
        """Should be closed after market closes."""
        dt = datetime(2024, 4, 10, 16, 0)
        assert is_market_open(dt) is False


class TestStrikeCalculation:

    def test_round_to_nifty_strike(self):
        """Should round to nearest 50 for NIFTY."""
        assert round_to_strike(22123, "NIFTY") == 22100
        assert round_to_strike(22130, "NIFTY") == 22150

    def test_round_to_banknifty_strike(self):
        """Should round to nearest 100 for BANKNIFTY."""
        assert round_to_strike(48260, "BANKNIFTY") == 48300
        assert round_to_strike(48240, "BANKNIFTY") == 48200

    def test_atm_strike(self):
        """ATM should be closest strike."""
        assert get_atm_strike(22130, "NIFTY") == 22150
        assert get_atm_strike(22100, "NIFTY") == 22100

    def test_otm_ce_strike(self):
        """OTM CE should be above ATM."""
        strike = get_otm_strike(22000, "NIFTY", "CE", steps=2)
        assert strike == 22100  # 22000 + 2*50

    def test_otm_pe_strike(self):
        """OTM PE should be below ATM."""
        strike = get_otm_strike(22000, "NIFTY", "PE", steps=2)
        assert strike == 21900  # 22000 - 2*50


class TestExpiry:

    def test_next_nifty_expiry(self):
        """Next NIFTY expiry should be a Thursday."""
        expiry = get_next_expiry("NIFTY", date(2024, 4, 8))  # Monday
        assert expiry.weekday() == 3  # Thursday

    def test_next_banknifty_expiry(self):
        """Next BANKNIFTY expiry should be a Wednesday."""
        expiry = get_next_expiry("BANKNIFTY", date(2024, 4, 8))
        assert expiry.weekday() == 2  # Wednesday

    def test_monthly_expiry_is_thursday(self):
        """Monthly expiry should be last Thursday."""
        expiry = get_monthly_expiry(date(2024, 4, 1))
        assert expiry.weekday() == 3
        assert expiry.month == 4

    def test_days_to_expiry(self):
        """DTE should be non-negative."""
        dte = days_to_expiry(date(2024, 4, 25), date(2024, 4, 20))
        assert dte > 0

    def test_expired_dte(self):
        """Past expiry should return 0."""
        dte = days_to_expiry(date(2024, 4, 10), date(2024, 4, 20))
        assert dte == 0


class TestPnL:

    def test_buy_pnl_profit(self):
        """Buy position with price increase = profit."""
        pnl = calculate_pnl(100, 120, 50, "BUY")
        assert pnl == 1000

    def test_buy_pnl_loss(self):
        """Buy position with price decrease = loss."""
        pnl = calculate_pnl(100, 80, 50, "BUY")
        assert pnl == -1000

    def test_pnl_pct(self):
        """PnL percentage calculation."""
        pct = calculate_pnl_pct(100, 130, "BUY")
        assert pct == 30.0
