"""Tests for Black-Scholes Greeks calculator."""

import math
import pytest

from src.utils.greeks import (
    bs_call_price,
    bs_put_price,
    calculate_greeks,
    implied_volatility,
)


class TestBlackScholes:
    """Test Black-Scholes pricing model."""

    def test_call_price_atm(self):
        """ATM call should have significant time value."""
        price = bs_call_price(S=22000, K=22000, T=7/365, r=0.07, sigma=0.15)
        assert price > 0
        assert price < 500  # Reasonable range for NIFTY ATM weekly

    def test_put_price_atm(self):
        """ATM put should have significant time value."""
        price = bs_put_price(S=22000, K=22000, T=7/365, r=0.07, sigma=0.15)
        assert price > 0
        assert price < 500

    def test_call_put_parity(self):
        """Test put-call parity: C - P = S - K*e^(-rT)."""
        S, K, T, r, sigma = 22000, 22000, 30/365, 0.07, 0.15
        call = bs_call_price(S, K, T, r, sigma)
        put = bs_put_price(S, K, T, r, sigma)
        expected = S - K * math.exp(-r * T)
        assert abs((call - put) - expected) < 1.0

    def test_deep_itm_call(self):
        """Deep ITM call should be close to intrinsic."""
        price = bs_call_price(S=23000, K=21000, T=7/365, r=0.07, sigma=0.15)
        assert price >= 2000  # At least intrinsic value

    def test_expired_call(self):
        """Expired call should return intrinsic value."""
        price = bs_call_price(S=22500, K=22000, T=0, r=0.07, sigma=0.15)
        assert abs(price - 500) < 0.01

    def test_expired_put_otm(self):
        """Expired OTM put should be 0."""
        price = bs_put_price(S=22500, K=22000, T=0, r=0.07, sigma=0.15)
        assert abs(price) < 0.01


class TestGreeks:
    """Test Greeks calculations."""

    def test_call_delta_range(self):
        """Call delta should be between 0 and 1."""
        greeks = calculate_greeks(22000, 22000, 7, 0.15, "CE")
        assert 0 < greeks.delta < 1

    def test_put_delta_range(self):
        """Put delta should be between -1 and 0."""
        greeks = calculate_greeks(22000, 22000, 7, 0.15, "PE")
        assert -1 < greeks.delta < 0

    def test_atm_delta_near_05(self):
        """ATM call delta should be near 0.5."""
        greeks = calculate_greeks(22000, 22000, 7, 0.15, "CE")
        assert 0.4 < greeks.delta < 0.6

    def test_gamma_positive(self):
        """Gamma should always be positive."""
        greeks = calculate_greeks(22000, 22000, 7, 0.15, "CE")
        assert greeks.gamma > 0

    def test_theta_negative_for_buyer(self):
        """Theta should be negative (time decay hurts buyer)."""
        greeks = calculate_greeks(22000, 22000, 7, 0.15, "CE")
        assert greeks.theta < 0

    def test_vega_positive(self):
        """Vega should be positive."""
        greeks = calculate_greeks(22000, 22000, 7, 0.15, "CE")
        assert greeks.vega > 0


class TestImpliedVolatility:
    """Test IV calculation."""

    def test_iv_round_trip(self):
        """Calculate price from vol, then back-calculate vol. Should match."""
        target_vol = 0.18
        price = bs_call_price(22000, 22000, 7/365, 0.07, target_vol)
        calc_vol = implied_volatility(price, 22000, 22000, 7, "CE")
        assert abs(calc_vol - target_vol) < 0.01

    def test_iv_put_round_trip(self):
        """Round-trip test for put IV."""
        target_vol = 0.20
        price = bs_put_price(22000, 22000, 14/365, 0.07, target_vol)
        calc_vol = implied_volatility(price, 22000, 22000, 14, "PE")
        assert abs(calc_vol - target_vol) < 0.01

    def test_iv_handles_zero_price(self):
        """Should handle near-zero option price gracefully."""
        vol = implied_volatility(0.05, 22000, 23000, 1, "CE")
        assert vol > 0
