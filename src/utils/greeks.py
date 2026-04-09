"""Black-Scholes-Merton Options Greeks calculator."""

import math
from dataclasses import dataclass
from typing import Optional

from src.utils.constants import RISK_FREE_RATE, TRADING_DAYS_PER_YEAR


@dataclass
class Greeks:
    """Container for option Greeks values."""

    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    iv: Optional[float] = None


def _norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    """Standard normal probability density function."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculate d1 in Black-Scholes formula."""
    if T <= 0 or sigma <= 0:
        return 0.0
    return (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))


def _d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculate d2 in Black-Scholes formula."""
    if T <= 0 or sigma <= 0:
        return 0.0
    return _d1(S, K, T, r, sigma) - sigma * math.sqrt(T)


def bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculate Black-Scholes call option price.

    Args:
        S: Spot price
        K: Strike price
        T: Time to expiry in years
        r: Risk-free rate
        sigma: Volatility (annualized)
    """
    if T <= 0:
        return max(S - K, 0)
    d_1 = _d1(S, K, T, r, sigma)
    d_2 = _d2(S, K, T, r, sigma)
    return S * _norm_cdf(d_1) - K * math.exp(-r * T) * _norm_cdf(d_2)


def bs_put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculate Black-Scholes put option price."""
    if T <= 0:
        return max(K - S, 0)
    d_1 = _d1(S, K, T, r, sigma)
    d_2 = _d2(S, K, T, r, sigma)
    return K * math.exp(-r * T) * _norm_cdf(-d_2) - S * _norm_cdf(-d_1)


def calculate_greeks(
    spot: float,
    strike: float,
    days_to_expiry: int,
    volatility: float,
    option_type: str = "CE",
    risk_free_rate: float = RISK_FREE_RATE,
) -> Greeks:
    """Calculate all Greeks for an option.

    Args:
        spot: Current spot/underlying price
        strike: Option strike price
        days_to_expiry: Calendar days to expiry
        volatility: Implied volatility (annualized, e.g., 0.15 for 15%)
        option_type: 'CE' for call, 'PE' for put
        risk_free_rate: Risk-free interest rate
    """
    T = max(days_to_expiry, 0.01) / 365.0  # Avoid division by zero
    S = spot
    K = strike
    r = risk_free_rate
    sigma = volatility

    d_1 = _d1(S, K, T, r, sigma)
    d_2 = _d2(S, K, T, r, sigma)
    sqrt_T = math.sqrt(T)

    if option_type == "CE":
        delta = _norm_cdf(d_1)
        theta = (
            -(S * _norm_pdf(d_1) * sigma) / (2 * sqrt_T)
            - r * K * math.exp(-r * T) * _norm_cdf(d_2)
        ) / TRADING_DAYS_PER_YEAR
        rho = K * T * math.exp(-r * T) * _norm_cdf(d_2) / 100
    else:  # PE
        delta = _norm_cdf(d_1) - 1
        theta = (
            -(S * _norm_pdf(d_1) * sigma) / (2 * sqrt_T)
            + r * K * math.exp(-r * T) * _norm_cdf(-d_2)
        ) / TRADING_DAYS_PER_YEAR
        rho = -K * T * math.exp(-r * T) * _norm_cdf(-d_2) / 100

    gamma = _norm_pdf(d_1) / (S * sigma * sqrt_T)
    vega = S * _norm_pdf(d_1) * sqrt_T / 100  # Per 1% change in IV

    return Greeks(
        delta=round(delta, 4),
        gamma=round(gamma, 6),
        theta=round(theta, 4),
        vega=round(vega, 4),
        rho=round(rho, 4),
        iv=volatility,
    )


def implied_volatility(
    option_price: float,
    spot: float,
    strike: float,
    days_to_expiry: int,
    option_type: str = "CE",
    risk_free_rate: float = RISK_FREE_RATE,
    max_iterations: int = 100,
    precision: float = 1e-5,
) -> float:
    """Calculate implied volatility using Newton-Raphson method.

    Args:
        option_price: Market price of the option
        spot: Current spot price
        strike: Strike price
        days_to_expiry: Calendar days to expiry
        option_type: 'CE' or 'PE'
        risk_free_rate: Risk-free rate
        max_iterations: Max iterations for convergence
        precision: Convergence threshold
    """
    T = max(days_to_expiry, 0.01) / 365.0
    S = spot
    K = strike
    r = risk_free_rate

    # Intrinsic value check
    if option_type == "CE":
        intrinsic = max(S - K, 0)
    else:
        intrinsic = max(K - S, 0)

    if option_price <= intrinsic:
        return 0.01  # Near-zero IV for deep ITM / no time value

    # Initial guess using Brenner-Subrahmanyam approximation
    sigma = math.sqrt(2 * math.pi / T) * option_price / S

    # Bound the initial guess
    sigma = max(0.01, min(sigma, 5.0))

    for _ in range(max_iterations):
        if option_type == "CE":
            price = bs_call_price(S, K, T, r, sigma)
        else:
            price = bs_put_price(S, K, T, r, sigma)

        diff = price - option_price

        if abs(diff) < precision:
            return sigma

        # Vega for Newton-Raphson
        d_1 = _d1(S, K, T, r, sigma)
        vega = S * _norm_pdf(d_1) * math.sqrt(T)

        if vega < 1e-10:
            break

        sigma -= diff / vega
        sigma = max(0.001, min(sigma, 5.0))

    return sigma
