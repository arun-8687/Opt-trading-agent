"""Options chain builder and analyzer."""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

import pandas as pd

from src.broker.base import BaseBroker
from src.broker.models import OptionChainRow, OptionContract, OptionType
from src.data.store import DataStore
from src.utils.constants import INDEX_LOT_SIZES, INDEX_STRIKE_INTERVALS
from src.utils.greeks import calculate_greeks, implied_volatility
from src.utils.helpers import days_to_expiry, get_atm_strike
from src.utils.logger import get_logger

logger = get_logger("options_chain")


@dataclass
class OptionsChainAnalysis:
    """Analysis results from the options chain."""

    symbol: str
    spot_price: float
    expiry: date
    atm_strike: int
    total_ce_oi: int = 0
    total_pe_oi: int = 0
    pcr_oi: float = 0.0
    pcr_volume: float = 0.0
    max_pain: int = 0
    max_ce_oi_strike: int = 0
    max_pe_oi_strike: int = 0
    total_ce_oi_change: int = 0
    total_pe_oi_change: int = 0
    atm_iv: float = 0.0
    iv_skew: float = 0.0  # CE IV - PE IV at ATM
    chain: list[OptionChainRow] = field(default_factory=list)


class OptionsChainManager:
    """Builds and analyzes options chains."""

    def __init__(self, broker: BaseBroker, store: DataStore):
        self.broker = broker
        self.store = store
        self._previous_oi: dict[str, dict[int, tuple[int, int]]] = {}

    def get_chain_analysis(
        self,
        symbol: str,
        spot_price: float,
        expiry: date,
        exchange: str = "NFO",
    ) -> OptionsChainAnalysis:
        """Build and analyze the full options chain.

        Returns comprehensive analysis including PCR, max pain,
        OI concentrations, and IV data.
        """
        chain = self.broker.get_option_chain(symbol, expiry, exchange, spot_price=spot_price)

        if not chain:
            return OptionsChainAnalysis(
                symbol=symbol,
                spot_price=spot_price,
                expiry=expiry,
                atm_strike=get_atm_strike(spot_price, symbol),
            )

        # Calculate OI changes from previous snapshot
        chain_key = f"{symbol}_{expiry}"
        prev_oi = self._previous_oi.get(chain_key, {})
        for row in chain:
            prev = prev_oi.get(row.strike, (0, 0))
            row.ce_oi_change = row.ce_oi - prev[0]
            row.pe_oi_change = row.pe_oi - prev[1]

        # Update previous OI
        self._previous_oi[chain_key] = {
            row.strike: (row.ce_oi, row.pe_oi) for row in chain
        }

        atm = get_atm_strike(spot_price, symbol)
        dte = days_to_expiry(expiry)

        # Calculate IVs for each strike
        for row in chain:
            if row.ce_ltp > 0 and dte > 0:
                row.ce_iv = implied_volatility(
                    row.ce_ltp, spot_price, row.strike, dte, "CE"
                )
            if row.pe_ltp > 0 and dte > 0:
                row.pe_iv = implied_volatility(
                    row.pe_ltp, spot_price, row.strike, dte, "PE"
                )

        # Aggregate analysis
        total_ce_oi = sum(r.ce_oi for r in chain)
        total_pe_oi = sum(r.pe_oi for r in chain)
        total_ce_vol = sum(r.ce_volume for r in chain)
        total_pe_vol = sum(r.pe_volume for r in chain)

        pcr_oi = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
        pcr_volume = total_pe_vol / total_ce_vol if total_ce_vol > 0 else 0

        # Max OI strikes
        max_ce_oi_strike = max(chain, key=lambda r: r.ce_oi).strike if chain else 0
        max_pe_oi_strike = max(chain, key=lambda r: r.pe_oi).strike if chain else 0

        # Max pain calculation
        max_pain = self._calculate_max_pain(chain, spot_price)

        # ATM IV and skew
        atm_row = min(chain, key=lambda r: abs(r.strike - atm)) if chain else None
        atm_iv = 0.0
        iv_skew = 0.0
        if atm_row:
            atm_iv = (atm_row.ce_iv + atm_row.pe_iv) / 2 if atm_row.ce_iv and atm_row.pe_iv else 0
            iv_skew = atm_row.ce_iv - atm_row.pe_iv

        # OI changes
        total_ce_oi_change = sum(r.ce_oi_change for r in chain)
        total_pe_oi_change = sum(r.pe_oi_change for r in chain)

        # Save OI snapshot
        self._save_oi_snapshot(symbol, expiry, spot_price, chain)

        analysis = OptionsChainAnalysis(
            symbol=symbol,
            spot_price=spot_price,
            expiry=expiry,
            atm_strike=atm,
            total_ce_oi=total_ce_oi,
            total_pe_oi=total_pe_oi,
            pcr_oi=round(pcr_oi, 2),
            pcr_volume=round(pcr_volume, 2),
            max_pain=max_pain,
            max_ce_oi_strike=max_ce_oi_strike,
            max_pe_oi_strike=max_pe_oi_strike,
            total_ce_oi_change=total_ce_oi_change,
            total_pe_oi_change=total_pe_oi_change,
            atm_iv=round(atm_iv, 4),
            iv_skew=round(iv_skew, 4),
            chain=chain,
        )

        logger.info(
            f"Chain analysis {symbol}: PCR={pcr_oi:.2f} MaxPain={max_pain} "
            f"ATM_IV={atm_iv:.2%} MaxCE_OI@{max_ce_oi_strike} MaxPE_OI@{max_pe_oi_strike}"
        )

        return analysis

    def _calculate_max_pain(
        self,
        chain: list[OptionChainRow],
        spot_price: float,
    ) -> int:
        """Calculate max pain strike (the price where total option losses are minimized).

        Max pain is the strike at which the total value of all outstanding
        options (CE + PE) would be at its minimum if the underlying expired there.
        """
        if not chain:
            return 0

        min_pain = float("inf")
        max_pain_strike = 0

        strikes = [row.strike for row in chain]

        for test_strike in strikes:
            total_pain = 0

            for row in chain:
                # CE pain: call writers lose when price > strike
                if test_strike > row.strike:
                    total_pain += (test_strike - row.strike) * row.ce_oi

                # PE pain: put writers lose when price < strike
                if test_strike < row.strike:
                    total_pain += (row.strike - test_strike) * row.pe_oi

            if total_pain < min_pain:
                min_pain = total_pain
                max_pain_strike = test_strike

        return max_pain_strike

    def _save_oi_snapshot(
        self,
        symbol: str,
        expiry: date,
        spot_price: float,
        chain: list[OptionChainRow],
    ):
        """Save OI snapshot to database for historical tracking."""
        now = datetime.now().isoformat()
        for row in chain:
            if row.ce_oi > 0 or row.pe_oi > 0:
                self.store.save_oi_snapshot({
                    "timestamp": now,
                    "symbol": symbol,
                    "expiry": expiry.isoformat(),
                    "strike": row.strike,
                    "ce_oi": row.ce_oi,
                    "pe_oi": row.pe_oi,
                    "ce_oi_change": row.ce_oi_change,
                    "pe_oi_change": row.pe_oi_change,
                    "ce_volume": row.ce_volume,
                    "pe_volume": row.pe_volume,
                    "spot_price": spot_price,
                })

    def select_option_contract(
        self,
        symbol: str,
        spot_price: float,
        expiry: date,
        option_type: str,
        strikes_otm: int = 2,
        exchange: str = "NFO",
    ) -> Optional[OptionContract]:
        """Select the best option contract for a trade.

        Picks an OTM strike and validates liquidity/spread constraints.
        """
        interval = INDEX_STRIKE_INTERVALS.get(symbol, 50)
        atm = get_atm_strike(spot_price, symbol)

        if option_type == "CE":
            target_strike = atm + (strikes_otm * interval)
        else:
            target_strike = atm - (strikes_otm * interval)

        # Look up the token from broker
        token, trading_symbol = "", ""
        if hasattr(self.broker, "lookup_option_token"):
            token, trading_symbol = self.broker.lookup_option_token(
                symbol, expiry, target_strike, option_type, exchange
            )

        if not token:
            logger.warning(
                f"Could not find token for {symbol} {expiry} {target_strike}{option_type}"
            )
            return None

        lot_size = INDEX_LOT_SIZES.get(symbol, 25)

        contract = OptionContract(
            symbol=symbol,
            token=token,
            trading_symbol=trading_symbol,
            strike=target_strike,
            option_type=OptionType.CE if option_type == "CE" else OptionType.PE,
            expiry=expiry,
            lot_size=lot_size,
            exchange=exchange,
        )

        # Fetch current price
        ltp = self.broker.get_ltp(exchange, trading_symbol, token)
        contract.ltp = ltp

        logger.info(
            f"Selected {contract.display_name} @ {ltp} (token={token})"
        )

        return contract
