"""Stock scanner — identifies trading opportunities from the F&O universe."""

from datetime import datetime, timedelta, time as time_type
from typing import Optional

import pandas as pd

from src.broker.base import BaseBroker
from src.data.historical import HistoricalDataManager
from src.scanner.filters import (
    ScannerFilterConfig,
    filter_by_change,
    filter_by_liquidity,
    filter_by_price_range,
    filter_by_trend_strength,
    filter_by_volume_surge,
    rank_stocks,
)
from src.signals.technical import calculate_adx
from src.utils.constants import FNO_STOCKS
from src.utils.logger import get_logger

logger = get_logger("scanner")

_MARKET_OPEN_MINS = 9 * 60 + 15   # 9:15 in minutes since midnight
_MARKET_TOTAL_MINS = 375           # 9:15–15:30


class StockScanner:
    """Scans F&O stock universe for trading opportunities."""

    def __init__(
        self,
        broker: BaseBroker,
        historical: HistoricalDataManager,
        config: ScannerFilterConfig = ScannerFilterConfig(),
        max_stocks: int = 15,
    ):
        self.broker = broker
        self.historical = historical
        self.config = config
        self.max_stocks = max_stocks
        self._last_scan_results: pd.DataFrame = pd.DataFrame()

    def scan(self, stock_list: Optional[list[str]] = None) -> pd.DataFrame:
        """Run a full scan of the stock universe.

        Args:
            stock_list: Optional custom list. Defaults to FNO_STOCKS.

        Returns:
            DataFrame of ranked stocks with columns:
            [symbol, ltp, change_pct, volume, avg_volume, volume_ratio, adx, scan_score]
        """
        stocks = stock_list or FNO_STOCKS
        logger.info(f"Scanning {len(stocks)} stocks...")

        scan_data = []

        for symbol in stocks:
            try:
                data = self._get_stock_data(symbol)
                if data:
                    scan_data.append(data)
            except Exception as e:
                logger.warning(f"Skip {symbol}: {e}")
                continue

        if not scan_data:
            logger.warning("No stocks passed initial data fetch")
            return pd.DataFrame()

        df = pd.DataFrame(scan_data)

        # Apply filters in sequence
        df = filter_by_price_range(df, self.config.min_price, self.config.max_price)
        logger.debug(f"After price filter: {len(df)} stocks")

        df = filter_by_liquidity(df, self.config.min_avg_volume)
        logger.debug(f"After liquidity filter: {len(df)} stocks")

        df = filter_by_volume_surge(df, self.config.min_volume_ratio)
        logger.debug(f"After volume surge filter: {len(df)} stocks")

        df = filter_by_trend_strength(df, self.config.min_adx)
        logger.debug(f"After trend filter: {len(df)} stocks")

        df["abs_change_pct"] = abs(df["change_pct"])
        df = filter_by_change(df, 0.5)
        logger.debug(f"After change filter: {len(df)} stocks")

        # Rank and limit
        df = rank_stocks(df)
        df = df.head(self.max_stocks)

        self._last_scan_results = df

        if not df.empty:
            top_symbols = df["symbol"].tolist()
            logger.info(
                f"Scan complete: {len(df)} stocks selected. "
                f"Top: {', '.join(top_symbols[:5])}"
            )

        return df

    def _get_stock_data(self, symbol: str) -> Optional[dict]:
        """Fetch data for a single stock and compute metrics."""
        token, trading_symbol = self.broker.lookup_instrument("NSE", symbol)
        if not token:
            return None

        # Get daily candles for price, volume, and ADX — avoids a separate LTP
        # API call per stock which hits Angel One rate limits during full scans.
        # Need 60+ calendar days to compute a valid 14-period ADX (≥28 trading days).
        daily_df = self.historical.get_daily_candles(
            symbol=symbol,
            token=token,
            exchange="NSE",
            days=60,
        )

        if daily_df.empty or len(daily_df) < 5:
            return None

        last = daily_df.iloc[-1]
        prev = daily_df.iloc[-2] if len(daily_df) >= 2 else last

        # Derive price and change from daily candles
        ltp = float(last["close"])
        change_pct = (
            ((ltp - float(prev["close"])) / float(prev["close"])) * 100
            if float(prev["close"]) > 0 else 0.0
        )

        if ltp <= 0:
            return None

        # Calculate average volume from historical daily candles
        avg_volume = float(daily_df["volume"].tail(20).mean())

        # Project end-of-day volume based on elapsed trading time so that the
        # volume-surge ratio is meaningful during intraday scans.
        current_volume = int(last["volume"])
        now = datetime.now().time()
        elapsed_mins = max(1, now.hour * 60 + now.minute - _MARKET_OPEN_MINS)
        elapsed_fraction = min(1.0, elapsed_mins / _MARKET_TOTAL_MINS)
        projected_volume = int(current_volume / elapsed_fraction)

        # Calculate ADX
        adx_series = calculate_adx(daily_df, 14)
        adx_value = float(adx_series.iloc[-1]) if not adx_series.empty else 0

        return {
            "symbol": symbol,
            "token": token,
            "ltp": ltp,
            "change_pct": change_pct,
            "volume": projected_volume,
            "avg_volume": avg_volume,
            "adx": adx_value,
            "oi": 0,
        }

    def get_last_results(self) -> pd.DataFrame:
        """Get results from the last scan."""
        return self._last_scan_results
