"""Stock scanner — identifies trading opportunities from the F&O universe."""

from datetime import datetime, timedelta
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
                logger.debug(f"Skip {symbol}: {e}")
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
        token = self.broker.lookup_token("NSE", symbol)
        if not token:
            return None

        # Get current quote
        quote = self.broker.get_quote("NSE", symbol, token)
        if quote.ltp <= 0:
            return None

        # Get daily candles for volume and ADX calculation
        daily_df = self.historical.get_daily_candles(
            symbol=symbol,
            token=token,
            exchange="NSE",
            days=30,
        )

        if daily_df.empty or len(daily_df) < 5:
            return None

        # Calculate average volume
        avg_volume = float(daily_df["volume"].tail(20).mean())

        # Calculate ADX
        adx_series = calculate_adx(daily_df, 14)
        adx_value = float(adx_series.iloc[-1]) if not adx_series.empty else 0

        return {
            "symbol": symbol,
            "token": token,
            "ltp": quote.ltp,
            "change_pct": quote.change_pct,
            "volume": quote.volume,
            "avg_volume": avg_volume,
            "adx": adx_value,
            "oi": quote.oi,
        }

    def get_last_results(self) -> pd.DataFrame:
        """Get results from the last scan."""
        return self._last_scan_results
