"""Historical data fetcher with local caching."""

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from src.broker.base import BaseBroker
from src.data.store import DataStore
from src.utils.constants import ANGEL_INTERVAL_MAP
from src.utils.logger import get_logger

logger = get_logger("historical")


class HistoricalDataManager:
    """Fetches and caches historical candle data."""

    def __init__(self, broker: BaseBroker, store: DataStore):
        self.broker = broker
        self.store = store

    def get_candles(
        self,
        symbol: str,
        token: str,
        exchange: str,
        interval: str = "5min",
        days: int = 30,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Get historical candles, using cache when available.

        First checks local SQLite cache. If data is missing or stale,
        fetches from broker API and updates cache.
        """
        if to_date is None:
            to_date = datetime.now()
        if from_date is None:
            from_date = to_date - timedelta(days=days)

        # Check cache first
        cached = self.store.get_candles(
            symbol=symbol,
            interval=interval,
            from_date=from_date,
            to_date=to_date,
            limit=10000,
        )

        if not cached.empty and len(cached) > 10:
            last_cached = pd.Timestamp(cached["timestamp"].max())
            first_cached = pd.Timestamp(cached["timestamp"].min())
            # Strip timezone so comparison with tz-naive from/to_date works correctly
            if last_cached.tzinfo is not None:
                last_cached = last_cached.tz_convert(None)
            if first_cached.tzinfo is not None:
                first_cached = first_cached.tz_convert(None)

            # If cache is recent enough (within 2 candle periods), use it —
            # but only if the cached range covers at least 90% of the requested range.
            gap_minutes = {"1min": 2, "5min": 10, "15min": 30, "1day": 1440}
            max_gap = timedelta(minutes=gap_minutes.get(interval, 10))
            requested_range = (to_date - from_date).total_seconds()
            covered_range = (last_cached - first_cached).total_seconds()
            coverage = covered_range / requested_range if requested_range > 0 else 1.0

            if (to_date - last_cached) <= max_gap and coverage >= 0.9:
                logger.debug(f"Using cached data for {symbol} ({len(cached)} candles)")
                return cached

        # Fetch from broker API
        df = self._fetch_from_broker(
            symbol=symbol,
            token=token,
            exchange=exchange,
            interval=interval,
            from_date=from_date,
            to_date=to_date,
        )

        if not df.empty:
            # Save to cache
            self.store.save_candles(symbol, exchange, interval, df)
            logger.info(f"Cached {len(df)} candles for {symbol} [{interval}]")

        return df

    def _fetch_from_broker(
        self,
        symbol: str,
        token: str,
        exchange: str,
        interval: str,
        from_date: datetime,
        to_date: datetime,
    ) -> pd.DataFrame:
        """Fetch historical data from broker, handling pagination."""
        all_data = []

        # Angel One limits: 30 days for intraday, 365 for daily
        max_days = 365 if interval == "1day" else 30
        current_from = from_date

        while current_from < to_date:
            current_to = min(
                current_from + timedelta(days=max_days),
                to_date,
            )

            df = self.broker.get_historical_candles(
                exchange=exchange,
                trading_symbol=symbol,
                token=token,
                interval=interval,
                from_date=current_from,
                to_date=current_to,
            )

            if not df.empty:
                all_data.append(df)

            current_from = current_to + timedelta(minutes=1)

        if not all_data:
            return pd.DataFrame()

        combined = pd.concat(all_data, ignore_index=True)
        combined = combined.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        return combined.reset_index(drop=True)

    def get_daily_candles(
        self,
        symbol: str,
        token: str,
        exchange: str = "NSE",
        days: int = 60,
    ) -> pd.DataFrame:
        """Convenience method for daily candles."""
        return self.get_candles(
            symbol=symbol,
            token=token,
            exchange=exchange,
            interval="1day",
            days=days,
        )

    def get_intraday_candles(
        self,
        symbol: str,
        token: str,
        exchange: str = "NSE",
        interval: str = "5min",
        days: int = 5,
    ) -> pd.DataFrame:
        """Convenience method for intraday candles."""
        return self.get_candles(
            symbol=symbol,
            token=token,
            exchange=exchange,
            interval=interval,
            days=days,
        )
