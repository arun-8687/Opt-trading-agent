"""Historical data loader for backtesting."""

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from src.broker.base import BaseBroker
from src.data.store import DataStore
from src.utils.logger import get_logger

logger = get_logger("backtest_data")


class BacktestDataLoader:
    """Loads historical data for backtesting purposes."""

    def __init__(self, broker: BaseBroker, store: DataStore):
        self.broker = broker
        self.store = store

    def load_intraday(
        self,
        symbol: str,
        token: str,
        exchange: str = "NSE",
        interval: str = "5min",
        days: int = 30,
    ) -> pd.DataFrame:
        """Load intraday candle data for backtesting.

        Fetches from broker API in chunks (30-day limit) and
        combines into a single DataFrame.
        """
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)

        all_data = []
        current_from = from_date

        while current_from < to_date:
            current_to = min(current_from + timedelta(days=29), to_date)

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
        combined = combined.reset_index(drop=True)

        logger.info(f"Loaded {len(combined)} intraday candles for {symbol}")
        return combined

    def load_daily(
        self,
        symbol: str,
        token: str,
        exchange: str = "NSE",
        days: int = 365,
    ) -> pd.DataFrame:
        """Load daily candle data for backtesting."""
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)

        df = self.broker.get_historical_candles(
            exchange=exchange,
            trading_symbol=symbol,
            token=token,
            interval="1day",
            from_date=from_date,
            to_date=to_date,
        )

        if not df.empty:
            logger.info(f"Loaded {len(df)} daily candles for {symbol}")

        return df
