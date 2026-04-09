"""Real-time market data manager with WebSocket and candle building."""

from collections import defaultdict
from datetime import datetime, timedelta
from threading import Lock
from typing import Callable, Optional

import pandas as pd

from src.broker.base import BaseBroker
from src.broker.models import Candle, Quote
from src.utils.logger import get_logger

logger = get_logger("market_data")


class MarketDataManager:
    """Manages real-time market data, LTP cache, and candle building."""

    def __init__(self, broker: BaseBroker):
        self.broker = broker
        self._ltp_cache: dict[str, float] = {}
        self._quote_cache: dict[str, Quote] = {}
        self._candles: dict[str, dict[str, list[Candle]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._current_candle: dict[str, dict[str, Candle]] = defaultdict(dict)
        self._lock = Lock()
        self._tick_callbacks: list[Callable] = []
        self._subscribed_tokens: set[str] = set()

    def subscribe(self, tokens: list[str]):
        """Subscribe to real-time ticks for given tokens."""
        new_tokens = set(tokens) - self._subscribed_tokens
        if new_tokens:
            self._subscribed_tokens.update(new_tokens)
            logger.info(f"Subscribed to {len(new_tokens)} new tokens")

    def add_tick_callback(self, callback: Callable):
        """Register a callback for incoming tick data."""
        self._tick_callbacks.append(callback)

    def on_tick(self, tick_data: dict):
        """Process incoming tick from WebSocket.

        Expected tick_data keys: token, ltp, volume, oi, best_bid, best_ask,
        best_bid_qty, best_ask_qty, timestamp
        """
        token = str(tick_data.get("token", ""))
        if not token:
            return

        ltp = float(tick_data.get("ltp", 0)) / 100  # Angel sends price * 100

        with self._lock:
            self._ltp_cache[token] = ltp

            self._quote_cache[token] = Quote(
                symbol=tick_data.get("symbol", ""),
                token=token,
                ltp=ltp,
                open=float(tick_data.get("open", 0)) / 100,
                high=float(tick_data.get("high", 0)) / 100,
                low=float(tick_data.get("low", 0)) / 100,
                close=float(tick_data.get("close", 0)) / 100,
                volume=int(tick_data.get("volume", 0)),
                oi=int(tick_data.get("oi", 0)),
                bid=float(tick_data.get("best_bid", 0)) / 100,
                ask=float(tick_data.get("best_ask", 0)) / 100,
                bid_qty=int(tick_data.get("best_bid_qty", 0)),
                ask_qty=int(tick_data.get("best_ask_qty", 0)),
                timestamp=datetime.now(),
            )

            self._update_candle(token, ltp, tick_data)

        for callback in self._tick_callbacks:
            try:
                callback(token, ltp, tick_data)
            except Exception as e:
                logger.error(f"Tick callback error: {e}")

    def _update_candle(self, token: str, ltp: float, tick_data: dict):
        """Build real-time candles from tick data."""
        now = datetime.now()
        volume = int(tick_data.get("volume", 0))

        for interval_name, minutes in [("1min", 1), ("5min", 5), ("15min", 15)]:
            candle_start = now.replace(
                minute=(now.minute // minutes) * minutes,
                second=0,
                microsecond=0,
            )

            current = self._current_candle.get(token, {}).get(interval_name)

            if current is None or current.timestamp != candle_start:
                # Save completed candle
                if current is not None:
                    self._candles[token][interval_name].append(current)
                    # Keep only last 500 candles in memory
                    if len(self._candles[token][interval_name]) > 500:
                        self._candles[token][interval_name] = (
                            self._candles[token][interval_name][-500:]
                        )

                # Start new candle
                if token not in self._current_candle:
                    self._current_candle[token] = {}
                self._current_candle[token][interval_name] = Candle(
                    timestamp=candle_start,
                    open=ltp,
                    high=ltp,
                    low=ltp,
                    close=ltp,
                    volume=volume,
                )
            else:
                # Update current candle
                current.high = max(current.high, ltp)
                current.low = min(current.low, ltp)
                current.close = ltp
                current.volume = volume

    def get_ltp(self, token: str) -> float:
        """Get cached LTP for a token."""
        with self._lock:
            return self._ltp_cache.get(token, 0.0)

    def get_quote(self, token: str) -> Optional[Quote]:
        """Get cached quote for a token."""
        with self._lock:
            return self._quote_cache.get(token)

    def get_candles_df(
        self,
        token: str,
        interval: str = "5min",
        count: int = 100,
    ) -> pd.DataFrame:
        """Get candles as a DataFrame for analysis."""
        with self._lock:
            candles = list(self._candles.get(token, {}).get(interval, []))
            current = self._current_candle.get(token, {}).get(interval)
            if current:
                candles.append(current)

        if not candles:
            return pd.DataFrame()

        candles = candles[-count:]
        data = [
            {
                "timestamp": c.timestamp,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume,
            }
            for c in candles
        ]
        return pd.DataFrame(data)

    def get_multiple_ltp(self, tokens: list[str]) -> dict[str, float]:
        """Get LTP for multiple tokens at once."""
        with self._lock:
            return {t: self._ltp_cache.get(t, 0.0) for t in tokens}

    def start(self, tokens: list[str]):
        """Start real-time data feed."""
        self.subscribe(tokens)
        self.broker.start_websocket(
            tokens=list(self._subscribed_tokens),
            on_tick=self.on_tick,
            on_connect=lambda: logger.info("Market data feed started"),
            on_disconnect=lambda: logger.warning("Market data feed disconnected"),
        )

    def stop(self):
        """Stop real-time data feed."""
        self.broker.stop_websocket()
        logger.info("Market data feed stopped")
