"""Angel One SmartAPI broker implementation."""

import json
import os
import time as time_module
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import pyotp
from SmartApi import SmartConnect
from SmartApi.smartWebSocketV2 import SmartWebSocketV2

from src.broker.base import BaseBroker
from src.broker.models import (
    Candle,
    OptionChainRow,
    OptionType,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    ProductType,
    Quote,
)
from src.utils.constants import ANGEL_INTERVAL_MAP
from src.utils.logger import get_logger

logger = get_logger("angel_one")

# Token mode constants
LTP_MODE = 1
QUOTE_MODE = 2
SNAP_QUOTE_MODE = 3


class AngelOneBroker(BaseBroker):
    """Angel One SmartAPI implementation."""

    def __init__(self):
        self.api_key = os.getenv("ANGEL_API_KEY", "")
        self.client_id = os.getenv("ANGEL_CLIENT_ID", "")
        self.password = os.getenv("ANGEL_PASSWORD", "")
        self.totp_secret = os.getenv("ANGEL_TOTP_SECRET", "")

        self.smart_api: Optional[SmartConnect] = None
        self.ws: Optional[SmartWebSocketV2] = None
        self.auth_token: str = ""
        self.feed_token: str = ""
        self.refresh_token: str = ""

        self._instruments: Optional[pd.DataFrame] = None
        self._instrument_cache_path = Path("instruments.json")
        self._last_request_time: float = 0
        self._min_request_interval: float = 0.1  # 10 req/sec

    def _rate_limit(self):
        """Enforce rate limiting between API calls."""
        elapsed = time_module.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time_module.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time_module.time()

    def login(self) -> bool:
        """Login to Angel One using TOTP-based authentication."""
        try:
            self.smart_api = SmartConnect(api_key=self.api_key)
            totp = pyotp.TOTP(self.totp_secret).now()

            data = self.smart_api.generateSession(
                clientCode=self.client_id,
                password=self.password,
                totp=totp,
            )

            if not data or data.get("status") is False:
                logger.error(f"Login failed: {data}")
                return False

            self.auth_token = data["data"]["jwtToken"]
            self.refresh_token = data["data"]["refreshToken"]
            self.feed_token = self.smart_api.getfeedToken()

            profile = self.smart_api.getProfile(self.refresh_token)
            logger.info(
                f"Logged in as: {profile['data'].get('name', self.client_id)}"
            )

            # Load instrument master
            self._load_instruments()

            return True

        except Exception as e:
            logger.error(f"Login error: {e}")
            return False

    def logout(self) -> None:
        """Logout and cleanup."""
        try:
            self.stop_websocket()
            if self.smart_api:
                self.smart_api.terminateSession(self.client_id)
                logger.info("Logged out successfully")
        except Exception as e:
            logger.error(f"Logout error: {e}")

    def _load_instruments(self):
        """Load instrument master list, using cache if fresh."""
        cache_path = self._instrument_cache_path
        cache_valid = False

        if cache_path.exists():
            mod_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
            if (datetime.now() - mod_time).total_seconds() < 86400:  # 24h
                cache_valid = True

        if cache_valid:
            try:
                self._instruments = pd.read_json(cache_path)
                logger.info(
                    f"Loaded {len(self._instruments)} instruments from cache"
                )
                return
            except Exception:
                cache_valid = False

        try:
            instruments = self.smart_api.getInstrumentList()
            if instruments:
                self._instruments = pd.DataFrame(instruments)
                self._instruments.to_json(cache_path)
                logger.info(
                    f"Downloaded {len(self._instruments)} instruments"
                )
        except Exception as e:
            logger.error(f"Failed to load instruments: {e}")
            self._instruments = pd.DataFrame()

    def get_ltp(self, exchange: str, trading_symbol: str, token: str) -> float:
        """Get last traded price."""
        self._rate_limit()
        try:
            data = self.smart_api.ltpData(exchange, trading_symbol, token)
            if data and data.get("data"):
                return float(data["data"]["ltp"])
        except Exception as e:
            logger.error(f"LTP fetch error for {trading_symbol}: {e}")
        return 0.0

    def get_quote(self, exchange: str, trading_symbol: str, token: str) -> Quote:
        """Get full quote data."""
        self._rate_limit()
        try:
            data = self.smart_api.ltpData(exchange, trading_symbol, token)
            if data and data.get("data"):
                d = data["data"]
                return Quote(
                    symbol=trading_symbol,
                    token=token,
                    ltp=float(d.get("ltp", 0)),
                    open=float(d.get("open", 0)),
                    high=float(d.get("high", 0)),
                    low=float(d.get("low", 0)),
                    close=float(d.get("close", 0)),
                    volume=int(d.get("volume", 0)),
                    oi=int(d.get("oi", 0)),
                    timestamp=datetime.now(),
                )
        except Exception as e:
            logger.error(f"Quote fetch error for {trading_symbol}: {e}")

        return Quote(
            symbol=trading_symbol, token=token, ltp=0, open=0,
            high=0, low=0, close=0, volume=0,
        )

    def get_option_chain(
        self,
        symbol: str,
        expiry: date,
        exchange: str = "NFO",
    ) -> list[OptionChainRow]:
        """Build options chain from instrument master and LTP data."""
        if self._instruments is None or self._instruments.empty:
            logger.error("Instruments not loaded")
            return []

        # Filter instruments for this symbol and expiry
        exp_str = expiry.strftime("%d%b%Y").upper()
        mask = (
            (self._instruments["name"] == symbol)
            & (self._instruments["exch_seg"] == exchange)
            & (self._instruments["expiry"] == exp_str)
            & (self._instruments["instrumenttype"].isin(["OPTIDX", "OPTSTK"]))
        )
        options = self._instruments[mask].copy()

        if options.empty:
            logger.warning(f"No options found for {symbol} expiry {exp_str}")
            return []

        chain = {}
        for _, row in options.iterrows():
            strike = int(float(row["strike"])) // 100  # Angel stores strike * 100
            opt_type = "CE" if "CE" in row["symbol"] else "PE"

            if strike not in chain:
                chain[strike] = OptionChainRow(strike=strike)

            # Fetch LTP for each option (rate-limited)
            token = str(row["token"])
            ltp = self.get_ltp(exchange, row["symbol"], token)

            if opt_type == "CE":
                chain[strike].ce_ltp = ltp
            else:
                chain[strike].pe_ltp = ltp

        return sorted(chain.values(), key=lambda x: x.strike)

    def get_historical_candles(
        self,
        exchange: str,
        trading_symbol: str,
        token: str,
        interval: str,
        from_date: datetime,
        to_date: datetime,
    ) -> pd.DataFrame:
        """Fetch historical candle data."""
        self._rate_limit()
        try:
            angel_interval = ANGEL_INTERVAL_MAP.get(interval, interval)
            params = {
                "exchange": exchange,
                "symboltoken": token,
                "interval": angel_interval,
                "fromdate": from_date.strftime("%Y-%m-%d %H:%M"),
                "todate": to_date.strftime("%Y-%m-%d %H:%M"),
            }

            data = self.smart_api.getCandleData(params)

            if not data or not data.get("data"):
                return pd.DataFrame()

            df = pd.DataFrame(
                data["data"],
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp").reset_index(drop=True)

            return df

        except Exception as e:
            logger.error(f"Historical data error for {trading_symbol}: {e}")
            return pd.DataFrame()

    def place_order(self, order: Order) -> str:
        """Place an order with Angel One."""
        self._rate_limit()
        try:
            order_params = {
                "variety": "NORMAL",
                "tradingsymbol": order.trading_symbol,
                "symboltoken": order.token,
                "transactiontype": order.side.value,
                "exchange": order.exchange,
                "ordertype": order.order_type.value,
                "producttype": order.product.value,
                "duration": "DAY",
                "quantity": str(order.quantity),
            }

            if order.order_type == OrderType.LIMIT:
                order_params["price"] = str(order.price)
            elif order.order_type in (
                OrderType.STOPLOSS_LIMIT,
                OrderType.STOPLOSS_MARKET,
            ):
                order_params["triggerprice"] = str(order.trigger_price)
                if order.order_type == OrderType.STOPLOSS_LIMIT:
                    order_params["price"] = str(order.price)

            if order.tag:
                order_params["ordertag"] = order.tag

            response = self.smart_api.placeOrder(order_params)

            if response:
                order_id = str(response)
                order.order_id = order_id
                order.status = OrderStatus.OPEN
                order.timestamp = datetime.now()
                logger.info(
                    f"ORDER PLACED: {order.side.value} {order.trading_symbol} "
                    f"qty={order.quantity} price={order.price} id={order_id}"
                )
                return order_id

        except Exception as e:
            order.status = OrderStatus.REJECTED
            logger.error(
                f"ORDER REJECTED: {order.side.value} {order.trading_symbol} "
                f"error={e}"
            )

        return ""

    def modify_order(
        self,
        order_id: str,
        price: Optional[float] = None,
        quantity: Optional[int] = None,
        trigger_price: Optional[float] = None,
        order_type: Optional[str] = None,
    ) -> bool:
        """Modify an existing order."""
        self._rate_limit()
        try:
            params = {"variety": "NORMAL", "orderid": order_id}
            if price is not None:
                params["price"] = str(price)
            if quantity is not None:
                params["quantity"] = str(quantity)
            if trigger_price is not None:
                params["triggerprice"] = str(trigger_price)
            if order_type is not None:
                params["ordertype"] = order_type

            response = self.smart_api.modifyOrder(params)
            if response:
                logger.info(f"ORDER MODIFIED: id={order_id} params={params}")
                return True

        except Exception as e:
            logger.error(f"Order modify error: {order_id} - {e}")

        return False

    def cancel_order(self, order_id: str, variety: str = "NORMAL") -> bool:
        """Cancel an order."""
        self._rate_limit()
        try:
            response = self.smart_api.cancelOrder(order_id, variety)
            if response:
                logger.info(f"ORDER CANCELLED: id={order_id}")
                return True
        except Exception as e:
            logger.error(f"Order cancel error: {order_id} - {e}")
        return False

    def get_order_book(self) -> list[dict]:
        """Get all orders for the day."""
        self._rate_limit()
        try:
            data = self.smart_api.orderBook()
            if data and data.get("data"):
                return data["data"]
        except Exception as e:
            logger.error(f"Order book error: {e}")
        return []

    def get_positions(self) -> list[dict]:
        """Get open positions."""
        self._rate_limit()
        try:
            data = self.smart_api.position()
            if data and data.get("data"):
                return data["data"]
        except Exception as e:
            logger.error(f"Positions error: {e}")
        return []

    def get_holdings(self) -> list[dict]:
        """Get portfolio holdings."""
        self._rate_limit()
        try:
            data = self.smart_api.holding()
            if data and data.get("data"):
                return data["data"]
        except Exception as e:
            logger.error(f"Holdings error: {e}")
        return []

    def get_margins(self) -> dict:
        """Get account margins."""
        self._rate_limit()
        try:
            data = self.smart_api.rmsLimit()
            if data and data.get("data"):
                return data["data"]
        except Exception as e:
            logger.error(f"Margins error: {e}")
        return {}

    def lookup_token(self, exchange: str, trading_symbol: str) -> str:
        """Look up token for a trading symbol from instrument master."""
        if self._instruments is None or self._instruments.empty:
            return ""

        mask = (
            (self._instruments["symbol"] == trading_symbol)
            & (self._instruments["exch_seg"] == exchange)
        )
        matches = self._instruments[mask]

        if not matches.empty:
            return str(matches.iloc[0]["token"])

        return ""

    def lookup_option_token(
        self,
        symbol: str,
        expiry: date,
        strike: int,
        option_type: str,
        exchange: str = "NFO",
    ) -> tuple[str, str]:
        """Look up token and trading symbol for an option contract.

        Returns (token, trading_symbol) tuple.
        """
        if self._instruments is None or self._instruments.empty:
            return "", ""

        exp_str = expiry.strftime("%d%b%Y").upper()

        mask = (
            (self._instruments["name"] == symbol)
            & (self._instruments["exch_seg"] == exchange)
            & (self._instruments["expiry"] == exp_str)
            & (self._instruments["strike"] == str(strike * 100))
            & (self._instruments["symbol"].str.endswith(option_type))
        )
        matches = self._instruments[mask]

        if not matches.empty:
            row = matches.iloc[0]
            return str(row["token"]), str(row["symbol"])

        return "", ""

    def start_websocket(
        self,
        tokens: list[str],
        on_tick: Callable,
        on_connect: Optional[Callable] = None,
        on_disconnect: Optional[Callable] = None,
    ) -> None:
        """Start WebSocket for real-time market data."""
        try:
            self.ws = SmartWebSocketV2(
                self.auth_token,
                self.api_key,
                self.client_id,
                self.feed_token,
            )

            def _on_data(wsapp, message):
                on_tick(message)

            def _on_open(wsapp):
                logger.info("WebSocket connected")
                # Subscribe to tokens
                token_list = [
                    {"exchangeType": 2, "tokens": tokens}  # NFO
                ]
                self.ws.subscribe("abc123", SNAP_QUOTE_MODE, token_list)
                if on_connect:
                    on_connect()

            def _on_error(wsapp, error):
                logger.error(f"WebSocket error: {error}")

            def _on_close(wsapp):
                logger.warning("WebSocket disconnected")
                if on_disconnect:
                    on_disconnect()

            self.ws.on_data = _on_data
            self.ws.on_open = _on_open
            self.ws.on_error = _on_error
            self.ws.on_close = _on_close

            self.ws.connect()

        except Exception as e:
            logger.error(f"WebSocket start error: {e}")

    def stop_websocket(self) -> None:
        """Stop WebSocket connection."""
        try:
            if self.ws:
                self.ws.close_connection()
                self.ws = None
                logger.info("WebSocket stopped")
        except Exception as e:
            logger.error(f"WebSocket stop error: {e}")

    def get_instrument_list(self) -> pd.DataFrame:
        """Return the instrument master DataFrame."""
        if self._instruments is None:
            self._load_instruments()
        return self._instruments if self._instruments is not None else pd.DataFrame()
