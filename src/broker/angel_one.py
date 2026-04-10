"""Angel One SmartAPI broker implementation."""

import json
import os
import time as time_module
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import pyotp
import requests
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
from src.utils.helpers import register_instrument_expiries
from src.utils.logger import get_logger

logger = get_logger("angel_one")

# Token mode constants
LTP_MODE = 1
QUOTE_MODE = 2
SNAP_QUOTE_MODE = 3

# Login retry settings
_LOGIN_MAX_RETRIES = 3
_LOGIN_RETRY_DELAY = 5  # seconds between retries
_TOTP_MIN_REMAINING_SECS = 5  # wait for fresh TOTP if fewer seconds remain in window

# Instrument master
_INSTRUMENT_MASTER_URL = (
    "https://margincalculator.angelone.in/OpenAPI_File/files/OpenAPIScripMaster.json"
)
_INSTRUMENT_CACHE_PATH = Path("data/instruments.json")


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
        self._instrument_cache_path = _INSTRUMENT_CACHE_PATH
        self._last_request_time: float = 0
        self._min_request_interval: float = 0.5  # 2 req/sec (Angel One allows 3/sec)
        self._session_expiry: Optional[datetime] = None
        self._session_lifetime_hours: float = 8.0  # Angel One sessions ~8h

    def _rate_limit(self):
        """Enforce rate limiting between API calls."""
        elapsed = time_module.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time_module.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time_module.time()

    def _get_fresh_totp(self) -> str:
        """Generate a TOTP code, waiting for a fresh window if too close to expiry."""
        remaining = 30 - (int(time_module.time()) % 30)
        if remaining < _TOTP_MIN_REMAINING_SECS:
            logger.debug(
                f"TOTP window expiring in {remaining}s — waiting {remaining + 1}s for fresh code"
            )
            time_module.sleep(remaining + 1)
        return pyotp.TOTP(self.totp_secret).now()

    def _is_session_valid(self) -> bool:
        """Return True if the current session is still valid."""
        if not self.auth_token or self._session_expiry is None:
            return False
        return datetime.now() < self._session_expiry

    def _refresh_session(self) -> bool:
        """Refresh the JWT token using the stored refresh token."""
        if not self.smart_api or not self.refresh_token:
            return False
        try:
            data = self.smart_api.generateToken(self.refresh_token)
            if data and data.get("data"):
                self.auth_token = data["data"].get("jwtToken", self.auth_token)
                self.feed_token = data["data"].get("feedToken", self.feed_token)
                self._session_expiry = datetime.now() + timedelta(hours=self._session_lifetime_hours)
                logger.info("Session token refreshed successfully")
                return True
            logger.warning(f"Token refresh returned unexpected response: {data}")
        except Exception as e:
            logger.error(f"Session refresh error: {e}")
        return False

    def ensure_session(self) -> bool:
        """Ensure a valid session exists, refreshing or re-logging in as needed."""
        if self._is_session_valid():
            return True
        if self.refresh_token:
            logger.info("Session expired — attempting token refresh")
            if self._refresh_session():
                return True
        logger.info("Token refresh failed or no refresh token — re-logging in")
        return self.login()

    def login(self) -> bool:
        """Login to Angel One using TOTP-based authentication, with retries."""
        for attempt in range(1, _LOGIN_MAX_RETRIES + 1):
            try:
                self.smart_api = SmartConnect(api_key=self.api_key)
                totp = self._get_fresh_totp()
                logger.info(f"Login attempt {attempt}/{_LOGIN_MAX_RETRIES} for client {self.client_id}")

                data = self.smart_api.generateSession(
                    clientCode=self.client_id,
                    password=self.password,
                    totp=totp,
                )

                if not data or data.get("status") is False:
                    logger.error(f"Login failed (attempt {attempt}): {data}")
                    if attempt < _LOGIN_MAX_RETRIES:
                        time_module.sleep(_LOGIN_RETRY_DELAY * attempt)
                    continue

                self.auth_token = data["data"]["jwtToken"]
                self.refresh_token = data["data"]["refreshToken"]
                self.feed_token = self.smart_api.getfeedToken()
                self._session_expiry = datetime.now() + timedelta(hours=self._session_lifetime_hours)

                profile = self.smart_api.getProfile(self.refresh_token)
                logger.info(
                    f"Logged in as: {profile['data'].get('name', self.client_id)}"
                )

                # Load instrument master
                self._load_instruments()

                return True

            except (ConnectionResetError, ConnectionError, OSError) as e:
                logger.warning(
                    f"Connection error on login attempt {attempt}/{_LOGIN_MAX_RETRIES}: {e}"
                )
                if attempt < _LOGIN_MAX_RETRIES:
                    wait = _LOGIN_RETRY_DELAY * attempt
                    logger.info(f"Retrying in {wait}s…")
                    time_module.sleep(wait)

            except Exception as e:
                logger.error(f"Login error (attempt {attempt}): {e}")
                if attempt < _LOGIN_MAX_RETRIES:
                    time_module.sleep(_LOGIN_RETRY_DELAY)

        logger.error(f"Login failed after {_LOGIN_MAX_RETRIES} attempts")
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
        """Load instrument master list from Angel One, using a 24h disk cache."""
        cache_path = self._instrument_cache_path
        cache_valid = False

        if cache_path.exists():
            mod_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
            if (datetime.now() - mod_time).total_seconds() < 86400:  # 24h
                cache_valid = True

        if cache_valid:
            try:
                self._instruments = pd.read_json(cache_path)
                logger.info(f"Loaded {len(self._instruments)} instruments from cache")
                self._register_expiries()
                return
            except Exception:
                cache_valid = False

        try:
            logger.info("Downloading instrument master from Angel One…")
            resp = requests.get(_INSTRUMENT_MASTER_URL, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            self._instruments = pd.DataFrame(data)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._instruments.to_json(cache_path, orient="records")
            logger.info(f"Downloaded and cached {len(self._instruments)} instruments")
            self._register_expiries()

        except Exception as e:
            logger.error(f"Failed to load instruments: {e}")
            self._instruments = pd.DataFrame()

    def _register_expiries(self) -> None:
        """Populate the helpers expiry registry from the loaded instrument master.

        This lets get_next_expiry() return real, holiday-adjusted NSE dates.
        """
        if self._instruments is None or self._instruments.empty:
            return

        today = date.today()
        opts = self._instruments[
            self._instruments["instrumenttype"].isin(["OPTIDX", "OPTSTK"])
            & (self._instruments["exch_seg"] == "NFO")
            & (self._instruments["expiry"] != "")
        ]

        for symbol, group in opts.groupby("name"):
            expiry_dates: list[date] = []
            for exp_str in group["expiry"].unique():
                try:
                    exp_date = datetime.strptime(str(exp_str), "%d%b%Y").date()
                    if exp_date >= today:
                        expiry_dates.append(exp_date)
                except ValueError:
                    continue
            if expiry_dates:
                register_instrument_expiries(symbol, expiry_dates)

        logger.info(f"Registered expiry dates for {len(opts['name'].unique())} symbols")

    # Maximum number of tokens per getMarketData call (Angel One limit: 50)
    _BATCH_LTP_LIMIT = 50

    def get_ltp(self, exchange: str, trading_symbol: str, token: str) -> float:
        """Get last traded price for a single instrument."""
        result = self._batch_get_ltp({exchange: [token]})
        return result.get(token, 0.0)

    @dataclass
    class _TokenMarketData:
        """Market data for a single token from FULL mode."""
        ltp: float = 0.0
        oi: int = 0
        volume: int = 0

    def _batch_get_ltp(
        self,
        exchange_tokens: dict[str, list[str]],
    ) -> dict[str, float]:
        """Fetch LTP for multiple tokens in batches using getMarketData.

        Args:
            exchange_tokens: e.g. {"NFO": ["12345", "12346"], "NSE": ["2885"]}

        Returns:
            Dict mapping token -> ltp
        """
        full_data = self._batch_get_market_data(exchange_tokens, mode="LTP")
        return {tok: d.ltp for tok, d in full_data.items()}

    def _batch_get_market_data(
        self,
        exchange_tokens: dict[str, list[str]],
        mode: str = "FULL",
    ) -> dict[str, "_TokenMarketData"]:
        """Fetch market data for multiple tokens using getMarketData.

        Args:
            exchange_tokens: e.g. {"NFO": ["12345", "12346"], "NSE": ["2885"]}
            mode: "LTP" for price only, "FULL" for price + OI + volume

        Returns:
            Dict mapping token -> _TokenMarketData
        """
        results: dict[str, AngelOneBroker._TokenMarketData] = {}

        # Flatten to list of (exchange, token) for batching
        flat: list[tuple[str, str]] = []
        for exch, tokens in exchange_tokens.items():
            for tok in tokens:
                flat.append((exch, tok))

        # Process in batches of _BATCH_LTP_LIMIT
        for i in range(0, len(flat), self._BATCH_LTP_LIMIT):
            batch = flat[i : i + self._BATCH_LTP_LIMIT]

            batch_map: dict[str, list[str]] = {}
            for exch, tok in batch:
                batch_map.setdefault(exch, []).append(tok)

            self._rate_limit()
            try:
                data = self.smart_api.getMarketData(mode, batch_map)
                if data and data.get("data") and data["data"].get("fetched"):
                    for item in data["data"]["fetched"]:
                        tok = str(item.get("symbolToken", ""))
                        results[tok] = AngelOneBroker._TokenMarketData(
                            ltp=float(item.get("ltp", 0)),
                            oi=int(item.get("opnInterest", 0)),
                            volume=int(item.get("tradeVolume", 0)),
                        )
                if data and data.get("data") and data["data"].get("unfetched"):
                    for item in data["data"]["unfetched"]:
                        logger.debug(f"Unfetched token: {item}")
            except Exception as e:
                err_str = str(e).lower()
                if "access rate" in err_str:
                    logger.warning(f"Rate limited on batch market data — waiting 5s ({len(batch)} tokens)")
                    time_module.sleep(5)
                    # Retry once
                    self._rate_limit()
                    try:
                        data = self.smart_api.getMarketData(mode, batch_map)
                        if data and data.get("data") and data["data"].get("fetched"):
                            for item in data["data"]["fetched"]:
                                tok = str(item.get("symbolToken", ""))
                                results[tok] = AngelOneBroker._TokenMarketData(
                                    ltp=float(item.get("ltp", 0)),
                                    oi=int(item.get("opnInterest", 0)),
                                    volume=int(item.get("tradeVolume", 0)),
                                )
                    except Exception:
                        logger.error("Batch market data retry also failed")
                else:
                    logger.error(f"Batch market data error: {e}")

        return results

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
            err_str = str(e)
            if "access rate" in err_str.lower():
                logger.warning(f"Rate limited on {trading_symbol} — skipping")
            else:
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
        strikes_around_atm: int = 10,
        spot_price: float = 0.0,
    ) -> list[OptionChainRow]:
        """Build options chain from instrument master + batch LTP.

        Limits to ±strikes_around_atm strikes around spot to minimize API calls.
        """
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

        # Parse strikes and build mapping
        options["strike_parsed"] = options["strike"].apply(
            lambda x: int(float(x)) // 100
        )

        # Limit strikes around ATM to reduce API calls
        if spot_price > 0:
            all_strikes = sorted(options["strike_parsed"].unique())
            atm_idx = min(range(len(all_strikes)), key=lambda i: abs(all_strikes[i] - spot_price))
            lo = max(0, atm_idx - strikes_around_atm)
            hi = min(len(all_strikes), atm_idx + strikes_around_atm + 1)
            valid_strikes = set(all_strikes[lo:hi])
            options = options[options["strike_parsed"].isin(valid_strikes)]

        # Collect all tokens for batch fetch
        token_info: list[tuple[str, int, str]] = []  # (token, strike, CE/PE)
        for _, row in options.iterrows():
            strike = row["strike_parsed"]
            opt_type = "CE" if "CE" in row["symbol"] else "PE"
            token_info.append((str(row["token"]), strike, opt_type))

        # Batch fetch FULL market data (LTP + OI + volume)
        mkt_data = self._batch_get_market_data(
            {exchange: [t[0] for t in token_info]}, mode="FULL"
        )

        # Build chain
        chain: dict[int, OptionChainRow] = {}
        for tok, strike, opt_type in token_info:
            if strike not in chain:
                chain[strike] = OptionChainRow(strike=strike)

            md = mkt_data.get(tok)
            if md is None:
                continue
            if opt_type == "CE":
                chain[strike].ce_ltp = md.ltp
                chain[strike].ce_oi = md.oi
                chain[strike].ce_volume = md.volume
            else:
                chain[strike].pe_ltp = md.ltp
                chain[strike].pe_oi = md.oi
                chain[strike].pe_volume = md.volume

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
        angel_interval = ANGEL_INTERVAL_MAP.get(interval, interval)
        params = {
            "exchange": exchange,
            "symboltoken": token,
            "interval": angel_interval,
            "fromdate": from_date.strftime("%Y-%m-%d %H:%M"),
            "todate": to_date.strftime("%Y-%m-%d %H:%M"),
        }

        for attempt in range(3):
            self._rate_limit()
            try:
                data = self.smart_api.getCandleData(params)

                if not data or not data.get("data"):
                    return pd.DataFrame()

                df = pd.DataFrame(
                    data["data"],
                    columns=["timestamp", "open", "high", "low", "close", "volume"],
                )
                # Normalize to tz-naive UTC (strip timezone) for consistent comparisons
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
                df = df.sort_values("timestamp").reset_index(drop=True)

                return df

            except Exception as e:
                err_str = str(e).lower()
                if "access rate" in err_str and attempt < 2:
                    wait = 5 * (attempt + 1)
                    logger.warning(
                        f"Rate limited on candle data for {trading_symbol} — "
                        f"waiting {wait}s (attempt {attempt + 1}/3)"
                    )
                    time_module.sleep(wait)
                else:
                    logger.error(f"Historical data error for {trading_symbol}: {e}")
                    return pd.DataFrame()

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

    def lookup_instrument(self, exchange: str, name: str) -> tuple[str, str]:
        """Look up token and actual trading symbol for any instrument.

        For NSE/BSE equities the instrument master stores symbols with a
        '-EQ' suffix (e.g. 'RELIANCE-EQ') while the plain name is 'RELIANCE'.
        This method handles both cases and returns (token, trading_symbol).
        """
        if self._instruments is None or self._instruments.empty:
            return "", ""

        # 1. Exact symbol match (options, futures, indices)
        mask = (
            (self._instruments["symbol"] == name)
            & (self._instruments["exch_seg"] == exchange)
        )
        matches = self._instruments[mask]
        if not matches.empty:
            row = matches.iloc[0]
            return str(row["token"]), str(row["symbol"])

        # 2. For NSE/BSE equities: master uses 'NAME-EQ' symbol; search by name field
        if exchange in ("NSE", "BSE"):
            mask2 = (
                (self._instruments["name"] == name)
                & (self._instruments["exch_seg"] == exchange)
                & (self._instruments["instrumenttype"] == "")
            )
            matches2 = self._instruments[mask2]
            if not matches2.empty:
                row = matches2.iloc[0]
                return str(row["token"]), str(row["symbol"])

        return "", ""

    def lookup_token(self, exchange: str, trading_symbol: str) -> str:
        """Look up token for a trading symbol from instrument master."""
        token, _ = self.lookup_instrument(exchange, trading_symbol)
        return token

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
