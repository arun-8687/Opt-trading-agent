"""NSE market constants, lot sizes, and trading parameters."""

from datetime import time

# NSE Market Hours (IST)
MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)
PRE_OPEN_START = time(9, 0)
PRE_OPEN_END = time(9, 8)

# Trading Windows
DEFAULT_TRADING_START = time(9, 20)  # Skip first 5 min volatility
DEFAULT_TRADING_END = time(15, 0)    # No new trades in last 30 min
DEFAULT_HARD_CUTOFF = time(15, 0)    # Exit all day trades

# Index Lot Sizes (updated Jan 2026 by NSE)
INDEX_LOT_SIZES = {
    "NIFTY": 65,
    "BANKNIFTY": 30,
    "FINNIFTY": 25,
    "MIDCPNIFTY": 50,
}

# Index Strike Intervals
INDEX_STRIKE_INTERVALS = {
    "NIFTY": 50,
    "BANKNIFTY": 100,
    "FINNIFTY": 50,
    "MIDCPNIFTY": 25,
}

# Weekly Expiry Days (0=Monday, 6=Sunday)
# NOTE: As of late 2023, SEBI allows only one weekly index contract per exchange.
# NSE kept NIFTY weekly (Thursday). BANKNIFTY, FINNIFTY, MIDCPNIFTY are monthly only.
# The algorithmic fallback in get_next_expiry() uses these values; live code uses
# the instrument master registry which handles holiday adjustments automatically.
INDEX_EXPIRY_DAYS = {
    "NIFTY": 3,           # Thursday (weekly)
    "BANKNIFTY": 2,        # Wednesday (monthly — last Wed of month)
    "FINNIFTY": 1,         # Tuesday (monthly — last Tue of month)
    "MIDCPNIFTY": 0,       # Monday (monthly — last Mon of month)
}

# Symbols with only monthly expiry (no weekly contracts)
MONTHLY_ONLY_SYMBOLS = {"BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"}

# Exchange Codes
EXCHANGE_NSE = "NSE"
EXCHANGE_NFO = "NFO"
EXCHANGE_BSE = "BSE"
EXCHANGE_BFO = "BFO"

# Angel One Exchange Mapping
ANGEL_EXCHANGE_MAP = {
    "NSE": "NSE",
    "NFO": "NFO",
    "BSE": "BSE",
    "BFO": "BFO",
}

# Order Types
ORDER_TYPE_MARKET = "MARKET"
ORDER_TYPE_LIMIT = "LIMIT"
ORDER_TYPE_SL = "STOPLOSS_LIMIT"
ORDER_TYPE_SLM = "STOPLOSS_MARKET"

# Order Sides
ORDER_BUY = "BUY"
ORDER_SELL = "SELL"

# Product Types
PRODUCT_INTRADAY = "INTRADAY"
PRODUCT_DELIVERY = "DELIVERY"
PRODUCT_CARRYFORWARD = "CARRYFORWARD"

# Order Varieties
VARIETY_NORMAL = "NORMAL"
VARIETY_STOPLOSS = "STOPLOSS"

# Order Status
STATUS_PENDING = "pending"
STATUS_OPEN = "open"
STATUS_COMPLETE = "complete"
STATUS_REJECTED = "rejected"
STATUS_CANCELLED = "cancelled"

# Option Types
OPTION_CE = "CE"
OPTION_PE = "PE"

# Candle Intervals
INTERVAL_1MIN = "ONE_MINUTE"
INTERVAL_5MIN = "FIVE_MINUTE"
INTERVAL_15MIN = "FIFTEEN_MINUTE"
INTERVAL_30MIN = "THIRTY_MINUTE"
INTERVAL_1HOUR = "ONE_HOUR"
INTERVAL_1DAY = "ONE_DAY"

# Angel One Interval Mapping
ANGEL_INTERVAL_MAP = {
    "1min": INTERVAL_1MIN,
    "5min": INTERVAL_5MIN,
    "15min": INTERVAL_15MIN,
    "30min": INTERVAL_30MIN,
    "1hour": INTERVAL_1HOUR,
    "1day": INTERVAL_1DAY,
}

# Signal Direction
DIRECTION_BULLISH = "BULLISH"
DIRECTION_BEARISH = "BEARISH"
DIRECTION_NEUTRAL = "NEUTRAL"

# Risk-Free Rate for Black-Scholes (India 10Y Govt Bond ~7%)
RISK_FREE_RATE = 0.07

# Trading Days per Year (India)
TRADING_DAYS_PER_YEAR = 252

# F&O Stock List (Top liquid stocks - complete list loaded from NSE)
FNO_STOCKS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK", "LT",
    "AXISBANK", "HINDUNILVR", "BAJFINANCE", "MARUTI", "TATAMOTORS",
    "TATASTEEL", "ADANIENT", "WIPRO", "HCLTECH", "TECHM",
    "SUNPHARMA", "NTPC", "POWERGRID", "ONGC", "COALINDIA",
    "ULTRACEMCO", "JSWSTEEL", "TITAN", "BAJAJFINSV", "ASIANPAINT",
    "NESTLEIND", "DIVISLAB", "DRREDDY", "CIPLA", "APOLLOHOSP",
    "HEROMOTOCO", "EICHERMOT", "TATACONSUM", "BRITANNIA", "GRASIM",
    "INDUSINDBK", "SBILIFE", "HDFCLIFE", "M&M", "BPCL",
    "HINDALCO", "VEDL", "BANKBARODA", "PNB", "IDFCFIRSTB",
    "LICHSGFIN", "MANAPPURAM", "DELTACORP", "CHAMBLFERT", "PEL",
    "VOLTAS", "MUTHOOTFIN", "FEDERALBNK", "CANBK", "SAIL",
    "BHEL", "GAIL", "IOC", "RECLTD", "PFC",
    "TATAPOWER", "IRCTC", "INDIGO", "ZOMATO", "PAYTM",
    "DLF", "GODREJPROP", "OBEROIRLTY", "PIDILITIND", "HAVELLS",
    "SIEMENS", "ABB", "BEL", "HAL", "LICI",
    "IDEA", "BANDHANBNK", "AUBANK", "MFSL", "CHOLAFIN",
    "SHRIRAMFIN", "SBICARD", "NAUKRI", "TRENT", "PERSISTENT",
    "LTIM", "COFORGE", "MPHASIS", "LAURUSLABS", "BIOCON",
]

# Stock Lot Sizes (subset - full list loaded from instrument master)
STOCK_LOT_SIZES = {
    "RELIANCE": 250,
    "TCS": 150,
    "HDFCBANK": 550,
    "INFY": 300,
    "ICICIBANK": 700,
    "SBIN": 750,
    "BHARTIARTL": 475,
    "ITC": 1600,
    "KOTAKBANK": 400,
    "LT": 150,
    "AXISBANK": 600,
    "HINDUNILVR": 300,
    "BAJFINANCE": 125,
    "TATAMOTORS": 575,
    "TATASTEEL": 1100,
    "WIPRO": 1500,
    "HCLTECH": 350,
    "TECHM": 300,
    "SUNPHARMA": 350,
    "M&M": 350,
}
