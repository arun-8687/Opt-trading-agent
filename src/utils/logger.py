"""Structured logging setup for the trading system."""

import logging
import logging.handlers
import os
from datetime import datetime


def setup_logger(
    name: str = "trading",
    log_file: str = "data/logs/trading.log",
    level: str = "INFO",
    max_bytes: int = 10_485_760,
    backup_count: int = 5,
) -> logging.Logger:
    """Set up a structured logger with file and console output."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if logger.handlers:
        return logger

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    # File handler with rotation
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
    )
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)

    # Trade-specific log file (separate for audit trail)
    trade_log = log_file.replace("trading.log", "trades.log")
    trade_handler = logging.handlers.RotatingFileHandler(
        trade_log,
        maxBytes=max_bytes,
        backupCount=backup_count,
    )
    trade_handler.setLevel(logging.INFO)
    trade_handler.setFormatter(file_fmt)
    trade_handler.addFilter(TradeFilter())
    logger.addHandler(trade_handler)

    return logger


class TradeFilter(logging.Filter):
    """Filter to only log trade-related messages to the trade log."""

    TRADE_KEYWORDS = {"ORDER", "TRADE", "POSITION", "SL_HIT", "TARGET_HIT", "EXIT", "ENTRY"}

    def filter(self, record: logging.LogRecord) -> bool:
        return any(kw in record.getMessage().upper() for kw in self.TRADE_KEYWORDS)


def get_logger(name: str) -> logging.Logger:
    """Get a child logger for a specific module."""
    return logging.getLogger(f"trading.{name}")
