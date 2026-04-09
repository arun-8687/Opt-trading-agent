"""Telegram notification bot for trade alerts."""

import os
from typing import Optional

import requests

from src.utils.logger import get_logger

logger = get_logger("telegram")


class TelegramAlerts:
    """Sends trading alerts via Telegram bot."""

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        enabled: bool = True,
    ):
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
        self.enabled = enabled and bool(self.bot_token) and bool(self.chat_id)

        if self.enabled:
            logger.info("Telegram alerts enabled")
        else:
            logger.info("Telegram alerts disabled")

    def send(self, message: str) -> bool:
        """Send a text message via Telegram."""
        if not self.enabled:
            return False

        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "HTML",
            }
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                return True
            else:
                logger.error(f"Telegram send failed: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Telegram error: {e}")
            return False

    def send_signal_alert(
        self,
        symbol: str,
        direction: str,
        score: float,
        option_type: str,
    ):
        """Send a signal alert."""
        emoji = "🟢" if direction == "BULLISH" else "🔴"
        msg = (
            f"{emoji} <b>SIGNAL: {symbol}</b>\n"
            f"Direction: {direction}\n"
            f"Score: {score:.0f}/100\n"
            f"Action: BUY {option_type}"
        )
        self.send(msg)

    def send_order_alert(
        self,
        action: str,
        symbol: str,
        price: float,
        quantity: int,
        strategy: str,
    ):
        """Send an order alert."""
        msg = (
            f"📋 <b>ORDER: {action} {symbol}</b>\n"
            f"Price: ₹{price:.2f}\n"
            f"Qty: {quantity}\n"
            f"Strategy: {strategy}"
        )
        self.send(msg)

    def send_exit_alert(
        self,
        symbol: str,
        exit_price: float,
        pnl: float,
        pnl_pct: float,
        reason: str,
    ):
        """Send a position exit alert."""
        emoji = "✅" if pnl > 0 else "❌"
        msg = (
            f"{emoji} <b>EXIT: {symbol}</b>\n"
            f"Price: ₹{exit_price:.2f}\n"
            f"P&L: ₹{pnl:+,.0f} ({pnl_pct:+.1f}%)\n"
            f"Reason: {reason}"
        )
        self.send(msg)

    def send_daily_summary(
        self,
        total_pnl: float,
        total_trades: int,
        winning: int,
        losing: int,
    ):
        """Send end-of-day summary."""
        emoji = "📈" if total_pnl > 0 else "📉"
        win_rate = (winning / total_trades * 100) if total_trades > 0 else 0
        msg = (
            f"{emoji} <b>DAILY SUMMARY</b>\n"
            f"P&L: ₹{total_pnl:+,.0f}\n"
            f"Trades: {total_trades} (W:{winning} L:{losing})\n"
            f"Win Rate: {win_rate:.0f}%"
        )
        self.send(msg)
