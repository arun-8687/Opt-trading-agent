"""Telegram notification bot for trade alerts with command handling."""

import os
import threading
import time as time_module
from datetime import datetime
from typing import Optional, Callable

import requests

from src.utils.logger import get_logger

logger = get_logger("telegram")


class TelegramAlerts:
    """Sends trading alerts via Telegram bot and handles incoming commands."""

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        enabled: bool = True,
    ):
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
        self.enabled = enabled and bool(self.bot_token) and bool(self.chat_id)
        self._last_update_id = 0
        self._polling = False
        self._poll_thread: Optional[threading.Thread] = None
        self._engine = None  # Set via attach_engine()
        self._command_handlers: dict[str, Callable] = {}

        if self.enabled:
            logger.info("Telegram alerts enabled")
        else:
            logger.info("Telegram alerts disabled")

    def attach_engine(self, engine):
        """Attach the trading engine for command control."""
        self._engine = engine
        self._register_commands()
        self._start_polling()

    def _register_commands(self):
        """Register all supported Telegram commands."""
        self._command_handlers = {
            "/start": self._cmd_start,
            "/help": self._cmd_help,
            "/status": self._cmd_status,
            "/positions": self._cmd_positions,
            "/pnl": self._cmd_pnl,
            "/signals": self._cmd_signals,
            "/close_all": self._cmd_close_all,
            "/stop": self._cmd_stop,
            "/scanner": self._cmd_scanner,
            "/risk": self._cmd_risk,
        }

    # ── Polling ─────────────────────────────────────────────

    def _start_polling(self):
        """Start background thread to poll for Telegram commands."""
        if not self.enabled or self._polling:
            return
        self._polling = True
        self._poll_thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="telegram-poller"
        )
        self._poll_thread.start()
        logger.info("Telegram command polling started")

    def stop_polling(self):
        """Stop the polling thread."""
        self._polling = False

    def _poll_loop(self):
        """Poll for new messages every 2 seconds."""
        while self._polling:
            try:
                self._process_updates()
            except Exception as e:
                logger.debug(f"Telegram poll error: {e}")
            time_module.sleep(2)

    def _process_updates(self):
        """Fetch and process new Telegram messages."""
        url = f"https://api.telegram.org/bot{self.bot_token}/getUpdates"
        params = {"offset": self._last_update_id + 1, "timeout": 5}
        try:
            resp = requests.get(url, params=params, timeout=10)
            data = resp.json()
        except Exception:
            return

        if not data.get("ok"):
            return

        for update in data.get("result", []):
            self._last_update_id = update["update_id"]
            msg = update.get("message", {})
            chat_id = str(msg.get("chat", {}).get("id", ""))
            text = (msg.get("text") or "").strip()

            # Only accept commands from authorized chat
            if chat_id != self.chat_id or not text:
                continue

            cmd = text.split()[0].lower().split("@")[0]  # handle /cmd@botname
            handler = self._command_handlers.get(cmd)
            if handler:
                try:
                    handler()
                except Exception as e:
                    self.send(f"⚠️ Command error: {e}")
                    logger.error(f"Command {cmd} error: {e}")
            else:
                self.send(
                    "❓ Unknown command. Send /help for available commands."
                )

    # ── Commands ────────────────────────────────────────────

    def _cmd_start(self):
        self.send(
            "🤖 <b>Trading Bot Active</b>\n"
            "Send /help for available commands."
        )

    def _cmd_help(self):
        self.send(
            "📚 <b>Commands:</b>\n\n"
            "/status — Engine status\n"
            "/positions — Open positions\n"
            "/pnl — Current P&L\n"
            "/signals — Recent signals\n"
            "/scanner — Last scan results\n"
            "/risk — Risk manager state\n"
            "/close_all — Close all positions\n"
            "/stop — Stop the engine\n"
            "/help — This message"
        )

    def _cmd_status(self):
        if not self._engine:
            self.send("⚠️ Engine not attached.")
            return

        e = self._engine
        now = datetime.now()
        open_pos = e.position_manager.get_open_positions()
        closed_pos = e.position_manager.get_closed_positions()
        mode = e.config.get("trading", {}).get("mode", "unknown")
        running = "✅ Running" if e._running else "🛑 Stopped"
        unrealized = e.position_manager.get_unrealized_pnl()
        realized = e.position_manager.get_realized_pnl()

        self.send(
            f"📊 <b>ENGINE STATUS</b>\n\n"
            f"State: {running}\n"
            f"Mode: {mode}\n"
            f"Time: {now.strftime('%H:%M:%S')}\n"
            f"Open positions: {len(open_pos)}\n"
            f"Closed today: {len(closed_pos)}\n"
            f"Unrealized P&L: ₹{unrealized:+,.0f}\n"
            f"Realized P&L: ₹{realized:+,.0f}\n"
            f"Strategies: {len(e.strategies)} active"
        )

    def _cmd_positions(self):
        if not self._engine:
            self.send("⚠️ Engine not attached.")
            return

        positions = self._engine.position_manager.get_open_positions()
        if not positions:
            self.send("📭 No open positions.")
            return

        lines = ["📋 <b>OPEN POSITIONS</b>\n"]
        for p in positions:
            emoji = "🟢" if p.pnl >= 0 else "🔴"
            duration = p.holding_duration_minutes
            lines.append(
                f"{emoji} <b>{p.trading_symbol}</b>\n"
                f"   Entry: ₹{p.entry_price:.2f} → ₹{p.current_price:.2f}\n"
                f"   P&L: ₹{p.pnl:+,.0f} ({p.pnl_pct:+.1f}%)\n"
                f"   SL: ₹{p.stop_loss:.2f} | Held: {duration:.0f}min\n"
                f"   Strategy: {p.strategy}\n"
            )
        self.send("\n".join(lines))

    def _cmd_pnl(self):
        if not self._engine:
            self.send("⚠️ Engine not attached.")
            return

        e = self._engine
        total = e.position_manager.get_total_pnl()
        realized = e.position_manager.get_realized_pnl()
        unrealized = e.position_manager.get_unrealized_pnl()
        closed = e.position_manager.get_closed_positions()
        winning = sum(1 for p in closed if p.pnl > 0)
        losing = sum(1 for p in closed if p.pnl < 0)
        win_rate = (winning / len(closed) * 100) if closed else 0

        emoji = "📈" if total >= 0 else "📉"
        self.send(
            f"{emoji} <b>P&L REPORT</b>\n\n"
            f"Total: ₹{total:+,.0f}\n"
            f"Realized: ₹{realized:+,.0f}\n"
            f"Unrealized: ₹{unrealized:+,.0f}\n"
            f"Trades: {len(closed)} (W:{winning} L:{losing})\n"
            f"Win Rate: {win_rate:.0f}%"
        )

    def _cmd_signals(self):
        if not self._engine:
            self.send("⚠️ Engine not attached.")
            return

        try:
            signals = self._engine.store.get_todays_signals()
            if signals.empty:
                self.send("📭 No signals generated today.")
                return

            # Show last 5 actionable signals
            actionable = signals[signals.get("is_actionable", False) == True]
            if actionable.empty:
                self.send(
                    f"📊 {len(signals)} signals today, none actionable yet.\n"
                    f"Min score needed: {self._engine.signal_engine.min_score}"
                )
                return

            lines = [f"📊 <b>RECENT SIGNALS</b> (last 5 of {len(actionable)})\n"]
            for _, row in actionable.tail(5).iterrows():
                direction = row.get("direction", "?")
                emoji = "🟢" if direction == "BULLISH" else "🔴"
                lines.append(
                    f"{emoji} {row.get('symbol', '?')} — "
                    f"{direction} {row.get('score', 0):.0f}/100"
                )
            self.send("\n".join(lines))
        except Exception as e:
            self.send(f"⚠️ Could not fetch signals: {e}")

    def _cmd_scanner(self):
        if not self._engine:
            self.send("⚠️ Engine not attached.")
            return

        results = self._engine.scanner.get_last_results()
        if results.empty:
            self.send("📭 No scanner results yet.")
            return

        lines = [f"🔍 <b>SCANNER RESULTS</b> ({len(results)} stocks)\n"]
        for _, row in results.head(10).iterrows():
            symbol = row.get("symbol", "?")
            score = row.get("score", 0)
            lines.append(f"• {symbol} — score: {score:.0f}")
        self.send("\n".join(lines))

    def _cmd_risk(self):
        if not self._engine:
            self.send("⚠️ Engine not attached.")
            return

        rm = self._engine.risk_manager
        open_count = len(self._engine.position_manager.get_open_positions())
        self.send(
            f"🛡️ <b>RISK STATUS</b>\n\n"
            f"Capital: ₹{rm.capital:,.0f}\n"
            f"Daily P&L: ₹{rm._daily_pnl:+,.0f}\n"
            f"Max daily loss: {rm.max_daily_loss_pct}%\n"
            f"Open/Max positions: {open_count}/{rm.max_open_positions}\n"
            f"Trades today: {rm._daily_trades}\n"
            f"Risk per trade: {rm.max_risk_per_trade_pct}%"
        )

    def _cmd_close_all(self):
        if not self._engine:
            self.send("⚠️ Engine not attached.")
            return

        positions = self._engine.position_manager.get_open_positions()
        if not positions:
            self.send("📭 No open positions to close.")
            return

        self._engine.position_manager.close_all_positions("TELEGRAM_CLOSE")
        self.send(
            f"✅ <b>CLOSED ALL</b>\n"
            f"Closed {len(positions)} positions via Telegram command."
        )

    def _cmd_stop(self):
        if not self._engine:
            self.send("⚠️ Engine not attached.")
            return

        self.send("🛑 <b>STOPPING ENGINE</b>\nShutting down gracefully...")
        self._engine._running = False

    # ── Alert Senders ───────────────────────────────────────

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
