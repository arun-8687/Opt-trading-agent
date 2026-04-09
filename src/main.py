"""Main trading engine orchestrator.

Ties together all components: data, signals, strategies, risk, and execution.
Runs the main trading loop during market hours.
"""

import os
import signal
import sys
import time as time_module
from datetime import datetime, date, time, timedelta
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv

from src.broker.angel_one import AngelOneBroker
from src.data.historical import HistoricalDataManager
from src.data.market_data import MarketDataManager
from src.data.options_chain import OptionsChainManager
from src.data.store import DataStore
from src.execution.order_manager import OrderManager
from src.execution.position_manager import PositionManager
from src.risk.position_sizer import PositionSizer
from src.risk.risk_manager import RiskManager
from src.risk.stop_loss import StopLossManager
from src.scanner.stock_scanner import StockScanner
from src.signals.signal_engine import SignalEngine
from src.strategies.base import BaseStrategy, TradeSetup
from src.strategies.breakout_buy import BreakoutBuyStrategy
from src.strategies.expiry_day import ExpiryDayStrategy
from src.strategies.momentum_buy import MomentumBuyStrategy
from src.strategies.oi_reversal import OIReversalStrategy
from src.utils.helpers import (
    get_next_expiry,
    is_market_open,
    is_trading_window,
)
from src.utils.logger import get_logger, setup_logger

logger = get_logger("main")


class TradingEngine:
    """Main trading engine that orchestrates all components."""

    def __init__(self, config_path: str = "config/settings.yaml"):
        load_dotenv()
        self.config = self._load_config(config_path)
        self._running = False

        # Initialize components
        self._init_components()
        self._init_strategies()

        # Track entry spot prices for underlying-based SL
        self._entry_spot_prices: dict[str, float] = {}

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Config loaded from {config_path}")
        return config

    def _init_components(self):
        """Initialize all system components."""
        trading_cfg = self.config.get("trading", {})
        signals_cfg = self.config.get("signals", {})
        sl_cfg = self.config.get("stop_loss", {})
        scanner_cfg = self.config.get("scanner", {})

        # Broker
        self.broker = AngelOneBroker()

        # Data
        self.store = DataStore(self.config.get("database", {}).get("path", "data/trading.db"))
        self.historical = HistoricalDataManager(self.broker, self.store)
        self.market_data = MarketDataManager(self.broker)
        self.options_chain = OptionsChainManager(self.broker, self.store)

        # Signal Engine
        self.signal_engine = SignalEngine(
            min_score=signals_cfg.get("min_score", 75),
            min_categories_agree=signals_cfg.get("min_categories_agree", 3),
            cooldown_minutes=signals_cfg.get("cooldown_minutes", 15),
            weights=signals_cfg.get("weights"),
        )

        # Risk
        mode = trading_cfg.get("mode", "paper")
        capital = trading_cfg.get("capital", 100000)

        self.risk_manager = RiskManager(
            capital=capital,
            max_risk_per_trade_pct=trading_cfg.get("risk_per_trade_pct", 2.0),
            max_daily_loss_pct=trading_cfg.get("max_daily_loss_pct", 5.0),
            max_open_positions=trading_cfg.get("max_open_positions", 5),
        )

        self.position_sizer = PositionSizer(
            capital=capital,
            risk_per_trade_pct=trading_cfg.get("risk_per_trade_pct", 2.0),
        )

        self.sl_manager = StopLossManager(
            initial_sl_pct=sl_cfg.get("initial_pct", 30),
            trailing_activation_pct=sl_cfg.get("trailing_activation_pct", 20),
            trailing_lock_pct=sl_cfg.get("trailing_lock_pct", 50),
            time_exit_minutes=sl_cfg.get("time_based_exit_minutes", 60),
        )

        # Execution
        self.order_manager = OrderManager(self.broker, mode=mode)
        self.position_manager = PositionManager(
            self.order_manager, self.sl_manager, self.store
        )

        # Scanner
        self.scanner = StockScanner(
            self.broker, self.historical,
            max_stocks=scanner_cfg.get("max_stocks_to_monitor", 15),
        )

        logger.info(f"Trading engine initialized | mode={mode} | capital={capital}")

    def _init_strategies(self):
        """Initialize and register trading strategies."""
        strat_cfg = self.config.get("strategies", {})
        self.strategies: list[BaseStrategy] = []

        if strat_cfg.get("momentum_buy", {}).get("enabled", True):
            s = MomentumBuyStrategy(
                min_score=strat_cfg.get("momentum_buy", {}).get("min_score", 75),
                target_pct=strat_cfg.get("momentum_buy", {}).get("target_pct", 40),
            )
            self.strategies.append(s)
            self.position_manager.register_strategy(s)

        if strat_cfg.get("breakout_buy", {}).get("enabled", True):
            s = BreakoutBuyStrategy(
                min_score=strat_cfg.get("breakout_buy", {}).get("min_score", 78),
                target_pct=strat_cfg.get("breakout_buy", {}).get("target_pct", 50),
            )
            self.strategies.append(s)
            self.position_manager.register_strategy(s)

        if strat_cfg.get("oi_reversal", {}).get("enabled", True):
            s = OIReversalStrategy(
                min_score=strat_cfg.get("oi_reversal", {}).get("min_score", 80),
                target_pct=strat_cfg.get("oi_reversal", {}).get("target_pct", 35),
            )
            self.strategies.append(s)
            self.position_manager.register_strategy(s)

        if strat_cfg.get("expiry_day", {}).get("enabled", True):
            s = ExpiryDayStrategy(
                min_score=strat_cfg.get("expiry_day", {}).get("min_score", 72),
                target_pct=strat_cfg.get("expiry_day", {}).get("target_pct", 75),
            )
            self.strategies.append(s)
            self.position_manager.register_strategy(s)

        logger.info(f"Strategies registered: {[s.name for s in self.strategies]}")

    def start(self):
        """Start the trading engine."""
        logger.info("=" * 60)
        logger.info("NSE OPTIONS TRADING ENGINE STARTING")
        logger.info(f"Date: {date.today()} | Mode: {self.config['trading']['mode']}")
        logger.info("=" * 60)

        # Login to broker
        if not self.broker.login():
            logger.error("Broker login failed. Exiting.")
            return

        # Setup graceful shutdown
        signal.signal(signal.SIGINT, self._shutdown_handler)
        signal.signal(signal.SIGTERM, self._shutdown_handler)

        self._running = True
        self.risk_manager.reset_daily()

        try:
            self._run_loop()
        except Exception as e:
            logger.error(f"Fatal error: {e}", exc_info=True)
        finally:
            self._shutdown()

    def _run_loop(self):
        """Main trading loop."""
        scan_done = False
        interval_seconds = 30  # Check every 30 seconds

        while self._running:
            now = datetime.now()

            # Wait for market to open
            if not is_market_open(now):
                if now.time() < time(9, 15):
                    logger.info("Waiting for market to open...")
                    time_module.sleep(60)
                    continue
                else:
                    # Market closed for the day
                    logger.info("Market closed. Generating daily summary.")
                    self._daily_summary()
                    break

            # Run scanner at start of day
            if not scan_done and now.time() >= time(9, 16):
                logger.info("Running stock scanner...")
                self.scanner.scan()
                scan_done = True

            # Main trading logic (within trading window)
            if is_trading_window(now):
                self._process_symbols()

            # Update open positions
            self._update_positions()

            # Check no-new-trades cutoff
            no_new_after = self.config.get("trading", {}).get("no_new_trades_after", "14:30")
            cutoff_h, cutoff_m = map(int, no_new_after.split(":"))
            if now.time() >= time(cutoff_h, cutoff_m):
                # Only monitor, no new trades
                pass

            # Hard cutoff — close all positions
            hard_cutoff_str = self.config.get("trading", {}).get("hard_cutoff", "15:00")
            hc_h, hc_m = map(int, hard_cutoff_str.split(":"))
            if now.time() >= time(hc_h, hc_m):
                open_positions = self.position_manager.get_open_positions()
                if open_positions:
                    logger.warning("HARD CUTOFF: Closing all positions")
                    self.position_manager.close_all_positions("HARD_CUTOFF")

            time_module.sleep(interval_seconds)

    def _process_symbols(self):
        """Process each monitored symbol for signals."""
        # Index symbols always monitored
        symbols_to_check = ["NIFTY", "BANKNIFTY"]

        # Add scanned stocks
        scan_results = self.scanner.get_last_results()
        if not scan_results.empty:
            symbols_to_check.extend(scan_results["symbol"].tolist()[:10])

        for symbol in symbols_to_check:
            try:
                self._process_single_symbol(symbol)
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")

    def _process_single_symbol(self, symbol: str):
        """Generate signal and potentially trade for a single symbol."""
        # Determine exchange
        exchange = "NSE"
        token = self.broker.lookup_token(exchange, symbol)
        if not token:
            return

        # Get spot price
        spot_price = self.broker.get_ltp(exchange, symbol, token)
        if spot_price <= 0:
            return

        # Get candle data
        intraday_df = self.historical.get_intraday_candles(
            symbol, token, exchange, "5min", days=2
        )
        daily_df = self.historical.get_daily_candles(
            symbol, token, exchange, days=30
        )

        # Get options chain
        expiry = get_next_expiry(symbol)
        chain_analysis = self.options_chain.get_chain_analysis(
            symbol, spot_price, expiry
        )

        # Generate signal
        sig = self.signal_engine.generate_signal(
            symbol=symbol,
            intraday_df=intraday_df,
            daily_df=daily_df,
            chain_analysis=chain_analysis,
            spot_price=spot_price,
        )

        # Save signal to database
        self.store.save_signal(sig.to_dict())

        # Check if actionable
        if not sig.is_actionable or not sig.option_type:
            return

        # Try each strategy
        now = datetime.now()
        for strategy in self.strategies:
            if not strategy.should_enter(sig, spot_price, now):
                continue

            # Select option contract
            strat_expiry = strategy.select_expiry(symbol, date.today())
            otm_strikes = self.config.get("options", {}).get("preferred_strikes_otm", 2)

            contract = self.options_chain.select_option_contract(
                symbol=symbol,
                spot_price=spot_price,
                expiry=strat_expiry,
                option_type=sig.option_type,
                strikes_otm=otm_strikes,
            )

            if not contract:
                continue

            # Create trade setup
            setup = strategy.create_setup(sig, contract, spot_price, self.risk_manager.capital)
            if not setup:
                continue

            # Risk check
            open_positions = self.position_manager.get_open_positions()
            risk_result = self.risk_manager.check_all(setup, open_positions)

            if not risk_result.passed:
                continue

            # Execute trade
            position = self.position_manager.open_position(setup)
            if position:
                self._entry_spot_prices[symbol] = spot_price
                break  # Only one trade per symbol per signal

    def _update_positions(self):
        """Update all open positions with current prices."""
        open_positions = self.position_manager.get_open_positions()
        if not open_positions:
            return

        # Build price map
        price_map = {}
        spot_prices = {}
        for pos in open_positions:
            ltp = self.broker.get_ltp(pos.exchange, pos.trading_symbol, pos.token)
            if ltp > 0:
                price_map[pos.token] = ltp

            spot_token = self.broker.lookup_token("NSE", pos.symbol)
            if spot_token:
                spot = self.broker.get_ltp("NSE", pos.symbol, spot_token)
                if spot > 0:
                    spot_prices[pos.symbol] = spot

        self.position_manager.update_positions(
            price_map, spot_prices, self._entry_spot_prices
        )

        # Update daily P&L in risk manager
        realized = self.position_manager.get_realized_pnl()
        self.risk_manager._daily_pnl = realized

    def _daily_summary(self):
        """Generate end-of-day summary."""
        total_pnl = self.position_manager.get_total_pnl()
        realized = self.position_manager.get_realized_pnl()
        closed = self.position_manager.get_closed_positions()

        winning = sum(1 for p in closed if p.pnl > 0)
        losing = sum(1 for p in closed if p.pnl < 0)

        self.store.save_daily_pnl({
            "date": date.today().isoformat(),
            "total_pnl": total_pnl,
            "realized_pnl": realized,
            "unrealized_pnl": self.position_manager.get_unrealized_pnl(),
            "total_trades": len(closed),
            "winning_trades": winning,
            "losing_trades": losing,
            "capital": self.risk_manager.capital,
        })

        logger.info("=" * 60)
        logger.info("DAILY SUMMARY")
        logger.info(f"Total P&L: {total_pnl:+.0f}")
        logger.info(f"Trades: {len(closed)} (W:{winning} L:{losing})")
        if closed:
            win_rate = (winning / len(closed)) * 100
            logger.info(f"Win Rate: {win_rate:.0f}%")
        logger.info("=" * 60)

    def _shutdown(self):
        """Clean shutdown."""
        self._running = False

        # Close any open positions
        open_pos = self.position_manager.get_open_positions()
        if open_pos:
            logger.warning(f"Closing {len(open_pos)} open positions on shutdown")
            self.position_manager.close_all_positions("SHUTDOWN")

        self._daily_summary()

        # Logout broker
        self.broker.logout()
        logger.info("Trading engine shut down")

    def _shutdown_handler(self, signum, frame):
        """Handle graceful shutdown on SIGINT/SIGTERM."""
        logger.info("Shutdown signal received")
        self._running = False


def main():
    """Entry point."""
    log_cfg = {}
    config_path = "config/settings.yaml"
    if Path(config_path).exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
            log_cfg = cfg.get("logging", {})

    setup_logger(
        log_file=log_cfg.get("file", "data/logs/trading.log"),
        level=log_cfg.get("level", "INFO"),
    )

    engine = TradingEngine(config_path)
    engine.start()


if __name__ == "__main__":
    main()
