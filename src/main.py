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
from src.strategies.gap_and_go import GapAndGoStrategy
from src.strategies.momentum_buy import MomentumBuyStrategy
from src.strategies.multi_timeframe import MultiTimeframeStrategy
from src.strategies.oi_reversal import OIReversalStrategy
from src.strategies.rsi_divergence import RSIDivergenceStrategy
from src.strategies.scalping import ScalpingStrategy
from src.strategies.straddle_breakout import StraddleBreakoutStrategy
from src.strategies.vwap_pullback import VWAPPullbackStrategy
from src.strategies.nifty_event_day import NiftyEventDayStrategy
from src.strategies.nifty_gamma_blast import NiftyGammaBlastStrategy
from src.strategies.nifty_gift_gap import NiftyGIFTGapStrategy
from src.strategies.nifty_oi_wall import NiftyOIWallStrategy
from src.strategies.nifty_orb import NiftyORBStrategy
from src.strategies.nifty_pcr_reversal import NiftyPCRReversalStrategy
from src.strategies.nifty_vix_regime import NiftyVIXRegimeStrategy
from src.strategies.earnings_play import EarningsPlayStrategy
from src.strategies.volume_profile import VolumeProfileStrategy
from src.strategies.bollinger_squeeze import BollingerSqueezeStrategy
from src.strategies.fibonacci_retracement import FibonacciRetracementStrategy
from src.strategies.sector_rotation import SectorRotationStrategy
from src.utils.helpers import (
    get_next_expiry,
    is_market_open,
    is_trading_window,
)
from src.alerts.telegram_bot import TelegramAlerts
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

        # Track NIFTY strategy state
        self._orb_range_set = False
        self._nifty_prev_close: float = 0.0
        self._nifty_chain_cache = None  # Avoid duplicate NIFTY chain fetch

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

        # Alerts
        alerts_cfg = self.config.get("alerts", {})
        self.telegram = TelegramAlerts(
            enabled=alerts_cfg.get("telegram_enabled", True),
        )
        self._alert_cfg = alerts_cfg

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

        if strat_cfg.get("vwap_pullback", {}).get("enabled", True):
            s = VWAPPullbackStrategy(
                min_score=strat_cfg.get("vwap_pullback", {}).get("min_score", 72),
                target_pct=strat_cfg.get("vwap_pullback", {}).get("target_pct", 35),
                stop_loss_pct=strat_cfg.get("vwap_pullback", {}).get("stop_loss_pct", 25),
            )
            self.strategies.append(s)
            self.position_manager.register_strategy(s)

        if strat_cfg.get("gap_and_go", {}).get("enabled", True):
            s = GapAndGoStrategy(
                min_score=strat_cfg.get("gap_and_go", {}).get("min_score", 70),
                target_pct=strat_cfg.get("gap_and_go", {}).get("target_pct", 45),
                stop_loss_pct=strat_cfg.get("gap_and_go", {}).get("stop_loss_pct", 30),
            )
            self.strategies.append(s)
            self.position_manager.register_strategy(s)

        if strat_cfg.get("rsi_divergence", {}).get("enabled", True):
            s = RSIDivergenceStrategy(
                min_score=strat_cfg.get("rsi_divergence", {}).get("min_score", 68),
                target_pct=strat_cfg.get("rsi_divergence", {}).get("target_pct", 40),
                stop_loss_pct=strat_cfg.get("rsi_divergence", {}).get("stop_loss_pct", 25),
            )
            self.strategies.append(s)
            self.position_manager.register_strategy(s)

        if strat_cfg.get("scalping", {}).get("enabled", True):
            s = ScalpingStrategy(
                min_score=strat_cfg.get("scalping", {}).get("min_score", 80),
                target_pct=strat_cfg.get("scalping", {}).get("target_pct", 20),
                stop_loss_pct=strat_cfg.get("scalping", {}).get("stop_loss_pct", 15),
                max_hold_minutes=strat_cfg.get("scalping", {}).get("max_hold_minutes", 20),
            )
            self.strategies.append(s)
            self.position_manager.register_strategy(s)

        if strat_cfg.get("straddle_breakout", {}).get("enabled", True):
            s = StraddleBreakoutStrategy(
                min_score=strat_cfg.get("straddle_breakout", {}).get("min_score", 70),
                target_pct=strat_cfg.get("straddle_breakout", {}).get("target_pct", 50),
                stop_loss_pct=strat_cfg.get("straddle_breakout", {}).get("stop_loss_pct", 35),
            )
            self.strategies.append(s)
            self.position_manager.register_strategy(s)

        if strat_cfg.get("multi_timeframe", {}).get("enabled", True):
            s = MultiTimeframeStrategy(
                min_score=strat_cfg.get("multi_timeframe", {}).get("min_score", 70),
                target_pct=strat_cfg.get("multi_timeframe", {}).get("target_pct", 50),
                stop_loss_pct=strat_cfg.get("multi_timeframe", {}).get("stop_loss_pct", 30),
                time_exit_minutes=strat_cfg.get("multi_timeframe", {}).get("time_exit_minutes", 120),
            )
            self.strategies.append(s)
            self.position_manager.register_strategy(s)

        # --- NIFTY-specific strategies ---
        if strat_cfg.get("nifty_orb", {}).get("enabled", True):
            s = NiftyORBStrategy(
                min_score=strat_cfg.get("nifty_orb", {}).get("min_score", 72),
                target_pct=strat_cfg.get("nifty_orb", {}).get("target_pct", 40),
                stop_loss_pct=strat_cfg.get("nifty_orb", {}).get("stop_loss_pct", 30),
            )
            self.strategies.append(s)
            self.position_manager.register_strategy(s)

        if strat_cfg.get("nifty_gamma_blast", {}).get("enabled", True):
            s = NiftyGammaBlastStrategy(
                min_score=strat_cfg.get("nifty_gamma_blast", {}).get("min_score", 65),
                target_pct=strat_cfg.get("nifty_gamma_blast", {}).get("target_pct", 100),
                stop_loss_pct=strat_cfg.get("nifty_gamma_blast", {}).get("stop_loss_pct", 40),
                max_consolidation_range=strat_cfg.get("nifty_gamma_blast", {}).get("max_consolidation_range", 50),
            )
            self.strategies.append(s)
            self.position_manager.register_strategy(s)

        if strat_cfg.get("nifty_vix_regime", {}).get("enabled", True):
            s = NiftyVIXRegimeStrategy(
                min_score=strat_cfg.get("nifty_vix_regime", {}).get("min_score", 72),
            )
            self.strategies.append(s)
            self.position_manager.register_strategy(s)

        if strat_cfg.get("nifty_pcr_reversal", {}).get("enabled", True):
            s = NiftyPCRReversalStrategy(
                min_score=strat_cfg.get("nifty_pcr_reversal", {}).get("min_score", 68),
                target_pct=strat_cfg.get("nifty_pcr_reversal", {}).get("target_pct", 40),
                stop_loss_pct=strat_cfg.get("nifty_pcr_reversal", {}).get("stop_loss_pct", 25),
                bullish_pcr_threshold=strat_cfg.get("nifty_pcr_reversal", {}).get("bullish_pcr_threshold", 1.3),
                bearish_pcr_threshold=strat_cfg.get("nifty_pcr_reversal", {}).get("bearish_pcr_threshold", 0.7),
            )
            self.strategies.append(s)
            self.position_manager.register_strategy(s)

        if strat_cfg.get("nifty_gift_gap", {}).get("enabled", True):
            s = NiftyGIFTGapStrategy(
                min_score=strat_cfg.get("nifty_gift_gap", {}).get("min_score", 70),
                min_gap_pct=strat_cfg.get("nifty_gift_gap", {}).get("min_gap_pct", 0.5),
                target_pct=strat_cfg.get("nifty_gift_gap", {}).get("target_pct", 45),
                stop_loss_pct=strat_cfg.get("nifty_gift_gap", {}).get("stop_loss_pct", 30),
            )
            self.strategies.append(s)
            self.position_manager.register_strategy(s)

        if strat_cfg.get("nifty_event_day", {}).get("enabled", True):
            s = NiftyEventDayStrategy(
                min_score=strat_cfg.get("nifty_event_day", {}).get("min_score", 65),
                pre_event_target_pct=strat_cfg.get("nifty_event_day", {}).get("pre_event_target_pct", 30),
                post_event_target_pct=strat_cfg.get("nifty_event_day", {}).get("post_event_target_pct", 60),
                post_event_sl_pct=strat_cfg.get("nifty_event_day", {}).get("post_event_sl_pct", 40),
            )
            self.strategies.append(s)
            self.position_manager.register_strategy(s)

        if strat_cfg.get("nifty_oi_wall", {}).get("enabled", True):
            s = NiftyOIWallStrategy(
                min_score=strat_cfg.get("nifty_oi_wall", {}).get("min_score", 68),
                bounce_target_pct=strat_cfg.get("nifty_oi_wall", {}).get("bounce_target_pct", 35),
                break_target_pct=strat_cfg.get("nifty_oi_wall", {}).get("break_target_pct", 50),
                bounce_sl_pct=strat_cfg.get("nifty_oi_wall", {}).get("bounce_sl_pct", 25),
                break_sl_pct=strat_cfg.get("nifty_oi_wall", {}).get("break_sl_pct", 30),
                proximity_points=strat_cfg.get("nifty_oi_wall", {}).get("proximity_points", 50),
            )
            self.strategies.append(s)
            self.position_manager.register_strategy(s)

        # --- Additional strategies ---
        if strat_cfg.get("earnings_play", {}).get("enabled", True):
            s = EarningsPlayStrategy(
                min_score=strat_cfg.get("earnings_play", {}).get("min_score", 68),
                pre_target_pct=strat_cfg.get("earnings_play", {}).get("pre_target_pct", 25),
                post_target_pct=strat_cfg.get("earnings_play", {}).get("post_target_pct", 60),
                pre_sl_pct=strat_cfg.get("earnings_play", {}).get("pre_sl_pct", 15),
                post_sl_pct=strat_cfg.get("earnings_play", {}).get("post_sl_pct", 35),
                min_gap_pct=strat_cfg.get("earnings_play", {}).get("min_gap_pct", 2.0),
            )
            self.strategies.append(s)
            self.position_manager.register_strategy(s)

        if strat_cfg.get("volume_profile", {}).get("enabled", True):
            s = VolumeProfileStrategy(
                min_score=strat_cfg.get("volume_profile", {}).get("min_score", 70),
                bounce_target_pct=strat_cfg.get("volume_profile", {}).get("bounce_target_pct", 35),
                break_target_pct=strat_cfg.get("volume_profile", {}).get("break_target_pct", 45),
                bounce_sl_pct=strat_cfg.get("volume_profile", {}).get("bounce_sl_pct", 25),
                break_sl_pct=strat_cfg.get("volume_profile", {}).get("break_sl_pct", 30),
                proximity_pct=strat_cfg.get("volume_profile", {}).get("proximity_pct", 0.3),
            )
            self.strategies.append(s)
            self.position_manager.register_strategy(s)

        if strat_cfg.get("bollinger_squeeze", {}).get("enabled", True):
            s = BollingerSqueezeStrategy(
                min_score=strat_cfg.get("bollinger_squeeze", {}).get("min_score", 72),
                target_pct=strat_cfg.get("bollinger_squeeze", {}).get("target_pct", 45),
                stop_loss_pct=strat_cfg.get("bollinger_squeeze", {}).get("stop_loss_pct", 30),
                max_bandwidth=strat_cfg.get("bollinger_squeeze", {}).get("max_bandwidth", 0.04),
                min_expansion_ratio=strat_cfg.get("bollinger_squeeze", {}).get("min_expansion_ratio", 1.5),
            )
            self.strategies.append(s)
            self.position_manager.register_strategy(s)

        if strat_cfg.get("fibonacci_retracement", {}).get("enabled", True):
            s = FibonacciRetracementStrategy(
                min_score=strat_cfg.get("fibonacci_retracement", {}).get("min_score", 70),
                target_pct=strat_cfg.get("fibonacci_retracement", {}).get("target_pct", 40),
                stop_loss_pct=strat_cfg.get("fibonacci_retracement", {}).get("stop_loss_pct", 25),
                proximity_pct=strat_cfg.get("fibonacci_retracement", {}).get("proximity_pct", 0.3),
            )
            self.strategies.append(s)
            self.position_manager.register_strategy(s)

        if strat_cfg.get("sector_rotation", {}).get("enabled", True):
            s = SectorRotationStrategy(
                min_score=strat_cfg.get("sector_rotation", {}).get("min_score", 72),
                target_pct=strat_cfg.get("sector_rotation", {}).get("target_pct", 40),
                stop_loss_pct=strat_cfg.get("sector_rotation", {}).get("stop_loss_pct", 28),
                min_sector_rs=strat_cfg.get("sector_rotation", {}).get("min_sector_rs", 1.05),
                min_stock_rs=strat_cfg.get("sector_rotation", {}).get("min_stock_rs", 1.03),
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
            self.telegram.send("❌ <b>ENGINE FAILED</b>\nBroker login failed.")
            return

        # Setup graceful shutdown
        signal.signal(signal.SIGINT, self._shutdown_handler)
        signal.signal(signal.SIGTERM, self._shutdown_handler)

        self._running = True
        self.risk_manager.reset_daily()

        # Attach engine to Telegram for command handling
        self.telegram.attach_engine(self)

        self.telegram.send(
            f"🚀 <b>ENGINE STARTED</b>\n"
            f"Date: {date.today()}\n"
            f"Mode: {self.config['trading']['mode']}\n"
            f"Capital: ₹{self.config['trading'].get('capital', 100000):,.0f}"
        )

        try:
            self._run_loop()
        except Exception as e:
            logger.error(f"Fatal error: {e}", exc_info=True)
            self.telegram.send(f"🔥 <b>FATAL ERROR</b>\n{str(e)[:200]}")
        finally:
            self._shutdown()

    def _run_loop(self):
        """Main trading loop."""
        scan_done = False
        interval_seconds = 30  # Check every 30 seconds

        while self._running:
            now = datetime.now()

            # Ensure broker session is still valid (refresh/re-login if needed)
            if not self.broker.ensure_session():
                logger.error("Could not maintain broker session. Stopping engine.")
                break

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
                    self.telegram.send(
                        f"⏰ <b>HARD CUTOFF</b>\n"
                        f"Closing {len(open_positions)} positions at {now.strftime('%H:%M')}"
                    )
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

        # Batch-resolve tokens and fetch spot prices in one go
        token_map: dict[str, str] = {}  # symbol -> token
        for symbol in symbols_to_check:
            tok = self.broker.lookup_token("NSE", symbol)
            if tok:
                token_map[symbol] = tok

        spot_ltp: dict[str, float] = {}
        if token_map and hasattr(self.broker, "_batch_get_ltp"):
            ltp_results = self.broker._batch_get_ltp(
                {"NSE": list(token_map.values())}
            )
            for sym, tok in token_map.items():
                spot_ltp[sym] = ltp_results.get(tok, 0.0)
        else:
            for sym, tok in token_map.items():
                spot_ltp[sym] = self.broker.get_ltp("NSE", sym, tok)

        # Prepare NIFTY-specific strategy state before processing symbols
        nifty_spot = spot_ltp.get("NIFTY", 0.0)
        self._nifty_chain_cache = None
        if nifty_spot > 0:
            try:
                nifty_expiry = get_next_expiry("NIFTY")
                self._nifty_chain_cache = self.options_chain.get_chain_analysis(
                    "NIFTY", nifty_spot, nifty_expiry
                )
                self._prepare_nifty_strategies(nifty_spot, self._nifty_chain_cache)
            except Exception as e:
                logger.warning(f"Error preparing NIFTY strategies: {e}")

        for symbol in symbols_to_check:
            try:
                tok = token_map.get(symbol)
                price = spot_ltp.get(symbol, 0.0)
                if tok and price > 0:
                    self._process_single_symbol(symbol, tok, price)
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")

    def _prepare_nifty_strategies(self, nifty_spot: float, nifty_chain_analysis):
        """Set up NIFTY-specific strategy state once per loop iteration.

        Wires: nifty_orb, nifty_gamma_blast, nifty_vix_regime,
               nifty_pcr_reversal, nifty_gift_gap, nifty_event_day, nifty_oi_wall
        """
        now = datetime.now()
        today = date.today()

        for s in self.strategies:
            # --- PCR Reversal: feed PCR + OI walls from chain analysis ---
            if isinstance(s, NiftyPCRReversalStrategy) and nifty_chain_analysis:
                s.set_pcr_data(
                    pcr=nifty_chain_analysis.pcr_oi,
                    max_put_oi_strike=nifty_chain_analysis.max_pe_oi_strike,
                    max_call_oi_strike=nifty_chain_analysis.max_ce_oi_strike,
                )

            # --- OI Wall: feed max OI strikes from chain analysis ---
            if isinstance(s, NiftyOIWallStrategy) and nifty_chain_analysis:
                s.set_oi_walls(
                    max_put_oi_strike=nifty_chain_analysis.max_pe_oi_strike,
                    max_call_oi_strike=nifty_chain_analysis.max_ce_oi_strike,
                    put_oi=nifty_chain_analysis.total_pe_oi,
                    call_oi=nifty_chain_analysis.total_ce_oi,
                )

            # --- VIX Regime: fetch India VIX LTP ---
            if isinstance(s, NiftyVIXRegimeStrategy):
                try:
                    vix_ltp = self.broker.get_ltp("NSE", "India VIX", "99926017")
                    if vix_ltp > 0:
                        # Approximate VIX change from yesterday's daily candle
                        vix_daily = self.historical.get_daily_candles(
                            "India VIX", "99926017", "NSE", days=3
                        )
                        vix_change = 0.0
                        if not vix_daily.empty and len(vix_daily) >= 2:
                            prev_close = vix_daily["close"].iloc[-2]
                            if prev_close > 0:
                                vix_change = vix_ltp - prev_close
                        s.set_vix(vix_ltp, vix_change)
                except Exception as e:
                    logger.debug(f"Could not fetch VIX: {e}")

            # --- ORB: set from first 15min candle of NIFTY ---
            if isinstance(s, NiftyORBStrategy) and not self._orb_range_set:
                if now.time() >= time(9, 31):
                    nifty_tok = self.broker.lookup_token("NSE", "NIFTY")
                    if nifty_tok:
                        df_15 = self.historical.get_intraday_candles(
                            "NIFTY", nifty_tok, "NSE", "15min", days=1
                        )
                        if not df_15.empty:
                            today_candles = df_15[
                                df_15["timestamp"].dt.date == today
                            ] if "timestamp" in df_15.columns else df_15
                            if not today_candles.empty:
                                orb_high = today_candles["high"].iloc[0]
                                orb_low = today_candles["low"].iloc[0]
                                s.set_orb_range(orb_high, orb_low)
                                self._orb_range_set = True

            # --- Gamma Blast: set day's high/low from today's candles ---
            if isinstance(s, NiftyGammaBlastStrategy) and nifty_spot > 0:
                nifty_tok = self.broker.lookup_token("NSE", "NIFTY")
                if nifty_tok:
                    df_5 = self.historical.get_intraday_candles(
                        "NIFTY", nifty_tok, "NSE", "5min", days=1
                    )
                    if not df_5.empty:
                        today_candles = df_5[
                            df_5["timestamp"].dt.date == today
                        ] if "timestamp" in df_5.columns else df_5
                        if not today_candles.empty:
                            s.set_day_range(
                                today_candles["high"].max(),
                                today_candles["low"].min(),
                            )

            # --- GIFT Gap: approximate from prev close vs today's open ---
            if isinstance(s, NiftyGIFTGapStrategy) and nifty_spot > 0:
                if self._nifty_prev_close <= 0:
                    nifty_tok = self.broker.lookup_token("NSE", "NIFTY")
                    if nifty_tok:
                        nifty_daily = self.historical.get_daily_candles(
                            "NIFTY", nifty_tok, "NSE", days=5
                        )
                        if not nifty_daily.empty and len(nifty_daily) >= 2:
                            self._nifty_prev_close = nifty_daily["close"].iloc[-2]
                if self._nifty_prev_close > 0:
                    gap_pct = ((nifty_spot - self._nifty_prev_close)
                               / self._nifty_prev_close * 100)
                    s.set_gap_data(
                        gift_gap_pct=gap_pct,
                        previous_close=self._nifty_prev_close,
                        fii_net_buy=False,  # FII data not available via broker API
                    )

            # --- Event Day: check config for upcoming events ---
            if isinstance(s, NiftyEventDayStrategy):
                events_cfg = self.config.get("events", [])
                for evt in events_cfg:
                    evt_date = date.fromisoformat(evt.get("date", ""))
                    days_away = (evt_date - today).days
                    if 0 <= days_away <= 3:
                        s.set_event(evt.get("name", "Unknown"), evt_date, today)
                        break
                else:
                    if s.event_phase.value != "NO_EVENT":
                        s.clear_event()

    def _prepare_symbol_strategies(self, symbol: str, token: str, exchange: str,
                                   intraday_df, daily_df, chain_analysis,
                                   spot_price: float):
        """Set up per-symbol strategy state (RSI divergence, multi-timeframe, etc.)."""
        for s in self.strategies:
            # RSI Divergence: check divergence on intraday data
            if isinstance(s, RSIDivergenceStrategy):
                if not intraday_df.empty:
                    s.check_divergence(symbol, intraday_df)

            # Multi-Timeframe: analyze all three timeframes
            if isinstance(s, MultiTimeframeStrategy):
                df_15min = self.historical.get_intraday_candles(
                    symbol, token, exchange, "15min", days=5
                )
                s.analyze_timeframes(symbol, intraday_df, df_15min, daily_df)

    def _process_single_symbol(self, symbol: str, token: str = "", spot_price: float = 0.0):
        """Generate signal and potentially trade for a single symbol."""
        exchange = "NSE"
        if not token:
            token = self.broker.lookup_token(exchange, symbol)
            if not token:
                return

        # Get spot price if not provided
        if spot_price <= 0:
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

        # Get options chain (reuse cached NIFTY chain to avoid duplicate API calls)
        expiry = get_next_expiry(symbol)
        if symbol == "NIFTY" and self._nifty_chain_cache is not None:
            chain_analysis = self._nifty_chain_cache
        else:
            chain_analysis = self.options_chain.get_chain_analysis(
                symbol, spot_price, expiry
            )

        # --- Wire strategies that need pre-processing ---
        self._prepare_symbol_strategies(symbol, token, exchange,
                                        intraday_df, daily_df, chain_analysis,
                                        spot_price)

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

        # Alert on actionable signal
        if self._alert_cfg.get("alert_on_signal", True):
            self.telegram.send_signal_alert(
                symbol=symbol,
                direction=sig.direction,
                score=sig.score,
                option_type=sig.option_type,
            )

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
                # Alert on order
                if self._alert_cfg.get("alert_on_order", True):
                    self.telegram.send_order_alert(
                        action="BUY",
                        symbol=position.trading_symbol,
                        price=position.entry_price,
                        quantity=position.quantity,
                        strategy=position.strategy,
                    )
                break  # Only one trade per symbol per signal

    def _update_positions(self):
        """Update all open positions with current prices."""
        open_positions = self.position_manager.get_open_positions()
        if not open_positions:
            return

        # Collect all tokens needed, then batch-fetch
        option_tokens: dict[str, list[str]] = {}
        spot_tokens: dict[str, str] = {}  # symbol -> token

        for pos in open_positions:
            option_tokens.setdefault(pos.exchange, []).append(pos.token)
            if pos.symbol not in spot_tokens:
                spot_tok = self.broker.lookup_token("NSE", pos.symbol)
                if spot_tok:
                    spot_tokens[pos.symbol] = spot_tok

        # Batch fetch option LTPs
        nse_spot_list = list(spot_tokens.values())
        if nse_spot_list:
            option_tokens.setdefault("NSE", []).extend(nse_spot_list)

        ltp_map_all = {}
        if hasattr(self.broker, "_batch_get_ltp"):
            ltp_map_all = self.broker._batch_get_ltp(option_tokens)
        else:
            # Fallback to single calls
            for pos in open_positions:
                ltp = self.broker.get_ltp(pos.exchange, pos.trading_symbol, pos.token)
                if ltp > 0:
                    ltp_map_all[pos.token] = ltp
            for sym, tok in spot_tokens.items():
                ltp = self.broker.get_ltp("NSE", sym, tok)
                if ltp > 0:
                    ltp_map_all[tok] = ltp

        # Build price maps for position manager
        price_map = {}
        spot_prices = {}
        for pos in open_positions:
            ltp = ltp_map_all.get(pos.token, 0.0)
            if ltp > 0:
                price_map[pos.token] = ltp

        for sym, tok in spot_tokens.items():
            spot = ltp_map_all.get(tok, 0.0)
            if spot > 0:
                spot_prices[sym] = spot

        # Track closed count before update to detect new exits
        closed_before = len(self.position_manager.get_closed_positions())

        self.position_manager.update_positions(
            price_map, spot_prices, self._entry_spot_prices
        )

        # Send exit alerts for newly closed positions
        closed_positions = self.position_manager.get_closed_positions()
        if len(closed_positions) > closed_before:
            for pos in closed_positions[closed_before:]:
                sl_hit = "SL" in (pos.exit_reason or "").upper()
                tp_hit = "TARGET" in (pos.exit_reason or "").upper()
                should_alert = (
                    (sl_hit and self._alert_cfg.get("alert_on_sl_hit", True))
                    or (tp_hit and self._alert_cfg.get("alert_on_target_hit", True))
                    or (not sl_hit and not tp_hit)
                )
                if should_alert:
                    self.telegram.send_exit_alert(
                        symbol=pos.trading_symbol,
                        exit_price=pos.exit_price,
                        pnl=pos.pnl,
                        pnl_pct=pos.pnl_pct,
                        reason=pos.exit_reason or "UNKNOWN",
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

        if self._alert_cfg.get("alert_on_daily_summary", True):
            self.telegram.send_daily_summary(
                total_pnl=total_pnl,
                total_trades=len(closed),
                winning=winning,
                losing=losing,
            )

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
        self.telegram.send("🛑 <b>ENGINE STOPPED</b>\nTrading engine shut down.")
        self.telegram.stop_polling()
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
