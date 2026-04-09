"""Backtesting engine — test strategies against historical data."""

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd

from src.data.options_chain import OptionsChainAnalysis
from src.signals.signal_engine import SignalEngine
from src.strategies.base import BaseStrategy, TradeSetup
from src.utils.helpers import get_atm_strike
from src.utils.logger import get_logger

logger = get_logger("backtest")


@dataclass
class BacktestTrade:
    """A single trade in the backtest."""

    symbol: str
    direction: str
    option_type: str
    strike: int
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: float = 0.0
    exit_reason: str = ""
    quantity: int = 1
    pnl: float = 0.0
    pnl_pct: float = 0.0
    strategy: str = ""
    signal_score: float = 0.0


@dataclass
class BacktestResult:
    """Complete backtest results."""

    start_date: date
    end_date: date
    initial_capital: float
    final_capital: float
    total_pnl: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    trades: list[BacktestTrade] = field(default_factory=list)
    daily_pnl: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        """Generate human-readable summary."""
        return (
            f"\n{'='*60}\n"
            f"BACKTEST RESULTS: {self.start_date} to {self.end_date}\n"
            f"{'='*60}\n"
            f"Initial Capital:  {self.initial_capital:>12,.0f}\n"
            f"Final Capital:    {self.final_capital:>12,.0f}\n"
            f"Total P&L:        {self.total_pnl:>+12,.0f} "
            f"({self.total_pnl/self.initial_capital*100:+.1f}%)\n"
            f"Total Trades:     {self.total_trades:>12}\n"
            f"Win Rate:         {self.win_rate:>11.1f}%\n"
            f"Avg Win:          {self.avg_win:>+12,.0f}\n"
            f"Avg Loss:         {self.avg_loss:>+12,.0f}\n"
            f"Profit Factor:    {self.profit_factor:>12.2f}\n"
            f"Max Drawdown:     {self.max_drawdown:>12,.0f} "
            f"({self.max_drawdown_pct:.1f}%)\n"
            f"Sharpe Ratio:     {self.sharpe_ratio:>12.2f}\n"
            f"{'='*60}"
        )


class BacktestEngine:
    """Event-driven backtesting engine."""

    def __init__(
        self,
        signal_engine: SignalEngine,
        strategies: list[BaseStrategy],
        initial_capital: float = 100000,
        slippage_pct: float = 0.5,
        risk_per_trade_pct: float = 2.0,
    ):
        self.signal_engine = signal_engine
        self.strategies = strategies
        self.initial_capital = initial_capital
        self.slippage_pct = slippage_pct
        self.risk_per_trade_pct = risk_per_trade_pct

    def run(
        self,
        symbol: str,
        intraday_data: pd.DataFrame,
        daily_data: pd.DataFrame,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> BacktestResult:
        """Run backtest on historical data.

        Args:
            symbol: Trading symbol
            intraday_data: Full intraday candle data (5min recommended)
            daily_data: Daily candle data
            start_date: Backtest start date
            end_date: Backtest end date
        """
        if intraday_data.empty:
            logger.error("No intraday data for backtest")
            return self._empty_result()

        capital = self.initial_capital
        trades: list[BacktestTrade] = []
        open_trade: Optional[BacktestTrade] = None
        high_price = 0.0
        daily_pnl_records = []
        peak_capital = capital
        max_drawdown = 0.0

        # Filter date range
        if start_date:
            intraday_data = intraday_data[
                intraday_data["timestamp"].dt.date >= start_date
            ]
        if end_date:
            intraday_data = intraday_data[
                intraday_data["timestamp"].dt.date <= end_date
            ]

        actual_start = intraday_data["timestamp"].dt.date.min()
        actual_end = intraday_data["timestamp"].dt.date.max()

        # Process candle by candle
        dates = sorted(intraday_data["timestamp"].dt.date.unique())

        for trade_date in dates:
            day_data = intraday_data[
                intraday_data["timestamp"].dt.date == trade_date
            ]
            day_pnl = 0.0

            # Get daily data up to this date for signals
            hist_daily = daily_data[daily_data["timestamp"].dt.date < trade_date].tail(30)

            for idx in range(20, len(day_data)):  # Need 20 candles for indicators
                candle_window = day_data.iloc[:idx + 1].copy()
                current = day_data.iloc[idx]
                current_time = current["timestamp"]
                current_price = current["close"]
                spot_price = current_price

                # Update open trade
                if open_trade:
                    if current_price > high_price:
                        high_price = current_price

                    # Check exit
                    should_exit, reason = self._check_exit(
                        open_trade, current_price, high_price, current_time
                    )

                    if should_exit:
                        exit_price = current_price * (1 - self.slippage_pct / 100)
                        open_trade.exit_price = exit_price
                        open_trade.exit_time = current_time
                        open_trade.exit_reason = reason
                        open_trade.pnl = (
                            (exit_price - open_trade.entry_price) * open_trade.quantity
                        )
                        open_trade.pnl_pct = (
                            (exit_price - open_trade.entry_price)
                            / open_trade.entry_price * 100
                        )

                        capital += open_trade.pnl
                        day_pnl += open_trade.pnl
                        trades.append(open_trade)
                        open_trade = None
                        high_price = 0.0
                        continue

                # Generate signal (only if no open trade)
                if open_trade is None:
                    signal = self.signal_engine.generate_signal(
                        symbol=symbol,
                        intraday_df=candle_window,
                        daily_df=hist_daily,
                        chain_analysis=None,
                        spot_price=spot_price,
                    )

                    if signal.is_actionable and signal.option_type:
                        for strategy in self.strategies:
                            if strategy.should_enter(signal, spot_price, current_time):
                                entry_price = current_price * (1 + self.slippage_pct / 100)
                                sl_distance = entry_price * 0.30
                                risk_amount = capital * (self.risk_per_trade_pct / 100)
                                quantity = max(1, int(risk_amount / sl_distance))

                                open_trade = BacktestTrade(
                                    symbol=symbol,
                                    direction=signal.direction.value,
                                    option_type=signal.option_type,
                                    strike=get_atm_strike(spot_price, symbol),
                                    entry_time=current_time,
                                    entry_price=entry_price,
                                    quantity=quantity,
                                    strategy=strategy.name,
                                    signal_score=signal.score,
                                )
                                high_price = entry_price
                                break

            # Force close at end of day
            if open_trade:
                last_price = day_data.iloc[-1]["close"]
                exit_price = last_price * (1 - self.slippage_pct / 100)
                open_trade.exit_price = exit_price
                open_trade.exit_time = day_data.iloc[-1]["timestamp"]
                open_trade.exit_reason = "EOD_CLOSE"
                open_trade.pnl = (exit_price - open_trade.entry_price) * open_trade.quantity
                open_trade.pnl_pct = (
                    (exit_price - open_trade.entry_price) / open_trade.entry_price * 100
                )
                capital += open_trade.pnl
                day_pnl += open_trade.pnl
                trades.append(open_trade)
                open_trade = None
                high_price = 0.0

            # Track daily P&L
            daily_pnl_records.append({"date": trade_date, "pnl": day_pnl, "capital": capital})

            # Track drawdown
            if capital > peak_capital:
                peak_capital = capital
            drawdown = peak_capital - capital
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        return self._compile_results(
            trades, daily_pnl_records, actual_start, actual_end,
            capital, max_drawdown, peak_capital,
        )

    def _check_exit(
        self,
        trade: BacktestTrade,
        current_price: float,
        high_price: float,
        current_time: datetime,
    ) -> tuple[bool, str]:
        """Check if backtest trade should exit."""
        entry = trade.entry_price

        # SL: 30% loss
        if current_price <= entry * 0.70:
            return True, "SL_HIT"

        # Target: 40% profit
        if current_price >= entry * 1.40:
            return True, "TARGET_HIT"

        # Trailing SL
        profit_from_peak = (high_price - entry) / entry * 100
        if profit_from_peak >= 20:
            trail_sl = entry + (high_price - entry) * 0.50
            if current_price <= trail_sl:
                return True, "TRAILING_SL"

        # Time: close before 3 PM
        if current_time.time() >= datetime.strptime("14:55", "%H:%M").time():
            return True, "TIME_EXIT"

        return False, ""

    def _compile_results(
        self,
        trades: list[BacktestTrade],
        daily_pnl: list[dict],
        start: date,
        end: date,
        final_capital: float,
        max_drawdown: float,
        peak_capital: float,
    ) -> BacktestResult:
        """Compile backtest statistics."""
        total_trades = len(trades)
        winning = [t for t in trades if t.pnl > 0]
        losing = [t for t in trades if t.pnl <= 0]

        total_pnl = final_capital - self.initial_capital
        win_rate = (len(winning) / total_trades * 100) if total_trades > 0 else 0
        avg_win = sum(t.pnl for t in winning) / len(winning) if winning else 0
        avg_loss = sum(t.pnl for t in losing) / len(losing) if losing else 0
        gross_profit = sum(t.pnl for t in winning)
        gross_loss = abs(sum(t.pnl for t in losing))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        max_dd_pct = (max_drawdown / peak_capital * 100) if peak_capital > 0 else 0

        # Sharpe ratio (approximate)
        daily_returns = pd.Series([d["pnl"] for d in daily_pnl])
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            sharpe = (daily_returns.mean() / daily_returns.std()) * (252 ** 0.5)
        else:
            sharpe = 0.0

        return BacktestResult(
            start_date=start,
            end_date=end,
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            total_pnl=total_pnl,
            total_trades=total_trades,
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_dd_pct,
            sharpe_ratio=sharpe,
            trades=trades,
            daily_pnl=daily_pnl,
        )

    def _empty_result(self) -> BacktestResult:
        return BacktestResult(
            start_date=date.today(),
            end_date=date.today(),
            initial_capital=self.initial_capital,
            final_capital=self.initial_capital,
            total_pnl=0, total_trades=0, winning_trades=0, losing_trades=0,
            win_rate=0, avg_win=0, avg_loss=0, profit_factor=0,
            max_drawdown=0, max_drawdown_pct=0, sharpe_ratio=0,
        )
