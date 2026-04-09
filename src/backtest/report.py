"""Backtest report generation."""

from datetime import date
from pathlib import Path

import pandas as pd

from src.backtest.engine import BacktestResult
from src.utils.logger import get_logger

logger = get_logger("backtest_report")


def generate_report(result: BacktestResult, output_dir: str = "reports") -> str:
    """Generate an HTML backtest report.

    Args:
        result: BacktestResult from the backtesting engine
        output_dir: Directory to save the report

    Returns:
        Path to the generated report file
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filename = f"backtest_{result.start_date}_{result.end_date}.html"
    filepath = Path(output_dir) / filename

    # Build trade table
    trade_rows = ""
    for t in result.trades:
        color = "green" if t.pnl > 0 else "red"
        trade_rows += f"""
        <tr>
            <td>{t.entry_time.strftime('%Y-%m-%d %H:%M') if t.entry_time else ''}</td>
            <td>{t.symbol}</td>
            <td>{t.direction}</td>
            <td>{t.option_type}</td>
            <td>{t.strategy}</td>
            <td>{t.entry_price:.2f}</td>
            <td>{t.exit_price:.2f}</td>
            <td style="color:{color}">{t.pnl:+,.0f}</td>
            <td style="color:{color}">{t.pnl_pct:+.1f}%</td>
            <td>{t.exit_reason}</td>
            <td>{t.signal_score:.0f}</td>
        </tr>"""

    # Build daily P&L table
    daily_rows = ""
    for d in result.daily_pnl:
        color = "green" if d["pnl"] > 0 else "red" if d["pnl"] < 0 else "black"
        daily_rows += f"""
        <tr>
            <td>{d['date']}</td>
            <td style="color:{color}">{d['pnl']:+,.0f}</td>
            <td>{d['capital']:,.0f}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Backtest Report: {result.start_date} to {result.end_date}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 20px 0; }}
        .stat-card {{ background: #f8f9fa; padding: 15px; border-radius: 6px; text-align: center; }}
        .stat-card .label {{ color: #666; font-size: 12px; text-transform: uppercase; }}
        .stat-card .value {{ font-size: 24px; font-weight: bold; margin-top: 5px; }}
        .positive {{ color: #4CAF50; }}
        .negative {{ color: #f44336; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; font-size: 13px; }}
        th {{ background: #4CAF50; color: white; padding: 10px 8px; text-align: left; }}
        td {{ padding: 8px; border-bottom: 1px solid #ddd; }}
        tr:hover {{ background: #f5f5f5; }}
    </style>
</head>
<body>
<div class="container">
    <h1>Backtest Report</h1>
    <p>Period: {result.start_date} to {result.end_date}</p>

    <div class="stats">
        <div class="stat-card">
            <div class="label">Total P&L</div>
            <div class="value {'positive' if result.total_pnl >= 0 else 'negative'}">
                {result.total_pnl:+,.0f}
            </div>
        </div>
        <div class="stat-card">
            <div class="label">Return</div>
            <div class="value {'positive' if result.total_pnl >= 0 else 'negative'}">
                {result.total_pnl/result.initial_capital*100:+.1f}%
            </div>
        </div>
        <div class="stat-card">
            <div class="label">Win Rate</div>
            <div class="value">{result.win_rate:.1f}%</div>
        </div>
        <div class="stat-card">
            <div class="label">Total Trades</div>
            <div class="value">{result.total_trades}</div>
        </div>
        <div class="stat-card">
            <div class="label">Profit Factor</div>
            <div class="value">{result.profit_factor:.2f}</div>
        </div>
        <div class="stat-card">
            <div class="label">Sharpe Ratio</div>
            <div class="value">{result.sharpe_ratio:.2f}</div>
        </div>
        <div class="stat-card">
            <div class="label">Max Drawdown</div>
            <div class="value negative">{result.max_drawdown:,.0f} ({result.max_drawdown_pct:.1f}%)</div>
        </div>
        <div class="stat-card">
            <div class="label">Avg Win / Avg Loss</div>
            <div class="value">{result.avg_win:+,.0f} / {result.avg_loss:+,.0f}</div>
        </div>
    </div>

    <h2>Trade Log ({result.total_trades} trades)</h2>
    <table>
        <tr>
            <th>Entry Time</th><th>Symbol</th><th>Direction</th><th>Type</th>
            <th>Strategy</th><th>Entry</th><th>Exit</th><th>P&L</th>
            <th>P&L%</th><th>Exit Reason</th><th>Score</th>
        </tr>
        {trade_rows}
    </table>

    <h2>Daily P&L</h2>
    <table>
        <tr><th>Date</th><th>P&L</th><th>Capital</th></tr>
        {daily_rows}
    </table>
</div>
</body>
</html>"""

    filepath.write_text(html)
    logger.info(f"Backtest report saved to {filepath}")
    return str(filepath)


def print_summary(result: BacktestResult):
    """Print backtest summary to console."""
    print(result.summary())
