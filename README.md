# NSE Options Trading System

Automated options trading system for the National Stock Exchange (NSE) of India, focused on **option buying** (directional CE/PE trades) for NIFTY, BANKNIFTY, and F&O stock options.

## Features

- **Multi-factor signal engine** — Combines technical analysis (EMA, RSI, MACD, SuperTrend, VWAP, ADX), Open Interest analysis (PCR, OI buildup/unwinding, max pain), IV analysis, and price action
- **4 trading strategies** — Momentum buying, breakout buying, OI reversal, and expiry day specials
- **Strict risk management** — 2% risk per trade, 5% daily loss cap, trailing stops, time-based exits, VIX-adjusted sizing
- **Stock scanner** — Scans F&O universe for high-momentum, liquid stocks
- **Backtesting engine** — Test strategies against historical data with detailed reports
- **Paper trading** — Simulate trades with real market data before going live
- **Real-time dashboard** — Streamlit-based monitoring with P&L, positions, and signals
- **Telegram alerts** — Get notified on signals, orders, and exits

## Architecture

```
Data Layer → Signal Engine → Strategy → Risk Check → Order Execution
    ↑              ↓                                        ↓
  Angel One    4 Components                          Position Manager
  SmartAPI     (Tech + OI +                          (SL/Target/Trail)
               IV + Price Action)
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure
```bash
cp .env.example .env
# Edit .env with your Angel One API credentials
```

### 3. Run (Paper Trading)
```bash
python -m src.main
```

### 4. Run Dashboard
```bash
streamlit run src/dashboard/app.py
```

### 5. Run Tests
```bash
pytest tests/ -v
```

## Configuration

All parameters are in `config/settings.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `trading.mode` | `paper` | `paper` or `live` |
| `trading.capital` | `100000` | Trading capital in INR |
| `signals.min_score` | `75` | Minimum signal score to trade (0-100) |
| `stop_loss.initial_pct` | `30` | Initial SL as % of premium |
| `stop_loss.trailing_activation_pct` | `20` | Activate trailing SL after this profit % |

## Broker Setup (Angel One)

1. Open an Angel One account with API access
2. Get your API Key from SmartAPI portal
3. Set up TOTP authentication
4. Add credentials to `.env` file

## Strategies

| Strategy | Min Score | Target | Stop Loss | Best For |
|----------|----------|--------|-----------|----------|
| Momentum Buy | 75 | 40% | 30% | Trending markets |
| Breakout Buy | 78 | 50% | 30% | Range breakouts |
| OI Reversal | 80 | 35% | 25% | Short covering rallies |
| Expiry Day | 72 | 75% | 40% | Weekly expiry plays |

## Disclaimer

This software is for educational purposes only. Trading in options involves substantial risk of loss. Past performance does not guarantee future results. Use at your own risk.
