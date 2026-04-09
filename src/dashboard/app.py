"""Streamlit dashboard for real-time trading monitoring.

Run with: streamlit run src/dashboard/app.py
"""

import sys
from datetime import date, datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd

from src.data.store import DataStore


def main():
    st.set_page_config(
        page_title="NSE Options Trading Dashboard",
        page_icon="📊",
        layout="wide",
    )

    st.title("NSE Options Trading Dashboard")

    store = DataStore("data/trading.db")

    # Sidebar
    st.sidebar.header("Controls")
    selected_date = st.sidebar.date_input("Date", date.today())
    auto_refresh = st.sidebar.checkbox("Auto Refresh (30s)", value=False)

    if auto_refresh:
        st.rerun()

    # Main content
    col1, col2, col3, col4 = st.columns(4)

    # Today's summary
    trades_df = store.get_trades(from_date=selected_date.isoformat())

    if not trades_df.empty:
        total_pnl = trades_df["pnl"].sum()
        total_trades = len(trades_df)
        winning = len(trades_df[trades_df["pnl"] > 0])
        losing = len(trades_df[trades_df["pnl"] < 0])
        win_rate = (winning / total_trades * 100) if total_trades > 0 else 0

        col1.metric("Total P&L", f"₹{total_pnl:+,.0f}")
        col2.metric("Total Trades", total_trades)
        col3.metric("Win Rate", f"{win_rate:.0f}%")
        col4.metric("W/L", f"{winning}/{losing}")
    else:
        col1.metric("Total P&L", "₹0")
        col2.metric("Total Trades", 0)
        col3.metric("Win Rate", "N/A")
        col4.metric("W/L", "0/0")

    # Trade log
    st.subheader("Trade Log")
    if not trades_df.empty:
        display_cols = [
            "symbol", "option_type", "strike", "strategy", "side",
            "entry_price", "exit_price", "pnl", "pnl_pct",
            "exit_reason", "signal_score", "entry_time", "exit_time",
        ]
        available_cols = [c for c in display_cols if c in trades_df.columns]
        st.dataframe(trades_df[available_cols], use_container_width=True)
    else:
        st.info("No trades for selected date")

    # Signals log
    st.subheader("Signals Generated")
    signals_query = f"SELECT * FROM signals WHERE timestamp >= '{selected_date}' ORDER BY timestamp DESC LIMIT 50"
    try:
        with store._connect() as conn:
            signals_df = pd.read_sql_query(signals_query, conn)
        if not signals_df.empty:
            st.dataframe(signals_df, use_container_width=True)
        else:
            st.info("No signals for selected date")
    except Exception:
        st.info("No signals data available")

    # Daily P&L history
    st.subheader("Daily P&L History")
    daily_df = store.get_daily_pnl_history(30)
    if not daily_df.empty:
        daily_df = daily_df.sort_values("date")
        st.bar_chart(daily_df.set_index("date")["total_pnl"])
    else:
        st.info("No daily P&L history available")

    # Strategy performance
    st.subheader("Strategy Performance")
    if not trades_df.empty and "strategy" in trades_df.columns:
        strat_perf = trades_df.groupby("strategy").agg(
            trades=("pnl", "count"),
            total_pnl=("pnl", "sum"),
            avg_pnl=("pnl", "mean"),
            win_rate=("pnl", lambda x: (x > 0).mean() * 100),
        ).round(1)
        st.dataframe(strat_perf, use_container_width=True)


if __name__ == "__main__":
    main()
