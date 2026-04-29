# -*- coding: utf-8 -*-
"""
52/LOW — Mean-Reversion Backtest Console
Project 3: An interactive Streamlit app built on the Project 1 (Group 2 Mini Case)
strategy: when the S&P 500 hits an N-day low, buy and hold for K days.

Run locally:
    streamlit run app.py
"""

import json
import os
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="52/LOW — S&P 500 Mean-Reversion Backtest",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# A bit of CSS to clean up Streamlit's default look
st.markdown(
    """
    <style>
    .main .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }
    h1, h2, h3 { font-family: Georgia, 'Times New Roman', serif; }
    [data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
    div[data-testid="stMetricLabel"] { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; opacity: 0.75; }
    .verdict-box { padding: 1rem 1.25rem; border-left: 4px solid; border-radius: 4px; margin: 1rem 0; }
    .verdict-pos { border-color: #10b981; background: rgba(16,185,129,0.08); }
    .verdict-mix { border-color: #f59e0b; background: rgba(245,158,11,0.08); }
    .verdict-neg { border-color: #ef4444; background: rgba(239,68,68,0.08); }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# DATA LOADING
# -----------------------------------------------------------------------------
@st.cache_data
def load_sp500_data():
    """Load embedded S&P 500 daily close data (Jan 2000 – Apr 2026).

    Anchored to actual historical end-of-month closing values, with realistic
    daily Brownian-bridge interpolation between anchors. This makes the dataset
    fully self-contained — no internet connection or API key required.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "sp500_data.json")) as f:
        raw = json.load(f)
    df = pd.DataFrame(raw, columns=["Date", "Close"])
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    return df


# -----------------------------------------------------------------------------
# STRATEGY ENGINE — direct port of Project 1 logic, with bugs fixed
# -----------------------------------------------------------------------------
def run_backtest(prices: pd.Series, lookback: int, holding_period: int, capital: float):
    """
    Project 1 strategy: buy when close == rolling N-day low, hold for K days, sell.

    Returns a dict of: trades_df, equity_df, signals_df, stats
    """
    closes = prices.values
    dates = prices.index
    n = len(closes)

    # Rolling N-day low (inclusive of today)
    rolling_low = pd.Series(closes, index=dates).rolling(window=lookback, min_periods=lookback).min()

    # Buy signals: today's close equals the N-day rolling low
    buy_mask = pd.Series(closes, index=dates) == rolling_low

    # Build trades
    trades = []
    for i in range(n):
        if buy_mask.iloc[i]:
            sell_idx = i + holding_period
            if sell_idx < n:
                buy_p = closes[i]
                sell_p = closes[sell_idx]
                trades.append({
                    "Buy Date": dates[i],
                    "Buy Price": buy_p,
                    "Sell Date": dates[sell_idx],
                    "Sell Price": sell_p,
                    "Return %": (sell_p - buy_p) / buy_p * 100,
                    "buy_idx": i,
                })
    trades_df = pd.DataFrame(trades)

    # Equity curve: compound each completed trade
    equity = capital
    exit_to_equity = {}
    for t in trades:
        equity = equity * (t["Sell Price"] / t["Buy Price"])
        exit_to_equity[t["Sell Date"]] = equity
    final_equity = equity

    # Daily equity series (flat between trade exits)
    running = capital
    eq_values = []
    for d in dates:
        if d in exit_to_equity:
            running = exit_to_equity[d]
        eq_values.append(running)

    bh_values = capital * (closes / closes[0])

    equity_df = pd.DataFrame({
        "Date": dates,
        "Strategy": eq_values,
        "Buy & Hold": bh_values,
    }).set_index("Date")

    # Signals dataframe for price chart
    signals_df = pd.DataFrame({
        "Date": dates,
        "Close": closes,
        "Rolling Low": rolling_low.values,
        "Buy Signal": np.where(buy_mask.values, closes, np.nan),
    }).set_index("Date")

    # Stats
    if len(trades) > 0:
        rets = trades_df["Return %"].values
        avg_return = rets.mean()
        win_rate = (rets > 0).mean() * 100
        best = rets.max()
        worst = rets.min()
        stddev = rets.std(ddof=1) if len(rets) > 1 else 0.0
    else:
        avg_return = win_rate = best = worst = stddev = 0.0

    total_return = (final_equity / capital - 1) * 100
    bh_return = (closes[-1] / closes[0] - 1) * 100
    years = (dates[-1] - dates[0]).days / 365.25
    annualized = ((final_equity / capital) ** (1 / years) - 1) * 100 if years > 0 else 0.0
    bh_annualized = ((closes[-1] / closes[0]) ** (1 / years) - 1) * 100 if years > 0 else 0.0

    # Max drawdown of equity curve
    eq_arr = np.array(eq_values)
    peaks = np.maximum.accumulate(eq_arr)
    drawdowns = (eq_arr - peaks) / peaks * 100
    max_dd = drawdowns.min() if len(drawdowns) else 0.0

    stats = {
        "trade_count": len(trades),
        "avg_return": avg_return,
        "win_rate": win_rate,
        "best": best,
        "worst": worst,
        "stddev": stddev,
        "final_equity": final_equity,
        "total_return": total_return,
        "annualized": annualized,
        "bh_return": bh_return,
        "bh_annualized": bh_annualized,
        "max_dd": max_dd,
        "years": years,
    }

    return {
        "trades": trades_df,
        "equity": equity_df,
        "signals": signals_df,
        "stats": stats,
    }


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
df = load_sp500_data()
data_min = df.index.min().date()
data_max = df.index.max().date()

# Initialize defaults in session state (on first run only)
if "lookback" not in st.session_state:
    st.session_state.lookback = 252
    st.session_state.holding_period = 5
    st.session_state.start_date = date(2005, 1, 1)
    st.session_state.end_date = data_max
    st.session_state.capital = 10_000


def apply_preset(lb, hp, start, end):
    """Set widget values via session_state. Must be called BEFORE widgets render."""
    st.session_state.lookback = lb
    st.session_state.holding_period = hp
    st.session_state.start_date = start
    st.session_state.end_date = end


# ---------- SIDEBAR: INPUTS ----------
st.sidebar.markdown("### ⚙️ Strategy Inputs")
st.sidebar.caption("Every output below updates instantly when these change.")

st.sidebar.markdown("**Lookback window (N)**")
lookback = st.sidebar.slider(
    "trading days to define a 'new low'",
    min_value=20, max_value=504, step=1,
    key="lookback",
    label_visibility="collapsed",
)
st.sidebar.caption(f"{lookback} days  (252 = one trading year)")

st.sidebar.markdown("**Holding period (K)**")
holding_period = st.sidebar.slider(
    "days to hold each position",
    min_value=1, max_value=60, step=1,
    key="holding_period",
    label_visibility="collapsed",
)
st.sidebar.caption(f"{holding_period} days  (Project 1 default = 5)")

st.sidebar.markdown("**Start date**")
start_date = st.sidebar.date_input(
    "Start",
    min_value=data_min, max_value=data_max,
    key="start_date",
    label_visibility="collapsed",
)

st.sidebar.markdown("**End date**")
end_date = st.sidebar.date_input(
    "End",
    min_value=data_min, max_value=data_max,
    key="end_date",
    label_visibility="collapsed",
)

st.sidebar.markdown("**Starting capital**")
capital = st.sidebar.number_input(
    "Starting capital",
    min_value=100, max_value=10_000_000, step=1000,
    key="capital",
    label_visibility="collapsed",
)

# Presets
st.sidebar.markdown("---")
st.sidebar.markdown("**Quick presets**")

preset_cols = st.sidebar.columns(2)
preset_cols[0].button(
    "Original\n(252 / 5)", use_container_width=True,
    on_click=apply_preset, args=(252, 5, date(2005, 1, 1), data_max),
)
preset_cols[1].button(
    "Patient\n(252 / 20)", use_container_width=True,
    on_click=apply_preset, args=(252, 20, date(2005, 1, 1), data_max),
)
preset_cols[0].button(
    "Quarterly low\n(63 / 10)", use_container_width=True,
    on_click=apply_preset, args=(63, 10, date(2018, 1, 1), date(2024, 12, 31)),
)
preset_cols[1].button(
    "GFC stress\n(252 / 5)", use_container_width=True,
    on_click=apply_preset, args=(252, 5, date(2008, 1, 1), date(2010, 12, 31)),
)

# ---------- HEADER ----------
st.markdown(
    """
    <div style='border-bottom: 2px solid #d97706; padding-bottom: 0.5rem; margin-bottom: 1rem;'>
        <div style='font-size: 0.7rem; letter-spacing: 0.3em; color: #d97706; font-weight: 700;'>
            PROJECT 3 · MEAN-REVERSION BACKTEST CONSOLE
        </div>
        <h1 style='margin: 0.2rem 0 0 0; font-size: 2.4rem;'>
            When the market hits a new low, should you <em>buy the dip?</em>
        </h1>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "Folk wisdom says **\"buy low, sell high.\"** This app tests that on 25+ years "
    "of S&P 500 data: every time the index hits an N-day low, buy and hold for K trading days. "
    "Use the sidebar to change the assumptions — every chart and number below updates instantly."
)

# Filter to the user's date range
mask = (df.index.date >= start_date) & (df.index.date <= end_date)
prices = df.loc[mask, "Close"]

if len(prices) < lookback + holding_period + 5:
    st.error(
        f"❌ Not enough data in the selected window. You need at least "
        f"{lookback + holding_period + 5} trading days, but only have {len(prices)}. "
        "Widen the date range or shorten the lookback."
    )
    st.stop()

# Run the backtest
result = run_backtest(prices, lookback, holding_period, capital)
trades = result["trades"]
equity = result["equity"]
signals = result["signals"]
stats = result["stats"]

# ---------- VERDICT ----------
def get_verdict(stats):
    if stats["trade_count"] == 0:
        return None
    beats_bh = stats["annualized"] > stats["bh_annualized"]
    positive = stats["avg_return"] > 0
    if beats_bh and positive:
        return ("pos", "▲ EDGE", "Strategy outperformed buy-and-hold",
                f"Buying {stats['trade_count']} times at {lookback}-day lows and holding "
                f"{holding_period} days produced <b>{stats['annualized']:+.2f}% annualized</b> "
                f"vs. <b>{stats['bh_annualized']:+.2f}%</b> for buy-and-hold. "
                f"Average per-trade return: <b>{stats['avg_return']:+.2f}%</b>.")
    if positive and not beats_bh:
        return ("mix", "● MIXED", "Strategy was profitable but lagged buy-and-hold",
                f"{stats['trade_count']} trades averaged <b>{stats['avg_return']:+.2f}%</b> each "
                f"(<b>{stats['annualized']:+.2f}%</b> annualized), but a passive buy-and-hold "
                f"returned <b>{stats['bh_annualized']:+.2f}%</b> annualized over the same period. "
                "Activity wasn't rewarded.")
    return ("neg", "▼ NO EDGE", "Strategy lost money on average",
            f"Buying at {lookback}-day lows is a bet that markets bounce — over this period, "
            f"with a {holding_period}-day hold, that bet failed. <b>{stats['trade_count']} trades</b> "
            f"averaged <b>{stats['avg_return']:+.2f}%</b> (win rate <b>{stats['win_rate']:.1f}%</b>). "
            "New lows tend to be followed by more lows in the short term.")


verdict = get_verdict(stats)
if verdict:
    tone, badge, title, body = verdict
    st.markdown(
        f"""
        <div class='verdict-box verdict-{tone}'>
            <div style='font-size: 0.75rem; font-weight: 700; letter-spacing: 0.15em;
                color: {"#10b981" if tone=="pos" else "#f59e0b" if tone=="mix" else "#ef4444"};'>
                {badge}
            </div>
            <div style='font-size: 1.25rem; font-weight: 600; margin: 0.25rem 0;'>{title}</div>
            <div style='opacity: 0.9;'>{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.warning("No buy signals fired in this configuration. Try widening the date range or shrinking the lookback.")
    st.stop()

# ---------- STATS GRID ----------
st.markdown("### 📊 Results")

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Trades", stats["trade_count"])
c2.metric("Win Rate", f"{stats['win_rate']:.1f}%")
c3.metric("Avg / Trade", f"{stats['avg_return']:+.2f}%")
c4.metric("Best Trade", f"{stats['best']:+.2f}%")
c5.metric("Worst Trade", f"{stats['worst']:+.2f}%")
c6.metric("Std Dev / Trade", f"{stats['stddev']:.2f}%")

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Final Equity", f"${stats['final_equity']:,.0f}",
          f"{stats['total_return']:+.1f}% total")
c2.metric("Annualized — Strategy", f"{stats['annualized']:+.2f}%")
c3.metric("Annualized — Buy & Hold", f"{stats['bh_annualized']:+.2f}%")
edge = stats["annualized"] - stats["bh_annualized"]
c4.metric("Strategy Edge", f"{edge:+.2f}%",
          "outperforms" if edge > 0 else "underperforms",
          delta_color="normal" if edge > 0 else "inverse")
c5.metric("Max Drawdown", f"{stats['max_dd']:+.2f}%")
c6.metric("Years in Sample", f"{stats['years']:.1f}")

# ---------- EQUITY CURVE ----------
st.markdown("### 📈 Equity curve — strategy vs. buy-and-hold")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=equity.index, y=equity["Buy & Hold"],
    name="Buy & Hold", line=dict(color="#9ca3af", width=1.5),
    hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra>Buy & Hold</extra>",
))
fig.add_trace(go.Scatter(
    x=equity.index, y=equity["Strategy"],
    name="Strategy", line=dict(color="#d97706", width=2.2),
    hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra>Strategy</extra>",
))
fig.add_hline(y=capital, line_dash="dot", line_color="#6b7280",
              annotation_text=f"Starting capital ${capital:,.0f}",
              annotation_position="bottom right",
              annotation_font_size=10)
fig.update_layout(
    height=380,
    margin=dict(l=10, r=10, t=10, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis_title=None,
    yaxis_title="Equity ($)",
    hovermode="x unified",
    template="plotly_white",
)
st.plotly_chart(fig, use_container_width=True)

# ---------- PRICE CHART WITH SIGNALS ----------
st.markdown(f"### 🎯 Buy signals on the S&P 500")
st.caption(f"Each amber dot is a day where the index closed at its {lookback}-day low and a buy was triggered.")

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=signals.index, y=signals["Close"],
    name="S&P 500", line=dict(color="#374151", width=1.2),
    hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.2f}<extra>S&P 500</extra>",
))
fig2.add_trace(go.Scatter(
    x=signals.index, y=signals["Rolling Low"],
    name=f"{lookback}-day low", line=dict(color="#dc2626", width=1, dash="dash"),
    hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.2f}<extra>Rolling low</extra>",
))
buy_pts = signals.dropna(subset=["Buy Signal"])
fig2.add_trace(go.Scatter(
    x=buy_pts.index, y=buy_pts["Buy Signal"],
    name="Buy signal", mode="markers",
    marker=dict(color="#d97706", size=8, line=dict(color="#7c2d12", width=1)),
    hovertemplate="%{x|%Y-%m-%d}<br>Buy @ %{y:,.2f}<extra>Buy Signal</extra>",
))
fig2.update_layout(
    height=380,
    margin=dict(l=10, r=10, t=10, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis_title=None,
    yaxis_title="Index level",
    hovermode="x unified",
    template="plotly_white",
)
st.plotly_chart(fig2, use_container_width=True)

# ---------- HISTOGRAM ----------
st.markdown("### 📉 Distribution of per-trade returns")
st.caption("The center of mass of this histogram tells you what the strategy actually delivers per trade.")

fig3 = go.Figure()
rets = trades["Return %"].values
fig3.add_trace(go.Histogram(
    x=rets,
    xbins=dict(size=1),
    marker=dict(
        color=["#ef4444" if r < 0 else "#10b981" for r in rets],
        line=dict(color="#1f2937", width=0.5),
    ),
    hovertemplate="Bin: %{x}%%<br>Count: %{y}<extra></extra>",
))
fig3.add_vline(x=0, line_dash="dot", line_color="#6b7280")
fig3.add_vline(x=stats["avg_return"], line_color="#d97706", line_width=2,
               annotation_text=f"avg {stats['avg_return']:+.2f}%",
               annotation_position="top right",
               annotation_font_color="#d97706")
fig3.update_layout(
    height=280,
    margin=dict(l=10, r=10, t=10, b=10),
    xaxis_title="Per-trade return (%)",
    yaxis_title="Number of trades",
    showlegend=False,
    template="plotly_white",
    bargap=0.05,
)
st.plotly_chart(fig3, use_container_width=True)

# ---------- TRADE LOG ----------
with st.expander(f"📋 Show all {len(trades)} trades"):
    display_df = trades[["Buy Date", "Buy Price", "Sell Date", "Sell Price", "Return %"]].copy()
    display_df["Buy Date"] = display_df["Buy Date"].dt.strftime("%Y-%m-%d")
    display_df["Sell Date"] = display_df["Sell Date"].dt.strftime("%Y-%m-%d")
    display_df["Buy Price"] = display_df["Buy Price"].round(2)
    display_df["Sell Price"] = display_df["Sell Price"].round(2)
    display_df["Return %"] = display_df["Return %"].round(2)

    def color_return(val):
        return f"color: {'#10b981' if val > 0 else '#ef4444'}; font-weight: 600"

    # pandas Styler.applymap was renamed to Styler.map in pandas 2.1+
    style_method = getattr(display_df.style, "map", None) or display_df.style.applymap
    styled = style_method(color_return, subset=["Return %"])
    st.dataframe(styled, use_container_width=True, height=400)

    csv = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download trade log as CSV",
        data=csv,
        file_name=f"trades_lookback{lookback}_hold{holding_period}.csv",
        mime="text/csv",
    )

# ---------- METHODOLOGY ----------
st.markdown("---")
m1, m2, m3 = st.columns(3)
with m1:
    st.markdown("**The Strategy**")
    st.caption(
        "For each trading day, compute the rolling minimum closing price over the previous "
        "N days. When today's close equals that minimum — i.e. the index is at a new N-day "
        "low — fire a buy signal. Hold the position for K trading days, then sell."
    )
with m2:
    st.markdown("**The Math**")
    st.caption(
        "Per-trade return = (sell − buy) / buy. The equity curve compounds each completed "
        "trade. Annualized return uses CAGR = (final/initial)^(1/years) − 1. Buy-and-hold "
        "benchmark: same starting capital invested at day-one close, marked to market daily."
    )
with m3:
    st.markdown("**Caveats**")
    st.caption(
        "No transaction costs, slippage, or taxes. The S&P 500 is a price index — dividends "
        "are excluded. Daily values are anchored to actual end-of-month historical closes "
        "with realistic intra-month interpolation. Past results never guarantee future returns."
    )

st.markdown(
    "<div style='text-align: center; opacity: 0.5; font-size: 0.75rem; "
    "letter-spacing: 0.2em; padding: 1rem; text-transform: uppercase;'>"
    "Project 3 · Built from Project 1 (Group 2 Mini Case) strategy"
    "</div>",
    unsafe_allow_html=True,
)
