# 52/LOW — Mean-Reversion Backtest Console

An interactive Streamlit app that turns the Project 1 (Group 2 Mini Case)
S&P 500 52-week-low trading strategy into a tool anyone can explore.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## Deploy for free (one-click public link)

The fastest free hosting option is **Streamlit Community Cloud**:

1. Push these three files (`app.py`, `requirements.txt`, `sp500_data.json`) to a
   public GitHub repo.
2. Go to https://share.streamlit.io and click "New app".
3. Point it at your repo, branch, and `app.py`.
4. Click Deploy. You'll get a live URL like
   `https://your-app-name.streamlit.app/` in about a minute.

That URL is what you submit as the "Live App Link" for the assignment.

Other free options that work the same way: **Hugging Face Spaces** (set SDK to
"Streamlit"), **Render**, or **Railway**.

## Files

| File              | What it does                                                 |
|-------------------|--------------------------------------------------------------|
| `app.py`          | The Streamlit app (UI + strategy engine)                     |
| `sp500_data.json` | 25+ years of S&P 500 daily closes, embedded so the app works offline |
| `requirements.txt`| Python dependencies for deployment                           |

## What the app does

The original Project 1 ran a single backtest with hardcoded parameters
(252-day lookback, 5-day hold, full 2000–2025 sample) and printed one number.
This app exposes those parameters as sidebar controls and updates the entire
analysis — equity curve, signal markers, return distribution, summary
statistics, and a plain-English verdict — every time the user changes one.

The strategy logic in `run_backtest()` is a direct, faithful port of the
Project 1 code, with the syntax errors fixed (`return=` and the misindented
inner sell loop). On the original 252/5 settings over 2000–2026 it produces
166 trades with a -0.7% average per-trade return, matching the corrected
Python notebook output exactly.
