"""
AI-Powered Technical Stock Analysis Dashboard
- Robust yfinance fetching for NSE/ETFs
- Safe Plotly image export with kaleido fallback
- Gemini (Google Generative AI) integration if GOOGLE_API_KEY provided
- Uses Streamlit secrets and environment variables for API key
- Full logging to error.log and console + quick log tail in sidebar
- Added: Williams %R indicator with configurable period and its own subplot
- Modified: Replaced 20-Day EMA with 8-Day EMA
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import google.generativeai as genai
import tempfile
import os
import json
from datetime import datetime, timedelta
import logging
import sys
from typing import Tuple

# -------- GLOBAL LOGGING CONFIGURATION --------
LOG_FILE = "error.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
        logging.StreamHandler(stream=sys.stdout)
    ]
)

logger = logging.getLogger("stock_dashboard")
logger.info("App started.")
# ----------------------------------------------

# Try to detect kaleido so we can inform the user early
try:
    import kaleido  # noqa: F401
    _HAS_KALEIDO = True
    logger.info("Kaleido detected.")
except Exception:
    _HAS_KALEIDO = False
    logger.info("Kaleido not available.")

# Get API key from Streamlit secrets or environment
# SAFE: Get API key from Streamlit secrets or environment without throwing errors
GOOGLE_API_KEY = ""

try:
    # Attempt to read st.secrets
    if hasattr(st, "secrets"):
        try:
            GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "")
        except Exception as e:
            logger.info(f"st.secrets not available: {e}")
            GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
    else:
        GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
except Exception as e:
    logger.info(f"Secrets read failed: {e}")
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

if not GOOGLE_API_KEY:
    logger.info("No GOOGLE_API_KEY found in st.secrets or environment.")


if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        MODEL_NAME = "gemini-2.0-flash"
        gen_model = genai.GenerativeModel(MODEL_NAME)
        logger.info("Gemini configured.")
    except Exception as e:
        gen_model = None
        logger.exception("Failed to configure Gemini: %s", e)
        st.warning("Gemini configuration failed. Check GOOGLE_API_KEY and network.")
else:
    gen_model = None
    st.warning("Gemini API key not configured. Set GOOGLE_API_KEY in Streamlit secrets or environment variables to enable AI analysis.")
    logger.info("Gemini API key not configured.")

# UI
st.set_page_config(layout="wide")
st.title("AI-Powered Technical Stock Analysis Dashboard")
st.sidebar.header("Configuration")

# Show a small log tail in the sidebar for quick debugging
def tail_logs(logfile: str, lines: int = 12) -> str:
    try:
        with open(logfile, "r", encoding="utf-8") as f:
            all_lines = f.readlines()
        return "".join(all_lines[-lines:])
    except Exception as e:
        logger.exception("Failed to read log file: %s", e)
        return f"Could not read log file: {e}"

if st.sidebar.checkbox("Show recent logs", value=False):
    st.sidebar.text_area("Recent logs (tail)", value=tail_logs(LOG_FILE, 20), height=300)

# Tickers input
tickers_input = st.sidebar.text_input(
    "Enter Stock Tickers (comma-separated):",
    "HDFCBANK.NS,ICICIBANK.NS,KOTAKBANK.NS,HINDUNILVR.NS,NIFTYBEES.NS,GOLDBEES.NS"
)
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

# Date pickers
end_date_default = datetime.today().date()
start_date_default = end_date_default - timedelta(days=365)
start_date_dt = st.sidebar.date_input("Start Date", value=start_date_default)
end_date_dt = st.sidebar.date_input("End Date", value=end_date_default)

start_date_str = start_date_dt.strftime("%Y-%m-%d")
end_date_str = end_date_dt.strftime("%Y-%m-%d")

# Indicators
st.sidebar.subheader("Technical Indicators")
indicators = st.sidebar.multiselect(
    "Select Indicators:",
    ["20-Day SMA", "8-Day EMA", "20-Day Bollinger Bands", "VWAP", "Williams %R"],
    default=["20-Day SMA"]
)

# Williams %R period input (only meaningful if user selects the indicator)
willr_period = st.sidebar.number_input("Williams %R Period (days)", min_value=5, max_value=100, value=14)

# Helper: robust history fetch
def fetch_symbol_history(symbol: str, start: str, end: str) -> pd.DataFrame:
    logger.info("Fetching history for: %s | %s → %s", symbol, start, end)
    try:
        tk = yf.Ticker(symbol)
        df = tk.history(start=start, end=end, interval="1d")
        if df is None or df.empty:
            logger.warning("No data returned for %s using start/end query. Trying fallback period...", symbol)
            try:
                days = (datetime.fromisoformat(end).date() - datetime.fromisoformat(start).date()).days
                period = f"{max(30, min(1095, days))}d"
                logger.info("Fallback period used for %s: %s", symbol, period)
            except Exception as e:
                logger.exception("Period computation failed for %s: %s", symbol, e)
                period = "365d"
            df = tk.history(period=period, interval="1d")

        if isinstance(df, pd.DataFrame):
            df = df.dropna(how="all")
            logger.info("Fetched %d rows for %s", len(df), symbol)
        else:
            logger.error("Fetched data for %s is not a DataFrame", symbol)
            df = pd.DataFrame()

        return df
    except Exception as e:
        logger.exception("Error fetching data for %s: %s", symbol, e)
        return pd.DataFrame()

# Fetch button
if st.sidebar.button("Fetch Data"):
    logger.info("User requested data fetch for tickers: %s", tickers)
    stock_data = {}
    for ticker in tickers:
        st.sidebar.info(f"Fetching {ticker} ...")
        try:
            df = fetch_symbol_history(ticker, start_date_str, end_date_str)
            if df.empty:
                st.warning(f"No data found for {ticker}. Try different dates or symbol.")
                logger.warning("No data for %s after fetch.", ticker)
            else:
                # standardize column names
                df = df.rename(columns={c: c.capitalize() for c in df.columns})
                stock_data[ticker] = df
        except Exception as e:
            logger.exception("Unhandled exception fetching %s: %s", ticker, e)
            st.error(f"Error fetching {ticker}: {e}")

    st.session_state["stock_data"] = stock_data
    if stock_data:
        st.success("Stock data loaded: " + ", ".join(stock_data.keys()))
        logger.info("Stock data loaded for tickers: %s", ", ".join(stock_data.keys()))
    else:
        st.error("No stock data loaded. Adjust tickers or dates and try again.")
        logger.warning("No stock data loaded after fetch attempt.")

# Analysis
if "stock_data" in st.session_state and st.session_state["stock_data"]:
    def analyze_ticker(ticker: str, data: pd.DataFrame) -> Tuple[go.Figure, dict]:
        logger.info("Analyzing ticker: %s", ticker)
        df = data.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
                logger.info("Converted index to DatetimeIndex for %s", ticker)
            except Exception as e:
                logger.exception("Failed to convert index to datetime for %s: %s", ticker, e)

        # Prepare subplot: main candlestick + indicators on top, Williams %R on bottom if selected
        rows = 2 if "Williams %R" in indicators else 1
        row_heights = [0.7, 0.3] if rows == 2 else [1.0]
        shared_x = True

        fig = make_subplots(rows=rows, cols=1, shared_xaxes=shared_x, row_heights=row_heights,
                            vertical_spacing=0.03, specs=[[{"secondary_y": False}]] * rows)

        # Candlestick in row 1
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df.get('Open'),
                high=df.get('High'),
                low=df.get('Low'),
                close=df.get('Close'),
                name="Candlestick"
            ), row=1, col=1
        )

        # Indicators on same (top) subplot
        try:
            if "20-Day SMA" in indicators and 'Close' in df:
                sma = df['Close'].rolling(window=20, min_periods=1).mean()
                fig.add_trace(go.Scatter(x=df.index, y=sma, mode='lines', name='SMA (20)'), row=1, col=1)
            if "8-Day EMA" in indicators and 'Close' in df:
                ema = df['Close'].ewm(span=8, adjust=False).mean()
                fig.add_trace(go.Scatter(x=df.index, y=ema, mode='lines', name='EMA (8)'), row=1, col=1)
            if "20-Day Bollinger Bands" in indicators and 'Close' in df:
                sma_bb = df['Close'].rolling(window=20, min_periods=1).mean()
                std = df['Close'].rolling(window=20, min_periods=1).std().fillna(0)
                bb_upper = sma_bb + 2 * std
                bb_lower = sma_bb - 2 * std
                fig.add_trace(go.Scatter(x=df.index, y=bb_upper, mode='lines', name='BB Upper'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=bb_lower, mode='lines', name='BB Lower'), row=1, col=1)
            if "VWAP" in indicators and 'Close' in df and 'Volume' in df:
                vol_cum = df['Volume'].cumsum()
                vol_cum_safe = vol_cum.replace({0: pd.NA}).ffill().fillna(method='bfill')
                vwap = (df['Close'] * df['Volume']).cumsum() / vol_cum_safe
                fig.add_trace(go.Scatter(x=df.index, y=vwap, mode='lines', name='VWAP'), row=1, col=1)

            # Williams %R calculation and plot in row 2 (separate panel)
            if "Williams %R" in indicators and {'High', 'Low', 'Close'}.issubset(df.columns):
                hh = df['High'].rolling(window=willr_period, min_periods=1).max()
                ll = df['Low'].rolling(window=willr_period, min_periods=1).min()
                denom = (hh - ll).replace(0, pd.NA)
                willr = (hh - df['Close']) / denom * -100
                # Ensure same index
                willr = willr.fillna(method='ffill').fillna(method='bfill')
                # Add to second row
                fig.add_trace(go.Scatter(x=df.index, y=willr, mode='lines', name=f"Williams %R ({willr_period})"), row=2, col=1)
                # Add overbought/oversold lines (-20, -80)
                fig.add_hline(y=-20, line_dash="dash", row=2, col=1, annotation_text="-20 (Overbought)", annotation_position="top left")
                fig.add_hline(y=-80, line_dash="dash", row=2, col=1, annotation_text="-80 (Oversold)", annotation_position="bottom left")

        except Exception as e:
            logger.exception("Error computing indicators for %s: %s", ticker, e)

        fig.update_layout(xaxis_rangeslider_visible=False, template="plotly_white")

        # Try image export (kaleido) safely
        image_bytes = None
        image_error = None
        tmpname = None
        if _HAS_KALEIDO:
            try:
                logger.info("Attempting image export for %s...", ticker)
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                    tmpname = tmpfile.name
                fig.write_image(tmpname, engine="kaleido")
                with open(tmpname, "rb") as f:
                    image_bytes = f.read()
                os.remove(tmpname)
                tmpname = None
                logger.info("Image exported successfully for %s", ticker)
            except Exception as e:
                image_error = str(e)
                logger.exception("Failed to write image for %s: %s", ticker, e)
                try:
                    if tmpname and os.path.exists(tmpname):
                        os.remove(tmpname)
                except Exception:
                    logger.exception("Failed to remove temporary image file %s", tmpname)
        else:
            image_error = "Kaleido not installed. Run: python -m pip install --upgrade kaleido"
            logger.info(image_error)

        # Build prompt and call Gemini (if configured)
        analysis_prompt = (
            f"You are a Stock Trader specializing in Technical Analysis at a top financial institution. "
            f"Analyze the stock chart for {ticker} based on its candlestick chart and the displayed technical indicators. "
            f"Provide a detailed justification of your analysis, explaining what patterns, signals, and trends you observe. "
            f"Then, based solely on the chart, provide a recommendation from the following options: "
            f"'Strong Buy', 'Buy', 'Weak Buy', 'Hold', 'Weak Sell', 'Sell', or 'Strong Sell'. "
            f"Return your output as a JSON object with two keys: 'action' and 'justification'."
        )

        contents = []
        contents.append({"role": "user", "parts": [analysis_prompt]})
        if image_bytes and gen_model is not None:
            image_part = {"data": image_bytes, "mime_type": "image/png"}
            contents.append({"role": "user", "parts": [image_part]})
            logger.info("Attached chart image to Gemini request for %s", ticker)
        elif not _HAS_KALEIDO:
            contents.append({"role": "user", "parts": [f"Note: Image export unavailable ({image_error}). Please analyze using textual description only."]})
            logger.info("Gemini will be asked to analyze without image for %s", ticker)

        result = {"action": "N/A", "justification": "AI analysis not run (Gemini API key not configured)."}
        if gen_model is not None:
            try:
                logger.info("Sending request to Gemini for %s", ticker)
                response = gen_model.generate_content(contents=contents)
                result_text = getattr(response, "text", "") or str(response)
                logger.info("Raw Gemini response (first 300 chars) for %s: %s", ticker, result_text[:300])
                json_start_index = result_text.find('{')
                json_end_index = result_text.rfind('}') + 1
                if json_start_index != -1 and json_end_index > json_start_index:
                    json_string = result_text[json_start_index:json_end_index]
                    try:
                        result = json.loads(json_string)
                        logger.info("Parsed Gemini JSON for %s: %s", ticker, result)
                    except Exception as je:
                        logger.exception("JSON parse failed for %s: %s", ticker, je)
                        result = {"action": "Error", "justification": f"Failed to parse JSON from Gemini response. Raw: {result_text}"}
                else:
                    logger.error("Gemini returned no valid JSON for %s.", ticker)
                    result = {"action": "Error", "justification": f"No JSON found. Raw response:\n{result_text}"}
            except Exception as e:
                logger.exception("Gemini API error for %s: %s", ticker, e)
                result = {"action": "Error", "justification": f"Gemini API error: {e}. {image_error or ''}"}

        if image_error:
            prev_just = result.get("justification", "") or ""
            result["justification"] = prev_just + f"\n\n[Note: image export issue: {image_error}]"

        return fig, result

    # Build tabs
    tab_names = ["Overall Summary"] + list(st.session_state["stock_data"].keys())
    tabs = st.tabs(tab_names)

    overall_results = []

    for i, ticker in enumerate(st.session_state["stock_data"]):
        try:
            data = st.session_state["stock_data"][ticker]
            fig, result = analyze_ticker(ticker, data)
            overall_results.append({"Stock": ticker, "Recommendation": result.get("action", "N/A")})
            with tabs[i + 1]:
                st.subheader(f"Analysis for {ticker}")
                st.plotly_chart(fig, use_container_width=True)
                st.write("**Detailed Justification:**")
                st.write(result.get("justification", "No justification provided."))
            logger.info("Analysis displayed for %s", ticker)
        except Exception as e:
            logger.exception("Error during analysis/display for %s: %s", ticker, e)
            with tabs[i + 1]:
                st.error(f"Analysis failed for {ticker}: {e}")

    with tabs[0]:
        st.subheader("Overall Structured Recommendations")
        df_summary = pd.DataFrame(overall_results)
        st.table(df_summary)

else:
    st.info("Please fetch stock data using the sidebar.")
    logger.info("No stock_data in session; waiting for user to fetch data.")
