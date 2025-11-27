import os
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime
from dotenv import load_dotenv

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

import google.generativeai as genai

# ===============================
# ENV + GEMINI SETUP
# ===============================
import streamlit as st
import google.generativeai as genai

# ===== STREAMLIT CLOUD SECRET HANDLING =====
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("❌ GEMINI_API_KEY missing in Streamlit Secrets!")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)
GEMINI_MODEL = "gemini-2.5-flash"



genai.configure(api_key=GEMINI_API_KEY)
GEMINI_MODEL = "gemini-2.5-flash"

# ===============================
# STREAMLIT PAGE CONFIG + CSS
# ===============================
st.set_page_config(
    page_title="Smart Trading Desk",
    page_icon="📈",
    layout="wide"
)

st.markdown(
    """
    <style>
    body {
        background-color: #050816;
    }
    .main-title {
        text-align: left;
        font-size: 34px;
        font-weight: 800;
        color: #e5f4ff;
        padding: 10px 0 0 0;
    }
    .sub-title {
        color: #98a6c7;
        font-size: 15px;
        margin-bottom: 25px;
    }
    .card {
        background: linear-gradient(135deg, rgba(18,25,52,0.98), rgba(8,17,34,0.97));
        padding: 18px 20px;
        border-radius: 18px;
        border: 1px solid rgba(135,206,250,0.25);
        box-shadow: 0 10px 25px rgba(0,0,0,0.45);
        margin-bottom: 18px;
    }
    .metric-card {
        background: radial-gradient(circle at top left, #273060, #060b1d);
        border-radius: 16px;
        padding: 14px 16px;
        border: 1px solid rgba(135,206,250,0.35);
        box-shadow: 0 8px 20px rgba(0,0,0,0.4);
    }
    .metric-label {
        font-size: 12px;
        text-transform: uppercase;
        color: #9da9d9;
        letter-spacing: 1.3px;
    }
    .metric-value {
        font-size: 22px;
        font-weight: 700;
        color: #e8f3ff;
        margin-top: 6px;
    }
    .metric-sub {
        font-size: 11px;
        color: #7c87b0;
        margin-top: 2px;
    }
    .ai-badge {
        display: inline-block;
        padding: 2px 9px;
        border-radius: 999px;
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 1.4px;
        background: rgba(0, 229, 255, 0.18);
        color: #7ee3ff;
        border: 1px solid rgba(0, 229, 255, 0.4);
        margin-bottom: 4px;
    }
    .ai-title {
        font-size: 20px;
        font-weight: 700;
        color: #f5fbff;
        margin-bottom: 6px;
    }
    .disclaimer {
        font-size: 11px;
        color: #808aa8;
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="main-title">Smart Trading Desk</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Real-time prices, ML-based forecasts, and an AI trader verdict — all in one dashboard. '
    'For educational use only, not financial advice.</div>',
    unsafe_allow_html=True
)

# ===============================
# SIDEBAR CONTROLS
# ===============================
with st.sidebar:
    st.header("⚙️ Controls")

    symbol = st.text_input("Stock symbol", "GOOG")
    start_date = st.text_input("Start date (YYYY-MM-DD)", "2015-01-01")
    end_date = st.text_input("End date (YYYY-MM-DD)", datetime.today().strftime("%Y-%m-%d"))

    model_choice = st.selectbox(
        "ML model",
        ["Linear Regression", "Random Forest", "SVR", "LSTM"]
    )

    future_year = st.number_input(
        "Year for forecast (ML)",
        min_value=datetime.today().year,
        max_value=datetime.today().year + 10,
        value=datetime.today().year + 1
    )

    st.markdown("---")
    use_ai_verdict = st.checkbox("🤖 Generate AI trading verdict (Gemini)", value=True)

# ===============================
# DATA LOADING
# ===============================
@st.cache_data(show_spinner=True)
def load_data(sym, start, end):
    df = yf.download(sym, start=start, end=end)
    df.reset_index(inplace=True)
    return df

data = load_data(symbol, start_date, end_date)

if data.empty:
    st.error("❌ Could not fetch data. Check stock symbol or date range.")
    st.stop()

# ===============================
# BASIC MARKET SNAPSHOT
# ===============================
latest = data.iloc[-1]
previous = data.iloc[-2] if len(data) > 1 else latest

current_price = float(latest["Close"])
prev_price = float(previous["Close"])
day_change = current_price - prev_price
day_change_pct = (day_change / prev_price * 100) if prev_price != 0 else 0.0

# 52-week high/low (last ~252 trading days)
last_252 = data.tail(252)
hi_52w = float(last_252["High"].max())
lo_52w = float(last_252["Low"].min())

volatility_pct = float(data["Close"].pct_change().std() * np.sqrt(252) * 100)  # annualized

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Last Price</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">${current_price:.2f}</div>', unsafe_allow_html=True)
    change_color = "#4caf50" if day_change >= 0 else "#ff5252"
    st.markdown(
        f'<div class="metric-sub" style="color:{change_color};">'
        f'{day_change:+.2f} ({day_change_pct:+.2f}%) today</div>',
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">52W Range</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="metric-value">${lo_52w:.2f} - ${hi_52w:.2f}</div>',
        unsafe_allow_html=True
    )
    st.markdown('<div class="metric-sub">Lowest & highest in last 52 weeks</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Volatility</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{volatility_pct:.1f}%</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-sub">Annualized (based on daily returns)</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Data Points</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{len(data):,}</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-sub">Daily candles in selected period</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# HISTORICAL PRICE CHART
# ===============================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("#### 📉 Historical Price Chart")

fig_hist, ax_hist = plt.subplots(figsize=(11, 4))
ax_hist.plot(data["Date"], data["Close"], linewidth=1.4)
ax_hist.set_title(f"{symbol} — Close Price History", color="#e8f3ff")
ax_hist.set_xlabel("Date")
ax_hist.set_ylabel("Price ($)")
ax_hist.grid(alpha=0.3)
for spine in ax_hist.spines.values():
    spine.set_color("#444b6a")
ax_hist.tick_params(colors="#c0c7e5")
st.pyplot(fig_hist)
st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# ML MODEL TRAINING + PREDICTION
# ===============================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("#### 🤖 ML Forecast: Actual vs Predicted")

# Prepare data
data["Date_ordinal"] = pd.to_datetime(data["Date"]).apply(lambda d: d.toordinal())
X = data[["Date_ordinal"]]
y = data["Close"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# ---------- LSTM helpers ----------
@st.cache_data
def prepare_lstm_sequences(close_series, seq_len=60):
    prices = close_series.values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices)

    X_seq = np.lib.stride_tricks.sliding_window_view(scaled[:-1, 0], seq_len)
    y_seq = scaled[seq_len:, 0]

    split = int(0.8 * len(X_seq))
    X_train_l = X_seq[:split].reshape(-1, seq_len, 1)
    X_test_l = X_seq[split:].reshape(-1, seq_len, 1)
    y_train_l = y_seq[:split]
    y_test_l = y_seq[split:]
    return X_train_l, X_test_l, y_train_l, y_test_l, scaler

def train_lstm_model(X_train_l, y_train_l):
    model = Sequential([
        LSTM(32, input_shape=(X_train_l.shape[1], 1)),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train_l, y_train_l, epochs=10, batch_size=32, verbose=0)
    return model

# ---------- Train selected model ----------
if model_choice == "Linear Regression":
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_test_eval = y_test

elif model_choice == "Random Forest":
    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=8,
        min_samples_split=8,
        min_samples_leaf=4,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_test_eval = y_test

elif model_choice == "SVR":
    scaler_svr = StandardScaler()
    X_train_scaled = scaler_svr.fit_transform(X_train)
    X_test_scaled = scaler_svr.transform(X_test)
    model = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_test_eval = y_test

elif model_choice == "LSTM":
    X_train_l, X_test_l, y_train_l, y_test_l, scaler_lstm = prepare_lstm_sequences(data["Close"])
    lstm_model = train_lstm_model(X_train_l, y_train_l)
    lstm_pred = lstm_model.predict(X_test_l, verbose=0)
    y_pred = scaler_lstm.inverse_transform(lstm_pred).flatten()
    y_test_eval = scaler_lstm.inverse_transform(y_test_l.reshape(-1, 1)).flatten()

# ---------- Metrics ----------
rmse = float(np.sqrt(mean_squared_error(y_test_eval, y_pred)))
mae = float(mean_absolute_error(y_test_eval, y_pred))
r2 = float(r2_score(y_test_eval, y_pred))

m1, m2, m3 = st.columns(3)
m1.metric("RMSE", f"{rmse:.2f}")
m2.metric("MAE", f"{mae:.2f}")
m3.metric("R² Score", f"{r2:.3f}")

# ---------- Actual vs Predicted Plot ----------
fig_ml, ax_ml = plt.subplots(figsize=(11, 4))
ax_ml.plot(data["Date"], data["Close"], label="Actual", linewidth=1.4)

if model_choice == "LSTM":
    # align predictions to last len(y_pred) points
    dates_pred = data["Date"].iloc[-len(y_pred):]
    ax_ml.plot(dates_pred, y_pred, label="Predicted", linewidth=1.4, linestyle="--")
else:
    test_dates = data.loc[X_test.index, "Date"]
    ax_ml.scatter(test_dates, y_pred, label="Predicted", s=18)

ax_ml.set_title(f"{symbol} — Actual vs Predicted ({model_choice})", color="#e8f3ff")
ax_ml.set_xlabel("Date")
ax_ml.set_ylabel("Price ($)")
ax_ml.legend()
ax_ml.grid(alpha=0.3)
for spine in ax_ml.spines.values():
    spine.set_color("#444b6a")
ax_ml.tick_params(colors="#c0c7e5")
st.pyplot(fig_ml)

# ---------- Future Price Prediction ----------
predicted_future_price = None
target_date = datetime(future_year, 1, 1)
target_ordinal = target_date.toordinal()

if model_choice == "LSTM":
    # sequence-based extrapolation
    seq_len = 60
    close_vals = data["Close"].values
    last_seq = close_vals[-seq_len:].reshape(-1, 1)
    scaled_last = scaler_lstm.transform(last_seq)[:, 0]

    days_ahead = (target_date.date() - data["Date"].iloc[-1].date()).days
    if days_ahead <= 0:
        st.warning("⚠️ Forecast year is not in the future relative to data end.")
    else:
        seq = scaled_last.copy()
        future_scaled = []
        for _ in range(days_ahead):
            inp = seq.reshape(1, seq_len, 1)
            next_scaled = lstm_model.predict(inp, verbose=0)[0, 0]
            future_scaled.append(next_scaled)
            seq = np.append(seq[1:], next_scaled)
        future_scaled = np.array(future_scaled).reshape(-1, 1)
        future_prices = scaler_lstm.inverse_transform(future_scaled)
        predicted_future_price = float(future_prices[-1, 0])
else:
    if model_choice == "SVR":
        target_feat = scaler_svr.transform([[target_ordinal]])
        predicted_future_price = float(model.predict(target_feat)[0])
    else:
        predicted_future_price = float(model.predict([[target_ordinal]])[0])

if predicted_future_price is not None:
    st.success(
        f"🔮 {model_choice} predicts **${predicted_future_price:.2f}** "
        f"around {future_year}-01-01"
    )
else:
    st.info("No future price computed (check forecast year / model).")

st.markdown('</div>', unsafe_allow_html=True)  # close ML card

# ===============================
# AI TRADING VERDICT (GEMINI)
# ===============================
if use_ai_verdict:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="ai-badge">AI TRADING VERDICT</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="ai-title">{symbol} — Positioning Insight</div>', unsafe_allow_html=True)

    verdict_context = f"""
    Symbol: {symbol}
    Current price: {current_price:.2f}
    Daily change: {day_change:+.2f} ({day_change_pct:+.2f}%)
    52W low: {lo_52w:.2f}
    52W high: {hi_52w:.2f}
    Annualized volatility (approx): {volatility_pct:.2f}%
    ML model used: {model_choice}
    ML RMSE: {rmse:.2f}
    ML MAE: {mae:.2f}
    ML R2: {r2:.3f}
    Future forecast year: {future_year}
    Future forecast price (if computed): {predicted_future_price if predicted_future_price is not None else 'N/A'}
    """

    prompt = f"""
    You are an institutional-grade Trading & News-Sentiment Assistant.
    Your job: Read all provided ML signals + all extracted news headlines (with URLs), perform sentiment analysis, and give a professional trader-level market verdict.

    You MUST:
    • Analyze each news item
    • Perform sentiment scoring (Positive / Negative / Neutral)
    • Infer market trend from combined news sentiment + ML context
    • Clearly explain WHY you think the stock is Bullish / Bearish / Sideways
    • Use the website links below for sentiment-based reasoning

    CONTEXT:
    {verdict_context}

    NEWS SOURCES (Use ALL with links in analysis):
    1. Bloomberg — https://www.bloomberg.com/
    2. Reuters (Markets) — https://www.reuters.com/markets/
    3. MarketWatch — https://www.marketwatch.com/
    4. Yahoo Finance — https://finance.yahoo.com/
    5. Wikipedia — https://www.wikipedia.org/
    6. TheStreet — https://www.thestreet.com/
    7. Seeking Alpha — https://seekingalpha.com/
    8. Investing.com (India) — https://in.investing.com/
    9. Moneycontrol — https://www.moneycontrol.com/
    10. Economic Times (ETMarkets) — https://economictimes.indiatimes.com/markets

    REQUIREMENTS:
    Output in EXACTLY 6 sections:

    1) **Market Overview**
    - Summarize overall environment based on ML signals + news sentiment.

    2) **Directional Bias**
    - Choose: Bullish / Bearish / Sideways
    - Provide 1–2 lines explaining WHY (based on sentiment + ML signals).

    3) **Trade Idea**
    - For equity: BUY / SELL / HOLD
    - For options: CALL / PUT / No clear opportunity
    - Mention timeframe: SHORT-TERM or LONG-TERM

    4) **Risk & Notes**
    - Mention risks based on negative headlines, volatility, uncertainty.

    5) **Top 10 News Sentiment Analysis (VERY IMPORTANT)**
    For EACH news item (include its website link):
    - Headline sentiment (Positive / Negative / Neutral)
    - Why this headline affects the stock
    - Whether it pushes trend UP or DOWN
    - Trend Impact: +Bullish / –Bearish / 0 Neutral

    Also add:
    - “Final Trend Based on ALL News Sentiments: Bullish / Bearish / Sideways”
    - Give top 2–3 reasons behind that trend prediction.

    6) **Buy/Sell Timing**
    - Give ideal entry and exit timing (e.g., Buy between 9:20–9:40 AM; Sell at 2:45–3:10 PM)

    CLOSING LINE:
    "This is not financial advice. Do your own research."
    """

    try:
        ai_response = genai.GenerativeModel(GEMINI_MODEL).generate_content(prompt)
        st.write(ai_response.text)
    except Exception as e:
        st.error(f"⚠️ AI verdict failed: {e}")

    st.markdown(
        '<div class="disclaimer">All outputs are for educational purposes only and do not constitute financial advice.</div>',
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)


#streamlit run major.py


