import streamlit as st
import google.generativeai as genai
import yfinance as yf

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="AI Trading Chat", page_icon="💬", layout="wide")

# -------------------------
# PAGE TITLE + CSS
# -------------------------
st.markdown("""
<style>
body { background-color: #050816; }

.chat-bubble-user {
    background: #1a233a;
    padding: 12px 15px;
    border-radius: 12px;
    color: #e8f3ff;
    margin-bottom: 8px;
    border: 1px solid rgba(135,206,250,0.25);
}

.chat-bubble-ai {
    background: #0b1225;
    padding: 12px 15px;
    border-radius: 12px;
    color: #d7e6ff;
    margin-bottom: 8px;
    border: 1px solid rgba(0, 229, 255, 0.25);
}

.title {
    font-size: 32px;
    font-weight: 800;
    color: #e5f4ff;
    padding-bottom: 0px;
}
.subtitle {
    font-size: 15px;
    color: #9da9d9;
    margin-bottom: 25px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">💬 AI Trading Chat Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Ask trading questions, technical analysis, options ideas, or market insights.</div>',
    unsafe_allow_html=True
)

# -------------------------
# GEMINI API SETUP
# -------------------------
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=GEMINI_API_KEY)
MODEL = "gemini-2.5-flash"

# -------------------------
# SESSION CHAT HISTORY
# -------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------------
# SMART MARKET CONTEXT FUNCTION
# -------------------------
def get_market_context(text):
    """Extract ticker and automatically pull market data for smarter answers."""
    words = text.upper().split()
    ticker = None

    # Detect possible tickers
    for w in words:
        if len(w) <= 5 and w.isalpha():
            ticker = w
            break

    if not ticker:
        return None

    try:
        df = yf.download(ticker, period="5d", progress=False)
        if df.empty:
            return None

        last = df["Close"].iloc[-1]
        prev = df["Close"].iloc[-2]
        change = last - prev
        pct = (change / prev) * 100

        return f"""
        Market Data for {ticker}:
        • Last Price: {last:.2f}
        • 1-Day Change: {change:+.2f} ({pct:+.2f}%)
        """
    except:
        return None


# -------------------------
# CHAT INPUT
# -------------------------
user_msg = st.chat_input("Ask anything about stocks, crypto, options, charts…")

if user_msg:
    st.session_state.chat_history.append(("user", user_msg))

    # Add auto-market context if ticker found
    market_info = get_market_context(user_msg)
    enriched_prompt = user_msg

    if market_info:
        enriched_prompt = f"""
You are a professional institutional trader.
Use this market data in your answer:

{market_info}

User Question:
{user_msg}
"""

    # Gemini Response
    try:
        ai_msg = genai.GenerativeModel(MODEL).generate_content(enriched_prompt).text
    except Exception as e:
        ai_msg = f"⚠️ Error: {e}"

    st.session_state.chat_history.append(("ai", ai_msg))

# -------------------------
# DISPLAY CHAT HISTORY
# -------------------------
for sender, msg in st.session_state.chat_history:
    if sender == "user":
        st.markdown(f"<div class='chat-bubble-user'>🧑 <b>You</b><br>{msg}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-bubble-ai'>🤖 <b>AI Trader</b><br>{msg}</div>", unsafe_allow_html=True)
