import streamlit as st
import google.generativeai as genai

st.set_page_config(page_title="AI Chat Assistant", page_icon="💬")

st.title("💬 AI Chat Assistant")

# Load key
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=GEMINI_API_KEY)

MODEL = "gemini-2.5-flash"

# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input box (Does NOT reload full app)
user_msg = st.chat_input("Ask anything…")

if user_msg:
    st.session_state.chat_history.append(("user", user_msg))

    # Call Gemini
    try:
        ai_msg = genai.GenerativeModel(MODEL).generate_content(user_msg).text
    except Exception as e:
        ai_msg = f"Error: {e}"

    st.session_state.chat_history.append(("ai", ai_msg))

# Show history
for sender, msg in st.session_state.chat_history:
    if sender == "user":
        st.markdown(f"🧑 **You:** {msg}")
    else:
        st.markdown(f"🤖 **AI:** {msg}")

