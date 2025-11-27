📈 AI Smart Trading Desk
Real-time stock analysis • ML forecasting • LSTM deep learning • AI trading verdict (Gemini 2.5 Flash)

🚀 Overview

AI Smart Trading Desk is a powerful Streamlit-based stock analysis platform that combines:

Real-time stock price data

AI-driven market sentiment (Gemini 2.5 Flash)

Machine Learning predictions

Deep Learning LSTM forecasting

Professional trading-style insights

Designed for traders, students, ML enthusiasts, and financial analysts.
⚠️ For educational use only. This is NOT financial advice.

⭐ Key Features
📊 Real-Time Market Dashboard

Live price updates

Daily price change

52-Week high/low

Price volatility

🤖 ML Models Included
Model	Best For	Notes
🔥 LSTM	Time-series forecasting	Most accurate
Linear Regression	Long-term trend	Fast but simple
Random Forest	Non-linear patterns	Handles noise well
SVR	Smooth predictions	Works after scaling
📉 Historical Charts

Candlestick/line charts

Auto-zoom and filter

Clean professional UI

🧠 Gemini 2.5 Flash Trading Verdict

Generates:

Market overview

Directional bias (Bullish / Bearish / Sideways)

Trade idea (Buy / Sell / Hold, Call/Put)

News sentiment analysis from 10 websites

Risk summary

Ideal entry/exit timing

⚙️ Custom Controls

Choose stock symbol

Select ML model

Choose prediction year

Toggle AI trading verdict

🛠️ Installation
1️⃣ Clone the repo
git clone https://github.com/exi-0/AI-smart-trading-desk.git
cd AI-smart-trading-desk

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Add your GEMINI API key

Create .env:

GEMINI_API_KEY=your_key_here

4️⃣ Run the app
streamlit run major.py

🌐 Deploy on Streamlit Cloud (Free)
Add to Streamlit → Secrets
GEMINI_API_KEY = "your_key_here"

Set entry point

Use:

major.py


Then deploy.

📁 Project Structure
AI-smart-trading-desk/
│── major.py               # Main Streamlit app
│── requirements.txt       # Dependencies
│── .env.example           # Example env file
│── README.md              # Project documentation
└── assets/                # Images, banners, preview assets

🔮 Future Improvements

Add ARIMA & Prophet forecasting

Add crypto price analysis

Add live news scraping

Add portfolio optimization

Build mobile-first UI

GPU-accelerated LSTM

💬 Contact

Developed by Shreyaan (exi-0)
📧 Open to collaboration
🔗 GitHub: https://github.com/exi-0

⚠️ Disclaimer

This project is strictly for educational and research purposes.
Not financial advice. Investments are risky.
