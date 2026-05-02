"""
From Tweets To Trades — Streamlit App
Pixel-perfect conversion matching the original Django site screenshots.
Run:  streamlit run app.py
"""

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests, os, re, json, warnings, sqlite3
warnings.filterwarnings("ignore")

# Load .env file so GEMINI_API_KEY and other secrets are available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
# ── optional heavy deps ───────────────────────────────────────────────────────
try:
    from tensorflow.keras.models import load_model
    import joblib
    ML_AVAILABLE = True
except Exception:
    ML_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch, torch.nn.functional as F
    FINBERT_AVAILABLE = True
except Exception:
    FINBERT_AVAILABLE = False

try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_AVAILABLE = True
except Exception:
    TRANSLATOR_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
try:
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    NEWS_API_KEY   = st.secrets.get("NEWS_API_KEY") or os.getenv("NEWS_API_KEY")
except Exception:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    NEWS_API_KEY   = os.getenv("NEWS_API_KEY")
LOOKBACK = 60

st.set_page_config(
    page_title="From Tweets To Trades",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
# DATABASE
# ══════════════════════════════════════════════════════════════════════════════
def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY, password TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS history
                 (username TEXT, ticker TEXT, last_price REAL, viewed_at TIMESTAMP,
                  PRIMARY KEY (username, ticker))''')
    c.execute('''CREATE TABLE IF NOT EXISTS watchlist
                 (username TEXT, ticker TEXT, added_at TIMESTAMP,
                  PRIMARY KEY (username, ticker))''')
    c.execute("""CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT,
                ticker TEXT,
                old_price REAL,
                new_price REAL,
                direction TEXT,
                detected_at TIMESTAMP
            )""")
    # Add default users if table is empty
    c.execute("SELECT COUNT(*) FROM users")
    if c.fetchone()[0] == 0:
        c.execute("INSERT INTO users VALUES (?,?)", ("admin", "admin123"))
        c.execute("INSERT INTO users VALUES (?,?)", ("demo", "demo123"))
    conn.commit()
    conn.close()

def add_user(username, password):
    try:
        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        c.execute("INSERT INTO users VALUES (?,?)", (username, password))
        conn.commit()
        conn.close()
        return True
    except: return False

def check_user(username, password):
    try:
        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
        user = c.fetchone()
        conn.close()
        return user is not None
    except: return False

def add_to_history(username, ticker, price):
    try:
        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        
        # Check for price change alert
        c.execute("SELECT last_price FROM history WHERE username=? AND ticker=? ORDER BY viewed_at DESC LIMIT 1", (username, ticker))
        last_row = c.fetchone()
        alert_info = None
        if last_row:
            old_p = last_row[0]
            if abs(price - old_p) > 0.01: # Significant change
                direction = "increased" if price > old_p else "decreased"
                c.execute("INSERT INTO alerts (username, ticker, old_price, new_price, direction, detected_at) VALUES (?, ?, ?, ?, ?, ?)",
                          (username, ticker, old_p, price, direction, datetime.now()))
                alert_info = (direction, abs(price - old_p))
        
        c.execute("INSERT OR REPLACE INTO history (username, ticker, last_price, viewed_at) VALUES (?, ?, ?, ?)",
                  (username, ticker, price, datetime.now()))
        conn.commit()
        conn.close()
        return alert_info
    except: return None

def get_user_alerts(username):
    try:
        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        c.execute("SELECT ticker, old_price, new_price, direction, detected_at FROM alerts WHERE username=? ORDER BY detected_at DESC", (username,))
        rows = c.fetchall()
        conn.close()
        return rows
    except: return []

def user_exists(username):
    try:
        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=?", (username,))
        user = c.fetchone()
        conn.close()
        return user is not None
    except: return False

def get_user_history(username):
    try:
        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        c.execute("SELECT ticker, last_price, viewed_at FROM history WHERE username=? ORDER BY viewed_at DESC", (username,))
        rows = c.fetchall()
        conn.close()
        return rows
    except: return []

def is_in_watchlist(username, ticker):
    try:
        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        c.execute("SELECT 1 FROM watchlist WHERE username=? AND ticker=?", (username, ticker))
        res = c.fetchone()
        conn.close()
        return res is not None
    except: return False

def toggle_watchlist(username, ticker):
    try:
        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        if is_in_watchlist(username, ticker):
            c.execute("DELETE FROM watchlist WHERE username=? AND ticker=?", (username, ticker))
            action = "removed"
        else:
            c.execute("INSERT INTO watchlist (username, ticker, added_at) VALUES (?, ?, ?)",
                      (username, ticker, datetime.now()))
            action = "added"
        conn.commit()
        conn.close()
        return action
    except: return None

def get_watchlist(username):
    try:
        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        c.execute("SELECT ticker, added_at FROM watchlist WHERE username=? ORDER BY added_at DESC", (username,))
        rows = c.fetchall()
        conn.close()
        return rows
    except: return []

@st.cache_resource
def init_db_cached():
    init_db()

init_db_cached()

@st.cache_data(ttl=300)
def get_ticker_history_cached(ticker, period="30d"):
    try:
        data = yf.download(ticker, period=period, progress=False)
        return data
    except: return pd.DataFrame()

@st.cache_data(ttl=60)
def get_batch_prices(tickers):
    if not tickers: return {}
    try:
        df = yf.download(tickers, period="1d", group_by='ticker', progress=False)
        prices = {}
        for ticker in tickers:
            try:
                if len(tickers) > 1:
                    val = df[ticker]['Close'].iloc[-1]
                else:
                    val = df['Close'].iloc[-1]
                prices[ticker] = float(val)
            except: prices[ticker] = None
        return prices
    except: return {tk: None for tk in tickers}



# ── Query Param Routing ──
if "page" in st.query_params:
    p = st.query_params["page"]
    if p in ["privacy", "terms", "home", "history", "watchlist"]:
        st.session_state.page = p
        # Note: Clearing query params might trigger a rerun in some versions, 
        # but it's needed to prevent sticking to a page on browser refresh.
        st.query_params.clear()

if "action" in st.query_params:
    act = st.query_params["action"]
    tk  = st.query_params.get("ticker")
    if act == "remove" and tk:
        user = st.session_state.username if st.session_state.get("signed_in") else "guest"
        toggle_watchlist(user, tk)
    st.query_params.clear()

# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&family=Space+Grotesk:wght@400;600;700;800&display=swap');

/* ── base ── */
html,body,[data-testid="stAppViewContainer"],[data-testid="stApp"]{
  background:linear-gradient(160deg,#0B132B 0%,#1C2541 55%,#2a3f5f 100%) !important;
  color:#fff !important; font-family:'Inter',sans-serif !important;
}
[data-testid="stHeader"]{background:transparent !important; display:none;}
[data-testid="stSidebar"]{display:none !important;}
.block-container, [data-testid="stMainBlockContainer"] { padding-top: 2rem !important; padding-bottom: 3rem !important; padding-left: 2.5rem !important; padding-right: 2.5rem !important; max-width: 1400px !important; }
section[data-testid="stMain"] > div { padding-top: 0 !important; }
[data-testid="stAppViewBlockContainer"] { padding-top: 2rem !important; }

/* ── hide streamlit chrome ── */
#MainMenu,footer,[data-testid="stToolbar"],.stDeployButton{visibility:hidden !important; display:none !important;}

/* ── navigation ── */
.topnav-wrapper {
  display:flex; align-items:center; justify-content:space-between;
  padding:14px 0 14px; border-bottom:1px solid rgba(255,255,255,0.07);
  margin-bottom:28px;
}
.brand-logo {
  width:48px; height:48px; border-radius:12px;
  background:linear-gradient(135deg,#00E0FF,#14FFEC);
  display:inline-flex; align-items:center; justify-content:center;
  font-weight:900; color:#001; font-size:13px;
  font-family:'Space Grotesk',sans-serif;
}
.brand-text { line-height: 1.2; }
.brand-text .line1 { font-weight: 800; font-size: 15px; color: #fff; }
.brand-text .line2 { font-weight: 400; font-size: 12px; color: rgba(255,255,255,0.4); }
.nav-center { display:flex; gap:32px; align-items:center; }

/* ── footer buttons (small integrated style) ── */
div[data-testid="stHorizontalBlock"]:has(button[key^="foot_"]) button {
    background: transparent !important; border: none !important; padding: 0 !important;
    color: rgba(217,226,236,0.4) !important; font-size: 13px !important;
    text-decoration: none !important; width: auto !important; box-shadow: none !important;
    min-height: 0 !important; line-height: 1 !important;
}
div[data-testid="stHorizontalBlock"]:has(button[key^="foot_"]) button:hover {
    color: #fff !important;
}

/* ── legal link overrides ── */
.legal-link {
    color: rgba(217,226,236,0.4) !important;
    font-size: 15px !important;
    text-decoration: none !important;
    transition: color 0.2s ease !important;
}
.legal-link:hover {
    color: #00E0FF !important;
}

/* ── CTA Container Styling ── */
div[data-testid="stVerticalBlock"]:has(span#cta-marker) {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.05) !important;
    border-radius: 24px !important;
    padding: 40px !important;
    margin-top: 40px !important;
}
div[data-testid="stVerticalBlock"]:has(span#cta-marker) [data-testid="stVerticalBlock"] {
    gap: 0 !important;
}

/* ── History/Watchlist Unified Containers ── */
div[data-testid="stVerticalBlock"]:has(span#history-marker),
div[data-testid="stVerticalBlock"]:has(span#watchlist-marker) {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 20px !important;
    padding: 35px 40px !important;
    margin-top: 20px !important;
}

/* ── Auth Page Button Scaling Removed so buttons span full width ── */

/* ── Specific Overrides for List Items ── */
.list-header { font-size: 13px; font-weight: 700; color: rgba(217,226,236,0.3); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px; }
.list-ticker { font-size: 19px; font-weight: 800; color: #fff; font-family: 'Space Grotesk', sans-serif; }
.list-price  { font-size: 17px; font-weight: 700; color: #14FFEC; }
.list-date   { font-size: 13px; color: rgba(217,226,236,0.5); }

/* ── Compact Remove Link ── */
.remove-link {
    background: rgba(255,107,107,0.1) !important;
    color: #ff6b6b !important;
    border: 1px solid rgba(255,107,107,0.3) !important;
    font-size: 13px !important;
    padding: 6px 18px !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    text-decoration: none !important;
    transition: all 0.2s ease !important;
    display: inline-block !important;
}
.remove-link:hover {
    background: rgba(255,107,107,0.2) !important;
    color: #ff6b6b !important;
    border-color: #ff6b6b !important;
    transform: translateY(-1px) !important;
}

/* ── buttons ── */
div[data-testid="stButton"] button {
  background: linear-gradient(90deg, #00E0FF, #14FFEC) !important;
  color: #011627 !important;
  font-weight: 800 !important;
  border: none !important;
  border-radius: 12px !important;
  font-size: 15px !important;
  padding: 12px 28px !important;
  width: 100% !important;
  transition: all 0.2s ease !important;
  box-shadow: 0 4px 15px rgba(0, 224, 255, 0.2) !important;
}
div[data-testid="stButton"] button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 25px rgba(0, 224, 255, 0.4) !important;
  color: #011627 !important;
}

/* Nav specific buttons (force single line) */
[data-testid="stHorizontalBlock"]:has(.brand-logo) div[data-testid="stButton"] button {
  background: transparent !important;
  color: rgba(255,255,255,0.75) !important;
  border: none !important; box-shadow: none !important;
  width: auto !important;
  white-space: nowrap !important;
  padding: 8px 10px !important;
  font-size: 13px !important;
}
[data-testid="stHorizontalBlock"]:has(.brand-logo) div[data-testid="stButton"] button:hover {
  color: #00E0FF !important;
  background: rgba(0,224,255,0.05) !important;
}
[data-testid="stHorizontalBlock"]:has(.brand-logo) > div:nth-child(10) div[data-testid="stButton"] button {
  background: linear-gradient(90deg, #00E0FF, #14FFEC) !important;
  color: #011627 !important;
  border-radius: 8px !important;
  padding: 8px 16px !important;
  font-weight: 800 !important;
}



/* ── inputs ── */
div[data-baseweb="select"], input {
  background: rgba(255,255,255,0.06) !important;
  border-radius: 8px !important; border: 1px solid rgba(255,255,255,0.1) !important;
}





/* ── Nav language selectbox — slim & clean ── */
div[data-testid="stHorizontalBlock"] > div:nth-child(8) [data-testid="stSelectbox"] > div > div {
  background: rgba(11,19,43,0.7) !important;
  border: 1px solid rgba(255,255,255,0.15) !important;
  border-radius: 8px !important;
  height: 38px !important;
  min-height: 38px !important;
  max-height: 38px !important;
  padding: 0 10px !important;
  font-size: 13px !important;
  font-weight: 600 !important;
  display: flex !important;
  align-items: center !important;
  overflow: hidden !important;
}
/* Target every nested div inside the selectbox control to center text */
div[data-testid="stHorizontalBlock"] > div:nth-child(8) [data-testid="stSelectbox"] > div > div > div,
div[data-testid="stHorizontalBlock"] > div:nth-child(8) [data-testid="stSelectbox"] > div > div > div > div,
div[data-testid="stHorizontalBlock"] > div:nth-child(8) [data-testid="stSelectbox"] [class*="ValueContainer"],
div[data-testid="stHorizontalBlock"] > div:nth-child(8) [data-testid="stSelectbox"] [class*="singleValue"],
div[data-testid="stHorizontalBlock"] > div:nth-child(8) [data-testid="stSelectbox"] [class*="placeholder"] {
  font-size: 13px !important;
  color: #fff !important;
  padding: 0 !important;
  margin: 0 !important;
  line-height: 38px !important;
  height: 38px !important;
  display: flex !important;
  align-items: center !important;
  top: 0 !important;
  transform: none !important;
  position: relative !important;
}
div[data-testid="stHorizontalBlock"] > div:nth-child(8) [data-testid="stSelectbox"] svg {
  fill: rgba(255,255,255,0.5) !important;
}
div[data-testid="stHorizontalBlock"] > div:nth-child(8) [data-testid="stSelectbox"] label {
  display: none !important;
}
div[data-testid="stHorizontalBlock"] > div:nth-child(8) [data-testid="stSelectbox"] label {
  display: none !important;
}

/* ── metrics ── */
[data-testid="metric-container"]{
  background:linear-gradient(180deg,rgba(11,19,43,0.85),rgba(11,19,43,0.75));
  border:1px solid rgba(0,224,255,0.1); border-radius:12px;
  padding:14px !important; box-shadow:0 8px 30px rgba(0,0,0,0.5);
}
[data-testid="metric-container"] label{color:rgba(217,226,236,0.7) !important; font-size:12px !important;}
[data-testid="stMetricValue"]{color:#14FFEC !important; font-weight:800 !important;}

/* ── dataframe ── */
[data-testid="stDataFrame"] thead tr th{
  background:rgba(0,224,255,0.12) !important; color:#00E0FF !important; font-weight:700 !important;
}
[data-testid="stDataFrame"]{border-radius:12px; overflow:hidden;}
[data-testid="stDataFrame"] td{color:#fff !important; font-size:13px !important;}

/* ── divider ── */
hr{border:1px solid rgba(255,255,255,0.08) !important; margin:16px 0 !important;}

/* ── shared card ── */
.card{
  background:linear-gradient(180deg,rgba(11,19,43,0.85),rgba(11,19,43,0.75));
  border-radius:16px; padding:24px;
  border:1px solid rgba(0,224,255,0.08);
  box-shadow:0 8px 30px rgba(0,0,0,0.5);
  margin-bottom:16px;
}
.card-title{font-size:18px; font-weight:800; color:#fff; margin:0 0 14px;}
.muted{color:rgba(217,226,236,0.65); font-size:13px;}

/* ── legal pages ── */
.glass-card {
    background: linear-gradient(145deg, rgba(255,255,255,0.05), rgba(255,255,255,0.01));
    border: 1px solid rgba(255,255,255,0.1);
    backdrop-filter: blur(10px);
    border-radius: 24px;
    box-shadow: 0 20px 50px rgba(0,0,0,0.3);
}
.legal-point {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(0, 224, 255, 0.1);
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 18px;
    transition: all 0.3s ease;
}
.legal-point:hover {
    background: rgba(0, 224, 255, 0.05);
    border-color: rgba(0, 224, 255, 0.3);
    transform: translateY(-2px);
}

/* ══════════════════════════════════════
   HOME PAGE
══════════════════════════════════════ */
.hero-eyebrow{
  display:inline-flex; align-items:center; padding:6px 14px; border-radius:999px;
  font-weight:700; font-size:13px; color:#00E0FF;
  background:rgba(0,224,255,0.1); border:1px solid rgba(0,224,255,0.2);
  margin-bottom:18px;
}
.hero-title{
  font-size:clamp(32px,4.5vw,54px); font-weight:800; line-height:1.08;
  color:#fff; margin:0 0 16px;
}
.accent-text{
  background:linear-gradient(90deg,#00E0FF,#14FFEC);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.hero-sub{color:rgba(217,226,236,0.8); font-size:16px; line-height:1.65; margin:0 0 16px; max-width:55ch;}
.hero-note{font-size:13px; color:rgba(217,226,236,0.55);}

/* showcase cards */
.showcase-card{
  position:relative;
  background:linear-gradient(145deg,rgba(255,255,255,0.055),rgba(255,255,255,0.015));
  border-radius:20px; padding:28px 32px;
  border:1px solid rgba(255,255,255,0.06); overflow:hidden; margin-bottom:16px;
}
.showcase-label{
  display:inline-block; padding:4px 12px; border-radius:999px;
  font-size:10px; font-weight:800; letter-spacing:.12em; text-transform:uppercase;
  color:#00E0FF; background:rgba(0,224,255,0.1); margin-bottom:12px;
}
.showcase-title{font-size:clamp(18px,2vw,24px); font-weight:800; color:#fff; margin:0 0 10px; line-height:1.2;}
.showcase-desc{font-size:14px; color:rgba(217,226,236,0.65); line-height:1.7; margin:0 0 14px;}
.showcase-cta{
  font-size:13px; font-weight:700; color:#00E0FF; text-decoration:none;
  border-bottom:1px solid rgba(0,224,255,0.3); padding-bottom:2px; cursor:pointer;
}

/* mock predictor */
.mock-predictor{background:rgba(0,0,0,0.3); border-radius:14px; padding:20px; margin-top:16px;}
.bar-row{display:flex; align-items:center; gap:12px; margin-bottom:10px;}
.bar-label{width:90px; font-size:12px; color:rgba(217,226,236,0.55); flex-shrink:0;}
.bar-track{flex:1; height:7px; background:rgba(255,255,255,0.08); border-radius:999px; overflow:hidden;}
.bar-fill{height:100%; border-radius:999px;}
.bar-val{width:36px; text-align:right; font-size:12px; font-weight:800; color:rgba(217,226,236,0.85);}

/* signals */
.signals-row{display:flex; gap:12px; margin-top:12px;}
.sig-badge{flex:1; border-radius:14px; padding:18px 8px; display:flex; flex-direction:column; align-items:center; gap:6px;}
.sig-badge--buy {background:rgba(116,242,155,0.08);}
.sig-badge--sell{background:rgba(255,123,132,0.08);}
.sig-badge--hold{background:rgba(255,200,80,0.08);}
.sig-icon{font-size:26px; font-weight:900;}
.sig-badge--buy  .sig-icon{color:#74f29b;}
.sig-badge--sell .sig-icon{color:#ff7b84;}
.sig-badge--hold .sig-icon{color:#ffc85a;}
.sig-lbl{font-size:10px; font-weight:800; letter-spacing:.1em; text-transform:uppercase;}
.sig-badge--buy  .sig-lbl{color:#74f29b;}
.sig-badge--sell .sig-lbl{color:#ff7b84;}
.sig-badge--hold .sig-lbl{color:#ffc85a;}

/* sentiment chips */
.sent-chip{display:inline-flex; align-items:center; gap:6px; padding:8px 14px; border-radius:999px;
  font-size:13px; font-weight:600; margin:4px;}
.sent-chip--pos{background:rgba(116,242,155,0.09); border:1px solid rgba(116,242,155,0.22); color:#74f29b;}
.sent-chip--neg{background:rgba(255,123,132,0.09); border:1px solid rgba(255,123,132,0.22); color:#ff7b84;}
.sent-chip--neu{background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.09); color:rgba(217,226,236,0.7);}

/* feature cards */
.feat-card{
  background:rgba(255,255,255,0.04); border-radius:16px; padding:24px;
  border:1px solid rgba(255,255,255,0.05); height:100%;
}
.feat-icon{
  width:52px; height:52px; border-radius:12px; font-size:22px;
  background:linear-gradient(180deg,rgba(0,224,255,0.08),rgba(20,255,236,0.03));
  display:flex; align-items:center; justify-content:center; margin-bottom:12px;
}

/* top companies */
.company-row-card{
  background:linear-gradient(180deg,rgba(255,255,255,0.025),rgba(255,255,255,0.01));
  border-radius:12px; padding:16px;
  border:1px solid rgba(255,255,255,0.05);
}
.logo-sm{
  width:44px; height:44px; border-radius:8px;
  background:rgba(255,255,255,0.07);
  display:inline-flex; align-items:center; justify-content:center;
  font-weight:800; color:#00E0FF; font-size:18px; flex-shrink:0;
}
.live-badge{font-weight:800; font-size:12px; color:#14FFEC; margin-top:2px;}

/* flow steps */
.flow-step{
  display:flex; align-items:flex-start; gap:14px; padding:16px 18px;
  border-radius:12px; background:linear-gradient(90deg,rgba(0,224,255,0.025),transparent);
  border:1px solid rgba(0,224,255,0.05); margin-bottom:10px;
}
.flow-num{
  min-width:42px; height:42px; border-radius:10px;
  background:linear-gradient(135deg,#14FFEC,#00E0FF);
  display:flex; align-items:center; justify-content:center;
  color:#001; font-weight:900; font-size:17px; flex-shrink:0;
  font-family:'Space Grotesk',sans-serif;
}

/* about */
.about-wrapper{
  background:linear-gradient(145deg,rgba(255,255,255,0.04),rgba(255,255,255,0.01));
  border-radius:20px; padding:40px 44px;
  border:1px solid rgba(255,255,255,0.06);
  margin-top:8px; margin-bottom:20px;
}
.about-stats-grid{display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-bottom:16px;}
.about-stat-box{
  background:rgba(0,0,0,0.2); border-radius:14px; padding:20px 14px; text-align:center;
}
.about-stat-num{font-family:'Space Grotesk',sans-serif; font-size:28px; font-weight:900; margin-bottom:5px;}
.about-stat-lbl{font-size:12px; color:rgba(217,226,236,0.5);}
.about-val{
  display:flex; align-items:flex-start; gap:14px; padding:14px 16px;
  border-radius:14px; background:rgba(0,224,255,0.02); margin-bottom:10px;
}
.about-val-ico{
  width:40px; height:40px; border-radius:10px; background:rgba(255,255,255,0.05);
  display:flex; align-items:center; justify-content:center; font-size:20px; flex-shrink:0;
}
.about-val-title{font-weight:800; font-size:14px; margin-bottom:3px;}
.about-val-desc{font-size:13px; color:rgba(217,226,236,0.55);}

/* ══════════════════════════════════════
   ANALYSIS PAGE
══════════════════════════════════════ */
.left-panel{
  background:rgba(11,19,43,0.92); border-radius:14px;
  border:1px solid rgba(0,224,255,0.18);
  padding:20px; backdrop-filter:blur(8px);
}
.left-panel-title{font-weight:800; font-size:15px; color:#D9E2EC; margin-bottom:10px;}
.left-panel-tip{color:rgba(217,226,236,0.6); font-size:13px; margin:8px 0 0; line-height:1.55;}

.welcome-card{
  text-align:center; padding:60px 24px;
  border:2px dashed rgba(0,224,255,0.3); border-radius:16px;
  background:linear-gradient(180deg,rgba(11,19,43,0.7),rgba(11,19,43,0.4));
}

.ks-card{
  background:linear-gradient(180deg,rgba(11,19,43,0.88),rgba(11,19,43,0.75));
  border-radius:16px; padding:24px;
  border:1px solid rgba(0,224,255,0.1);
  box-shadow:0 8px 30px rgba(0,0,0,0.5);
  margin-bottom:20px;
}
.ks-title{font-size:20px; font-weight:800; color:#fff; margin-bottom:18px;}
.stats-row{
  display:flex; justify-content:space-between; padding:9px 0;
  border-bottom:1px solid rgba(255,255,255,0.06);
}
.stats-lbl{font-size:13px; font-weight:500; color:rgba(255,255,255,0.6);}
.stats-val{font-size:13px; font-weight:700; color:#14FFEC;}

/* pred tiles */
.pred-tile{
  background:rgba(0,0,0,0.3); padding:12px 8px; border-radius:10px;
  border:1px solid rgba(0,224,255,0.08); text-align:center; margin-bottom:8px;
}
.pred-tile-date{font-size:11px; color:rgba(217,226,236,0.55); margin-bottom:4px;}
.pred-tile-price{font-size:15px; font-weight:800; color:#fff;}

/* news */
.news-item{padding:14px 0; border-bottom:1px solid rgba(255,255,255,0.07);}
.news-link{color:#fff; font-weight:700; font-size:14px; text-decoration:none; line-height:1.5; display:block;}
.news-link:hover{color:#00E0FF;}
.news-meta{font-size:12px; color:rgba(217,226,236,0.55); margin-top:5px;}
.badge{display:inline-block; padding:3px 10px; border-radius:20px; font-size:11px; font-weight:700; margin-top:6px;}
.badge-pos{background:rgba(20,255,236,0.12); color:#14FFEC; border:1px solid rgba(20,255,236,0.35);}
.badge-neg{background:rgba(255,0,0,0.12); color:#ff6b6b; border:1px solid rgba(255,0,0,0.35);}
.badge-neu{background:rgba(255,255,255,0.07); color:#D9E2EC; border:1px solid rgba(255,255,255,0.18);}

/* risk / signal */
.risk-bar-wrap{
  width:100%; height:18px; border-radius:20px; overflow:hidden; margin:14px 0 10px;
  background:linear-gradient(90deg,rgba(20,255,236,0.15),rgba(0,224,255,0.1),rgba(255,107,107,0.15));
}
.risk-fill{
  height:100%; border-radius:20px;
  background:linear-gradient(90deg,#14FFEC 0%,#00E0FF 50%,#ff6b6b 100%);
  box-shadow:0 0 20px rgba(0,224,255,0.5);
}
.signal-box{
  margin-top:20px; padding:20px; border-radius:18px;
  background:linear-gradient(145deg,rgba(255,255,255,0.04),rgba(0,0,0,0.3));
  border:1px solid rgba(255,255,255,0.08); backdrop-filter:blur(8px);
}
.signal-main{
  display:flex; justify-content:space-between; align-items:center;
  font-size:26px; font-weight:900; letter-spacing:1px; margin-bottom:10px;
  font-family:'Space Grotesk',sans-serif;
}
.signal-buy{color:#14FFEC;}
.signal-sell{color:#ff6b6b;}
.signal-hold{color:#7FDBFF;}

/* ══════════════════════════════════════
   COMPARISON PAGE
══════════════════════════════════════ */
.comp-table{width:100%; border-collapse:collapse; font-size:13px;}
.comp-table th{
  background:rgba(0,224,255,0.1); color:#fff; font-weight:700;
  padding:13px 16px; text-align:center; border-bottom:1px solid rgba(255,255,255,0.07);
}
.comp-table th:first-child{text-align:left; color:#00E0FF;}
.comp-table td{
  padding:13px 16px; text-align:center;
  border-bottom:1px solid rgba(255,255,255,0.05); color:#fff;
}
.comp-table td:first-child{font-weight:700; color:#00E0FF; text-align:left;}
.comp-table tr:hover td{background:rgba(0,224,255,0.02);}

.vcard{
  background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08);
  border-radius:16px; padding:18px; height:100%;
}
.vcard.winner{
  border:1.5px solid rgba(0,224,255,0.6) !important;
  background:rgba(0,224,255,0.05) !important;
  box-shadow:0 0 30px rgba(0,224,255,0.08);
}
.vcard-rank{font-size:10px; font-weight:700; color:rgba(255,255,255,0.4); margin-bottom:4px;}
.vcard-badge{
  display:inline-block; font-size:9px; font-weight:800; padding:3px 10px;
  border-radius:20px; margin-bottom:10px;
  background:linear-gradient(90deg,#00E0FF,#14FFEC); color:#001;
  letter-spacing:.07em; text-transform:uppercase;
}
.vcard-header{display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;}
.vcard-ticker{font-size:20px; font-weight:900; font-family:'Space Grotesk',sans-serif;}
.vcard-bar-wrap{height:5px; background:rgba(255,255,255,0.08); border-radius:3px; overflow:hidden; margin-bottom:4px;}
.vcard-bar{height:100%; border-radius:3px;}
.vcard-score-lbl{
  font-size:10px; color:rgba(217,226,236,0.35);
  display:flex; justify-content:space-between; margin-bottom:14px;
}
.vcard-meta{border-top:1px solid rgba(255,255,255,0.07); padding-top:12px; font-size:13px;}
.vcard-row{display:flex; justify-content:space-between; margin-bottom:6px;}
.vcard-row span:first-child{color:rgba(255,255,255,0.45);}
.vcard-row span:last-child{font-weight:700;}
.verdict-summary{
  margin-top:16px; padding:20px 24px;
  border-left:4px solid #00e5ff;
  background:rgba(255,255,255,0.03);
  border-radius:12px; color:#dbe7ff;
  font-size:14px; line-height:1.75;
}
.verdict-summary strong{color:#00E0FF; font-weight:800;}
.verdict-disc{margin-top:12px; font-size:11px; color:rgba(255,255,255,0.2); text-align:center;}
/* Default Grid Definitions */
.grid-history-head { display:grid; grid-template-columns:1.2fr 1fr 1.5fr; gap:20px; padding:0 10px; }
.grid-watchlist-head { display:grid; grid-template-columns:1.2fr 1fr 0.8fr; gap:20px; padding:0 10px; }
.grid-history-row { display:grid; grid-template-columns:1.2fr 1fr 1.5fr; gap:20px; padding:15px 10px; align-items:center; }
.grid-watchlist-row { display:grid; grid-template-columns:1.2fr 1fr 0.8fr; gap:20px; padding:15px 10px; align-items:center; }
.grid-alerts-head { display:grid; grid-template-columns:1fr 1fr 1fr 1fr; gap:20px; padding:0 10px; }
.grid-alerts-row { display:grid; grid-template-columns:1fr 1fr 1fr 1fr; gap:20px; padding:15px 10px; align-items:center; }

@media (max-width: 768px) {
  /* Reduce padding on main containers */
  .block-container, [data-testid="stMainBlockContainer"] { padding-left: 1rem !important; padding-right: 1rem !important; }
  .about-wrapper, .welcome-card, .vcard, .ks-card, .feat-card, .showcase-card { padding: 20px 15px !important; }
  
  /* Typography Scaling */
  .hero-title { font-size: 28px !important; }
  .showcase-title { font-size: 20px !important; }
  .brand-text .line1 { font-size: 13px !important; }
  
  /* Navbar Wrapping - Override Streamlit column stacking */
  div[data-testid="stHorizontalBlock"]:has(.brand-logo) {
      display: flex !important;
      flex-wrap: wrap !important;
      justify-content: center !important;
      gap: 10px !important;
  }
  div[data-testid="stHorizontalBlock"]:has(.brand-logo) > div {
      min-width: unset !important;
      width: auto !important;
      flex: 0 0 auto !important;
  }

  /* Grid Resets for Mobile */
  .about-stats-grid,
  .grid-history-head, .grid-history-row,
  .grid-watchlist-head, .grid-watchlist-row,
  .grid-alerts-head, .grid-alerts-row {
      grid-template-columns: 1fr !important;
      gap: 8px !important;
  }
  
  .grid-history-head, .grid-watchlist-head, .grid-alerts-head {
      display: none !important; /* Hide headers on mobile to save space */
  }
  
  .grid-history-row, .grid-watchlist-row, .grid-alerts-row {
      text-align: center;
      padding: 15px 0;
      border-bottom: 1px solid rgba(255,255,255,0.05);
  }
  .list-date, .list-price, .remove-link {
      text-align: center !important;
      justify-content: center;
  }
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════
COMPANY_NAMES = {
    "ADANIENT":"Adani Enterprises Limited","ADANIPORTS":"Adani Ports & SEZ Limited",
    "APOLLOHOSP":"Apollo Hospitals Enterprises Limited","ASIANPAINT":"Asian Paints Limited",
    "AXISBANK":"Axis Bank Limited","BAJAJ-AUTO":"Bajaj Auto Limited",
    "BAJAJFINSV":"Bajaj Finserv Limited","BAJFINANCE":"Bajaj Finance Limited",
    "BEL":"Bharat Electronics Limited","BHARTIARTL":"Bharti Airtel Limited",
    "CIPLA":"Cipla Limited","COALINDIA":"Coal India Limited",
    "DRREDDY":"Dr. Reddy's Laboratories Limited","EICHERMOT":"Eicher Motors Limited",
    "ETERNAL":"Eternal Limited","GRASIM":"Grasim Industries Limited",
    "HCLTECH":"HCL Technologies Limited","HDFCBANK":"HDFC Bank Limited",
    "HDFCLIFE":"HDFC Life Insurance Company Limited","HINDALCO":"Hindalco Industries Limited",
    "HINDUNILVR":"Hindustan Unilever Limited","ICICIBANK":"ICICI Bank Limited",
    "INDIGO":"InterGlobe Aviation Limited","INFY":"Infosys Limited",
    "ITC":"ITC Limited","JIOFIN":"Jio Financial Services Limited",
    "JSWSTEEL":"JSW Steel Limited","KOTAKBANK":"Kotak Mahindra Bank Limited",
    "LT":"Larsen & Toubro Limited","M&M":"Mahindra & Mahindra Limited",
    "MARUTI":"Maruti Suzuki India Limited","MAXHEALTH":"Max Healthcare Institute Limited",
    "NESTLEIND":"Nestle India Limited","NTPC":"NTPC Limited",
    "ONGC":"Oil & Natural Gas Corporation Limited","POWERGRID":"Power Grid Corporation of India Limited",
    "RELIANCE":"Reliance Industries Limited","SBILIFE":"SBI Life Insurance Company Limited",
    "SBIN":"State Bank of India","SHRIRAMFIN":"Shriram Finance Limited",
    "SUNPHARMA":"Sun Pharmaceuticals Industries Limited","TATACONSUM":"Tata Consumer Products Limited",
    "TATASTEEL":"Tata Steel Limited","TCS":"Tata Consultancy Services Limited",
    "TECHM":"Tech Mahindra Limited","TITAN":"Titan Company Limited",
    "TMPV":"Tata Motors Passenger Vehicles Limited","TRENT":"TRENT Limited",
    "ULTRACEMCO":"UltraTech Cement Limited","WIPRO":"Wipro Limited",
}
COMPANY_ALIASES = {
    "ADANIENT":["adani enterprises"],"ADANIPORTS":["adani ports","adani ports & sez"],
    "APOLLOHOSP":["apollo hospitals","apollo hospital","apollo"],
    "ASIANPAINT":["asian paints"],"AXISBANK":["axis bank","axis"],
    "BAJAJ-AUTO":["bajaj auto","bajaj"],"BAJAJFINSV":["bajaj finserv"],
    "BAJFINANCE":["bajaj finance","bajaj fin"],"BEL":["bharat electronics","bel"],
    "BHARTIARTL":["bharti airtel","airtel"],"CIPLA":["cipla"],
    "COALINDIA":["coal india"],"DRREDDY":["dr reddy","dr. reddy","reddy"],
    "EICHERMOT":["eicher motors","royal enfield","eicher"],"ETERNAL":["eternal"],
    "GRASIM":["grasim"],"HCLTECH":["hcl tech","hcl technologies","hcl"],
    "HDFCBANK":["hdfc bank"],"HDFCLIFE":["hdfc life"],
    "HINDALCO":["hindalco"],"HINDUNILVR":["hindustan unilever","hul","unilever"],
    "ICICIBANK":["icici bank","icici"],"INDIGO":["interglobe aviation","indigo"],
    "INFY":["infosys"],"ITC":["itc"],"JIOFIN":["jio financial","jio finance","jio"],
    "JSWSTEEL":["jsw steel","jsw"],"KOTAKBANK":["kotak bank","kotak mahindra","kotak"],
    "LT":["larsen & toubro","l&t","lt"],"M&M":["mahindra","m&m"],
    "MARUTI":["maruti","maruti suzuki"],"MAXHEALTH":["max healthcare","max health"],
    "NESTLEIND":["nestle india","nestle"],"NTPC":["ntpc"],
    "ONGC":["ongc","oil and natural gas"],"POWERGRID":["power grid","powergrid"],
    "RELIANCE":["reliance","reliance industries","ril"],"SBILIFE":["sbi life"],
    "SBIN":["state bank of india","sbi"],"SHRIRAMFIN":["shriram finance","shriram"],
    "SUNPHARMA":["sun pharma","sun pharmaceutical"],"TATACONSUM":["tata consumer"],
    "TATASTEEL":["tata steel"],"TCS":["tcs","tata consultancy"],
    "TECHM":["tech mahindra"],"TITAN":["titan"],"TMPV":["tata motors"],
    "TRENT":["trent"],"ULTRACEMCO":["ultratech cement"],"WIPRO":["wipro"],
}

# ══════════════════════════════════════════════════════════════════════════════
# LOAD ASSETS
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_ml_assets():
    if not ML_AVAILABLE: return None, None, None
    try:
        base = os.path.dirname(os.path.abspath(__file__))
        m   = load_model(os.path.join(base,"ml_models","stock_model.keras"))
        sc  = joblib.load(os.path.join(base,"ml_models","stock_scaler.joblib"))
        sid = joblib.load(os.path.join(base,"ml_models","stock_id_mapping.joblib"))
        return m, sc, sid
    except: return None, None, None

@st.cache_resource(show_spinner=False)
def load_finbert():
    if not FINBERT_AVAILABLE: return None, None
    try:
        tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        mdl = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        return tok, mdl
    except: return None, None

ml_model, scalers, stock_to_id = load_ml_assets()
finbert_tok, finbert_mdl       = load_finbert()

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def safe_float(*args):
    if not args: return 0.0
    x = args[0]
    default = args[1] if len(args) > 1 else 0.0
    try:
        if x is None: return default
        return float(x.values[0]) if hasattr(x, "values") else float(x)
    except: return default

def safe_int(*args):
    if not args: return 0
    x = args[0]
    default = args[1] if len(args) > 1 else 0
    try:
        if x is None: return default
        return int(x.values[0]) if hasattr(x, "values") else int(x)
    except: return default

def get_sentiment(text):
    if finbert_mdl is not None:
        try:
            inp = finbert_tok(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad(): out = finbert_mdl(**inp)
            probs = F.softmax(out.logits, dim=1)
            conf, cls = torch.max(probs, dim=1)
            lbl = {0:"Negative",1:"Neutral",2:"Positive"}[cls.item()]
            c = conf.item()
            return (c if lbl=="Positive" else (-c if lbl=="Negative" else 0.0)), lbl, c
        except: pass
    try:
        from textblob import TextBlob
        s = TextBlob(text).sentiment.polarity
        lbl = "Positive" if s>0.05 else ("Negative" if s<-0.05 else "Neutral")
        return s, lbl, abs(s)
    except: return 0.0, "Neutral", 0.0

def alias_match(title, aliases):
    t = title.lower().replace("-"," ")
    for a in aliases:
        if re.search(r""+re.escape(a.lower().replace("-"," "))+r"", t): return True
    return False

@st.cache_data(show_spinner=False)
def translate_text(text, lang):
    if not text or lang=="en" or not TRANSLATOR_AVAILABLE: return text
    try: return GoogleTranslator(source="auto", target=lang).translate(text)
    except: return text

@st.cache_data(ttl=300, show_spinner=False)
def fetch_news(ticker, lang="en"):
    headlines, sents, seen = [], [], set()
    sym     = ticker.split(".")[0]
    aliases = COMPANY_ALIASES.get(sym, [sym.lower()])
    try:
        for item in (yf.Ticker(ticker).news or [])[:15]:
            # Handle new yfinance structure where data is under 'content'
            content = item.get("content", item)
            title = content.get("title", "")
            if not title or title in seen: continue
            
            seen.add(title)
            score, label, conf = get_sentiment(title)
            sents.append(score)
            
            ts = content.get("pubDate") or content.get("providerPublishTime")
            try:
                if isinstance(ts, str): dt = pd.to_datetime(ts)
                else: dt = datetime.fromtimestamp(ts) if ts else datetime.now()
            except: dt = datetime.now()
            
            headlines.append({"source":content.get("provider",{}).get("displayName", item.get("publisher","Yahoo Finance")),
                "title":translate_text(title,lang),"original_title":title,
                "url":content.get("canonicalUrl",{}).get("url", item.get("link","#")),
                "published_at":dt.strftime("%d %b %Y • %I:%M %p"),
                "raw_time":dt,"sentiment_score":round(score,3),
                "sentiment_label":label,"confidence":round(conf*100,2)})
    except Exception as e:
        print(f"DEBUG: yf news failed for {ticker}: {e}")
        pass
    try:
        q = " OR ".join(f'"{a}"' for a in aliases)
        r = requests.get("https://newsapi.org/v2/everything",
            params={"q":q,"language":lang,"sortBy":"publishedAt","pageSize":15,"apiKey":NEWS_API_KEY},
            timeout=15).json()
        if r.get("status")=="ok":
            for a in r.get("articles",[]):
                title = a.get("title","")
                if not title or title in seen: continue
                if not alias_match(title, aliases): continue
                seen.add(title)
                score, label, conf = get_sentiment(title)
                sents.append(score)
                pub = a.get("publishedAt","")
                try: raw_dt=datetime.strptime(pub,"%Y-%m-%dT%H:%M:%SZ"); fmt=raw_dt.strftime("%d %b %Y • %I:%M %p")
                except: raw_dt=datetime.now(); fmt=pub
                headlines.append({"source":a.get("source",{}).get("name","NewsAPI"),
                    "title":translate_text(title,lang),"original_title":title,
                    "url":a.get("url","#"),"published_at":fmt,"raw_time":raw_dt,
                    "sentiment_score":round(score,3),"sentiment_label":label,"confidence":round(conf*100,2)})
    except: pass
    headlines.sort(key=lambda x:x.get("raw_time",datetime.min),reverse=True)
    for h in headlines: h.pop("raw_time",None)
    return float(np.mean(sents)) if sents else 0.0, headlines[:5]

def compute_rsi(series, period=14):
    d=series.diff(); g=d.clip(lower=0).rolling(period).mean(); l=(-d.clip(upper=0)).rolling(period).mean()
    return 100-(100/(1+g/l))

def linear_predict(prices, days=14):
    x=np.arange(len(prices)); c=np.polyfit(x,prices,1)
    return np.polyval(c,np.arange(len(prices),len(prices)+days)).tolist()

def generate_ai_summary(ticker, headlines):
    sym=ticker.split(".")[0]; name=COMPANY_NAMES.get(sym,sym)
    if not headlines:
        return f"No major recent news for {name}. Prediction based on historical price trends and technical indicators."
    pos=sum(1 for n in headlines if n["sentiment_label"]=="Positive")
    neg=sum(1 for n in headlines if n["sentiment_label"]=="Negative")
    neu=sum(1 for n in headlines if n["sentiment_label"]=="Neutral")
    tone="mostly positive" if pos>neg else ("mostly negative" if neg>pos else "mixed")
    tops=[n.get("original_title") or n.get("title","") for n in headlines[:2]]
    s=(f"Recent news sentiment for {name} is {tone}. Out of {len(headlines)} items, "
       f"{pos} were positive, {neg} negative, and {neu} neutral. ")
    if tops: s+="Key topics include: "+"; ".join(tops)+". "
    return s+"This summary helps explain the market mood around the stock."

@st.cache_data(ttl=300, show_spinner=False)
def get_stock_prediction(ticker, lang="en"):
    try:
        df = yf.download(tickers=ticker, period="6mo", interval="1d", progress=False)
        df.reset_index(inplace=True)
        if isinstance(df.columns,pd.MultiIndex): df.columns=df.columns.get_level_values(0)
        if "Adj Close" in df.columns:   df["price"]=df["Adj Close"]
        elif "Close" in df.columns:     df["price"]=df["Close"]
        else: return {"error":"No price column"}
        df=df.drop(columns=[c for c in ["Close","Adj Close"] if c in df.columns])
        df=df.sort_values("Date")
        if df.empty: return {"error":f"No data for {ticker}"}
        try: info = yf.Ticker(ticker).info or {}
        except: info = {}
        sentiment, news = fetch_news(ticker, lang=lang)
        summary = generate_ai_summary(ticker, news)
        prices_arr = df["price"].values.astype(float)

        if ml_model is not None and scalers is not None:
            sid=(stock_to_id or {}).get(ticker,0)
            feat=df.copy(); feat["sentiment"]=sentiment
            feat["MA5"]=feat["price"].rolling(5).mean(); feat["MA10"]=feat["price"].rolling(10).mean()
            feat["RSI"]=compute_rsi(feat["price"]); feat["stock_id"]=sid
            feat=feat.bfill().ffill()
            feats=feat[["price","sentiment","MA5","MA10","RSI","stock_id"]].values
            scaler=scalers.get(ticker)
            if scaler:
                scaled=scaler.transform(feats[-LOOKBACK:]).reshape(1,LOOKBACK,6)
                ps=ml_model.predict(scaled,verbose=0)[0]
                dummy=np.zeros((14,6)); dummy[:,0]=ps
                predicted=scaler.inverse_transform(dummy)[:,0].tolist()
            else: predicted=linear_predict(prices_arr)
        else: predicted=linear_predict(prices_arr)

        last_close=float(prices_arr[-1])
        predicted[0]=0.7*last_close+0.3*predicted[0]

        # CALCULATE SIGNAL (Sentiment + Trend)
        trend = (predicted[-1] - last_close) / last_close
        
        # Weigh sentiment and price trend
        # Sentiment is -1 to 1, Trend is usually -0.1 to 0.1
        # Combined score:
        combined_score = (sentiment * 0.3) + (trend * 0.7 * 10) 
        
        if combined_score > 0.08:     signal="BUY"
        elif combined_score < -0.08:  signal="SELL"
        else:                          signal="HOLD"

        signal_strength = round(min(100, abs(combined_score) * 250), 2)
        # ── Data-Driven Risk Assessment ──
        beta = safe_float(info.get("beta"), 1.0)
        # Calculate annualized volatility from recent prices
        price_vol = np.std(prices_arr[-30:]) / (np.mean(prices_arr[-30:]) or 1)
        
        # Risk Components:
        # 1. Market Risk (Beta): Higher beta = higher sensitivity/risk
        r_beta = min(100, beta * 40) 
        # 2. Volatility Risk: Higher price swings = higher risk
        r_vol = min(100, price_vol * 2000) 
        # 3. Sentiment Uncertainty: Neutral/Negative sentiment adds caution
        r_sent = (1 - (sentiment + 1) / 2) * 50 + (abs(sentiment) * 20)
        
        risk_score = round(min(99, r_beta * 0.4 + r_vol * 0.4 + r_sent * 0.2), 2)
        confidence = max(40, min(95, round((1 - price_vol * 5) * 100, 2)))

        h30=df.sort_values("Date").dropna(subset=["price"]).iloc[-30:].copy()
        h30["Date"]=pd.to_datetime(h30["Date"])
        pc=pd.to_numeric(h30["price"],errors="coerce"); mask=pc.notna()
        hist_graph={"dates":h30.loc[mask,"Date"].dt.strftime("%Y-%m-%d").tolist(),
                    "prices":pc.loc[mask].round(2).tolist()}

        last_date=pd.to_datetime(h30["Date"].iloc[-1]).date(); future_dates=[]
        d=last_date
        while len(future_dates)<14:
            d+=timedelta(days=1)
            if d.weekday()<5: future_dates.append(d.strftime("%Y-%m-%d"))
        pred_graph={"dates":future_dates,"prices":[float(p) for p in predicted]}

        yd=yf.download(ticker,period="1y",interval="1d")
        if isinstance(yd.columns,pd.MultiIndex): yd.columns=yd.columns.get_level_values(0)
        yr_low=safe_float(yd["Low"].min()) if not yd.empty else 0
        yr_high=safe_float(yd["High"].max()) if not yd.empty else 0
        lr=df.tail(1)
        key_stats={"open":safe_float(lr["Open"]),"high":safe_float(lr["High"]),
                   "low":safe_float(lr["Low"]),"close":safe_float(lr["price"]),
                   "volume":safe_int(lr["Volume"]),"last_close":safe_float(lr["price"]),
                   "market_cap":safe_int(info.get("marketCap")),
                   "pe_ratio":safe_float(info.get("trailingPE")),
                   "beta":safe_float(info.get("beta")),
                   "eps_basic":safe_float(info.get("epsTrailingTwelveMonths")),
                   "forward_pe":safe_float(info.get("forwardPE")),
                   "dividend_yield":safe_float(info.get("dividendYield")),
                   "days_range":f"{yr_low:.2f} - {yr_high:.2f}"}
        res = {"ticker":ticker,"key_stats":key_stats,"latest_news":news,
                "historical_graph_data":hist_graph,"prediction_graph_data":pred_graph,
                "confidence":confidence,"ai_summary":summary,"signal":signal,
                "signal_strength":signal_strength,"risk_score":risk_score,
                "final_sentiment":round(sentiment,3),
                "explanation":{"BUY":"Positive sentiment detected from news and social media. Uptrend possible.",
                               "SELL":"Negative sentiment detected. Downward pressure likely.",
                               "HOLD":"Mixed or weak sentiment."}[signal]}
        
        if lang != "en":
            res["explanation"] = translate_text(res["explanation"], lang)
            if res["ai_summary"]:
                res["ai_summary"] = translate_text(res["ai_summary"], lang)
        return res

    except Exception as e: return {"error":str(e)}


def fmt_mcap(v):
    if not v or v<=0: return "—"
    if v>=1e12: return f"₹{v/1e12:.1f}T"
    if v>=1e9:  return f"₹{v/1e9:.1f}B"
    return f"₹{int(v):,}"

def score_stock(info):
    score=0
    pe=info.get("trailingPE"); fpe=info.get("forwardPE")
    beta=info.get("beta"); mcap=info.get("marketCap"); vol=info.get("volume")
    if pe and pe>0:    score+=25 if pe<=15 else(20 if pe<=25 else(10 if pe<=35 else 5))
    if pe and pe>0 and fpe and fpe>0:
        d=(pe-fpe)/pe*100; score+=20 if d>=20 else(15 if d>=10 else(10 if d>=0 else 5))
    if beta is not None: score+=(8 if beta<0 else(20 if beta<=0.8 else(15 if beta<=1.2 else(8 if beta<=1.6 else 3))))
    if mcap and mcap>0: score+=(20 if mcap>=5e12 else(16 if mcap>=1e12 else(10 if mcap>=5e11 else 5)))
    if vol and vol>0:   score+=(15 if vol>=1e7 else(12 if vol>=5e6 else(8 if vol>=1e6 else 4)))
    return int(np.clip(score,0,100))

def get_gemini_verdict(winner, stocks):
    if not GEMINI_API_KEY: return None
    tickers_str = ", ".join([s["ticker"] for s in stocks])
    prompt = (
        f"You are a professional financial analyst AI. Analyze these NIFTY 50 stocks and explain your verdict.\n\n"
        f"WINNER: {winner['ticker']} (Score: {winner['score']}/100)\n"
        f"ALL STOCKS COMPARED: {json.dumps(stocks, indent=2)}\n\n"
        f"Write a detailed 5-6 paragraph investment analysis explaining:\n"
        f"1. Why {winner['ticker']} ranked highest with specific numbers\n"
        f"2. How it compares to the other stocks ({tickers_str}) on P/E, Beta, Market Cap\n"
        f"3. Key risk factors to watch out for\n"
        f"4. A final investment recommendation\n\n"
        f"Write like a Bloomberg analyst. Use actual numbers from the data. "
        f"Output clean HTML using only <p> and <strong> tags. No bullet points. No headings."
    )
    for model in ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.0-flash-lite"]:
        try:
            r = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}",
                headers={"Content-Type": "application/json"},
                json={"contents": [{"parts": [{"text": prompt}]}],
                      "generationConfig": {"temperature": 0.6, "topP": 0.9, "maxOutputTokens": 40000}},
                timeout=40)
            if r.status_code == 200:
                data = r.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                # If response was cut off, append a closing sentence
                finish = data["candidates"][0].get("finishReason", "STOP")
                if finish != "STOP":
                    text += f"<p><em>(Analysis truncated — please try again for the full verdict.)</em></p>"
                return text
            elif r.status_code == 429:
                continue  # Try next model
        except: continue
    return None


CHART_LAYOUT = dict(
    paper_bgcolor="rgba(11,19,43,0.92)",
    plot_bgcolor="rgba(11,19,43,0.92)",
    font=dict(color="#D9E2EC", family="Inter"),
    margin=dict(l=50, r=20, t=80, b=50),
    xaxis=dict(gridcolor="rgba(255,255,255,0.07)", color="rgba(255,255,255,0.6)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.07)", color="rgba(255,255,255,0.6)",
               autorange=True, rangemode="normal"),
    shapes=[dict(type="rect", xref="paper", yref="paper",
                 x0=0, y0=0, x1=1, y1=1,
                 line=dict(color="rgba(0,224,255,0.1)", width=1),
                 fillcolor="rgba(0,0,0,0)")])

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════


# Initialize Session State
for k,v in [("page","home"),("lang","English"),("lang_code","en"),("signed_in",False),("username",None),
              ("scroll_to",None),("auth_mode","login"),("auth_redirect",None)]:
    if k not in st.session_state: st.session_state[k]=v

# ══════════════════════════════════════════════════════════════════════════════
# TOP NAV BAR
# ══════════════════════════════════════════════════════════════════════════════
n0,n1,n2,n3,nh,nw,nb,n6,n7,n8 = st.columns([2.8, 0.7, 0.8, 1.1, 0.8, 0.9, 0.4, 0.9, 0.8, 0.8])

with n0:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;padding:6px 0;">
      <div class="brand-logo">T→T</div>
      <div class="brand-text">
        <div class="line1">Tweets to Trades</div>
        <div class="line2">AI-powered market signals</div>
      </div>
    </div>""", unsafe_allow_html=True)

with n1:
    if st.button(translate_text("Home", st.session_state.lang_code), key="nav_home"):
        st.session_state.page="home"; st.rerun()
with n2:
    if st.button(translate_text("Analysis", st.session_state.lang_code), key="nav_analysis"):
        if st.session_state.signed_in:
            st.session_state.page="analysis"; st.rerun()
        else:
            st.session_state.auth_mode="login"
            st.session_state.auth_redirect="analysis"
            st.session_state.page="auth"; st.rerun()
with n3:
    if st.button(translate_text("Compare Stocks", st.session_state.lang_code), key="nav_compare"):
        if st.session_state.signed_in:
            st.session_state.page="comparison"; st.rerun()
        else:
            st.session_state.auth_mode="login"
            st.session_state.auth_redirect="comparison"
            st.session_state.page="auth"; st.rerun()
with nh:
    if st.button(translate_text("History", st.session_state.lang_code), key="nav_history"):
        st.session_state.page="history"; st.rerun()
with nw:
    if st.button(translate_text("Watchlist", st.session_state.lang_code), key="nav_watchlist"):
        st.session_state.page="watchlist"; st.rerun()
with nb:
    # Bell Icon for Alerts
    if st.button("🔔", key="nav_alerts"):
        st.session_state.page="alerts"; st.rerun()

with n6:
    lang_opts = ["English","हिन्दी","मराठी","ಕನ್ನಡ","മലയാളം"]
    lang_map  = {"English":"en","हिन्दी":"hi","मराठी":"mr","ಕನ್ನಡ":"kn","മലയാളം":"ml"}
    lang = st.selectbox("Lang", lang_opts,
                        index=lang_opts.index(st.session_state.lang),
                        label_visibility="collapsed", key="lang_sel")
    if lang != st.session_state.lang:
        st.session_state.lang = lang
        st.session_state.lang_code = lang_map[lang]
        st.rerun()

with n7:
    if st.session_state.signed_in:
        greet = translate_text("Hi", st.session_state.lang_code)
        st.markdown(f'<div style="font-size:13px;font-weight:700;color:#fff;padding:10px 4px;white-space:nowrap;">{greet}, {st.session_state.username}!</div>', unsafe_allow_html=True)
    else:
        if st.button(translate_text("Sign In", st.session_state.lang_code), key="nav_signin"):
            st.session_state.auth_mode="login"
            st.session_state.auth_redirect=None
            st.session_state.page="auth"; st.rerun()

with n8:
    if st.session_state.signed_in:
        if st.button(translate_text("Sign Out", st.session_state.lang_code), key="nav_signout"):
            st.session_state.signed_in=False
            st.session_state.username=None
            st.session_state.page="home"; st.rerun()
    else:
        if st.button(translate_text("Sign Up", st.session_state.lang_code), key="nav_signup"):
            st.session_state.auth_mode="signup"
            st.session_state.auth_redirect=None
            st.session_state.page="auth"; st.rerun()

st.markdown("<div style='border-bottom:1px solid rgba(255,255,255,0.07);margin-bottom:28px;'></div>", unsafe_allow_html=True)

# ── Scroll-to anchor via components.v1.html (actually executes JS) ──────────
_scroll = st.session_state.get("scroll_to")
if _scroll:
    import streamlit.components.v1 as _components
    _components.html(f"""
    <script>
    (function() {{
      var tries = 0;
      function doScroll() {{
        // Try parent document first (Streamlit renders in iframe)
        var doc = window.parent ? window.parent.document : document;
        var el = doc.getElementById("{_scroll}");
        if (!el) el = doc.querySelector('[data-anchor="{_scroll}"]');
        if (el) {{
          el.scrollIntoView({{behavior:"smooth", block:"start"}});
        }} else if (tries < 40) {{
          tries++;
          setTimeout(doScroll, 100);
        }}
      }}
      setTimeout(doScroll, 200);
    }})();
    </script>
    """, height=0, scrolling=False)
    st.session_state.scroll_to = None

# ── Page Routing ──
lc = st.session_state.get("lang_code", "en")

if st.session_state.page == "auth":
    st.markdown('<span id="auth-marker"></span>', unsafe_allow_html=True)
    mode = st.session_state.auth_mode
    if "auth_error" not in st.session_state: st.session_state.auth_error = ""

    st.markdown("""
    <style>
    /* ── AUTH CARD CONTAINER ── */
    [data-testid="stHorizontalBlock"]:has(div[data-testid="stTextInput"]) > div:nth-child(2) > div[data-testid="stVerticalBlock"] {
        background: rgba(11, 19, 43, 0.94) !important;
        border: 1px solid rgba(0, 224, 255, 0.22) !important;
        border-radius: 28px !important;
        box-shadow: 0 30px 90px rgba(0,0,0,0.85), 0 0 40px rgba(0,224,255,0.08) !important;
        padding: 44px 48px 40px !important;
        backdrop-filter: blur(12px);
    }
    
    /* ── LOGO GLOW ── */
    .auth-logo-glow {
        width: 60px; height: 60px; border-radius: 16px;
        background: linear-gradient(135deg,#00E0FF,#14FFEC);
        display: flex; align-items: center; justify-content: center;
        margin: 0 auto 20px;
        box-shadow: 0 0 35px rgba(0,224,255,0.5);
        font-weight: 900; color: #001; font-size: 18px;
        font-family: 'Space Grotesk', sans-serif;
    }

    /* ── INPUTS ── */
    div[data-testid="stTextInput"] label {
        font-size: 13px !important; font-weight: 700 !important;
        color: rgba(217,226,236,0.6) !important; margin-bottom: 6px !important;
    }
    div[data-testid="stTextInput"] input {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 12px !important; color: #fff !important;
        padding: 12px 16px !important; transition: all 0.2s;
    }
    div[data-testid="stTextInput"] input:focus {
        border-color: #00E0FF !important;
        background: rgba(255,255,255,0.08) !important;
        box-shadow: 0 0 0 4px rgba(0,224,255,0.1) !important;
    }

    /* ── MAIN SUBMIT BUTTON ── */
    div[data-testid="stVerticalBlock"] > div[data-testid="stButton"]:not(.auth-link-row) > button {
        background: linear-gradient(90deg, #00E0FF, #14FFEC) !important;
        color: #001 !important; font-weight: 800 !important;
        border-radius: 14px !important; border: none !important;
        padding: 14px 24px !important; font-size: 16px !important;
        width: 100% !important; min-height: 54px !important;
        margin-top: 10px !important;
        box-shadow: 0 10px 25px rgba(0,224,255,0.3) !important;
        transition: all 0.2s !important;
    }
    div[data-testid="stVerticalBlock"] > div[data-testid="stButton"]:not(.auth-link-row) > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 15px 35px rgba(0,224,255,0.45) !important;
    }

    /* ── BOTTOM LINK BUTTON ── */
    .auth-link-row button {
        background: transparent !important;
        color: #00E0FF !important; font-weight: 700 !important;
        border: none !important; padding: 0 !important;
        font-size: 14px !important; box-shadow: none !important;
        text-decoration: underline !important; text-underline-offset: 4px !important;
        width: auto !important; margin: 0 auto !important; display: block !important;
    }
    .auth-link-row button:hover {
        color: #14FFEC !important; background: transparent !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <style>
    /* Card Container */
    div[data-testid="stVerticalBlock"]:has(div[data-testid="stTextInput"]) {
        background: rgba(11, 19, 43, 0.9) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 20px !important;
        padding: 40px !important;
        box-shadow: 0 15px 35px rgba(0,0,0,0.4) !important;
    }
    </style>
    """, unsafe_allow_html=True)

    _, mid, _ = st.columns([1, 1.2, 1])
    with mid:
        # Simple Logo
        st.markdown("""
        <div style="text-align:center;margin-bottom:20px;">
          <div style="width:54px;height:54px;border-radius:12px;
               background:linear-gradient(135deg,#00E0FF,#14FFEC);
               display:inline-flex;align-items:center;justify-content:center;
               font-weight:900;color:#001;font-size:16px;
               font-family:'Space Grotesk',sans-serif;">T→T</div>
        </div>""", unsafe_allow_html=True)
        
        # Title
        lc = st.session_state.lang_code
        title = translate_text("Welcome back!", lc) if mode == "login" else translate_text("Create Account", lc)
        st.markdown(f"<h2 style='text-align:center;color:#fff;margin-bottom:24px;'>{title}</h2>", unsafe_allow_html=True)

        # Error
        if st.session_state.auth_error:
            st.error(translate_text(st.session_state.auth_error, lc))

        if mode == "login":
            st.text_input(translate_text("Username", lc), placeholder=translate_text("Enter your username", lc), key="li_user")
            st.text_input(translate_text("Password", lc), placeholder=translate_text("Enter your password", lc), key="li_pass", type="password")
            st.markdown("<div style='margin:10px 0'></div>", unsafe_allow_html=True)
            if st.button(translate_text("Sign In", lc), key="do_login", use_container_width=True):
                u = st.session_state.get("li_user","")
                p = st.session_state.get("li_pass","")
                if not u or not p:
                    st.session_state.auth_error = "Please fill in all fields."
                elif not check_user(u, p):
                    st.session_state.auth_error = "Invalid username or password."
                else:
                    st.session_state.signed_in = True; st.session_state.username = u

                    st.session_state.auth_error = ""
                    st.session_state.page = st.session_state.get("auth_redirect") or "home"
                    st.rerun()
            st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
            st.markdown(f'''<div style="text-align:center;font-size:14px;color:rgba(255,255,255,0.4);">{translate_text("Don't have an account?", lc)}</div>''', unsafe_allow_html=True)
            if st.button(translate_text("Switch to Sign Up", lc), key="go_signup", use_container_width=True):
                st.session_state.auth_mode="signup"; st.session_state.auth_error=""; st.rerun()
        else:
            st.text_input(translate_text("Username", lc),         placeholder=translate_text("Choose a username", lc),      key="su_user")
            st.text_input(translate_text("Password", lc),         placeholder=translate_text("Minimum 6 characters", lc),   key="su_pass",  type="password")
            st.text_input(translate_text("Confirm Password", lc), placeholder=translate_text("Re-enter your password", lc), key="su_pass2", type="password")
            st.markdown("<div style='margin:10px 0'></div>", unsafe_allow_html=True)
            if st.button(translate_text("Create Account", lc), key="do_signup", use_container_width=True):
                u  = st.session_state.get("su_user","")
                p  = st.session_state.get("su_pass","")
                p2 = st.session_state.get("su_pass2","")
                if not u or not p or not p2:
                    st.session_state.auth_error = "Please fill in all fields."
                elif len(p) < 6:
                    st.session_state.auth_error = "Password must be at least 6 characters."
                elif p != p2:
                    st.session_state.auth_error = "Passwords do not match."
                elif user_exists(u):
                    st.session_state.auth_error = "Username already taken."
                else:
                    if add_user(u, p):
                        st.session_state.signed_in = True; st.session_state.username = u

                        st.session_state.page = st.session_state.get("auth_redirect") or "home"
                        st.rerun()
                    else:
                        st.session_state.auth_error = "Database error."
            st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
            st.markdown(f'''<div style="text-align:center;font-size:14px;color:rgba(255,255,255,0.4);">{translate_text("Already have an account?", lc)}</div>''', unsafe_allow_html=True)
            if st.button(translate_text("Switch to Sign In", lc), key="go_login", use_container_width=True):
                st.session_state.auth_mode="login"; st.session_state.auth_error=""; st.rerun()



# ══════════════════════════════════════════════════════════════════════════════
# HOME PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "home":

    # HERO
    hl, hr = st.columns([1.1, 1], gap="large")
    with hl:
        lc = st.session_state.lang_code
        st.markdown(f"""
        <div style="padding-top:24px;">
          <div class="hero-eyebrow">⚡ {translate_text("Real-time", lc)} &nbsp;•&nbsp; AI &nbsp;•&nbsp; {translate_text("Sentiment", lc)}</div>
          <div class="hero-title">
            {translate_text("Never Miss a", lc)} <span class="accent-text">{translate_text("Market-Moving", lc)}</span><br>{translate_text("Moment Again", lc)}
          </div>
          <p class="hero-sub">
            {translate_text("From Tweets to Trades — convert social chatter into real-time trading signals and market alerts.", lc)}
          </p>
        </div>""", unsafe_allow_html=True)

    with hr:
        snap = get_ticker_history_cached("HDFCBANK.NS", period="30d")
        if not snap.empty:
            if isinstance(snap.columns, pd.MultiIndex): snap.columns=snap.columns.get_level_values(0)
            pc = "Adj Close" if "Adj Close" in snap.columns else "Close"
            lp = float(snap[pc].iloc[-1])
            fig_h = go.Figure()
            fig_h.add_trace(go.Scatter(x=snap.index, y=snap[pc], mode="lines",
                line=dict(color="#00E0FF", width=2.5),
                fill="tozeroy", fillcolor="rgba(0,224,255,0.05)"))
            fig_h.update_layout(
                paper_bgcolor="rgba(11,19,43,0.92)", plot_bgcolor="rgba(11,19,43,0.92)",
                font=dict(color="#D9E2EC", family="Inter"),
                height=300, showlegend=False,
                margin=dict(l=50,r=20,t=60,b=50),
                xaxis=dict(gridcolor="rgba(255,255,255,0.07)", color="rgba(255,255,255,0.55)",
                           tickformat="%b %d\n%Y", tickfont=dict(size=10)),
                yaxis=dict(gridcolor="rgba(255,255,255,0.07)", color="rgba(255,255,255,0.55)"),
                title=dict(text=f"🔥 HDFCBANK.NS &nbsp;&nbsp; ₹{lp:,.2f}",
                           font=dict(size=15, color="#fff"), x=0.04, xanchor="left"),
                shapes=[dict(type="rect",xref="paper",yref="paper",x0=0,y0=0,x1=1,y1=1,
                             line=dict(color="rgba(0,224,255,0.1)",width=1),fillcolor="rgba(0,0,0,0)")])
            st.plotly_chart(fig_h, use_container_width=True)

    st.markdown("<div style='margin:20px 0;'></div>", unsafe_allow_html=True)

    # EVERYTHING YOU NEED
    st.markdown(f'''
    <div style="text-align:center;margin:0 0 36px;">
      <h2 style="font-size:clamp(24px,3.5vw,42px);color:#fff;margin:0 0 12px;font-weight:800;">
        {translate_text("Everything you need to trade smarter", lc)}
      </h2>
      <p style="color:rgba(217,226,236,0.65);font-size:15px;max-width:55ch;margin:0 auto;line-height:1.65;">
        {translate_text("From sentiment signals to AI predictions — our platform gives you an edge at every step of your research.", lc)}
      </p>
    </div>''', unsafe_allow_html=True)

    # PREDICTOR card
    st.markdown(f'''
    <div class="showcase-card">
      <div class="showcase-label">{translate_text("PREDICTOR", lc)}</div>
      <div class="showcase-title">{translate_text("Predict stock movement", lc)} <span class="accent-text">{translate_text("before it happens", lc)}</span></div>
      <p class="showcase-desc">{translate_text("Our model analyses historical data, social sentiment, and volume patterns to forecast short-term price direction with a confidence score.", lc)}</p>
      <a class="showcase-cta">{translate_text("Try the Predictor", lc)} →</a>
      <div class="mock-predictor">
        <div style="font-weight:800;font-size:16px;margin-bottom:14px;">TCS.NS</div>
        <div class="bar-row">
          <span class="bar-label">{translate_text("Confidence", lc)}</span>
          <div class="bar-track"><div class="bar-fill" style="width:78%;background:linear-gradient(90deg,#14FFEC,#00E0FF);"></div></div>
          <span class="bar-val">78%</span>
        </div>
        <div class="bar-row">
          <span class="bar-label">{translate_text("Sentiment", lc)}</span>
          <div class="bar-track"><div class="bar-fill" style="width:65%;background:linear-gradient(90deg,#74f29b,#00E0FF);"></div></div>
          <span class="bar-val">65%</span>
        </div>
        <div class="bar-row">
          <span class="bar-label">{translate_text("Risk Score", lc)}</span>
          <div class="bar-track"><div class="bar-fill" style="width:32%;background:linear-gradient(90deg,#ff7b84,#ffb347);"></div></div>
          <span class="bar-val" style="color:#ff7b84;">32%</span>
        </div>
      </div>
    </div>''', unsafe_allow_html=True)

    # SIGNAL + COMPARISON row
    sc1, sc2 = st.columns(2, gap="medium")
    with sc1:
        lc = st.session_state.lang_code
        st.markdown(f'''
        <div class="showcase-card">
          <div class="showcase-label">{translate_text("SIGNAL ENGINE", lc)}</div>
          <div class="showcase-title">{translate_text("Clear", lc)} <span class="accent-text">Buy / Sell / Hold</span> {translate_text("signals", lc)}</div>
          <p class="showcase-desc">{translate_text("No more second-guessing. Our engine combines price action, social momentum, and sentiment drift to give you a clear action signal.", lc)}</p>
          <a class="showcase-cta">{translate_text("See Live Signals", lc)} →</a>
          <div class="signals-row">
            <div class="sig-badge sig-badge--buy">
              <div class="sig-icon">↑</div><div class="sig-lbl">{translate_text("BUY", lc)}</div>
            </div>
            <div class="sig-badge sig-badge--sell">
              <div class="sig-icon">↓</div><div class="sig-lbl">{translate_text("SELL", lc)}</div>
            </div>
            <div class="sig-badge sig-badge--hold">
              <div class="sig-icon">→</div><div class="sig-lbl">{translate_text("HOLD", lc)}</div>
            </div>
          </div>
        </div>''', unsafe_allow_html=True)
    with sc2:
        st.markdown(f'''
        <div class="showcase-card">
          <div class="showcase-label">{translate_text("COMPARISON TOOL", lc)}</div>
          <div class="showcase-title">{translate_text("Compare stocks", lc)} <span class="accent-text">{translate_text("side by side", lc)}</span></div>
          <p class="showcase-desc">{translate_text("Pit any two NSE stocks head-to-head across sentiment, price trend, volume — so you always pick the stronger trade.", lc)}</p>
          <a class="showcase-cta">{translate_text("Compare Now", lc)} →</a>
          <div style="display:flex;align-items:center;justify-content:space-around;
               background:rgba(0,0,0,0.25);border-radius:14px;padding:20px;margin-top:14px;">
            <span style="font-family:'Space Grotesk',sans-serif;font-size:52px;font-weight:900;color:#74f29b;">TCS</span>
            <span style="font-size:18px;font-weight:800;color:rgba(217,226,236,0.4);">{translate_text("vs", lc)}</span>
            <span style="font-family:'Space Grotesk',sans-serif;font-size:52px;font-weight:900;color:#ff7b84;">INFY</span>
          </div>
        </div>''', unsafe_allow_html=True)

    # SENTIMENT card
    st.markdown(f'''
    <div class="showcase-card">
      <div class="showcase-label">{translate_text("SENTIMENT ANALYSIS", lc)}</div>
      <div class="showcase-title">{translate_text("Track social mood", lc)} <span class="accent-text">{translate_text("in real time", lc)}</span></div>
      <p class="showcase-desc">{translate_text("We ingest thousands of tweets, news articles, and forum posts every hour — scoring market sentiment by ticker, sector, and event type so you always know the crowd's pulse.", lc)}</p>
      <a class="showcase-cta">{translate_text("View Sentiment Dashboard", lc)} →</a>
      <div style="margin-top:14px;">
        <span class="sent-chip sent-chip--pos">📈 {translate_text("Banking", lc)} <strong>+72%</strong></span>
        <span class="sent-chip sent-chip--neg">📉 {translate_text("IT", lc)} <strong>-18%</strong></span>
        <span class="sent-chip sent-chip--neu">➡ {translate_text("Energy", lc)} <strong>+4%</strong></span>
        <span class="sent-chip sent-chip--pos">📈 {translate_text("FMCG", lc)} <strong>+55%</strong></span>
        <span class="sent-chip sent-chip--neg">📉 {translate_text("Pharma", lc)} <strong>-31%</strong></span>
      </div>
    </div>''', unsafe_allow_html=True)

    # FEATURE CARDS
    f1,f2,f3 = st.columns(3, gap="medium")
    for col,icon,title,desc in [
        (f1,"⚡","Market Trend Analysis","Continuously analyze market data to identify emerging trends and highlight sectors showing unusual momentum or investor attention."),
        (f2,"🤖","AI-driven Insights","Proprietary models filter noise and surface high-confidence trade ideas with sentiment and event scoring."),
        (f3,"📰","Sentiment Tracking","Track sentiment trends across sectors and map social momentum to price movement with clear visualizations."),
    ]:
        t_title = translate_text(title, st.session_state.lang_code)
        t_desc  = translate_text(desc, st.session_state.lang_code)
        col.markdown(f"""
        <div class="feat-card">
          <div class="feat-icon">{icon}</div>
          <div style="font-size:17px;font-weight:800;margin-bottom:8px;">{t_title}</div>
          <p style="color:rgba(217,226,236,0.7);font-size:14px;margin:0;line-height:1.65;">{t_desc}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin:16px 0;'></div>", unsafe_allow_html=True)

    # TOP TRENDING COMPANIES
    st.markdown('<div id="top-companies" style="position:relative;top:-80px;visibility:hidden;"></div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="margin:8px 0 18px;">
      <div style="font-size:22px;font-weight:800;margin-bottom:4px;">{translate_text("Top Trending Companies", st.session_state.lang_code)}</div>
      <div class="muted">{translate_text("Live social and market movement", st.session_state.lang_code)}</div>
    </div>""", unsafe_allow_html=True)

    top_list = [("HDFCBANK","HDFC Bank","H","HDFC • Finance"),
                ("RELIANCE","Reliance Ind.","R","REL • Technology"),
                ("TCS","TCS","T","TCS • IT Services"),
                ("ICICIBANK","ICICI","I","ICICI • Finance")]
    tc = st.columns(4, gap="small")
    for col,(sym,name,letter,sub) in zip(tc,top_list):
        h2 = yf.download(f"{sym}.NS",period="2d",interval="1d")
        try: inf2 = yf.Ticker(f"{sym}.NS").info or {}
        except: inf2 = {}
        if isinstance(h2.columns,pd.MultiIndex): h2.columns=h2.columns.get_level_values(0)
        pc2="Adj Close" if "Adj Close" in h2.columns else "Close"
        cur=float(h2[pc2].iloc[-1]) if not h2.empty else 0
        vol=inf2.get("volume",0)
        col.markdown(f"""
        <div class="company-row-card">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:8px;margin-bottom:10px;">
            <div style="display:flex;gap:10px;align-items:center;">
              <div class="logo-sm">{letter}</div>
              <div>
                <div style="font-weight:800;font-size:15px;">{name}</div>
                <div style="font-size:12px;color:rgba(255,255,255,0.5);">{sub}</div>
              </div>
            </div>
            <div style="text-align:right;">
              <div style="font-weight:800;font-size:16px;font-family:'Space Grotesk',sans-serif;">₹{cur:,.2f}</div>
              <div class="live-badge">● LIVE</div>
            </div>
          </div>
          <div style="font-size:13px;color:rgba(217,226,236,0.55);">Vol: {vol:,}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin:16px 0;'></div>", unsafe_allow_html=True)

    # HOW IT WORKS
    st.markdown('<div id="how-it-works" style="position:relative;top:-80px;visibility:hidden;"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="margin:10px 0 20px;">
      <div style="font-size:24px;font-weight:800;margin-bottom:6px;">From Tweets to Trends — How it works</div>
      <div class="muted">Raw social chatter → signal processing → trade-ready alerts</div>
    </div>""", unsafe_allow_html=True)

    hw1,hw2 = st.columns(2, gap="medium")
    steps=[
        ("1","Collect","We ingest tweets, threads, news, Reddit posts and message board chatter in real time across global feeds."),
        ("2","Filter & Classify","Our NLP pipeline removes bots and noise, classifies event types, and tags tickers & topics automatically."),
        ("3","Score","Signals are scored for confidence, volume, and sentiment drift. Only high-impact signals generate alerts."),
        ("4","Analysis","Takes the historical data and sentiments in account for further analysis."),
        ("5","Stock Prediction","After thorough analysis, we can predict the stock price movement and suggest trades."),
        ("6","Visualizations","See trend timelines, sentiment heatmaps, and correlation views to validate ideas before trading."),
    ]
    for i,(num,title,desc) in enumerate(steps):
        col=hw1 if i<3 else hw2
        t_title = translate_text(title, st.session_state.lang_code)
        t_desc  = translate_text(desc, st.session_state.lang_code)
        col.markdown(f"""
        <div class="flow-step">
          <div class="flow-num">{num}</div>
          <div>
            <div style="font-weight:800;font-size:15px;margin-bottom:4px;">{t_title}</div>
            <p style="margin:0;color:rgba(217,226,236,0.7);font-size:13px;line-height:1.55;">{t_desc}</p>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin:16px 0;'></div>", unsafe_allow_html=True)

    # ABOUT
    ab_l, ab_r = st.columns([1.3, 1], gap="large")
    with ab_l:
        st.markdown(f'''
        <div style="padding:8px 0;">
          <div style="display:inline-block;padding:5px 14px;border-radius:999px;font-size:12px;
               font-weight:800;color:#00E0FF;background:rgba(0,224,255,0.1);
               border:1px solid rgba(0,224,255,0.2);margin-bottom:16px;">{translate_text("Our Mission", lc)}</div>
          <div style="font-size:clamp(24px,3vw,36px);font-weight:800;line-height:1.2;margin-bottom:18px;">
            {translate_text("Built for traders who", lc)}<br>
            <span style="background:linear-gradient(90deg,#14FFEC,#00E0FF);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">{translate_text("demand more", lc)}</span>
            {translate_text("from their data", lc)}
          </div>
          <p style="color:rgba(217,226,236,0.78);font-size:15px;line-height:1.7;margin:0 0 14px;">
            {translate_text("We're building From Tweets to Trades — to help you express your trading and investing ideas, and to help you analyse Indian markets better.", lc)}
          </p>
          <p style="color:rgba(217,226,236,0.65);font-size:14px;line-height:1.7;margin:0 0 14px;">
            {translate_text("Stock Markets are the true indicator of the growth of any country's economy. We are bullish on India's prospects to be one of the largest economies in the world. We believe that Stock Markets provide a unique opportunity for all Indians to participate in India's growth story — and we are enabling the same.", lc)}
          </p>
          <p style="color:rgba(217,226,236,0.65);font-size:14px;line-height:1.7;margin:0;">
            {translate_text("Most screening, trading, and investing platforms available today have not evolved with time. We plan to change that — a technology-led platform built for super traders and long-term investors, powered by real-time AI and social intelligence.", lc)}
          </p>
        </div>''', unsafe_allow_html=True)

    with ab_r:
        st.markdown(f'''
        <div class="about-stats-grid">
          <div class="about-stat-box">
            <div class="about-stat-num" style="color:#00E0FF;">50</div>
            <div class="about-stat-lbl">{translate_text("NSE Stocks Tracked", lc)}</div>
          </div>
          <div class="about-stat-box">
            <div class="about-stat-num" style="color:#14FFEC;font-size:18px;">{translate_text("Real-time", lc)}</div>
            <div class="about-stat-lbl">{translate_text("Social Sentiment Feeds", lc)}</div>
          </div>
          <div class="about-stat-box">
            <div class="about-stat-num" style="color:#00E0FF;">14 {translate_text("Days", lc)}</div>
            <div class="about-stat-lbl">{translate_text("Predictions", lc)}</div>
          </div>
          <div class="about-stat-box">
            <div class="about-stat-num" style="color:#14FFEC;">5</div>
            <div class="about-stat-lbl">{translate_text("Languages Supported", lc)}</div>
          </div>
        </div>
        <div class="about-val">
          <div class="about-val-ico">🎯</div>
          <div>
            <div class="about-val-title">{translate_text("Precision over noise", lc)}</div>
            <div class="about-val-desc">{translate_text("Only high-confidence signals make it through our NLP pipeline.", lc)}</div>
          </div>
        </div>
        <div class="about-val">
          <div class="about-val-ico">🇮🇳</div>
          <div>
            <div class="about-val-title">{translate_text("Built for India", lc)}</div>
            <div class="about-val-desc">{translate_text("NSE, multilingual support — designed for the Indian market.", lc)}</div>
          </div>
        </div>
        <div class="about-val" style="margin-bottom:0;">
          <div class="about-val-ico">⚡</div>
          <div>
            <div class="about-val-title">{translate_text("Speed matters", lc)}</div>
            <div class="about-val-desc">{translate_text("Market-moving signals delivered in real time, not hours later.", lc)}</div>
          </div>
        </div>''', unsafe_allow_html=True)

    st.markdown("<div style='margin:20px 0;'></div>", unsafe_allow_html=True)

    # CTA BLOCK (FIXED UNIFIED CONTAINER)
    with st.container():
        st.markdown('<span id="cta-marker"></span>', unsafe_allow_html=True)
        
        row1_col1, row1_col2 = st.columns([1.6, 1])
        with row1_col1:
            st.markdown(f'''
                <div style="font-weight:800; font-size:24px; color:#fff; font-family:'Space Grotesk', sans-serif;">{translate_text("Ready to turn chatter into alpha?", lc)}</div>
                <div style="color:rgba(217,226,236,0.4); margin-top:8px; font-size:16px;">{translate_text("Start a free trial — get alerts for your watchlist in minutes.", lc)}</div>
            ''', unsafe_allow_html=True)
            
        with row1_col2:
            st.markdown('<div style="margin-top:-5px;">', unsafe_allow_html=True)
            in_col1, in_col2 = st.columns([1.5, 1])
            with in_col1:
                st.text_input("email_cta", placeholder=translate_text("Your work email", lc), label_visibility="collapsed")
            with in_col2:
                st.button(translate_text("Start Free Trial", lc), key="cta_main_btn", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        st.markdown(f'''
        <div style="margin-top:20px; display:flex; justify-content:space-between; align-items:center;">
            <div style="color:rgba(217,226,236,0.3); font-size:14px; display:flex; align-items:center; gap:8px;">
                <span style="font-size:16px;">©</span> {translate_text("From Tweets to Trades", lc)} — 2025
            </div>
            <div style="display:flex; gap:25px; margin-right:60px;">
                <a href="?page=privacy" target="_self" class="legal-link">{translate_text("Privacy Policy", lc)}</a>
                <a href="?page=terms" target="_self" class="legal-link">{translate_text("Terms of Service", lc)}</a>
            </div>
        </div>''', unsafe_allow_html=True)



# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "analysis":
    if not st.session_state.signed_in:
        st.session_state.auth_redirect = "analysis"
        st.session_state.page = "auth"
        st.rerun()
        st.stop()

    # ── CSS that styles Streamlit's own column containers for the left panel ──
    st.markdown("""
    <style>
    /* Style the first column in analysis as the left panel */
    [data-testid="stHorizontalBlock"]:has(> div:nth-child(1) [data-testid="stSelectbox"]) > div:first-child {
        background: rgba(11,19,43,0.92) !important;
        border: 1px solid rgba(0,224,255,0.2) !important;
        border-radius: 14px !important;
        padding: 20px 16px !important;
    }
    /* Left panel selectbox label override */
    [data-testid="stSelectbox"] label {
        font-weight: 800 !important;
        font-size: 15px !important;
        color: #D9E2EC !important;
        margin-bottom: 8px !important;
    }
    /* Generate Insights button — restore cyan gradient inside left panel */
    [data-testid="stHorizontalBlock"]:has(> div:nth-child(1) [data-testid="stSelectbox"]) [data-testid="stButton"] > button {
        background: linear-gradient(90deg,#00E0FF,#14FFEC) !important;
        color: #001 !important;
        font-weight: 800 !important;
        border-radius: 999px !important;
        border: none !important;
        padding: 11px 24px !important;
        font-size: 14px !important;
        box-shadow: 0 4px 15px rgba(0,224,255,0.2) !important;
        width: 100% !important;
    }
    [data-testid="stHorizontalBlock"]:has(> div:nth-child(1) [data-testid="stSelectbox"]) [data-testid="stButton"] > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(0,224,255,0.35) !important;
    }
    /* Key stats container */
    .ks-outer {
        background: linear-gradient(180deg,rgba(11,19,43,0.92),rgba(11,19,43,0.80));
        border-radius: 16px; padding: 24px 28px;
        border: 1px solid rgba(0,224,255,0.1);
        box-shadow: 0 8px 30px rgba(0,0,0,0.5);
        margin-bottom: 20px;
    }
    /* Chart card — title above, plotly sits inside same dark card via paper_bgcolor */
    .chart-card {
        background: rgba(11,19,43,0.92);
        border-radius: 16px 16px 0 0;
        border: 1px solid rgba(0,224,255,0.1);
        border-bottom: none;
        padding: 18px 24px 8px;
        margin-bottom: 0;
    }
    .chart-card-title { font-size:19px; font-weight:800; color:#fff; margin:0 0 2px; }
    .chart-card-sub   { font-size:13px; color:rgba(217,226,236,0.5); margin:0; }
    /* Plotly chart sits below the header card, gets bottom border-radius */
    [data-testid="stPlotlyChart"] {
        background: rgba(11,19,43,0.92) !important;
        border-radius: 0 0 16px 16px !important;
        border: 1px solid rgba(0,224,255,0.1) !important;
        border-top: none !important;
        margin-bottom: 16px !important;
        overflow: hidden !important;
    }
    /* Tiles grid */
    .tiles-outer {
        background: rgba(11,19,43,0.88);
        border-radius: 16px; padding: 20px 22px;
        border: 1px solid rgba(0,224,255,0.08);
        margin-bottom: 16px;
    }
    .pred-tile {
        background: rgba(255,255,255,0.04);
        padding: 12px 6px; border-radius: 10px;
        border: 1px solid rgba(0,224,255,0.1);
        text-align: center; margin-bottom: 8px;
        transition: background .15s;
    }
    .pred-tile:hover { background: rgba(0,224,255,0.06); }
    .pred-tile-date  { font-size: 11px; color:rgba(217,226,236,0.5); margin-bottom:5px; }
    .pred-tile-price { font-size: 14px; font-weight:800; color:#fff; }
    /* News card */
    .news-outer {
        background: rgba(11,19,43,0.88);
        border-radius: 16px; padding: 22px 24px;
        border: 1px solid rgba(0,224,255,0.08);
    }
    .news-item { padding:12px 0; border-bottom:1px solid rgba(255,255,255,0.06); }
    .news-item:last-child { border-bottom: none; padding-bottom: 0; }
    .news-link {
        color:#fff; font-weight:700; font-size:14px;
        text-decoration:none; line-height:1.5; display:block;
    }
    .news-link:hover { color:#00E0FF; }
    .news-meta { font-size:12px; color:rgba(217,226,236,0.5); margin-top:5px; }
    .badge { display:inline-block; padding:3px 10px; border-radius:20px; font-size:11px; font-weight:700; margin-top:6px; }
    .badge-pos { background:rgba(20,255,236,0.12); color:#14FFEC; border:1px solid rgba(20,255,236,0.35); }
    .badge-neg { background:rgba(255,0,0,0.12);    color:#ff6b6b; border:1px solid rgba(255,0,0,0.35); }
    .badge-neu { background:rgba(255,255,255,0.07); color:#D9E2EC; border:1px solid rgba(255,255,255,0.18); }
    /* Risk/Signal card */
    .risk-outer {
        background: rgba(11,19,43,0.88);
        border-radius: 16px; padding: 22px 24px;
        border: 1px solid rgba(0,224,255,0.08);
    }
    .risk-section-title { font-size:17px; font-weight:800; color:#fff; margin:0 0 14px; }
    .risk-bar-wrap {
        width:100%; height:16px; border-radius:999px; overflow:hidden;
        margin: 10px 0 8px;
        background: rgba(255,255,255,0.07);
    }
    .risk-fill {
        height:100%; border-radius:999px;
        background: linear-gradient(90deg,#14FFEC 0%,#00E0FF 50%,#ff6b6b 100%);
        box-shadow: 0 0 16px rgba(0,224,255,0.4);
    }
    .risk-label { font-weight:700; font-size:14px; margin-bottom:3px; }
    .risk-note  { font-size:12px; color:rgba(217,226,236,0.5); }
    .signal-divider { border:none; border-top:1px solid rgba(255,255,255,0.07); margin:16px 0; }
    .signal-box-inner {
        padding: 18px 20px; border-radius:14px;
        background: rgba(0,0,0,0.25);
        border: 1px solid rgba(255,255,255,0.07);
    }
    .signal-row {
        display:flex; justify-content:space-between; align-items:center;
        margin-bottom: 10px;
    }
    .signal-word { font-size:28px; font-weight:900; font-family:'Space Grotesk',sans-serif; }
    .signal-arrow{ font-size:28px; font-weight:900; }
    .signal-buy  { color:#14FFEC; }
    .signal-sell { color:#ff6b6b; }
    .signal-hold { color:#7FDBFF; }
    .signal-strength { font-size:14px; color:rgba(217,226,236,0.7); margin-bottom:7px; }
    .signal-explanation { font-size:13px; color:rgba(217,226,236,0.5); line-height:1.6; }
    .signal-disclaimer { font-size:11px; color:rgba(255,255,255,0.2); margin-top:16px; }
    /* Stats rows */
    .stats-row   { display:flex; justify-content:space-between; align-items:center;
                   padding:9px 0; border-bottom:1px solid rgba(255,255,255,0.05); }
    .stats-row:last-child { border-bottom: none; }
    .stats-lbl   { font-size:13px; color:rgba(255,255,255,0.55); }
    .stats-val   { font-size:13px; font-weight:700; color:#14FFEC; }
    /* Welcome card */
    .welcome-card {
        text-align:center; padding:70px 24px;
        border:2px dashed rgba(0,224,255,0.25); border-radius:16px;
        background: rgba(11,19,43,0.5);
    }
    </style>
    """, unsafe_allow_html=True)

    left_col, right_col = st.columns([1, 3.2], gap="large")

    with left_col:
        st.markdown("**Choose one company (NIFTY 50)**")
        options = ["-- Select a company --"] + [f"{t} - {n}" for t,n in COMPANY_NAMES.items()]
        selected = st.selectbox("Company", options, label_visibility="collapsed")
        st.markdown('<p style="color:rgba(217,226,236,0.6);font-size:13px;margin:6px 0 14px;line-height:1.55;">Tip: pick one ticker. Charts and news will update for the selected company.</p>', unsafe_allow_html=True)
        gen_btn = st.button("Generate Insights", use_container_width=True)

    with right_col:
        # Only show welcome if no ticker is being actively viewed or persisted
        current_ticker = selected.split(" - ")[0].strip() if selected != "-- Select a company --" else None
        is_analyzing = selected != "-- Select a company --" and (gen_btn or (st.session_state.get("analyzed_ticker") == current_ticker))
        
        if not is_analyzing:
            st.markdown("""
            <div class="welcome-card">
              <div style="font-size:52px;margin-bottom:16px;">📈</div>
              <div style="font-size:22px;font-weight:800;margin-bottom:10px;">Welcome!</div>
              <div style="color:rgba(217,226,236,0.6);font-size:14px;line-height:1.65;">
                Select a NIFTY 50 company from the left panel<br>
                to view charts, news, predictions and signals.
              </div>
            </div>""", unsafe_allow_html=True)

    if selected != "-- Select a company --" and (gen_btn or st.session_state.get("analyzed_ticker")):
        # Persist the selection
        if gen_btn:
            st.session_state.analyzed_ticker = selected.split(" - ")[0].strip()
        
        tc  = st.session_state.analyzed_ticker
        tns = f"{tc}.NS"
        with st.spinner(f"Fetching data for {tns}…"):
            data = get_stock_prediction(tns, lang=st.session_state.lang_code)
        if "error" in data:
            st.error(f"Error: {data['error']}")
            st.stop()

        ks = data["key_stats"]
        
        # ── Update History & Alerts ──
        user = st.session_state.username if st.session_state.signed_in else "guest"
        alert = add_to_history(user, tns, ks['close'])
        if alert:
            a_type, diff = alert
            emoji = "📈" if a_type == "increased" else "📉"
            st.toast(f"{emoji} {tns} {translate_text(a_type, st.session_state.lang_code)} by ₹{diff:.2f}!", icon=emoji)

        with right_col:
            # ── KEY STATS ──────────────────────────────────────────────────
            left_stats  = [("OPEN",f"₹{ks['open']:.2f}"),("HIGH",f"₹{ks['high']:.2f}"),
                           ("LOW",f"₹{ks['low']:.2f}"),("CLOSE",f"₹{ks['close']:.2f}"),
                           ("VOLUME",f"{ks['volume']:,}"),("LAST CLOSE",f"₹{ks['last_close']:.2f}"),
                           ("MARKET CAP",f"{ks['market_cap']:,}")]
            right_stats = [("PE RATIO",f"{ks['pe_ratio']:.2f}"),("BETA",f"{ks['beta']:.2f}"),
                           ("EPS BASIC",f"{ks['eps_basic']:.2f}"),("FORWARD PE",f"{ks['forward_pe']:.2f}"),
                           ("DIVIDEND YIELD",f"₹{ks['dividend_yield']:.2f}"),("DAYS RANGE",ks["days_range"])]

            ls_html = "".join(f'<div class="stats-row"><span class="stats-lbl">{translate_text(k, st.session_state.lang_code)}</span><span class="stats-val">{v}</span></div>' for k,v in left_stats)
            rs_html = "".join(f'<div class="stats-row"><span class="stats-lbl">{translate_text(k, st.session_state.lang_code)}</span><span class="stats-val">{v}</span></div>' for k,v in right_stats)

            st.markdown(f"""
            <div class="ks-outer">
              <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:18px;">
                <div style="font-size:20px;font-weight:800;color:#fff;">{translate_text("Key Stats", st.session_state.lang_code)}</div>
                <div id="watchlist-btn-placeholder"></div>
              </div>
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:0 48px;">
                <div>{ls_html}</div>
                <div>{rs_html}</div>
              </div>
            </div>""", unsafe_allow_html=True)
            
            # Watchlist Toggle
            w_user = st.session_state.username if st.session_state.signed_in else "guest"
            in_w = is_in_watchlist(w_user, tns)
            w_label = "⭐ " + translate_text("In Watchlist", st.session_state.lang_code) if in_w else "☆ " + translate_text("Add to Watchlist", st.session_state.lang_code)
            if st.button(w_label, key="watchlist_toggle", use_container_width=True):
                act = toggle_watchlist(w_user, tns)
                st.toast(f"{tns} {act}!")
                st.rerun()

        hg   = data["historical_graph_data"]
        pg   = data["prediction_graph_data"]
        conf = data.get("confidence","--")

        # ── HISTORICAL CHART ───────────────────────────────────────────────
        st.markdown(f"""
        <div class="chart-card">
          <div class="chart-card-title">{translate_text("Historical Price (Past 30 days)", st.session_state.lang_code)}</div>
          <div class="chart-card-sub">{translate_text("Source: sample data", st.session_state.lang_code)} &nbsp;•&nbsp; {tns}</div>
        </div>""", unsafe_allow_html=True)

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=hg["dates"], y=hg["prices"],
            mode="lines+markers",
            name=f"{tc}.NS (Past 30 Days)",
            line=dict(color="#00E0FF", width=2.5),
            marker=dict(size=5, color="#00E0FF", line=dict(color="#0B132B",width=1)),
            fill="tozeroy", fillcolor="rgba(0,224,255,0.04)"))
        min_p1 = min(hg["prices"]) if hg["prices"] else 0
        max_p1 = max(hg["prices"]) if hg["prices"] else 0
        pad1 = (max_p1 - min_p1) * 0.1 if max_p1 != min_p1 else max_p1 * 0.05
        
        fig1.update_layout(
            paper_bgcolor="rgba(11,19,43,0.92)",
            plot_bgcolor="rgba(11,19,43,0.92)",
            font=dict(color="#D9E2EC", family="Inter"),
            height=380, showlegend=True,
            margin=dict(l=60,r=20,t=10,b=50),
            xaxis=dict(gridcolor="rgba(255,255,255,0.06)", color="rgba(255,255,255,0.5)",
                       tickfont=dict(size=11), showline=False),
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)", color="rgba(255,255,255,0.5)",
                       autorange=False, range=[min_p1 - pad1, max_p1 + pad1], tickfont=dict(size=11)),
            legend=dict(orientation="h", y=1.05, font=dict(color="#D9E2EC", size=12),
                        bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig1, use_container_width=True)

        st.markdown("<div style='margin:4px 0;'></div>", unsafe_allow_html=True)

        # ── PREDICTION CHART ───────────────────────────────────────────────
        st.markdown(f"""
        <div class="chart-card">
          <div class="chart-card-title">{translate_text("14-Day Price Prediction", st.session_state.lang_code)}</div>
          <div class="chart-card-sub">{translate_text("Model: Ensemble", st.session_state.lang_code)} &nbsp;•&nbsp; {translate_text("Confidence", st.session_state.lang_code)}: {conf}%</div>
        </div>""", unsafe_allow_html=True)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=pg["dates"], y=pg["prices"],
            mode="lines+markers",
            name=translate_text("Predicted Price (₹)", st.session_state.lang_code),
            line=dict(color="#00E0FF", width=2.5),
            marker=dict(size=6, color="#00E0FF", line=dict(color="#0B132B",width=1)),
            fill="tozeroy", fillcolor="rgba(0,224,255,0.04)"))
        min_p2 = min(pg["prices"]) if pg["prices"] else 0
        max_p2 = max(pg["prices"]) if pg["prices"] else 0
        pad2 = (max_p2 - min_p2) * 0.1 if max_p2 != min_p2 else max_p2 * 0.05
        
        fig2.update_layout(
            paper_bgcolor="rgba(11,19,43,0.92)",
            plot_bgcolor="rgba(11,19,43,0.92)",
            font=dict(color="#D9E2EC", family="Inter"),
            height=380, showlegend=True,
            margin=dict(l=60,r=20,t=10,b=50),
            xaxis=dict(gridcolor="rgba(255,255,255,0.06)", color="rgba(255,255,255,0.5)",
                       tickfont=dict(size=11), showline=False),
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)", color="rgba(255,255,255,0.5)",
                       autorange=False, range=[min_p2 - pad2, max_p2 + pad2], tickfont=dict(size=11)),
            legend=dict(orientation="h", y=1.05, font=dict(color="#D9E2EC", size=12),
                        bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig2, use_container_width=True)

        # ── PREDICTED VALUE TILES ──────────────────────────────────────────
        tiles_row1 = "".join(
            f'<div style="flex:1;min-width:0;">'
            f'<div class="pred-tile">'
            f'<div class="pred-tile-date">{d}</div>'
            f'<div class="pred-tile-price">₹{p:,.2f}</div>'
            f'</div></div>'
            for d,p in zip(pg["dates"][:7], pg["prices"][:7]))
        tiles_row2 = "".join(
            f'<div style="flex:1;min-width:0;">'
            f'<div class="pred-tile">'
            f'<div class="pred-tile-date">{d}</div>'
            f'<div class="pred-tile-price">₹{p:,.2f}</div>'
            f'</div></div>'
            for d,p in zip(pg["dates"][7:], pg["prices"][7:]))

        st.markdown(f"""
        <div class="tiles-outer">
          <div style="font-size:18px;font-weight:800;color:#fff;margin-bottom:4px;">{translate_text("Predicted Values (14 Days)", st.session_state.lang_code)}</div>
          <div style="font-size:13px;color:rgba(217,226,236,0.5);margin-bottom:14px;">{translate_text("Click a tile to highlight", st.session_state.lang_code)}</div>
          <div style="display:flex;gap:8px;margin-bottom:8px;">{tiles_row1}</div>
          <div style="display:flex;gap:8px;">{tiles_row2}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<div style='margin:8px 0;'></div>", unsafe_allow_html=True)

        # ── NEWS + RISK/SIGNAL ─────────────────────────────────────────────
        nc, rc = st.columns([3, 2], gap="small")

        with nc:
            news = data.get("latest_news", [])
            news_html = ""
            if news:
                for n in news:
                    cls = {"Positive":"badge-pos","Negative":"badge-neg"}.get(n["sentiment_label"],"badge-neu")
                    news_html += (
                        f'<div class="news-item">'
                        f'<a href="{n["url"]}" target="_blank" class="news-link">{n["title"]}</a>'
                        f'<div class="news-meta">{n["source"]} &bull; {n.get("published_at","--")}</div>'
                        f'<span class="badge {cls}">{n["sentiment_label"]} | {n["sentiment_score"]}</span>'
                        f'</div>')
            else:
                news_html = '<div style="color:rgba(217,226,236,0.5);padding:20px 0;text-align:center;font-size:14px;">• No recent news found.</div>'

            st.markdown(f"""
            <div class="news-outer">
              <div style="font-size:18px;font-weight:800;color:#fff;margin-bottom:14px;">Latest News ({tns})</div>
              {news_html}
            </div>""", unsafe_allow_html=True)

        with rc:
            risk     = data.get("risk_score", 50)
            rl_raw   = "Low Risk" if risk<30 else ("Moderate Risk" if risk<60 else "High Risk")
            risk_lbl = translate_text(rl_raw, st.session_state.lang_code)
            risk_color = "#14FFEC" if risk<30 else ("#FFC85A" if risk<60 else "#ff6b6b")
            sig      = data["signal"]
            arrow    = {"BUY":"↑","SELL":"↓","HOLD":"→"}[sig]
            sc_class = f"signal-{sig.lower()}"

            st.markdown(f"""
            <div class="risk-outer">
              <div class="risk-section-title">{translate_text("Risk Score Meter", st.session_state.lang_code)}</div>
              <div class="risk-bar-wrap">
                <div class="risk-fill" style="width:{risk:.0f}%;"></div>
              </div>
              <div class="risk-label" style="color:{risk_color};">{risk_lbl}</div>
              <div class="risk-note">{translate_text("Risk shows how safe or risky the decision is.", st.session_state.lang_code)}</div>

              <hr class="signal-divider">

              <div class="risk-section-title">{translate_text("Buy / Sell / Hold Signal", st.session_state.lang_code)}</div>
              <div class="signal-box-inner">
                <div class="signal-row">
                  <span class="signal-word {sc_class}">{translate_text(sig, st.session_state.lang_code)}</span>
                  <span class="signal-arrow {sc_class}">{arrow}</span>
                </div>
                <div class="signal-strength">{translate_text("Signal Strength", st.session_state.lang_code)}: {data.get("signal_strength","--")}%</div>
                <div class="signal-explanation">{data.get("explanation", translate_text("Awaiting prediction data...", st.session_state.lang_code))}</div>
              </div>
              <div class="signal-disclaimer">
                {translate_text("Prototype — dropdown uses NIFTY 50 constituents (source: NIFTY / Wikipedia).", st.session_state.lang_code)}
              </div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# COMPARISON PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "comparison":
    if not st.session_state.signed_in:
        st.session_state.auth_redirect = "comparison"
        st.session_state.page = "auth"
        st.rerun()
        st.stop()

    st.markdown("""
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:24px;">
      <div>
        <div style="font-size:24px;font-weight:800;">Compare Stocks</div>
        <div class="muted" style="margin-top:4px;">Compare multiple NIFTY 50 companies side by side</div>
      </div>
    </div>""", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<div style="font-size:18px;font-weight:800;margin-bottom:14px;">Select Multiple Companies (NIFTY 50)</div>', unsafe_allow_html=True)
        all_labels = [f"{t} - {n}" for t,n in COMPANY_NAMES.items()]
        chosen = st.multiselect(
            "Select companies", all_labels,
            label_visibility="collapsed",
            placeholder="Click to select companies…",
            key="comp_select")
        st.markdown('<div class="muted" style="margin:6px 0 12px;">Hold Ctrl (Windows) or Cmd (Mac) to select multiple companies.</div>', unsafe_allow_html=True)
        cmp_btn = st.button("Compare Selected Stocks", use_container_width=True)

    if chosen and cmp_btn:
        if len(chosen) < 2:
            st.warning("Please select at least 2 stocks.")
            st.stop()

        tickers = [c.split(" - ")[0].strip() for c in chosen]

        all_hist, all_info_map = {}, {}
        with st.spinner("Comparing stocks…"):
            for t in tickers:
                h = get_ticker_history_cached(f"{t}.NS", period="3mo")
                if isinstance(h.columns,pd.MultiIndex): h.columns=h.columns.get_level_values(0)
                pc = "Adj Close" if "Adj Close" in h.columns else "Close"
                if not h.empty: h["price"]=h[pc]
                all_hist[t] = h
                all_info_map[t] = yf.Ticker(f"{t}.NS").info or {}

        # Stock comparison chart
        with st.container(border=True):
            st.markdown('<div style="font-size:18px;font-weight:800;margin-bottom:14px;">Stock Comparison Chart</div>', unsafe_allow_html=True)
            COLORS=["#00E0FF","#ff6b6b","#FFC85A","#74f29b","#a78bfa"]
            fig=go.Figure()
            for idx,t in enumerate(tickers):
                h=all_hist[t]
                if not h.empty and "price" in h.columns:
                    x_vals = h["Date"] if "Date" in h.columns else list(range(1, len(h) + 1))
                    first_price = h["price"].iloc[0] if h["price"].iloc[0] != 0 else 1
                    y_vals = ((h["price"] / first_price) - 1) * 100
                    fig.add_trace(go.Scatter(
                        x=x_vals, y=y_vals,
                        mode="lines+markers", name=t,
                        line=dict(color=COLORS[idx%len(COLORS)],width=2.5),
                        marker=dict(size=5,color=COLORS[idx%len(COLORS)])))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#D9E2EC",family="Inter"),
                margin=dict(l=50,r=20,t=20,b=50), height=420,
                xaxis=dict(gridcolor="rgba(255,255,255,0.07)",color="rgba(255,255,255,0.6)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.07)",color="rgba(255,255,255,0.6)", ticksuffix="%"),
                legend=dict(orientation="h",y=1.06,font=dict(color="#D9E2EC",size=13)))
            st.plotly_chart(fig, use_container_width=True)

        # Key metrics table
        with st.container(border=True):
            st.markdown('<div style="font-size:18px;font-weight:800;margin-bottom:14px;">Key Metrics Comparison</div>', unsafe_allow_html=True)
            header_cells = "".join(f"<th>{t}</th>" for t in tickers)
            rows_html = ""
            metrics = [
                ("Open",       lambda t: f"₹{safe_float(all_hist[t]['Open'].iloc[-1]):,.2f}" if not all_hist[t].empty else "—"),
                ("Close",      lambda t: f"₹{safe_float(all_hist[t]['price'].iloc[-1]):,.2f}" if not all_hist[t].empty and 'price' in all_hist[t].columns else "—"),
                ("Market Cap", lambda t: f"{all_info_map[t].get('marketCap',0):,}" if all_info_map[t].get('marketCap') else "—"),
                ("P/E Ratio",  lambda t: f"{all_info_map[t].get('trailingPE'):.2f}" if isinstance(all_info_map[t].get('trailingPE'),float) else "—"),
                ("Forward P/E",lambda t: f"{all_info_map[t].get('forwardPE'):.2f}"  if isinstance(all_info_map[t].get('forwardPE'),float)  else "—"),
                ("Beta",       lambda t: f"{all_info_map[t].get('beta'):.2f}"       if isinstance(all_info_map[t].get('beta'),float)       else "—"),
                ("Volume",     lambda t: f"{all_info_map[t].get('volume',0):,}"),
            ]
            for m_label, m_fn in metrics:
                cells = "".join(f"<td>{m_fn(t)}</td>" for t in tickers)
                rows_html += f"<tr><td>{m_label}</td>{cells}</tr>"
            st.markdown(f"""
            <table class="comp-table">
              <thead><tr><th>Metric</th>{header_cells}</tr></thead>
              <tbody>{rows_html}</tbody>
            </table>""", unsafe_allow_html=True)

        # Investment verdict
        with st.container(border=True):
            st.markdown(f'<div style="font-size:18px;font-weight:800;margin-bottom:18px;">{translate_text("Investment Verdict", lc)}</div>', unsafe_allow_html=True)
            scored = sorted(
                [{"ticker":t,"name":COMPANY_NAMES.get(t,t),
                  "score":score_stock(all_info_map[t]),"info":all_info_map[t]} for t in tickers],
                key=lambda x:x["score"], reverse=True)
            winner=scored[0]

            vc = st.columns(len(scored), gap="small")
            for col,s in zip(vc,scored):
                is_w = s["ticker"]==winner["ticker"]
                i = s["info"]
                pe  = f"{i.get('trailingPE'):.1f}" if isinstance(i.get('trailingPE'),float) else "—"
                fpe = f"{i.get('forwardPE'):.1f}"  if isinstance(i.get('forwardPE'),float)  else "—"
                bet = f"{i.get('beta'):.2f}"        if isinstance(i.get('beta'),float)       else "—"
                mcap= fmt_mcap(i.get('marketCap'))
                vol = f"{i.get('volume',0):,}"
                rank= scored.index(s)+1
                bar_bg="linear-gradient(90deg,#00E0FF,#14FFEC)" if is_w else "rgba(255,255,255,0.2)"
                sc_color="#14FFEC" if is_w else "#fff"
                badge=f'<div class="vcard-badge" style="position:absolute;top:10px;right:10px;background:#00E0FF;color:#001;font-size:10px;padding:2px 8px;border-radius:4px;font-weight:900;">{translate_text("TOP PICK", lc)}</div>' if is_w else ""
                col.markdown(f"""
                <div class="vcard {'winner' if is_w else ''}" style="position:relative; background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); border-radius:16px; padding:24px;">
                  <div class="vcard-rank" style="position:absolute; bottom:12px; right:16px; font-size:12px; font-weight:900; color:rgba(255,255,255,0.2);">#{rank}</div>
                  {badge}
                  <div class="vcard-header" style="display:flex;justify-content:space-between;align-items:center;">
                    <span class="vcard-ticker" style="font-size:24px; font-weight:900; font-family:'Space Grotesk',sans-serif; color:#fff;">{s['ticker']}</span>
                    <span style="font-size:18px;font-weight:900;color:{sc_color};">{s['score']}/100</span>
                  </div>
                  <div class="vcard-bar-wrap" style="height:6px; background:rgba(255,255,255,0.1); border-radius:99px; margin:16px 0 8px;">
                    <div class="vcard-bar" style="width:{s['score']}%;background:{bar_bg};height:100%;border-radius:99px;"></div>
                  </div>
                  <div class="vcard-score-lbl" style="display:flex;justify-content:space-between;font-size:12px;color:rgba(255,255,255,0.4);margin-bottom:16px;">
                    <span>{translate_text("Investment Score", lc)}</span><span>{s['score']}/100</span>
                  </div>
                  <div class="vcard-meta" style="font-size:13px;border-top:1px solid rgba(255,255,255,0.05);padding-top:12px;">
                    <div style="display:flex;justify-content:space-between;margin-bottom:6px;"><span>P/E</span><span>{pe}</span></div>
                    <div style="display:flex;justify-content:space-between;margin-bottom:6px;"><span>Fwd P/E</span><span>{fpe}</span></div>
                    <div style="display:flex;justify-content:space-between;margin-bottom:6px;"><span>Beta</span><span>{bet}</span></div>
                    <div style="display:flex;justify-content:space-between;margin-bottom:6px;"><span>Market Cap</span><span>{mcap}</span></div>
                    <div style="display:flex;justify-content:space-between;"><span>Volume</span><span>{vol}</span></div>
                  </div>
                </div>""", unsafe_allow_html=True)

            payload=[{"ticker":s["ticker"],"name":s["name"],"score":s["score"],
                      "pe":all_info_map[s["ticker"]].get("trailingPE"),
                      "forwardPe":all_info_map[s["ticker"]].get("forwardPE"),
                      "beta":all_info_map[s["ticker"]].get("beta"),
                      "marketCap":all_info_map[s["ticker"]].get("marketCap"),
                      "volume":all_info_map[s["ticker"]].get("volume")} for s in scored]

            with st.spinner(translate_text("Generating AI verdict…", lc)):
                explanation = get_gemini_verdict(payload[0], payload)
            
            if not explanation:
                w = payload[0]
                others = ", ".join([f"<strong>{s['ticker']}</strong> ({s['score']}/100)" for s in payload[1:]])
                pe = w.get('pe') or "N/A"
                mcap = w.get('marketCap')
                mcap_str = f"₹{mcap/1e12:.1f}T" if mcap else "N/A"
                explanation = (
                    f"<p>Based on quantitative analysis, <strong>{w['ticker']}</strong> ranked first with an "
                    f"Investment Score of <strong>{w['score']}/100</strong>, outperforming {others}.</p>"
                    f"<p>Key metrics driving this ranking: a trailing P/E of <strong>{pe}</strong>, "
                    f"Market Cap of <strong>{mcap_str}</strong>, and a Beta of <strong>{w.get('beta', 'N/A')}</strong>. "
                    f"These fundamentals collectively suggest a stronger risk-adjusted return profile.</p>"
                    f"<p><em>Note: Full AI analysis unavailable — API quota exceeded. The verdict above is computed from live market data.</em></p>"
                )

            st.markdown(f"""
            <div style="background:rgba(0,224,255,0.04); border:1px solid rgba(0,224,255,0.15);
                        border-radius:16px; padding:28px 32px; margin-top:24px;">
                <div style="display:flex; align-items:center; gap:10px; margin-bottom:18px;">
                    <div style="width:10px; height:10px; border-radius:50%; background:#00E0FF;
                                box-shadow: 0 0 8px #00E0FF; animation: pulse 2s infinite;"></div>
                    <div style="font-weight:800; font-size:16px; color:#00E0FF; font-family:'Space Grotesk',sans-serif;">
                        {translate_text("AI Analyst Verdict", lc)}
                    </div>
                    <div style="margin-left:auto; font-size:11px; color:rgba(255,255,255,0.3);
                                background:rgba(255,255,255,0.05); padding:3px 10px; border-radius:20px;">
                        Powered by Gemini
                    </div>
                </div>
                <div style="font-size:14.5px; line-height:1.85; color:#D9E2EC; letter-spacing:0.01em;">
                    {explanation}
                </div>
            </div>""", unsafe_allow_html=True)
    else:
        st.info("👆 " + translate_text("Select at least 2 companies above and click Compare Selected Stocks.", lc))

# ══════════════════════════════════════════════════════════════════════════════
# PRIVACY POLICY PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "privacy":
    if st.button("← " + translate_text("Back to Home", lc), key="back_home_priv", use_container_width=True):
        st.session_state.page = "home"; st.rerun()
    
    st.markdown(f'''<div class="glass-card" style="padding:40px; margin-top:20px; max-width:900px; margin:0 auto;">
<h1 style="color:#00E0FF; margin-bottom:30px; font-size:32px;">{translate_text("Privacy Policy", lc)}</h1>
<div class="legal-point">
<h3 style="color:#fff; margin-top:0; font-size:18px;">1. {translate_text("Information We Collect", lc)}</h3>
<p style="color:rgba(217,226,236,0.7); line-height:1.7; margin-bottom:0;">{translate_text("We may collect basic account information such as your name, email address, and login credentials when you register on our platform.", lc)}</p>
</div>
<div class="legal-point">
<h3 style="color:#fff; margin-top:0; font-size:18px;">2. {translate_text("How We Use Your Information", lc)}</h3>
<p style="color:rgba(217,226,236,0.7); line-height:1.7; margin-bottom:0;">{translate_text("Your information is used to provide stock analysis services, improve user experience, generate insights, and communicate important updates.", lc)}</p>
</div>
<div class="legal-point">
<h3 style="color:#fff; margin-top:0; font-size:18px;">3. {translate_text("Data Security", lc)}</h3>
<p style="color:rgba(217,226,236,0.7); line-height:1.7; margin-bottom:0;">{translate_text("We implement appropriate security measures to protect your data from unauthorized access, alteration, or disclosure.", lc)}</p>
</div>
<div class="legal-point">
<h3 style="color:#fff; margin-top:0; font-size:18px;">4. {translate_text("Third-Party Services", lc)}</h3>
<p style="color:rgba(217,226,236,0.7); line-height:1.7; margin-bottom:0;">{translate_text("Our platform may use third-party APIs such as market data providers. We are not responsible for their privacy practices.", lc)}</p>
</div>
<div class="legal-point">
<h3 style="color:#fff; margin-top:0; font-size:18px;">5. {translate_text("Cookies", lc)}</h3>
<p style="color:rgba(217,226,236,0.7); line-height:1.7; margin-bottom:0;">{translate_text("We may use cookies to enhance user experience and maintain session authentication.", lc)}</p>
</div>
<div class="legal-point">
<h3 style="color:#fff; margin-top:0; font-size:18px;">6. {translate_text("Changes to This Policy", lc)}</h3>
<p style="color:rgba(217,226,236,0.7); line-height:1.7; margin-bottom:0;">{translate_text("We may update this Privacy Policy from time to time. Changes will be reflected on this page.", lc)}</p>
</div>
<div class="legal-point">
<h3 style="color:#fff; margin-top:0; font-size:18px;">7. {translate_text("Contact Us", lc)}</h3>
<p style="color:rgba(217,226,236,0.7); line-height:1.7; margin-bottom:0;">{translate_text("If you have any questions regarding this Privacy Policy, please contact us through the platform.", lc)}</p>
</div>
</div>''', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TERMS OF SERVICE PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "terms":
    if st.button("← " + translate_text("Back to Home", lc), key="back_home_terms", use_container_width=True):
        st.session_state.page = "home"; st.rerun()
    
    st.markdown(f'''<div class="glass-card" style="padding:40px; margin-top:20px; max-width:900px; margin:0 auto;">
<h1 style="color:#00E0FF; margin-bottom:30px; font-size:32px;">{translate_text("Terms of Service", lc)}</h1>
<div class="legal-point">
<h3 style="color:#fff; margin-top:0; font-size:18px;">1. {translate_text("Acceptance of Terms", lc)}</h3>
<p style="color:rgba(217,226,236,0.7); line-height:1.7; margin-bottom:0;">{translate_text("By accessing and using this platform, you agree to comply with these Terms & Conditions.", lc)}</p>
</div>
<div class="legal-point">
<h3 style="color:#fff; margin-top:0; font-size:18px;">2. {translate_text("Educational Purpose Only", lc)}</h3>
<p style="color:rgba(217,226,236,0.7); line-height:1.7; margin-bottom:0;">{translate_text("All stock predictions, sentiment scores, and analytics provided on this platform are for educational and informational purposes only.", lc)}</p>
</div>
<div class="legal-point">
<h3 style="color:#fff; margin-top:0; font-size:18px;">3. {translate_text("No Financial Advice", lc)}</h3>
<p style="color:rgba(217,226,236,0.7); line-height:1.7; margin-bottom:0;">{translate_text("We do not provide financial, investment, or trading advice. Users are responsible for their own investment decisions.", lc)}</p>
</div>
<div class="legal-point">
<h3 style="color:#fff; margin-top:0; font-size:18px;">4. {translate_text("User Responsibilities", lc)}</h3>
<p style="color:rgba(217,226,236,0.7); line-height:1.7; margin-bottom:0;">{translate_text("Users must provide accurate information during registration and maintain the confidentiality of their account credentials.", lc)}</p>
</div>
<div class="legal-point">
<h3 style="color:#fff; margin-top:0; font-size:18px;">5. {translate_text("Limitation of Liability", lc)}</h3>
<p style="color:rgba(217,226,236,0.7); line-height:1.7; margin-bottom:0;">{translate_text("We are not liable for any financial losses or damages resulting from the use of this platform.", lc)}</p>
</div>
<div class="legal-point">
<h3 style="color:#fff; margin-top:0; font-size:18px;">6. {translate_text("Modifications", lc)}</h3>
<p style="color:rgba(217,226,236,0.7); line-height:1.7; margin-bottom:0;">{translate_text("We reserve the right to modify or discontinue any part of the platform at any time without notice.", lc)}</p>
</div>
<div class="legal-point">
<h3 style="color:#fff; margin-top:0; font-size:18px;">7. {translate_text("Governing Law", lc)}</h3>
<p style="color:rgba(217,226,236,0.7); line-height:1.7; margin-bottom:0;">{translate_text("These terms shall be governed by applicable laws of the jurisdiction in which the platform operates.", lc)}</p>
</div>
</div>''', unsafe_allow_html=True)

elif st.session_state.page == "history":
    if not st.session_state.signed_in:
        st.session_state.auth_redirect = "history"
        st.session_state.page = "auth"
        st.rerun()
        st.stop()
    user = st.session_state.username if st.session_state.signed_in else "guest"
    
    with st.container():
        st.markdown('<span id="history-marker"></span>', unsafe_allow_html=True)
        st.markdown(f'<h1 style="color:#00E0FF; margin-bottom:30px; font-family:\'Space Grotesk\', sans-serif;">{translate_text("Search History", lc)}</h1>', unsafe_allow_html=True)
        
        history = get_user_history(user)
        if not history:
            st.info(translate_text("No history found. Start searching stocks to see them here!", lc))
        else:
            # Header Row
            st.markdown(f'''
            <div class="grid-history-head">
                <div class="list-header">{translate_text("Ticker", lc)}</div>
                <div class="list-header">{translate_text("Last Seen Price", lc)}</div>
                <div class="list-header" style="text-align:right;">{translate_text("Date & Time", lc)}</div>
            </div>
            <div style="border-bottom:1px solid rgba(255,255,255,0.1); margin:10px 0;"></div>
            ''', unsafe_allow_html=True)
            
            for ticker, price, dt in history:
                clean_dt = dt.split(".")[0]
                st.markdown(f'''
                <div class="grid-history-row">
                    <div class="list-ticker">{ticker}</div>
                    <div class="list-price">₹{price:.2f}</div>
                    <div class="list-date" style="text-align:right;">{clean_dt}</div>
                </div>
                <div style="border-bottom:1px solid rgba(255,255,255,0.05);"></div>
                ''', unsafe_allow_html=True)

elif st.session_state.page == "watchlist":
    if not st.session_state.signed_in:
        st.session_state.auth_redirect = "watchlist"
        st.session_state.page = "auth"
        st.rerun()
        st.stop()
    user = st.session_state.username if st.session_state.signed_in else "guest"
    
    with st.container():
        st.markdown('<span id="watchlist-marker"></span>', unsafe_allow_html=True)
        st.markdown(f'<h1 style="color:#00E0FF; margin-bottom:30px; font-family:\'Space Grotesk\', sans-serif;">{translate_text("Your Watchlist", lc)}</h1>', unsafe_allow_html=True)
        
        watchlist = get_watchlist(user)
        if not watchlist:
            st.info(translate_text("Your watchlist is empty. Add stocks from the analysis page!", lc))
        else:
            # Batch fetch current prices with caching for speed
            tickers = [tk for tk, _ in watchlist]
            prices = get_batch_prices(tickers)

            # Header Row
            st.markdown(f'''
            <div class="grid-watchlist-head">
                <div class="list-header">{translate_text("Ticker", lc)}</div>
                <div class="list-header">{translate_text("Current Price", lc)}</div>
                <div class="list-header" style="text-align:right;">{translate_text("Action", lc)}</div>
            </div>
            <div style="border-bottom:1px solid rgba(255,255,255,0.1); margin:10px 0;"></div>
            ''', unsafe_allow_html=True)

            for ticker, dt in watchlist:
                price_val = prices.get(ticker)
                price_display = f"₹{price_val:.2f}" if price_val else "—"
                price_html = f'<div class="list-price">{price_display}</div>'
                
                st.markdown(f'''
                <div class="grid-watchlist-row">
                    <div class="list-ticker">{ticker}</div>
                    {price_html}
                    <div style="text-align:right;">
                        <a href="?page=watchlist&action=remove&ticker={ticker}" target="_self" class="remove-link">{translate_text("Remove", lc)}</a>
                    </div>
                </div>
                <div style="border-bottom:1px solid rgba(255,255,255,0.05);"></div>
                ''', unsafe_allow_html=True)

elif st.session_state.page == "alerts":
    if not st.session_state.signed_in:
        st.session_state.auth_redirect = "alerts"
        st.session_state.page = "auth"
        st.rerun()
        st.stop()
    user = st.session_state.username if st.session_state.signed_in else "guest"
    
    with st.container():
        st.markdown('<span id="alerts-marker"></span>', unsafe_allow_html=True)
        
        # Header with Refresh Button
        head_col, btn_col = st.columns([3, 1])
        with head_col:
            st.markdown(f'<h1 style="color:#00E0FF; margin-bottom:10px; font-family:\'Space Grotesk\', sans-serif;">🔔 {translate_text("Price Alerts", lc)}</h1>', unsafe_allow_html=True)
            st.markdown(f'<p style="color:rgba(255,255,255,0.5); margin-bottom:30px;">{translate_text("Tracking price changes for your recently viewed and watchlist stocks.", lc)}</p>', unsafe_allow_html=True)
        
        with btn_col:
            st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)
            if st.button("🔄 " + translate_text("Refresh Watchlist", lc), key="refresh_alerts_btn", use_container_width=True):
                with st.spinner(translate_text("Scanning market...", lc)):
                    # Logic to scan watchlist for changes
                    conn = sqlite3.connect("users.db")
                    c = conn.cursor()
                    c.execute("SELECT ticker FROM watchlist WHERE username=?", (user,))
                    tickers = [r[0] for r in c.fetchall()]
                    conn.close()
                    
                    if tickers:
                        # Batch fetch
                        prices = get_batch_prices(tickers)
                        alerts_found = 0
                        for tk, p in prices.items():
                            if p is not None:
                                alert = add_to_history(user, tk, p)
                                if alert: alerts_found += 1
                        
                        if alerts_found > 0:
                            st.success(f"✅ {alerts_found} {translate_text('new price movements detected!', lc)}")
                        else:
                            st.info(translate_text("No significant price movements since last check.", lc))
                    else:
                        st.warning(translate_text("Add stocks to your Watchlist first to use this feature!", lc))
                st.rerun()

        alerts = get_user_alerts(user)
        if not alerts:
            st.info(translate_text("No alerts detected yet. View stocks periodically to see price changes here!", lc))
        else:
            # Header Row
            st.markdown(f'''
            <div class="grid-alerts-head">
                <div class="list-header">{translate_text("Ticker", lc)}</div>
                <div class="list-header">{translate_text("Old Price", lc)}</div>
                <div class="list-header">{translate_text("New Price", lc)}</div>
                <div class="list-header" style="text-align:right;">{translate_text("Alert", lc)}</div>
            </div>
            <div style="border-bottom:1px solid rgba(255,255,255,0.1); margin:10px 0;"></div>
            ''', unsafe_allow_html=True)

            for ticker, old_p, new_p, direction, dt in alerts:
                icon = "📈" if direction == "increased" else "📉"
                color = "#14FFEC" if direction == "increased" else "#FF6B6B"
                st.markdown(f'''
                <div class="grid-alerts-row">
                    <div class="list-ticker">{ticker}</div>
                    <div class="list-date">₹{old_p:.2f}</div>
                    <div class="list-price">₹{new_p:.2f}</div>
                    <div style="text-align:right; font-weight:800; color:{color};">
                        {icon} {translate_text(direction, lc)}
                    </div>
                </div>
                <div style="border-bottom:1px solid rgba(255,255,255,0.05);"></div>
                ''', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div style='margin-top:80px;'></div>", unsafe_allow_html=True)
st.markdown(f'<div style="text-align:center; padding:20px; border-top:1px solid rgba(255,255,255,0.05); margin-bottom:40px;"><span style="font-size:12px; color:rgba(217,226,236,0.4);">© 2024 From Tweets to Trades. {translate_text("All rights reserved.", lc)}</span></div>', unsafe_allow_html=True)