# ----------------------------------------------------
# This is train_model.py (UPDATED)
# ----------------------------------------------------

import yfinance as yf
import requests
import numpy as np
import pandas as pd
from textblob import TextBlob
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import os
import sys

# --- CONFIG ---
import sys

TICKERS = [
    "ADANIENT.NS","ADANIPORTS.NS","APOLLOHOSP.NS","ASIANPAINT.NS","AXISBANK.NS","BAJAJ-AUTO.NS","BAJAJFINSV.NS","BAJFINANCE.NS","BEL.NS","BHARTIARTL.NS",
    "CIPLA.NS","COALINDIA.NS","DRREDDY.NS","EICHERMOT.NS","ETERNAL.NS","GRASIM.NS","HCLTECH.NS","HDFCBANK.NS","HDFCLIFE.NS","HINDALCO.NS",
    "HINDUNILVR.NS","ICICIBANK.NS","INDIGO.NS","INFY.NS","ITC.NS","JIOFIN.NS","JSWSTEEL.NS","KOTAKBANK.NS","LT.NS","M&M.NS","MARUTI.NS","MAXHEALTH.NS",
    "NESTLEIND.NS","NTPC.NS","ONGC.NS","POWERGRID.NS","RELIANCE.NS","SBILIFE.NS","SBIN.NS","SHRIRAMFIN.NS","SUNPHARMA.NS","TATACONSUM.NS",
    "TATASTEEL.NS","TCS.NS","TECHM.NS","TITAN.NS","TMPV.NS","TRENT.NS","ULTRACEMCO.NS","WIPRO.NS"
]

NEWS_API_KEY = "84fc43ca1f7045ad88e16857dbb0f901"
TRAIN_START_DATE = "2020-01-01"
TRAIN_END_DATE = datetime.now().strftime('%Y-%m-%d')
LOOKBACK = 60
EPOCHS = 60

# Create stock ID mapping
stock_to_id = {ticker: idx for idx, ticker in enumerate(TICKERS)}
print("Stock ID Mapping:", stock_to_id)

# --- FUNCTIONS ---
def fetch_stock_data(ticker, start, end):
    print(f"Fetching stock data for {ticker}...")
    data = yf.download(
        tickers=ticker,
        start=start,
        end=end,
        interval="1d",
        group_by="column",
        auto_adjust=False
    )

    if data is None or data.empty:
        print("❌ No stock data received!")
        return pd.DataFrame()

    data.reset_index(inplace=True)
    # 🔒 FORCE single Close column (handles MultiIndex safely)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    if "Adj Close" in data.columns:
        data["Close"] = data["Adj Close"]

    # Use Adj Close for better stability (splits/dividends)
    if "Adj Close" in data.columns:
        data["Close"] = data["Adj Close"]

    return data


def fetch_news_sentiment_daily(query, start, end):
    """
    Returns a dataframe with columns:
    date, sentiment
    """
    print("Fetching daily news sentiment...")

    if not NEWS_API_KEY:
        print("NewsAPI key not set. Using neutral sentiment.")
        return pd.DataFrame(columns=["date", "sentiment"])

    url = "https://newsapi.org/v2/everything"

    params = {
        "q": query,
        "from": start,
        "to": end,
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": 100,   # max per request
        "apiKey": NEWS_API_KEY
    }

    try:
        res = requests.get(url, params=params, timeout=15)
        data = res.json()

        if data.get("status") != "ok":
            print("⚠️ NewsAPI error:", data)
            return pd.DataFrame(columns=["date", "sentiment"])

        articles = data.get("articles", [])
        if not articles:
            print("⚠️ No news articles found.")
            return pd.DataFrame(columns=["date", "sentiment"])

        rows = []
        for a in articles:
            title = a.get("title")
            published = a.get("publishedAt")  # ISO string

            if not title or not published:
                continue

            # Convert to date only (YYYY-MM-DD)
            dt = pd.to_datetime(published).date()

            polarity = TextBlob(title).sentiment.polarity
            rows.append((dt, polarity))

        if not rows:
            return pd.DataFrame(columns=["date", "sentiment"])

        df = pd.DataFrame(rows, columns=["date", "sentiment"])

        # Daily mean sentiment
        df = df.groupby("date", as_index=False)["sentiment"].mean()

        return df

    except Exception as e:
        print("⚠️ Sentiment fetch failed:", e)
        return pd.DataFrame(columns=["date", "sentiment"])

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

def prepare_features(stock_data, sentiment_df):
    print("Preparing features (MAs + daily sentiment)...")

    # Stock date column -> date only
    stock_data["date"] = pd.to_datetime(stock_data["Date"]).dt.date

    # Merge daily sentiment with stock dates
    if sentiment_df is None or sentiment_df.empty:
        stock_data["sentiment"] = 0.0
    else:
        stock_data = stock_data.merge(sentiment_df, on="date", how="left")
        stock_data["sentiment"] = stock_data["sentiment"].fillna(0.0)

    # Moving averages
    stock_data["MA5"] = stock_data["Close"].rolling(5).mean()
    stock_data["MA10"] = stock_data["Close"].rolling(10).mean()
    stock_data["RSI"] = compute_rsi(stock_data["Close"])

    stock_data.dropna(inplace=True)
    return stock_data


def create_sequences(data, lookback=60, horizon=14):
    X, y = [], []
    for i in range(lookback, len(data) - horizon):
        X.append(data[i - lookback:i, :])
        y.append(data[i:i + horizon, 0])
    return np.array(X), np.array(y)


def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dense(14)   # 14-day forecast
    ])

    model.compile(optimizer="adam", loss="mse")
    return model

# --- MAIN SCRIPT TO RUN ---
if __name__ == "__main__":
    print(f"\n=== Starting AI Model Training ===")

    os.makedirs("ml_models", exist_ok=True)

    # 1. Fetch & combine data from all companies
    all_data = []

    for ticker in TICKERS:
        print(f"\n--- Processing {ticker} ---")

        stock_data = fetch_stock_data(ticker, TRAIN_START_DATE, TRAIN_END_DATE)
        if stock_data.empty:
            print(f"Skipping {ticker}, no data.")
            continue

        sentiment_df = pd.DataFrame(columns=["date", "sentiment"])
        stock_data = prepare_features(stock_data, sentiment_df)

        stock_id = stock_to_id[ticker]
        features_df = stock_data.loc[:, ["Close", "sentiment", "MA5", "MA10", "RSI"]].copy()
        features_df["stock_id"] = stock_id
        all_data.append(features_df)

    # 🔒 SAFETY CHECK
    if not all_data:
        raise Exception("No data collected from any ticker!")

    # 🔒 CONCAT ONCE
    combined_data = pd.concat(all_data, axis=0, ignore_index=True)

    # 🔒 FORCE FINAL FEATURE SET
    combined_data = combined_data[["Close", "sentiment", "MA5", "MA10", "RSI", "stock_id"]]

    print("FINAL combined_data shape:", combined_data.shape)
    print("Columns:", combined_data.columns.tolist())

    # 2. Normalize
    print("\nCreating sequences stock-wise...")
    X_all, y_all = [], []
    scalers = {}
    for ticker in TICKERS:
        stock_id = stock_to_id[ticker]
        stock_df = combined_data[combined_data["stock_id"] == stock_id]

        if len(stock_df) < LOOKBACK + 14:
            print(f"Skipping {ticker}, not enough data.")
            continue

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_stock = scaler.fit_transform(stock_df)
        scalers[ticker] = scaler

        X_stock, y_stock = create_sequences(scaled_stock, LOOKBACK, 14)

        X_all.append(X_stock)
        y_all.append(y_stock)

    X = np.concatenate(X_all)
    y = np.concatenate(y_all)

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # 4. Build & train model
    model = build_lstm_model(input_shape=(LOOKBACK, 6))

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
        verbose=1
    )

    print("\nTraining model... This may take a few minutes.")
    split_index = int(len(X) * 0.8)
    X_train, X_val = X[:split_index], X[split_index:]
    y_train, y_val = y[:split_index], y[split_index:]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    # 5. Save model and scaler
    model.save("ml_models/stock_model.keras")
    joblib.dump(scalers, "ml_models/stock_scaler.joblib")
    joblib.dump(stock_to_id, "ml_models/stock_id_mapping.joblib")

    print("✅ Model saved.")
    print("✅ Model saved to ml_models/stock_model.keras")
    print("✅ Scaler saved to ml_models/stock_scaler.joblib")