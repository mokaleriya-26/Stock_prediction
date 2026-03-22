# In predictor/views.py

from django.http import JsonResponse
from django.conf import settings # To find your model files
from tensorflow.keras.models import load_model
import joblib
import yfinance as yf
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from deep_translator import GoogleTranslator
from django.utils import translation
import re
from .twitter_utils import get_twitter_sentiment_for_ticker

COMPANY_NAMES = {
    "ADANIENT": "Adani Enterprises Limited",
    "ADANIPORTS": "Adani Ports & SEZ Limited",
    "APOLLOHOSP": "Apollo Hospitals Enterprises Limited",
    "ASIANPAINT": "Asian Paints Limited",
    "AXISBANK": "Axis Bank Limited",
    "BAJAJ-AUTO": "Bajaj Auto Limited",
    "BAJAJFINSV": "Bajaj Finserv Limited",
    "BAJFINANCE": "Bajaj Finance Limited",
    "BEL": "Bharat Electronics Limited",
    "BHARTIARTL": "Bharti Airtel Limited",
    "CIPLA": "Cipla Limited",
    "COALINDIA": "Coal India Limited",
    "DRREDDY": "Dr. Reddy's Laboratories Limited",
    "EICHERMOT": "Eicher Motors Limited",
    "ETERNAL": "Eternal Limited",
    "GRASIM": "Grasim Industries Limited",
    "HCLTECH": "HCL Technologies Limited",
    "HDFCBANK": "HDFC Bank Limited",
    "HDFCLIFE": "HDFC Life Insurance Company Limited",
    "HINDALCO": "Hindalco Industries Limited",
    "HINDUNILVR": "Hindustan Unilever Limited",
    "ICICIBANK": "ICICI Bank Limited",
    "INDIGO": "InterGlobe Aviation Limited",
    "INFY": "Infosys Limited",
    "ITC": "ITC Limited",
    "JIOFIN": "Jio Financial Services Limited",
    "JSWSTEEL": "JSW Steel Limited",
    "KOTAKBANK": "Kotak Mahindra Bank Limited",
    "LT": "Larsen & Toubro Limited",
    "M&M": "Mahindra & Mahindra Limited",
    "MARUTI": "Maruti Suzuki India Limited",
    "MAXHEALTH": "Max Healthcare Institute Limited",
    "NESTLEIND": "Nestle India Limited",
    "NTPC": "NTPC Limited",
    "ONGC": "Oil & Natural Gas Corporation Limited",
    "POWERGRID": "Power Grid Corporation of India Limited",
    "RELIANCE": "Reliance Industries Limited",
    "SBILIFE": "SBI Life Insurance Company Limited",
    "SBIN": "State Bank of India",
    "SHRIRAMFIN": "Shriram Finance Limited",
    "SUNPHARMA": "Sun Pharmaceuticals Industries Limited",
    "TATACONSUM": "Tata Consumer Products Limited",
    "TATASTEEL": "Tata Steel Limited",
    "TCS": "Tata Consultancy Services Limited",
    "TECHM": "Tech Mahindra Limited",
    "TITAN": "Titan Company Limited",
    "TMPV": "Tata Motors Passenger Vehicles Limited",
    "TRENT": "TRENT Limited",
    "ULTRACEMCO": "UltraTech Cement Limited",
    "WIPRO": "Wipro Limited",
}

COMPANY_ALIASES = {
    "ADANIENT": ["adani enterprises"],
    "ADANIPORTS": ["adani ports", "adani ports & sez"],
    "APOLLOHOSP": ["apollo hospitals", "apollo hospital", "apollo"],
    "ASIANPAINT": ["asian paints"],
    "AXISBANK": ["axis bank", "axis"],
    "BAJAJ-AUTO": ["bajaj auto", "bajaj"],
    "BAJAJFINSV": ["bajaj finserv", "bajaj"],
    "BAJFINANCE": ["bajaj finance","bajaj fin"],
    "BEL": ["bharat electronics", "bel"],
    "BHARTIARTL": ["bharti airtel", "airtel"],
    "CIPLA": ["cipla"],
    "COALINDIA": ["coal india"],
    "DRREDDY": ["dr reddy", "dr. reddy", "reddy laboratories"],
    "EICHERMOT": ["eicher motors", "royal enfield", "eicher"],
    "ETERNAL": ["eternal"],
    "GRASIM": ["grasim"],
    "HCLTECH": ["hcl tech", "hcl technologies", "hcl"],
    "HDFCBANK": ["hdfc bank"],
    "HDFCLIFE": ["hdfc life","hdfc"],
    "HINDALCO": ["hindalco"],
    "HINDUNILVR": ["hindustan unilever", "hul", "unilever"],
    "ICICIBANK": ["icici bank", "icici"],
    "INDIGO": ["interglobe aviation","indigo airlines","indigo shares","indigo"],
    "INFY": ["infosys"],
    "ITC": ["itc"],
    "JIOFIN": ["jio financial", "jio finance", "jio"],
    "JSWSTEEL": ["jsw steel", "jsw"],
    "KOTAKBANK": ["kotak bank", "kotak mahindra","kotak"],
    "LT": ["larsen & toubro", "l&t", "lt"],
    "M&M": ["mahindra", "mahindra & mahindra", "m&m"],
    "MARUTI": ["maruti", "maruti suzuki"],
    "MAXHEALTH": ["max healthcare", "max health"],
    "NESTLEIND": ["nestle india", "nestle"],
    "NTPC": ["ntpc"],
    "ONGC": ["ongc", "oil and natural gas"],
    "POWERGRID": ["power grid", "powergrid"],
    "RELIANCE": ["reliance", "reliance industries", "ril"],
    "SBILIFE": ["sbi life"],
    "SBIN": ["state bank of india", "sbi"],
    "SHRIRAMFIN": ["shriram finance", "shriram"],
    "SUNPHARMA": ["sun pharma", "sun pharmaceutical"],
    "TATACONSUM": ["tata consumer", "tata consumer products"],
    "TATASTEEL": ["tata steel"],
    "TCS": ["tcs", "tata consultancy services"],
    "TECHM": ["tech mahindra"],
    "TITAN": ["titan"],
    "TMPV": ["tata motors", "tata motors passenger"],
    "TRENT": ["trent"],
    "ULTRACEMCO": ["ultratech cement",],
    "WIPRO": ["wipro"],
}

# ===================== CONFIG =====================
NEWS_API_KEY = "84fc43ca1f7045ad88e16857dbb0f901" # <-- PUT YOUR KEY HERE
LOOKBACK = 60 # Same lookback as training

# ===================== LOAD THE "BRAIN" (ONCE, WHEN SERVER STARTS) =====================
print("Loading AI model and scaler...")
try:
    MODEL_PATH = os.path.join(settings.ML_MODELS_DIR, 'stock_model.keras')
    SCALER_PATH = os.path.join(settings.ML_MODELS_DIR, 'stock_scaler.joblib')

    model = load_model(MODEL_PATH)
    scalers = joblib.load(SCALER_PATH)

    print("✅ Model and scaler loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model/scaler: {e}")
    model = None
    scalers = None

print("Loading FinBERT model...")
try:
    finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    print("✅ FinBERT loaded successfully.")
except Exception as e:
    print("❌ FinBERT loading error:", e)
    finbert_model = None

STOCK_ID_PATH = os.path.join(settings.ML_MODELS_DIR, 'stock_id_mapping.joblib')
stock_to_id = joblib.load(STOCK_ID_PATH)
# ===================== HELPER FUNCTIONS =====================

def compare_stocks(request):
    tickers_param = request.GET.get("tickers")

    if not tickers_param:
        return JsonResponse({"error": "No tickers provided"}, status=400)

    tickers = tickers_param.split(",")
    comparison_data = []

    for ticker in tickers:
        try:
            ticker_obj = yf.Ticker(ticker)
            fast_info = getattr(ticker_obj, "fast_info", {}) or {}

            stock_data = yf.download(ticker, period="3mo", interval="1d")

            if stock_data.empty:
                continue

            stock_data.reset_index(inplace=True)

            if isinstance(stock_data.columns, pd.MultiIndex):
                stock_data.columns = stock_data.columns.get_level_values(0)

            if "Adj Close" in stock_data.columns:
                stock_data["price"] = stock_data["Adj Close"]
            elif "Close" in stock_data.columns:
                stock_data["price"] = stock_data["Close"]
            else:
                continue

            start_price = float(stock_data["price"].iloc[0])

            current_price = (
                safe_float(fast_info.get("lastPrice")) or
                safe_float(fast_info.get("regularMarketPrice")) or
                safe_float(stock_data["price"].iloc[-1])
            )

            percent_change = ((current_price - start_price) / start_price) * 100
            print("COMPARE DEBUG:", ticker, current_price, percent_change)
            comparison_data.append({
                "ticker": ticker,
                "current_price": round(current_price, 2),
                "percent_change_3m": round(percent_change, 2)
            })

        except Exception as e:
            print("Comparison error:", ticker, e)

    print("FINAL comparison_data:", comparison_data)
    return JsonResponse({"comparison": comparison_data})

def get_finbert_sentiment(text):
    if finbert_model is None:
        return 0.0, "Neutral", 0.0

    inputs = finbert_tokenizer(text, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = finbert_model(**inputs)

    probs = F.softmax(outputs.logits, dim=1)
    confidence, predicted_class = torch.max(probs, dim=1)

    label_map = {
        0: "Negative",
        1: "Neutral",
        2: "Positive"
    }

    label = label_map[predicted_class.item()]
    confidence_score = confidence.item()

    # Convert to numeric score
    if label == "Positive":
        score = confidence_score
    elif label == "Negative":
        score = -confidence_score
    else:
        score = 0.0

    return score, label, confidence_score

def alias_match(title, aliases):
    title = title.lower().replace("-", " ")
    for alias in aliases:
        alias_clean = alias.lower().replace("-", " ")
        pattern = r"\b" + re.escape(alias_clean) + r"\b"
        if re.search(pattern, title):
            return True
    return False

def fetch_company_news(ticker, from_date, to_date, lang="en"):
    headlines = []
    sentiments = []
    seen_titles = set()
    symbol = ticker.split(".")[0]
    aliases = COMPANY_ALIASES.get(symbol, [])

    try:
        # =============================
        # 1️⃣ YAHOO FINANCE NEWS
        # =============================
        ticker_obj = yf.Ticker(ticker)
        yahoo_news = ticker_obj.news or []

        for item in yahoo_news[:15]:
            title = item.get("title")
            if not title or title in seen_titles:
                continue
            link = item.get("link")
            publisher = item.get("publisher")
            provider_time = item.get("providerPublishTime")

            alias_hit = alias_match(title, aliases)
            if not alias_hit:
                continue

            # Safe datetime handling
            if provider_time:
                dt = datetime.fromtimestamp(provider_time)
                formatted_time = dt.strftime("%d %b %Y • %I:%M %p")
            else:
                dt = datetime.now()
                formatted_time = None

            score, label, confidence = get_finbert_sentiment(title)
           
            seen_titles.add(title)
            sentiments.append(score)

            translated_title = title
            if lang != "en":
                try:
                    translated_title = GoogleTranslator(source="auto", target=lang).translate(title)
                except Exception as e:
                    print("Yahoo title translation error:", e)
                    translated_title = title

            headlines.append({
                "source": publisher or "Yahoo Finance",
                "title": translated_title,
                "original_title": title,
                "url": link,
                "published_at": formatted_time,
                "raw_time": dt,
                "sentiment_score": round(score, 3),
                "sentiment_label": label,
                "confidence": round(confidence * 100, 2)
            })

    except Exception as e:
        print("Yahoo news error:", e)

    try:
        # =============================
        # 2️⃣ NEWS API (FALLBACK ADDITION)
        # =============================
        url = "https://newsapi.org/v2/everything"
        query_string = " OR ".join(f'"{a}"' for a in aliases) if aliases else symbol

        params = {
            "q": query_string,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 15,
            "apiKey": NEWS_API_KEY
        }

        response = requests.get(url, params=params, timeout=15)
        data = response.json()
        if data.get("status") == "ok":
            articles = data.get("articles", [])

            for a in articles:
                title = a.get("title")
                if not title or title in seen_titles:
                    continue
                alias_hit = alias_match(title, aliases)
                if not alias_hit:
                    continue
                seen_titles.add(title)
                score, label, confidence = get_finbert_sentiment(title)
                
                sentiments.append(score)
                translated_title = title
                if lang != "en":
                    try:
                        translated_title = GoogleTranslator(
                            source="auto",
                            target=lang
                        ).translate(title)
                    except:
                        translated_title = title
                published_at = a.get("publishedAt")
                formatted_time = None

                raw_dt = None
                if published_at:
                    try:
                        raw_dt = datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%SZ")
                        formatted_time = raw_dt.strftime("%d %b %Y • %I:%M %p")
                    except:
                        formatted_time = published_at
                else:
                    raw_dt = datetime.now()

                headlines.append({
                    "source": a.get("source", {}).get("name", "NewsAPI"),
                    "title": translated_title,
                    "original_title": title,
                    "url": a.get("url", "#"),
                    "published_at": formatted_time,
                    "raw_time": raw_dt,
                    "sentiment_score": round(score, 3),
                    "sentiment_label": label,
                    "confidence": round(confidence * 100, 2)
                })

    except Exception as e:
        print("NewsAPI error:", e)

    # =============================
    # 3️⃣ FINAL PROCESSING
    # =============================
    sentiment_score = float(np.mean(sentiments)) if sentiments else 0.0

    # Sort by published time (latest first)
    headlines = sorted(
        headlines,
        key=lambda x: x.get("raw_time", datetime.min),
        reverse=True
    )
    for h in headlines:
        h.pop("raw_time", None)
    return sentiment_score, headlines[:5]

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def prepare_features(stock_data, sentiment_score):
    """Prepares data for prediction."""
    stock_data["sentiment"] = sentiment_score
    stock_data["MA5"] = stock_data["price"].rolling(5).mean()
    stock_data["MA10"] = stock_data["price"].rolling(10).mean()
    return stock_data

# ===================== THE API VIEW =====================

def safe_float(x):
    try:
        if hasattr(x, "values"):
            return float(x.values[0])
        return float(x)
    except:
        return float(0.0)

def safe_int(x):
    try:
        if hasattr(x, "values"):
            return int(x.values[0])
        return int(x)
    except:
        return int(0)

def generate_ai_summary(ticker, news_headlines, lang="en"):
    symbol = ticker.split(".")[0]
    company_name = COMPANY_NAMES.get(symbol, symbol)

    if not news_headlines:
        summary_en = (
            f"No major recent news was found for {company_name}. "
            f"The prediction is mainly based on historical price trends and technical indicators."
        )
    else:
        positive = sum(1 for item in news_headlines if item.get("sentiment_label") == "Positive")
        negative = sum(1 for item in news_headlines if item.get("sentiment_label") == "Negative")
        neutral = sum(1 for item in news_headlines if item.get("sentiment_label") == "Neutral")

        top_headlines = []
        for item in news_headlines[:2]:
            title = item.get("original_title") or item.get("title")
            if title:
                top_headlines.append(title)

        if positive > negative:
            tone = "mostly positive"
        elif negative > positive:
            tone = "mostly negative"
        else:
            tone = "mixed"

        summary_en = (
            f"Recent news sentiment for {company_name} is {tone}. "
            f"Out of the latest {len(news_headlines)} news items, "
            f"{positive} were positive, {negative} were negative, and {neutral} were neutral. "
        )

        if top_headlines:
            summary_en += "Key topics include: " + "; ".join(top_headlines[:2]) + ". "

        summary_en += (
            "This summary is generated from recent headlines and helps explain the market mood around the stock."
        )

    if lang != "en":
        try:
            return GoogleTranslator(source="en", target=lang).translate(summary_en)
        except Exception as e:
            print("AI summary translation error:", e)

    return summary_en

def get_stock_prediction(request, ticker):
    """
    This is the function that runs when a user visits your API URL.
    """
    try:
        if model is None or scalers is None:
            return JsonResponse({"error": "Model not loaded"}, status=500)
        # ----- 1. GET ALL DATA -----
        lang = translation.get_language() or "en"
        print(f"\n--- Request received for {ticker} ---")
        
        stock_data = yf.download(tickers=ticker, period="6mo", interval="1d")
        stock_data.reset_index(inplace=True)
        ticker_obj = yf.Ticker(ticker)   # ⭐ NEW
        info = ticker_obj.info           # ⭐ NEW

        # 🔒 FIX: flatten MultiIndex columns (CRITICAL)
        if isinstance(stock_data.columns, pd.MultiIndex):
            stock_data.columns = stock_data.columns.get_level_values(0)

        # 🔒 CREATE A SINGLE SOURCE OF TRUTH FOR PRICE
        if "Adj Close" in stock_data.columns:
            stock_data["price"] = stock_data["Adj Close"]
        elif "Close" in stock_data.columns:
            stock_data["price"] = stock_data["Close"]
        else:
            raise Exception(f"No price column found: {stock_data.columns.tolist()}")

        # 🔒 DROP ambiguity
        stock_data = stock_data.drop(columns=[c for c in ["Close", "Adj Close"] if c in stock_data.columns])

        print("DEBUG — last 5 prices:")
        print(stock_data[["Date", "price"]].tail(5))

        if stock_data.empty and ticker.endswith(".NS"):
            alt_ticker = ticker.replace(".NS", ".BO")
            stock_data = yf.download(tickers=alt_ticker, period="6mo", interval="1d")
            stock_data.reset_index(inplace=True)

            if isinstance(stock_data.columns, pd.MultiIndex):
                stock_data.columns = stock_data.columns.get_level_values(0)

            if "Adj Close" in stock_data.columns:
                stock_data["price"] = stock_data["Adj Close"]
            elif "Close" in stock_data.columns:
                stock_data["price"] = stock_data["Close"]

        # 🔒 Always keep correct order
        stock_data = stock_data.sort_values("Date")

        if stock_data.empty:
            return JsonResponse({"error": f"No stock data found for {ticker}"}, status=404)

        news_end = datetime.now()
        news_start = news_end - timedelta(days=30) 
        sentiment, news_headlines = fetch_company_news(
            ticker, 
            news_start.strftime('%Y-%m-%d'), 
            news_end.strftime('%Y-%m-%d'),
            lang
        )
        ai_summary = generate_ai_summary(ticker, news_headlines, lang)

        # ================== SENTIMENT + SIGNAL LOGIC ==================
        twitter_sentiment = get_twitter_sentiment_for_ticker(ticker)
        print(f"Current sentiment for {ticker}: {sentiment:.4f}")

        # Normalize twitter sentiment
        if isinstance(twitter_sentiment, dict):
            tw_sent = twitter_sentiment.get("score", 0.0)
        else:
            tw_sent = twitter_sentiment if twitter_sentiment is not None else 0.0

        # Final combined sentiment
        sentiment_score = (0.6 * sentiment) + (0.4 * tw_sent)
        if request.method != "GET":
            return JsonResponse({"error": "Invalid request method"}, status=405)
        
        # ----- 2. PREPARE DATA FOR PREDICTION -----
        stock_id = stock_to_id.get(ticker, 0)
        stock_data_features = prepare_features(stock_data.copy(), sentiment)
        stock_data_features["RSI"] = compute_rsi(stock_data_features["price"])
        stock_data_features["stock_id"] = stock_id
        stock_data_features = stock_data_features.bfill().ffill()

        features_to_predict = stock_data_features[["price", "sentiment", "MA5", "MA10", "RSI", "stock_id"]].values
        last_sequence_unscaled = features_to_predict[-LOOKBACK:]
        
        stock_scaler = scalers.get(ticker)

        if stock_scaler is None:
            raise Exception(f"No scaler found for {ticker}")

        current_seq_scaled = stock_scaler.transform(last_sequence_unscaled)
        current_seq_scaled = current_seq_scaled.reshape((1, LOOKBACK, 6))

        # ----- 3. RUN 14-DAY PREDICTION (CORRECT WAY) -----
        print("Running 14-day prediction (multi-output model)...")

        # Predict ONCE
        pred_scaled = model.predict(current_seq_scaled, verbose=0)[0]  # shape (14,)

        # Inverse scale correctly
        dummy = np.zeros((14, 6))
        dummy[:, 0] = pred_scaled  # Close column only

        predicted_prices = stock_scaler.inverse_transform(dummy)[:, 0]
        predicted_prices = [float(p) for p in predicted_prices]
        # ===== PRICE TREND =====
        last_close_series = stock_data["price"].iloc[-1]

        if hasattr(last_close_series, "values"):
            last_close = float(last_close_series.values[0])
        else:
            last_close = float(last_close_series)

        price_trend = (predicted_prices[-1] - last_close) / last_close

        final_score = (0.6 * sentiment_score) + (0.4 * price_trend)

       # ================== PURE NEWS-BASED SIGNAL ==================

        positive_count = sum(1 for n in news_headlines if n["sentiment_label"] == "Positive")
        negative_count = sum(1 for n in news_headlines if n["sentiment_label"] == "Negative")
        neutral_count = sum(1 for n in news_headlines if n["sentiment_label"] == "Neutral")

        total_news = len(news_headlines) if news_headlines else 1

        # Majority logic
        if positive_count / total_news > 0.6:
            signal = "BUY"

        elif negative_count / total_news > 0.6:
            signal = "SELL"

        else:
            # Mixed → use sentiment score
            if sentiment_score > 0.1:
                signal = "BUY"
            elif sentiment_score < -0.1:
                signal = "SELL"
            else:
                signal = "HOLD"

        # Signal strength (based on sentiment only)
        signal_strength = round(abs(sentiment_score) * 100, 2)

        pred_std = np.std(predicted_prices)
        pred_mean = np.mean(predicted_prices) if np.mean(predicted_prices) != 0 else 1
        volatility_ratio = pred_std / pred_mean
        confidence = max(40, min(95, round((1 - volatility_ratio) * 100, 2)))

        # ================== RISK SCORE ==================

        risk_from_volatility = min(100, volatility_ratio * 100)
        risk_from_sentiment = (1 - abs(final_score)) * 100

        risk_score = round((0.6 * risk_from_volatility + 0.4 * risk_from_sentiment), 2)

        # Explanation message
        if signal == "BUY":
            explanation = "Positive sentiment detected from news and social media. Uptrend possible."
        elif signal == "SELL":
            explanation = "Negative sentiment detected. Downward pressure likely."
        else:
            explanation = "Neutral sentiment. Sideways consolidation likely."

        # 🔒 Optional stabilizer (VERY IMPORTANT)
        last_close_series = stock_data["price"].iloc[-1]

        if hasattr(last_close_series, "values"):
            last_close = float(last_close_series.values[0])
        else:
            last_close = float(last_close_series)
        predicted_prices[0] = 0.7 * last_close + 0.3 * predicted_prices[0]

        # ----- 5. PREPARE ALL DASHBOARD DATA (WITH ALL FIXES) -----
    
       # 1. Key Stats (FIXED)
        print("DEBUG: Processing key_stats...")
        last_row = stock_data.tail(1)
        # Calculate 52-week range manually (BEST METHOD)
        one_year_data = yf.download(ticker, period="1y", interval="1d")
        if isinstance(one_year_data.columns, pd.MultiIndex):
            one_year_data.columns = one_year_data.columns.get_level_values(0)
        year_low = safe_float(one_year_data["Low"].min())
        year_high = safe_float(one_year_data["High"].max())
        fast_info = getattr(ticker_obj, "fast_info", {}) or {}

        key_stats = {
            # Price data (from history)
            "open": safe_float(last_row["Open"]),
            "high": safe_float(last_row["High"]),
            "low": safe_float(last_row["Low"]),
            "close": safe_float(last_row["price"]),
            "volume": safe_int(last_row["Volume"]),
            "last_close": safe_float(last_row["price"]),
            # Company stats (from info)
            "market_cap": safe_int(info.get("marketCap")),
            "pe_ratio": safe_float(info.get("trailingPE")),
            "beta": safe_float(info.get("beta")),
            "eps_basic": safe_float(info.get("epsTrailingTwelveMonths")),
            "forward_pe": safe_float(info.get("forwardPE")),
            "dividend_yield": safe_float(info.get("dividendYield")),
            # 52 week range
            "days_range": f"{year_low:.2f} - {year_high:.2f}",
        }
        # 2. Historical Graph (FIXED - SAFE)
        print("DEBUG: Processing historical_graph...")
        historical_data = (
            stock_data
            .sort_values("Date")
            .dropna(subset=["price"])
            .iloc[-30:]
            .copy()
        )

        # Ensure Date is clean
        historical_data["Date"] = pd.to_datetime(historical_data["Date"])

        price_series = pd.to_numeric(historical_data["price"], errors="coerce")

        mask = price_series.notna()
        dates = historical_data.loc[mask, "Date"]
        prices = price_series.loc[mask]

        historical_graph = {
            "dates": dates.dt.strftime("%Y-%m-%d").tolist(),
            "prices": prices.round(2).tolist()
        }

        # 3. Prediction Graph
        print("DEBUG: Processing prediction_graph...")
        future_dates_list = []
        last_date = pd.to_datetime(historical_data["Date"].iloc[-1]).date()

        next_day = last_date
        while len(future_dates_list) < 14:
            next_day = next_day + timedelta(days=1)
            # Monday=0 ... Sunday=6
            if next_day.weekday() < 5:
                future_dates_list.append(next_day.strftime('%Y-%m-%d'))
            
        prediction_graph = {
            "dates": future_dates_list,
            "prices": [float(p) for p in predicted_prices]
        }

        # ----- 6. RETURN THE FINAL, COMPLETE JSON -----
        print("--- Request complete. Sending JSON response. ---")
        print("Current Django language:", translation.get_language())
        lang = (translation.get_language() or "en").split("-")[0]
        print("LANG USED:", lang)
        return JsonResponse({
            "ticker": ticker,
            "key_stats": key_stats,
            "latest_news": news_headlines,
            "historical_graph_data": historical_graph,
            "prediction_graph_data": prediction_graph,
            "twitter_sentiment": twitter_sentiment,
            "confidence": confidence,
            "ai_summary": ai_summary,
            "signal": signal,
            "signal_strength": signal_strength,
            "risk_score": risk_score,
            "final_sentiment": round(final_score, 3),
            "explanation": explanation,
        })

    except Exception as e:
        print(f"❌ An error occurred during prediction: {e}")
        return JsonResponse({"error": str(e)}, status=500)    