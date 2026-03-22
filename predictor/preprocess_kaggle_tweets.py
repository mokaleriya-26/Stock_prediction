# predictor/preprocess_kaggle_tweets.py
import os
import re
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "fttt_project", "dataset")

RAW_FILE = os.path.join(DATASET_DIR, "tweets.csv")
OUTPUT_FILE = os.path.join(DATASET_DIR, "twitter_sentiment_processed_new.csv")

analyzer = SentimentIntensityAnalyzer()

def clean_tweet(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def get_sentiment(text):
    score = analyzer.polarity_scores(text)["compound"]
    if score >= 0.05:
        return "Positive", score
    elif score <= -0.05:
        return "Negative", score
    else:
        return "Neutral", score

def convert_stock_name_to_ticker(stock_name):
    if pd.isna(stock_name):
        return None
    stock_name = str(stock_name).strip().upper()

    # If dataset already contains names like AAPL, TSLA, MSFT etc.
    # You can later map Indian stocks manually if needed.
    return stock_name

def main():
    print(f"Reading raw file from: {RAW_FILE}")
    print(f"Will save processed file to: {OUTPUT_FILE}")

    df = pd.read_csv(RAW_FILE)
    df.columns = [c.strip() for c in df.columns]

    print("Available columns:", df.columns.tolist())

    tweet_col = None
    stock_col = None
    company_col = None
    date_col = None

    for c in df.columns:
        cl = c.lower()
        if cl in ["tweet", "text", "content", "body"]:
            tweet_col = c
        elif cl in ["stock name", "stock_name", "ticker", "symbol", "company"]:
            stock_col = c
        elif cl in ["company name", "company_name"]:
            company_col = c
        elif cl in ["date", "datetime", "created_at", "timestamp"]:
            date_col = c

    if tweet_col is None:
        raise Exception(f"No tweet/text column found. Available columns: {df.columns.tolist()}")

    if stock_col is None:
        raise Exception(f"No stock/company column found. Available columns: {df.columns.tolist()}")

    df["clean_text"] = df[tweet_col].apply(clean_tweet)
    df = df[df["clean_text"] != ""]

    df["company"] = df[stock_col].apply(convert_stock_name_to_ticker)
    df["sentiment"] = df["clean_text"].apply(lambda x: get_sentiment(x)[0])

    output_df = pd.DataFrame({
        "company": df["company"],
        "tweet": df["clean_text"],
        "sentiment": df["sentiment"]
    })

    output_df.to_csv(OUTPUT_FILE, index=False)

    print(f"Processed file saved at: {OUTPUT_FILE}")
    print(f"Total processed rows: {len(output_df)}")
    print(output_df.head())

if __name__ == "__main__":
    main()