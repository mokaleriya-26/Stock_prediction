# predictor/twitter_utils.py

import os
import pandas as pd
from django.conf import settings


def normalize_symbol(value):
    if pd.isna(value):
        return ""

    value = str(value).upper().strip()

    # remove leading $
    value = value.replace("$", "")

    # remove exchange suffixes
    for suffix in [".NS", ".NSE", ".BO", ".BSE"]:
        if value.endswith(suffix):
            value = value[: -len(suffix)]

    return value.strip()


def get_twitter_sentiment_for_ticker(ticker):
    try:
        csv_path = os.path.join(
            settings.BASE_DIR,
            "fttt_project",
            "dataset",
            "twitter_sentiment_processed_new.csv"
        )

        if not os.path.exists(csv_path):
            print(f"Processed twitter file not found: {csv_path}")
            return {
                "tweet_count": 0,
                "avg_sentiment": 0,
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
                "latest_tweets": []
            }

        df = pd.read_csv(csv_path)

        if df.empty:
            return {
                "tweet_count": 0,
                "avg_sentiment": 0,
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
                "latest_tweets": []
            }

        df.columns = [c.strip().lower() for c in df.columns]

        required_cols = {"company", "tweet", "sentiment"}
        if not required_cols.issubset(df.columns):
            print("CSV must contain columns: company, tweet, sentiment")
            return {
                "tweet_count": 0,
                "avg_sentiment": 0,
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
                "latest_tweets": []
            }

        base_ticker = normalize_symbol(ticker)

        df["company"] = df["company"].apply(normalize_symbol)
        df["tweet"] = df["tweet"].astype(str).fillna("").str.strip()
        df["sentiment"] = df["sentiment"].astype(str).str.strip().str.capitalize()

        matched = df[df["company"] == base_ticker].copy()

        print(f"Requested ticker: {ticker}")
        print(f"Normalized ticker: {base_ticker}")
        print(f"Matched rows: {len(matched)}")

        if matched.empty:
            return {
                "tweet_count": 0,
                "avg_sentiment": 0,
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
                "latest_tweets": []
            }

        matched = matched[matched["tweet"] != ""]

        def sentiment_to_score(label):
            if label == "Positive":
                return 1.0
            elif label == "Negative":
                return -1.0
            return 0.0

        matched["sentiment_score"] = matched["sentiment"].apply(sentiment_to_score)

        positive_count = int((matched["sentiment"] == "Positive").sum())
        negative_count = int((matched["sentiment"] == "Negative").sum())
        neutral_count = int((matched["sentiment"] == "Neutral").sum())

        latest_tweets = []
        for _, row in matched.head(5).iterrows():
            latest_tweets.append({
                "clean_text": row["tweet"],
                "sentiment_label": row["sentiment"],
                "sentiment_score": float(row["sentiment_score"])
            })

        avg_sentiment = round(float(matched["sentiment_score"].mean()), 3)

        return {
            "tweet_count": int(len(matched)),
            "avg_sentiment": avg_sentiment,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "neutral_count": neutral_count,
            "latest_tweets": latest_tweets
        }

    except Exception as e:
        print("Twitter sentiment CSV error:", e)
        return {
            "tweet_count": 0,
            "avg_sentiment": 0,
            "positive_count": 0,
            "negative_count": 0,
            "neutral_count": 0,
            "latest_tweets": []
        }