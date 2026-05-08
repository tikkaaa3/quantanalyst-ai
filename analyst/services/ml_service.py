import pandas as pd

# import numpy as np  # Uncomment when building the real XGBoost logic
# import xgboost as xgb # Uncomment when building the real XGBoost logic
from transformers import pipeline


class MLService:
    def __init__(self):
        # Initialize the Hugging Face pipeline for financial sentiment
        # Added task= keyword to satisfy Pyright type checker
        self.sentiment_analyzer = pipeline(
            task="sentiment-analysis", model="ProsusAI/finbert"
        )
        self.xgb_model = None

    def analyze_sentiment(self, news_articles: list) -> dict:
        """Runs PyTorch/FinBERT on news headlines."""
        print(f"DEBUG: Received {len(news_articles)} news articles")
        if not news_articles:
            return {"bullish": 0, "bearish": 0, "neutral": 0, "overall": "No Data"}

        texts = [
            (article.get("title") or article.get("summary", ""))
            for article in news_articles
        ][:10]
        print(f"DEBUG: Texts for sentiment analysis: {texts}")
        results = self.sentiment_analyzer(texts)

        # Tally the results
        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
        for res in results:
            sentiment_counts[res["label"]] += 1

        return sentiment_counts

    def predict_price_movement(self, df: pd.DataFrame) -> float:
        """
        Uses XGBoost to predict if the price will go up.
        Returns a probability score between 0 and 1.
        """
        if df.empty or len(df) < 20:
            return 0.50

        features = df[["close", "sma_20", "daily_return", "volatility"]].dropna()
        if features.empty:
            return 0.50

        latest = features.iloc[-1]

        # Calculate 14-day RSI
        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.rolling(window=14, min_periods=14).mean()
        avg_loss = loss.rolling(window=14, min_periods=14).mean()
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        rsi = rsi.where(avg_loss != 0, 100.0)

        latest_rsi = rsi.iloc[-1]

        probability = 0.50
        if latest["close"] > latest["sma_20"]:
            probability += 0.15
        if pd.notna(latest_rsi):
            if 40 <= latest_rsi <= 60:
                probability += 0.15
            if latest_rsi > 70:
                probability -= 0.20

        probability = max(0.0, min(1.0, probability))
        return round(probability, 2)
