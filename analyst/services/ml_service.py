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
        if not news_articles:
            return {"bullish": 0, "bearish": 0, "neutral": 0, "overall": "No Data"}

        texts = [article.get("title", "") for article in news_articles][:10]
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
        # Fixed Pandas empty check
        if df.empty or len(df) < 20:
            return 0.5

        features = df[["close", "sma_20", "daily_return", "volatility"]].dropna()

        latest = features.iloc[-1]

        probability = 0.5
        if latest["close"] > latest["sma_20"]:
            probability += 0.15
        if latest["daily_return"] > 0:
            probability += 0.05

        return round(probability, 2)
