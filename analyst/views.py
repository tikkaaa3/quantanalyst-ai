from django.shortcuts import render
from .services.data_service import DataService
from .services.ml_service import MLService
from .services.agent_graph import QuantAgent


def analysis_lab(request):
    ticker = request.GET.get("ticker", "AAPL").upper()
    data_service = DataService()
    ml_service = MLService()

    context = {"ticker": ticker}

    try:
        # 1. Fetch Data
        df = data_service.fetch_stock_data(ticker)
        news = data_service.fetch_news_sentiment(ticker)

        # 2. Run ML Models
        xgboost_prob = ml_service.predict_price_movement(df)
        sentiment = ml_service.analyze_sentiment(news)

        # 3. Agent Synthesis
        agent = QuantAgent()
        ai_report = agent.run(
            ticker=ticker,
            price=round(df["close"].iloc[-1], 2),
            score=xgboost_prob,
            sentiment=sentiment,
        )

        # 4. Populate Context
        context.update(
            {
                "latest_price": round(df["close"].iloc[-1], 2),
                "latest_sma": round(df["sma_20"].iloc[-1], 2),
                "news_count": len(news),
                "news": news[:5],
                "prediction_score": xgboost_prob * 100,  # Convert to percentage
                "sentiment": sentiment,
                "ai_report": ai_report,
            }
        )
    except Exception as e:
        context["error"] = str(e)

    return render(request, "analyst/index.html", context)
