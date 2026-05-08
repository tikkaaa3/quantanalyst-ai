import os
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI


class AgentState(TypedDict):
    ticker: str
    price: float
    prediction_score: float
    sentiment: dict
    final_report: str


class QuantAgent:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
        model = os.getenv("OPENAI_MODEL", "meta-llama/llama-3.3-70b-instruct")

        try:
            self.llm = ChatOpenAI(
                api_key=api_key,
                base_url=base_url,
                model=model,
                temperature=0.7,
            )
        except Exception:
            self.llm = None

        workflow = StateGraph(AgentState)
        workflow.add_node("draft_report", self.draft_report)
        workflow.set_entry_point("draft_report")
        workflow.add_edge("draft_report", END)
        self.graph = workflow.compile()

    def draft_report(self, state: AgentState) -> dict:
        if self.llm is None:
            return {
                "final_report": "Agent synthesis currently unavailable. Please check OpenRouter API keys."
            }

        system_prompt = (
            "Act as a Quant Analyst. Read this XGBoost probability score and PyTorch news sentiment. "
            "Write a concise, 2-paragraph markdown report for the ticker."
        )

        user_prompt = (
            f"Ticker: {state['ticker']}\n"
            f"Current Price: ${state['price']}\n"
            f"XGBoost Bullish Probability: {state['prediction_score']:.2%}\n"
            f"News Sentiment: {state['sentiment']}\n\n"
            f"Generate the investment report now."
        )

        try:
            response = self.llm.invoke(
                [
                    ("system", system_prompt),
                    ("human", user_prompt),
                ]
            )
            return {"final_report": response.content}
        except Exception:
            return {
                "final_report": "Agent synthesis currently unavailable. Please check OpenRouter API keys."
            }

    def run(self, ticker: str, price: float, score: float, sentiment: dict) -> str:
        try:
            result = self.graph.invoke(
                {
                    "ticker": ticker,
                    "price": price,
                    "prediction_score": score,
                    "sentiment": sentiment,
                    "final_report": "",
                }
            )
            return result.get(
                "final_report",
                "Agent synthesis currently unavailable. Please check OpenRouter API keys.",
            )
        except Exception:
            return "Agent synthesis currently unavailable. Please check OpenRouter API keys."
