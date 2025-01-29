import os
import matplotlib.pyplot as plt
import mplfinance as mpf
import yfinance as yf
import pandas as pd
from typing import List, Dict, Any
from phi.assistant import Assistant
from phi.llm.groq import Groq
from dotenv import load_dotenv
from duckduckgo_search import DDGS
from typing import ClassVar

load_dotenv()
plt.switch_backend('Agg')  # Headless mode for plotting

class FinancialAnalyst(Assistant):
    ddgs: ClassVar = DDGS  # Annotate as a ClassVar to prevent Pydantic errors
    def get_stock_data(self, ticker: str, period: str = "1y") -> Dict[str, Any]:
        """Fetch historical stock market data"""
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period=period)
            if data.empty:
                return {"error": f"No data found for {ticker}"}

            data = data.astype(float, errors='ignore').dropna()
            return {
                "ticker": ticker,
                "period": period,
                "data": data.reset_index().to_dict(orient="records"),
                "latest_close": data.iloc[-1].Close if not data.empty else None
            }
        except Exception as e:
            return {"error": f"Error fetching data for {ticker}: {str(e)}"}


    def get_financial_news(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Fetch financial news using DuckDuckGo's latest API"""
        try:
            with DDGS() as ddgs:
                results = list(ddgs.news(query, max_results=max_results))
            return {"news": [{"title": r.get("title", ""), "url": r.get("url", "")} for r in results]}
        except Exception as e:
            return {"error": f"Error fetching news: {str(e)}"}



    def generate_financial_chart(self, ticker: str, period: str = "1y", chart_type: str = "line") -> Dict[str, Any]:
        """Generate stock price charts"""
        os.makedirs("financial_charts", exist_ok=True)
        data_response = self.get_stock_data(ticker, period)
        if "error" in data_response:
            return data_response

        try:
            data = pd.DataFrame(data_response["data"])
            if data.empty or "Date" not in data.columns:
                return {"error": "Invalid data format"}

            data["Date"] = pd.to_datetime(data["Date"])
            data.set_index("Date", inplace=True)
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            data.dropna(inplace=True)

            filename = f"financial_charts/{ticker}_{period}_{chart_type}.png"
            if chart_type.lower() == "candlestick":
                mpf.plot(data, type='candle', style='charles',
                         title=f"{ticker} {period} Candlestick Chart",
                         ylabel='Price', volume=True,
                         savefig=dict(fname=filename, dpi=100, bbox_inches='tight'))
            elif chart_type.lower() == "line":
                plt.figure(figsize=(12, 6))
                data['Close'].plot(title=f"{ticker} {period} Price Chart")
                plt.xlabel("Date")
                plt.ylabel("Price ($)")
                plt.grid(True)
                plt.savefig(filename, bbox_inches='tight')
                plt.close()
            else:
                return {"error": "Unsupported chart type. Use 'line' or 'candlestick'"}

            return {"success": True, "chart_path": filename}
        except Exception as e:
            return {"error": f"Chart generation failed: {str(e)}"}

    def generate_technical_analysis_chart(self, ticker: str, indicators: List[str] = ["sma"]) -> Dict[str, Any]:
        """Generate stock charts with technical indicators"""
        data_response = self.get_stock_data(ticker, "6mo")
        if "error" in data_response:
            return data_response

        try:
            data = pd.DataFrame(data_response["data"])
            if data.empty or "Date" not in data.columns:
                return {"error": "Invalid data format"}

            data["Date"] = pd.to_datetime(data["Date"])
            data.set_index("Date", inplace=True)
            data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
            data.dropna(inplace=True)

            plt.figure(figsize=(12, 8))
            ax1 = plt.subplot(2, 1, 1)
            data['Close'].plot(ax=ax1, title=f"{ticker} Technical Analysis", label='Price')

            if "sma" in indicators:
                sma20 = data['Close'].rolling(window=20).mean()
                sma50 = data['Close'].rolling(window=50).mean()
                sma20.plot(ax=ax1, label='20-day SMA')
                sma50.plot(ax=ax1, label='50-day SMA')
                ax1.legend()

            filename = f"financial_charts/{ticker}_technical_analysis.png"
            plt.tight_layout()
            plt.savefig(filename, bbox_inches='tight')
            plt.close()
            return {"success": True, "chart_path": filename}
        except Exception as e:
            return {"error": f"Technical analysis failed: {str(e)}"}

    def __init__(self):
        super().__init__(
            name="Financial Analyst AI",
            role="Analyze financial markets and generate visualizations",
            llm=Groq(model="mixtral-8x7b-32768", api_key=os.getenv("GROQ_API_KEY")),
            tools=[
                self.generate_financial_chart,
                self.generate_technical_analysis_chart,  # Now correctly referenced
                self.get_financial_news,
                self.get_stock_data,
            ],
            description="AI financial analyst with advanced visualization capabilities",
            instructions=[
                "Convert time periods to valid yfinance formats (1d, 5d, 1mo, 3mo, 6mo, 1y, 5y, max)",
                "Handle both stock tickers and cryptocurrency symbols",
                "Validate user inputs before fetching data",
                "Add proper error handling for data requests",
                "Format dates in charts appropriately",
            ],
            add_datetime=True,
            markdown=True,
        )

if __name__ == "__main__":
    analyst = FinancialAnalyst()
    print(analyst.get_financial_news("Microsoft Stock"))
    print(analyst.generate_financial_chart("NVDA", "1y", "candlestick"))

