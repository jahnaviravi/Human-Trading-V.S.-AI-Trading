import pandas as pd
from lumibot.strategies.strategy import Strategy
from lumibot.brokers import Alpaca
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
from datetime import datetime 
from alpaca_trade_api import REST 
from timedelta import Timedelta 
from finbert_utils import estimate_sentiment

class MLTrader(Strategy):
    def select_top_stocks(self, start_date, end_date, num_stocks=5):
        stock_metrics = {}
        for symbol in self.symbols:
            # Fetch historical data for the stock
            historical_data = self.broker.get_historical_data(symbol, start_date, end_date)
            if historical_data.empty:
                continue

            # Calculate performance metrics (e.g., average daily returns)
            returns = historical_data['close'].pct_change().mean() * 252
            volatility = historical_data['close'].pct_change().std() * (252 ** 0.5)
            sharpe_ratio = returns / volatility if volatility != 0 else 0

            # Store metrics for the stock
            stock_metrics[symbol] = {
                'returns': returns,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio
            }

        # Create DataFrame from metrics
        metrics_df = pd.DataFrame.from_dict(stock_metrics, orient='index')

        # Rank stocks based on chosen metric (e.g., Sharpe ratio)
        ranked_stocks = metrics_df['sharpe_ratio'].nlargest(num_stocks).index.tolist()

        return ranked_stocks

    def initialize(self, num_stocks=5, cash_at_risk=0.5):
        self.num_stocks = num_stocks
        # Other initialization code...

    def on_start(self):
        # Select top 5 stocks
        top_stocks = self.select_top_stocks(self.start_date, self.end_date, self.num_stocks)
        print("Top 5 stocks selected by the strategy:")
        for i, stock in enumerate(top_stocks, start=1):
            print(f"{i}. {stock}")
        
        # Store selected stocks for further use if needed
        self.selected_stocks = top_stocks

# Example usage:
start_date = datetime(2020, 1, 1)
end_date = datetime(2023, 12, 31)
symbols_to_trade = ["JNJ", "XOM", "AAPL", "WMT", "JPM"]
broker = Alpaca(ALPACA_CREDS)
strategy = MLTrader(name='mlstrat', broker=broker, parameters={"symbols": symbols_to_trade, "cash_at_risk": .5})
strategy.backtest(YahooDataBacktesting, start_date, end_date, parameters={"symbols": symbols_to_trade, "cash_at_risk": .5})
trader = Trader()
trader.add_strategy(strategy)
trader.run_all()

# Print top stocks
print("Top 5 stocks selected by the strategy:")
for i, stock in enumerate(strategy.selected_stocks, start=1):
    print(f"{i}. {stock}")

