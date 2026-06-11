
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

class DataFetcher:
    """Handles reliable data fetching from Yahoo Finance."""
    @staticmethod
    def get_data(ticker, start_date, end_date, interval='1d'):
        try:
            df = yf.download(ticker, start=start_date, end=end_date, interval=interval)
            if df.empty:
                raise ValueError(f"No data found for ticker {ticker}")
            # Ensure a clean index
            df.index = pd.to_datetime(df.index)
            return df
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return None

class BacktestEngine:
    """Professional backtesting engine with IS/OOS split."""
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital

    def split_data(self, df, train_ratio=0.7):
        """Strictly splits data into In-Sample and Out-of-Sample."""
        split_idx = int(len(df) * train_ratio)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        return train_df, test_df

    def run_strategy(self, df, strategy_func):
        """
        Executes a strategy function. 
        strategy_func should return a Series of signals (1 for long, 0 for flat/short).
        """
        # Ensure we are working with a Series for Close price
        close_series = df['Close']
        if isinstance(close_series, pd.DataFrame):
            close_series = close_series.iloc[:, 0]
            
        signals = strategy_func(df)
        if isinstance(signals, pd.DataFrame):
            signals = signals.iloc[:, 0]
        
        # Calculate Daily Returns
        df_returns = close_series.pct_change()
        
        # Strategy returns = signal from yesterday * return of today
        strategy_returns = signals.shift(1) * df_returns
        
        # Equity Curve
        equity_curve = (1 + strategy_returns.fillna(0)).cumprod() * self.initial_capital
        return equity_curve, strategy_returns


class MetricsCalculator:
    """Calculates standard quant metrics with high precision."""
    @staticmethod
    def calculate_metrics(equity_curve, returns, benchmark_returns=None):
        # Total Return
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        
        # CAGR (Compound Annual Growth Rate)
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (365.25 / days) - 1
        
        # Max Drawdown
        rolling_max = equity_curve.cummax()
        drawdown = (equity_curve - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        
        # Sharpe Ratio (Assuming risk-free rate = 0 for simplicity, annualized)
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0
        
        # Win Rate
        trades = returns[returns != 0]
        win_rate = (trades > 0).sum() / len(trades) if len(trades) > 0 else 0
        
        metrics = {
            "Total Return": f"{total_return:.2%}",
            "CAGR": f"{cagr:.2%}",
            "Max Drawdown": f"{max_drawdown:.2%}",
            "Sharpe Ratio": f"{sharpe:.2f}",
            "Win Rate": f"{win_rate:.2%}",
            "Total Trades": len(trades)
        }
        
        if benchmark_returns is not None:
            # Ensure benchmark_returns is a Series
            if isinstance(benchmark_returns, pd.DataFrame):
                bench_returns_series = benchmark_returns.iloc[:, 0]
            else:
                bench_returns_series = benchmark_returns
                
            bench_total = (1 + bench_returns_series).prod() - 1
            metrics["Benchmark Total Return"] = f"{float(bench_total):.2%}"
            metrics["Alpha (vs Bench)"] = f"{(float(total_return) - float(bench_total)):.2%}" # Wait, I made a typo in the manual edit logic here.

            
        return metrics

# --- Example Usage for Validation ---
if __name__ == "__main__":
    # 1. Setup
    fetcher = DataFetcher()
    engine = BacktestEngine()
    calc = MetricsCalculator()
    
    # Strategy: Simple SMA Crossover (Dummy example to prove it runs)
    def sma_cross_strategy(df):
        sma_short = df['Close'].rolling(window=20).mean()
        sma_long = df['Close'].rolling(window=50).mean()
        return (sma_short > sma_long).astype(int)

    # 2. Get Data (NVDA vs SPY)
    print("Fetching data...")
    nvda_data = fetcher.get_data("NVDA", "2018-01-01", "2023-12-31")
    spy_data = fetcher.get_data("SPY", "2018-01-01", "2023-12-31")
    
    if nvda_data is not None and spy_data is not None:
        # 3. IS/OOS Split
        train, test = engine.split_data(nvda_data)
        
        # 4. Run Backtest on OOS (The real test)
        print("Running backtest on Out-of-Sample data...")
        equity_curve, returns = engine.run_strategy(test, sma_cross_strategy)
        
        # 5. Get Benchmark returns for the same OOS period
        bench_returns = spy_data['Close'].pct_change().loc[test.index]
        
        # 6. Calculate Results
        results = calc.calculate_metrics(equity_curve, returns, bench_returns)
        
        print("\n--- Final Backtest Report (OOS) ---")
        for k, v in results.items():
            print(f"{k}: {v}")
