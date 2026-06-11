
import numpy as np
import pandas as pd
from backtest_engine import BacktestEngine, MetricsCalculator

class StressTestEngine:
    """
    Advanced stress testing module to validate strategy robustness.
    """
    def __init__(self, initial_capital=10000):
        self.engine = BacktestEngine(initial_capital)
        self.calc = MetricsCalculator()

    def parameter_sweep(self, df, strategy_factory, param_range):
        """
        Tests a range of parameters to find stability zones.
        strategy_factory: a function that takes (df, param) and returns signals.
        """
        results = []
        for p in param_range:
            signals = strategy_factory(df, p)
            equity_curve, returns = self.engine.run_strategy(df, lambda x: signals)
            metrics = self.calc.calculate_metrics(equity_curve, returns)
            results.append({"param": p, "return": float(metrics["Total Return"].strip('%'))/100, "sharpe": float(metrics["Sharpe Ratio"])})
        
        results_df = pd.DataFrame(results)
        # Stability check: Is the best parameter surrounded by other profitable parameters?
        best_idx = results_df['return'].idxmax()
        # Check neighbors
        is_stable = False
        if 0 < best_idx < len(results_df) - 1:
            if results_df.iloc[best_idx-1]['return'] > 0 and results_df.iloc[best_idx+1]['return'] > 0:
                is_stable = True
        
        return results_df, is_stable

    def monte_carlo_simulation(self, returns, iterations=1000):
        """
        Shuffles returns to see the distribution of equity curves.
        """
        trade_returns = returns[returns != 0].values
        if len(trade_returns) == 0:
            return None, 0
        
        final_returns = []
        max_drawdowns = []
        
        for _ in range(iterations):
            shuffled = np.random.choice(trade_returns, size=len(trade_returns), replace=False)
            equity_curve = (1 + shuffled).cumprod()
            
            rolling_max = np.maximum.accumulate(equity_curve)
            drawdown = (equity_curve - rolling_max) / rolling_max
            
            final_returns.append(equity_curve[-1] - 1)
            max_drawdowns.append(np.min(drawdown))
            
        return np.array(final_returns), np.array(max_drawdowns)

    def random_baseline_test(self, df, actual_returns, iterations=1000):
        """
        Compares strategy against 1000 random signal generators.
        """
        random_returns_list = []
        # Ensure we use a Series for close returns
        close_series = df['Close']
        if isinstance(close_series, pd.DataFrame):
            close_series = close_series.iloc[:, 0]
        df_returns = close_series.pct_change().fillna(0)
        
        for _ in range(iterations):
            random_signals = np.random.randint(0, 2, size=len(df))
            strat_ret = pd.Series(random_signals, index=df.index).shift(1).fillna(0) * df_returns
            random_returns_list.append(strat_ret.sum())
            
        # Ensure actual_total_return is a float
        actual_total_return = float(actual_returns.sum())
        win_count = sum(1 for r in random_returns_list if actual_total_return > float(r))
        p_value = win_count / iterations
        
        return p_value # p < 0.05 means significantly better than random


# --- Validation Run ---
if __name__ == "__main__":
    import yfinance as yf
    
    # 1. Setup
    df = yf.download("NVDA", start="2020-01-01", end="2023-12-31")
    stress_engine = StressTestEngine()
    
    # Strategy Factory for EMA
    def ema_strategy_factory(data, period):
        ema = data['Close'].ewm(span=period).mean()
        return (data['Close'] > ema).astype(int)

    # 2. Parameter Sweep (EMA 10 to 50)
    print("Running Parameter Sweep...")
    param_range = range(10, 51)
    sweep_results, stable = stress_engine.parameter_sweep(df, ema_strategy_factory, param_range)
    print(f"Is parameter zone stable? {stable}")

    # 3. Monte Carlo & Random Baseline
    # Get actual returns for a fixed param (e.g., EMA 20)
    signals = ema_strategy_factory(df, 20)
    _, actual_returns = stress_engine.engine.run_strategy(df, lambda x: signals)
    
    print("Running Monte Carlo...")
    final_rets, mdds = stress_engine.monte_carlo_simulation(actual_returns)
    print(f"Average Final Return: {np.mean(final_rets):.2%}, Worst MDD: {np.min(mdds):.2%}")
    
    print("Running Random Baseline...")
    p_val = stress_engine.random_baseline_test(df, actual_returns)
    print(f"P-Value (vs Random): {p_val:.4f} ({'Significant' if p_val < 0.05 else 'Not Significant'})")
