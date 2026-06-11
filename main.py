
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Import our custom engines
from extractor_prototype import StrategyExtractor
from backtest_engine import DataFetcher, BacktestEngine, MetricsCalculator
from stress_test_engine import StressTestEngine
import numpy as np
import pandas as pd

app = FastAPI()

# Mount static files for the UI
app.mount("/static", StaticFiles(directory="static"), name="static")

class StrategyRequest(BaseModel):
    input_text: str
    language: str = "zh"

@app.get("/")
async def read_index():
    return FileResponse('static/index.html')

@app.post("/analyze")
async def analyze_strategy(request: StrategyRequest):
    try:
        # 1. Extraction Phase
        extractor = StrategyExtractor()
        extraction_result = extractor.extract(request.input_text)
        
        if extraction_result["status"] == "incomplete":
            raise HTTPException(status_code=400, detail="Could not extract a valid strategy.")
        
        strat = extraction_result["strategy"]
        assumptions = extraction_result["assumptions"]
        
        # 2. Data Fetching
        fetcher = DataFetcher()
        # For demo, we use a fixed 5-year range
        df = fetcher.get_data(strat["ticker"], "2019-01-01", "2024-01-01")
        if df is None:
            raise HTTPException(status_code=404, detail="Market data not found.")
            
        # 3. Backtesting (IS/OOS)
        engine = BacktestEngine()
        calc = MetricsCalculator()
        
        # Define the strategy logic for the engine
        def strategy_logic(data):
            # Simple EMA Crossover based on extracted period
            period = strat["parameters"].get("ema_period", 20)
            ema = data['Close'].ewm(span=period).mean()
            return (data['Close'] > ema).astype(int)
            
        train_df, test_df = engine.split_data(df)
        equity_curve, returns = engine.run_strategy(test_df, strategy_logic)
        
        # Benchmark (SPY)
        spy_df = fetcher.get_data("SPY", "2019-01-01", "2024-01-01")
        bench_returns = spy_df['Close'].pct_change().loc[test_df.index] if spy_df is not None else None
        
        metrics = calc.calculate_metrics(equity_curve, returns, bench_returns)
        
        # 4. Stress Testing
        stress_engine = StressTestEngine()
        # Parameter sweep (around the extracted period)
        p_start = max(5, strat["parameters"].get("ema_period", 20) - 10)
        p_end = strat["parameters"].get("ema_period", 20) + 10
        sweep_results, stable = stress_engine.parameter_sweep(df, 
            lambda d, p: (d['Close'] > d['Close'].ewm(span=p).mean()).astype(int), 
            range(p_start, p_end))
            
        # Monte Carlo & Random Baseline
        final_rets, mdds = stress_engine.monte_carlo_simulation(returns)
        p_val = stress_engine.random_baseline_test(df, returns)
        
        # 5. Final Verdict Logic
        # PASS Conditions: Stable params AND p_val < 0.05 AND positive return
        is_pass = stable and p_val < 0.05 and float(metrics["Total Return"].strip('%')) > 0
        
        return {
            "verdict": "PASS" if is_pass else "FAIL",
            "metrics": metrics,
            "strategy": strat,
            "assumptions": assumptions,
            "stress_test": {
                "is_stable": stable,
                "p_value": p_val,
                "worst_mdd": f"{np.min(mdds):.2%}" if mdds is not None else "N/A"
            },
            "equity_curve": {
                "dates": equity_curve.index.strftime('%Y-%m-%d').tolist(),
                "values": equity_curve.values.tolist()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import os
    if not os.path.exists("static"):
        os.makedirs("static")
    uvicorn.run(app, host="0.0.0.0", port=8080)
