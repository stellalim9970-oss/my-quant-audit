
import json

class StrategyExtractor:
    """
    This class simulates the LLM extraction logic.
    In the final version, this will call GPT-4o or Claude 3.5.
    """
    def extract(self, text):
        # This is a mock representation of what the LLM will output
        # based on the strict requirements provided by the user.
        
        # In real implementation, this would be:
        # response = llm.complete(prompt=f"Extract strategy from: {text}")
        # return json.loads(response)
        
        # Mocking a successful extraction for demonstration
        if "EMA" in text and "NVDA" in text:
            return {
                "status": "success",
                "strategy": {
                    "ticker": "NVDA",
                    "timeframe": "1h",
                    "entry_rule": "Price crosses above 20 EMA",
                    "exit_rule": "Price crosses below 20 EMA",
                    "stop_loss": "2%",
                    "take_profit": "6%",
                    "position_sizing": "10% of equity",
                    "parameters": {"ema_period": 20}
                },
                "assumptions": [
                    "Timeframe was not explicitly mentioned, assumed '1h' based on common EMA strategies.",
                    "Position sizing was not mentioned, assumed '10% of equity' for safety."
                ]
            }
        else:
            return {
                "status": "incomplete",
                "error": "Could not extract a complete strategy. Please provide more details.",
                "extracted_partial": {
                    "ticker": "Unknown",
                    "entry_rule": "Unknown"
                }
            }

# Test the extractor
if __name__ == "__main__":
    extractor = StrategyExtractor()
    test_text = "I trade NVDA using a simple 20 EMA crossover. When price goes above EMA 20, I buy. When it goes below, I sell."
    result = extractor.extract(test_text)
    print(json.dumps(result, indent=4, ensure_ascii=False))
