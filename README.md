# Decision Engine

Trading rule evaluation and signal generation service for the systematic trading platform.

## Overview

The Decision Engine consumes technical indicator events from the Analytics Service, evaluates configurable trading rules, and produces:

1. **Individual Decisions** (`trading.decisions`) - BUY/SELL/WATCH signals per stock
2. **Rankings** (`trading.rankings`) - Cross-stock comparison: "Which stock should I buy first?"

## Your Trading Strategy

The engine implements your core strategy: **"Buy dips in uptrending stocks"**

### Primary Rules

| Rule | Condition | Signal |
|------|-----------|--------|
| **Buy Dip in Uptrend** | RSI < 40 AND SMA_20 > SMA_50 | BUY |
| **Strong Buy Signal** | RSI < 35 AND SMA_20 > SMA_50 > SMA_200 | BUY (highest confidence) |
| Weekly Uptrend | SMA_20 > SMA_50 | Required for buy |
| Monthly Uptrend | SMA_50 > SMA_200 | Preferred, not required |

### Supporting Rules

- RSI Oversold/Overbought
- MACD Bullish/Bearish Crossover
- Full Trend Alignment (Golden Cross)
- Trend Break Warning

## Ranking

When you have multiple BUY signals, the ranker answers: "Which should I buy first?"

Ranking weights:
- **Dip Depth** (30%): Lower RSI = better entry
- **Trend Strength** (30%): Full SMA alignment preferred
- **Confidence** (25%): More rules agreeing = higher confidence
- **Volatility** (15%): Moderate volatility preferred

## Adding New Rules

1. Describe your rule in plain English
2. I translate it to a Python rule class
3. Add to `config/rules.yaml`

Example rule in `rules/composite_rules.py`:
```python
class BuyDipInUptrendRule(Rule):
    """
    YOUR PRIMARY RULE:
    Natural Language: "Buy when RSI dips to 35-40 AND weekly uptrend is intact"
    """
    @property
    def description(self) -> str:
        return "Buy when RSI < 40 AND weekly uptrend intact (SMA_20 > SMA_50)"
```

## Configuration

Edit `config/rules.yaml` to:
- Enable/disable rules
- Adjust thresholds (RSI levels, etc.)
- Set rule weights for ranking

## Running

```bash
# With Docker Compose (recommended)
docker-compose up decision-engine

# Local development
cp .env.example .env
# Edit .env with your settings
python -m decision_engine.main
```

## Event Schemas

### Input: INDICATOR_UPDATE
```json
{
  "event_type": "INDICATOR_UPDATE",
  "data": {
    "symbol": "AAPL",
    "indicators": {
      "RSI_14": 28.5,
      "MACD": 0.12,
      "SMA_20": 150.00,
      "SMA_50": 148.00,
      "SMA_200": 145.00
    }
  }
}
```

### Output: DECISION_UPDATE
```json
{
  "event_type": "DECISION_UPDATE",
  "data": {
    "symbol": "AAPL",
    "signal": "BUY",
    "confidence": 0.85,
    "primary_reasoning": "BUY DIP: deep dip (RSI: 28.5) in strong uptrend",
    "rules_triggered": [...]
  }
}
```

### Output: RANKING_UPDATE
```json
{
  "event_type": "RANKING_UPDATE",
  "data": {
    "signal_type": "BUY",
    "rankings": [
      {"rank": 1, "symbol": "WPM", "score": 0.92},
      {"rank": 2, "symbol": "AAPL", "score": 0.85}
    ]
  }
}
```

## Backtesting Integration

Rules are compatible with the backtesting service via the `RuleBasedStrategy` adapter:

```python
from decision_engine.adapters.backtesting import RuleBasedStrategy
from decision_engine.rules.composite_rules import BuyDipInUptrendRule

strategy = RuleBasedStrategy(
    rules=[BuyDipInUptrendRule()],
    min_confidence=0.6
)
```
