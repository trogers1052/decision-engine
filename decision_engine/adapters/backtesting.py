"""
Adapter to make decision-engine rules compatible with backtesting-service.

The backtesting service expects a Strategy with:
- on_price_update(event: PriceEvent) -> Optional[Signal]
- on_indicator_update(event: IndicatorEvent) -> Optional[Signal]

This adapter wraps decision-engine rules into that interface.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from ..rules.base import Rule, SymbolContext, SignalType


# Backtesting service signal type (matches backtesting-service/backtesting/strategy.py)
class BacktestSignal(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class IndicatorEvent:
    """Indicator event for backtesting."""
    symbol: str
    indicators: Dict[str, float]
    timestamp: datetime


@dataclass
class PriceEvent:
    """Price event for backtesting."""
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    timestamp: datetime


class RuleBasedStrategy:
    """
    Wraps decision-engine rules into backtesting Strategy interface.

    Usage:
        from decision_engine.rules.composite_rules import BuyDipInUptrendRule
        from decision_engine.rules.rsi_rules import RSIOversoldRule

        strategy = RuleBasedStrategy(
            rules=[
                BuyDipInUptrendRule(rsi_threshold=40),
                RSIOversoldRule(threshold=30),
            ],
            name="Buy Dip Strategy",
            min_confidence=0.6,
        )

        # Use with backtesting service
        backtester = Backtester(strategy=strategy)
    """

    def __init__(
        self,
        rules: List[Rule],
        name: str = "Rule-Based Strategy",
        min_confidence: float = 0.6,
        require_consensus: bool = False,
        profit_target: float = 0.07,  # 7% profit target
        stop_loss: float = 0.05,      # 5% stop loss
    ):
        """
        Initialize the strategy adapter.

        Args:
            rules: List of Rule instances to evaluate
            name: Strategy name for logging
            min_confidence: Minimum confidence to generate a signal
            require_consensus: If True, require multiple rules to agree
            profit_target: Exit when up this percentage
            stop_loss: Exit when down this percentage
        """
        self.rules = rules
        self.name = name
        self.min_confidence = min_confidence
        self.require_consensus = require_consensus
        self.profit_target = profit_target
        self.stop_loss = stop_loss

        # Track state per symbol
        self._entry_prices: Dict[str, float] = {}
        self._positions: Dict[str, Optional[str]] = {}

    def on_price_update(self, event: PriceEvent) -> Optional[BacktestSignal]:
        """
        Handle price updates (for exit conditions).

        Args:
            event: PriceEvent from backtesting service

        Returns:
            BacktestSignal.SELL if exit condition met, None otherwise
        """
        symbol = event.symbol

        # Check if we have a position to manage
        if symbol not in self._positions or self._positions[symbol] is None:
            return None

        entry_price = self._entry_prices.get(symbol)
        if entry_price is None:
            return None

        current_price = event.close

        # Check profit target
        if current_price >= entry_price * (1 + self.profit_target):
            self._positions[symbol] = None
            return BacktestSignal.SELL

        # Check stop loss
        if current_price <= entry_price * (1 - self.stop_loss):
            self._positions[symbol] = None
            return BacktestSignal.SELL

        return None

    def on_indicator_update(self, event: IndicatorEvent) -> Optional[BacktestSignal]:
        """
        Handle indicator updates (main rule evaluation).

        Args:
            event: IndicatorEvent from backtesting service

        Returns:
            BacktestSignal.BUY, BacktestSignal.SELL, or None
        """
        # Build context for rules
        context = SymbolContext(
            symbol=event.symbol,
            indicators=event.indicators,
            timestamp=event.timestamp,
            current_position=self._positions.get(event.symbol),
        )

        # Evaluate all applicable rules
        buy_results = []
        sell_results = []

        for rule in self.rules:
            if not rule.can_evaluate(context):
                continue

            result = rule.evaluate(context)

            if result.triggered:
                if result.signal == SignalType.BUY:
                    buy_results.append(result)
                elif result.signal == SignalType.SELL:
                    sell_results.append(result)

        # Determine final signal
        if buy_results:
            avg_confidence = sum(r.confidence for r in buy_results) / len(buy_results)

            if self.require_consensus and len(buy_results) < 2:
                return None

            if avg_confidence >= self.min_confidence:
                # Don't buy if already in position
                if self._positions.get(event.symbol) == 'long':
                    return None

                self._positions[event.symbol] = 'long'
                # Would need price from somewhere to set entry
                # In practice, backtester handles this
                return BacktestSignal.BUY

        if sell_results:
            avg_confidence = sum(r.confidence for r in sell_results) / len(sell_results)

            if self.require_consensus and len(sell_results) < 2:
                return None

            if avg_confidence >= self.min_confidence:
                if self._positions.get(event.symbol) == 'long':
                    self._positions[event.symbol] = None
                    return BacktestSignal.SELL

        return None

    def reset(self):
        """Reset state for new backtest run."""
        self._entry_prices.clear()
        self._positions.clear()

    def get_position(self, symbol: str) -> Optional[str]:
        """Get current position for a symbol."""
        return self._positions.get(symbol)

    def set_entry_price(self, symbol: str, price: float):
        """Set entry price for a symbol (called by backtester)."""
        self._entry_prices[symbol] = price

    def __repr__(self) -> str:
        rule_names = [r.name for r in self.rules]
        return f"RuleBasedStrategy(name='{self.name}', rules={rule_names})"
