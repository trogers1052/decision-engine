"""
Base classes for trading rules.

Rules are designed to be:
1. Self-documenting with name and description
2. Easy to translate from natural language
3. Return signals with confidence and reasoning
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class SignalType(Enum):
    """Trading signal types."""
    BUY = "BUY"
    SELL = "SELL"
    WATCH = "WATCH"  # Interested but not actionable yet


@dataclass
class RuleResult:
    """
    Result of a rule evaluation with full explainability.

    Attributes:
        triggered: Whether the rule conditions were met
        signal: The signal type if triggered (BUY/SELL/WATCH)
        confidence: Confidence level from 0.0 to 1.0
        reasoning: Human-readable explanation
        contributing_factors: Key indicators/values that contributed
    """
    triggered: bool
    signal: Optional[SignalType] = None
    confidence: float = 0.0
    reasoning: str = ""
    contributing_factors: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.triggered and self.signal is None:
            raise ValueError("Triggered rules must have a signal")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")


@dataclass
class SymbolContext:
    """
    Current state and indicator data for a symbol.

    This is passed to rules for evaluation.
    """
    symbol: str
    indicators: Dict[str, float]
    timestamp: datetime
    data_quality: Optional[Dict[str, Any]] = None

    # Historical context (populated by state manager)
    previous_signals: List[Any] = field(default_factory=list)
    bars_since_last_signal: int = 0
    current_position: Optional[str] = None  # 'long', 'short', None

    def get_indicator(self, name: str, default: float = 0.0) -> float:
        """Safely get an indicator value."""
        return self.indicators.get(name, default)

    def has_indicators(self, *names: str) -> bool:
        """Check if all specified indicators are present and not None."""
        for name in names:
            if name not in self.indicators or self.indicators[name] is None:
                return False
        return True


class Rule(ABC):
    """
    Base class for trading rules.

    To create a new rule:
    1. Subclass this class
    2. Implement name, description, and evaluate()
    3. Optionally override required_indicators

    Example:
        class MyRule(Rule):
            @property
            def name(self) -> str:
                return "My Custom Rule"

            @property
            def description(self) -> str:
                return "Buy when RSI < 30 and price above 200 SMA"

            @property
            def required_indicators(self) -> List[str]:
                return ["RSI_14", "SMA_200"]

            def evaluate(self, context: SymbolContext) -> RuleResult:
                rsi = context.get_indicator("RSI_14")
                # ... evaluation logic ...
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable rule name."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """
        Natural language description of the rule.

        Example: "Buy when RSI drops below 30 (oversold condition)"
        """
        pass

    @property
    def required_indicators(self) -> List[str]:
        """
        List of indicators this rule needs.
        Override to specify dependencies.

        Returns:
            List of indicator names (e.g., ['RSI_14', 'MACD'])
        """
        return []

    @abstractmethod
    def evaluate(self, context: SymbolContext) -> RuleResult:
        """
        Evaluate the rule against current symbol context.

        Args:
            context: Current symbol state and indicators

        Returns:
            RuleResult with signal, confidence, and reasoning
        """
        pass

    def can_evaluate(self, context: SymbolContext) -> bool:
        """Check if all required indicators are present."""
        return context.has_indicators(*self.required_indicators)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
