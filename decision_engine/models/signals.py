"""
Signal models for aggregated trading decisions.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..rules.base import SignalType


@dataclass
class Signal:
    """Individual signal from a single rule."""
    rule_name: str
    rule_description: str
    signal_type: SignalType
    confidence: float
    reasoning: str
    contributing_factors: Dict[str, Any]
    timestamp: datetime

    def to_dict(self) -> dict:
        return {
            "rule_name": self.rule_name,
            "rule_description": self.rule_description,
            "signal_type": self.signal_type.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "contributing_factors": self.contributing_factors,
            "timestamp": self.timestamp.isoformat() + "Z",
        }


@dataclass
class AggregatedSignal:
    """Aggregated signal from multiple rules for a single symbol."""
    symbol: str
    signal_type: SignalType
    aggregate_confidence: float
    primary_reasoning: str
    contributing_signals: List[Signal]
    timestamp: datetime

    # Metadata
    rules_triggered: int = 0
    rules_evaluated: int = 0

    # Market regime context (populated from context-service via Redis)
    regime_id: str = "UNKNOWN"
    regime_confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "signal_type": self.signal_type.value,
            "aggregate_confidence": round(self.aggregate_confidence, 3),
            "primary_reasoning": self.primary_reasoning,
            "contributing_signals": [s.to_dict() for s in self.contributing_signals],
            "rules_triggered": self.rules_triggered,
            "rules_evaluated": self.rules_evaluated,
            "regime_id": self.regime_id,
            "regime_confidence": round(self.regime_confidence, 3),
            "timestamp": self.timestamp.isoformat() + "Z",
        }


class ConfidenceAggregator:
    """
    Aggregates confidence scores from multiple rules.

    Strategies:
    - weighted_average: Simple weighted average
    - highest: Take highest confidence
    - consensus_boost: Boost when multiple rules agree
    """

    @staticmethod
    def weighted_average(
        signals: List[Signal],
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """Compute weighted average confidence."""
        if not signals:
            return 0.0

        if weights is None:
            weights = {s.rule_name: 1.0 for s in signals}

        total_weight = sum(weights.get(s.rule_name, 1.0) for s in signals)
        weighted_sum = sum(
            s.confidence * weights.get(s.rule_name, 1.0)
            for s in signals
        )

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    @staticmethod
    def highest(signals: List[Signal]) -> float:
        """Return the highest confidence among signals."""
        if not signals:
            return 0.0
        return max(s.confidence for s in signals)

    @staticmethod
    def consensus_boost(signals: List[Signal]) -> float:
        """
        Boost confidence when multiple rules agree.

        Base confidence is weighted average, with a boost
        for each additional agreeing rule (up to +15%).
        """
        if not signals:
            return 0.0

        base_confidence = ConfidenceAggregator.weighted_average(signals)

        # Boost based on number of agreeing rules
        # Each additional rule adds 5%, up to 15% max
        agreement_boost = min((len(signals) - 1) * 0.05, 0.15)

        return min(base_confidence + agreement_boost, 1.0)
