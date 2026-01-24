"""
Ranker for comparing and ranking symbols.

Answers the question: "Of all my BUY signals, which stock should I buy first?"
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from .models.signals import AggregatedSignal
from .rules.base import SignalType

logger = logging.getLogger(__name__)


class RankingCriteria(Enum):
    """Criteria for ranking symbols."""
    CONFIDENCE = "confidence"
    DIP_DEPTH = "dip_depth"  # Lower RSI = better entry
    TREND_STRENGTH = "trend_strength"
    COMPOSITE = "composite"


@dataclass
class RankedSymbol:
    """A symbol with its ranking score and details."""
    symbol: str
    rank: int
    score: float
    signal: AggregatedSignal
    ranking_factors: Dict[str, float]

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "rank": self.rank,
            "score": round(self.score, 3),
            "signal_type": self.signal.signal_type.value,
            "confidence": round(self.signal.aggregate_confidence, 3),
            "reasoning": self.signal.primary_reasoning,
            "ranking_factors": {
                k: round(v, 3) for k, v in self.ranking_factors.items()
            },
        }


@dataclass
class RankingResult:
    """Complete ranking of symbols."""
    signal_type: SignalType
    ranked_symbols: List[RankedSymbol]
    timestamp: datetime
    criteria_used: RankingCriteria

    def top(self, n: int = 5) -> List[RankedSymbol]:
        """Get top N ranked symbols."""
        return self.ranked_symbols[:n]

    def to_dict(self) -> dict:
        return {
            "signal_type": self.signal_type.value,
            "criteria": self.criteria_used.value,
            "timestamp": self.timestamp.isoformat() + "Z",
            "total_symbols": len(self.ranked_symbols),
            "rankings": [r.to_dict() for r in self.ranked_symbols],
        }


class SymbolRanker:
    """
    Ranks symbols based on signal strength and your priorities.

    Your ranking weights (from your strategy):
    - Dip Depth (30%): Lower RSI = better entry point
    - Trend Strength (30%): Full SMA alignment preferred
    - Confidence (25%): How many rules triggered
    - Volatility (15%): Based on ATR (prefer moderate)
    """

    def __init__(
        self,
        criteria: RankingCriteria = RankingCriteria.COMPOSITE,
        weights: Optional[Dict[str, float]] = None,
    ):
        self.criteria = criteria

        # Your preferred weights
        self.weights = weights or {
            "dip_depth": 0.30,
            "trend_strength": 0.30,
            "confidence": 0.25,
            "volatility": 0.15,
        }

    def rank(
        self,
        signals: Dict[str, AggregatedSignal],
        signal_type: SignalType = SignalType.BUY,
    ) -> RankingResult:
        """
        Rank symbols with the specified signal type.

        Args:
            signals: Dictionary of symbol -> AggregatedSignal
            signal_type: Filter to only rank this signal type

        Returns:
            RankingResult with ordered list of symbols
        """
        # Filter to only the requested signal type
        filtered = {
            symbol: sig
            for symbol, sig in signals.items()
            if sig.signal_type == signal_type
        }

        if not filtered:
            return RankingResult(
                signal_type=signal_type,
                ranked_symbols=[],
                timestamp=datetime.utcnow(),
                criteria_used=self.criteria,
            )

        # Score each symbol
        scored = []
        for symbol, signal in filtered.items():
            score, factors = self._calculate_score(signal)
            scored.append((symbol, signal, score, factors))

        # Sort by score descending
        scored.sort(key=lambda x: x[2], reverse=True)

        # Build ranked result
        ranked = [
            RankedSymbol(
                symbol=symbol,
                rank=i + 1,
                score=score,
                signal=signal,
                ranking_factors=factors,
            )
            for i, (symbol, signal, score, factors) in enumerate(scored)
        ]

        result = RankingResult(
            signal_type=signal_type,
            ranked_symbols=ranked,
            timestamp=datetime.utcnow(),
            criteria_used=self.criteria,
        )

        if ranked:
            logger.info(
                f"Ranked {len(ranked)} {signal_type.value} signals. "
                f"Top: {ranked[0].symbol} (score: {ranked[0].score:.3f})"
            )

        return result

    def _calculate_score(
        self,
        signal: AggregatedSignal,
    ) -> tuple[float, Dict[str, float]]:
        """
        Calculate composite score for a signal.

        Extracts indicator values from contributing signals and
        calculates weighted score based on your priorities.
        """
        factors = {}

        # Extract all contributing factors from signals
        all_factors = {}
        for s in signal.contributing_signals:
            all_factors.update(s.contributing_factors)

        # === Confidence Score (0-1) ===
        factors["confidence"] = signal.aggregate_confidence

        # === Dip Depth Score (0-1) ===
        # Lower RSI = higher score for BUY signals
        if "RSI_14" in all_factors:
            rsi = all_factors["RSI_14"]
            if signal.signal_type == SignalType.BUY:
                # RSI 20 = score 1.0, RSI 50 = score 0.0
                factors["dip_depth"] = max(0, (50 - rsi) / 30)
            elif signal.signal_type == SignalType.SELL:
                # RSI 80 = score 1.0, RSI 50 = score 0.0
                factors["dip_depth"] = max(0, (rsi - 50) / 30)
            else:
                factors["dip_depth"] = 0.5
        else:
            factors["dip_depth"] = 0.5

        # === Trend Strength Score (0-1) ===
        trend_score = 0.5  # Default neutral

        # Check for trend spread in factors
        if "spread_20_50" in all_factors:
            spread = all_factors["spread_20_50"]
            if signal.signal_type == SignalType.BUY:
                trend_score = min(0.5 + spread / 10, 1.0)
            else:
                trend_score = min(0.5 - spread / 10, 1.0)

        # Check for full alignment
        if all_factors.get("trend_quality") == "strong":
            trend_score = min(trend_score + 0.2, 1.0)
        elif all_factors.get("dip_quality") == "deep":
            trend_score = min(trend_score + 0.1, 1.0)

        factors["trend_strength"] = max(0, trend_score)

        # === Volatility Score (0-1) ===
        # We prefer moderate volatility (not too high, not too low)
        # This would use ATR if available
        # For now, default to neutral
        factors["volatility"] = 0.5

        # === Calculate Weighted Composite Score ===
        composite = sum(
            factors.get(key, 0.5) * weight
            for key, weight in self.weights.items()
        )

        return composite, factors

    def get_recommendation(
        self,
        signals: Dict[str, AggregatedSignal],
    ) -> Optional[str]:
        """
        Get a simple recommendation string.

        Returns something like:
        "BUY WPM (score: 0.85) - Deep dip in strong uptrend"
        """
        # Get best BUY signal
        buy_ranking = self.rank(signals, SignalType.BUY)

        if not buy_ranking.ranked_symbols:
            return None

        best = buy_ranking.ranked_symbols[0]
        return (
            f"BUY {best.symbol} (score: {best.score:.2f}, "
            f"confidence: {best.signal.aggregate_confidence:.2f}) - "
            f"{best.signal.primary_reasoning}"
        )
