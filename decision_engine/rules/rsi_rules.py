"""
RSI-based trading rules.
"""

from .base import Rule, RuleResult, SignalType, SymbolContext


class RSIOversoldRule(Rule):
    """
    Natural Language: "Buy when RSI drops below 30"

    This rule identifies oversold conditions where the stock may be
    due for a bounce. Confidence increases as RSI gets lower.
    """

    def __init__(self, threshold: float = 30.0, extreme_threshold: float = 20.0):
        self.threshold = threshold
        self.extreme_threshold = extreme_threshold

    @property
    def name(self) -> str:
        return "RSI Oversold"

    @property
    def description(self) -> str:
        return f"Buy when RSI drops below {self.threshold} (oversold condition)"

    @property
    def required_indicators(self) -> list:
        return ["RSI_14"]

    def evaluate(self, context: SymbolContext) -> RuleResult:
        rsi = context.get_indicator("RSI_14")

        if rsi >= self.threshold:
            return RuleResult(
                triggered=False,
                reasoning=f"RSI at {rsi:.1f}, above threshold {self.threshold}"
            )

        # Calculate confidence based on how oversold
        # More oversold = higher confidence
        if rsi <= self.extreme_threshold:
            confidence = 0.9
            reasoning = f"RSI extremely oversold at {rsi:.1f}"
        else:
            # Linear scale from threshold to extreme
            range_size = self.threshold - self.extreme_threshold
            distance = self.threshold - rsi
            confidence = 0.5 + 0.4 * (distance / range_size) if range_size > 0 else 0.7
            reasoning = f"RSI oversold at {rsi:.1f}"

        return RuleResult(
            triggered=True,
            signal=SignalType.BUY,
            confidence=min(confidence, 1.0),
            reasoning=reasoning,
            contributing_factors={"RSI_14": rsi}
        )


class RSIOverboughtRule(Rule):
    """
    Natural Language: "Watch/Sell when RSI rises above 70"

    This identifies overbought conditions - potential exit points.
    """

    def __init__(self, threshold: float = 70.0, extreme_threshold: float = 80.0):
        self.threshold = threshold
        self.extreme_threshold = extreme_threshold

    @property
    def name(self) -> str:
        return "RSI Overbought"

    @property
    def description(self) -> str:
        return f"Watch/Sell when RSI rises above {self.threshold} (overbought)"

    @property
    def required_indicators(self) -> list:
        return ["RSI_14"]

    def evaluate(self, context: SymbolContext) -> RuleResult:
        rsi = context.get_indicator("RSI_14")

        if rsi <= self.threshold:
            return RuleResult(
                triggered=False,
                reasoning=f"RSI at {rsi:.1f}, below threshold {self.threshold}"
            )

        if rsi >= self.extreme_threshold:
            confidence = 0.85
            reasoning = f"RSI extremely overbought at {rsi:.1f} - consider selling"
            signal = SignalType.SELL
        else:
            range_size = self.extreme_threshold - self.threshold
            distance = rsi - self.threshold
            confidence = 0.4 + 0.3 * (distance / range_size) if range_size > 0 else 0.5
            reasoning = f"RSI overbought at {rsi:.1f} - watch closely"
            signal = SignalType.WATCH

        return RuleResult(
            triggered=True,
            signal=signal,
            confidence=min(confidence, 1.0),
            reasoning=reasoning,
            contributing_factors={"RSI_14": rsi}
        )


class RSIApproachingOversoldRule(Rule):
    """
    Natural Language: "Watch when RSI approaches 35"

    WATCH signal - interesting but not yet actionable.
    Good for setting up alerts.
    """

    def __init__(self, watch_threshold: float = 40.0, buy_threshold: float = 30.0):
        self.watch_threshold = watch_threshold
        self.buy_threshold = buy_threshold

    @property
    def name(self) -> str:
        return "RSI Approaching Oversold"

    @property
    def description(self) -> str:
        return f"Watch when RSI between {self.buy_threshold} and {self.watch_threshold}"

    @property
    def required_indicators(self) -> list:
        return ["RSI_14"]

    def evaluate(self, context: SymbolContext) -> RuleResult:
        rsi = context.get_indicator("RSI_14")

        if self.buy_threshold <= rsi <= self.watch_threshold:
            return RuleResult(
                triggered=True,
                signal=SignalType.WATCH,
                confidence=0.4,
                reasoning=f"RSI at {rsi:.1f}, approaching oversold territory",
                contributing_factors={"RSI_14": rsi}
            )

        return RuleResult(
            triggered=False,
            reasoning=f"RSI at {rsi:.1f}, not in watch zone"
        )
