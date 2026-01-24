"""
MACD-based trading rules.
"""

from .base import Rule, RuleResult, SignalType, SymbolContext


class MACDBullishCrossoverRule(Rule):
    """
    Natural Language: "Buy signal when MACD crosses above signal line"

    Indicates bullish momentum building.
    """

    def __init__(self, histogram_threshold: float = 0.1):
        # Only trigger on fresh crossovers (small histogram)
        self.histogram_threshold = histogram_threshold

    @property
    def name(self) -> str:
        return "MACD Bullish Crossover"

    @property
    def description(self) -> str:
        return "Buy when MACD line crosses above signal line (bullish momentum)"

    @property
    def required_indicators(self) -> list:
        return ["MACD", "MACD_SIGNAL"]

    def evaluate(self, context: SymbolContext) -> RuleResult:
        macd = context.get_indicator("MACD")
        signal = context.get_indicator("MACD_SIGNAL")
        histogram = context.indicators.get("MACD_HISTOGRAM", macd - signal)

        # MACD above signal indicates bullish
        if macd <= signal:
            return RuleResult(
                triggered=False,
                reasoning=f"MACD ({macd:.3f}) below signal ({signal:.3f})"
            )

        # Check for fresh crossover (histogram small and positive)
        if 0 < histogram < self.histogram_threshold:
            # Fresh crossover - higher confidence
            confidence = min(0.65 + abs(histogram) * 3, 0.85)
            reasoning = f"Fresh bullish crossover: MACD ({macd:.3f}) just crossed above signal ({signal:.3f})"
        elif histogram >= self.histogram_threshold:
            # Already crossed a while ago - lower confidence
            confidence = 0.5
            reasoning = f"MACD ({macd:.3f}) above signal ({signal:.3f}), crossover already occurred"
        else:
            return RuleResult(
                triggered=False,
                reasoning="No bullish crossover detected"
            )

        return RuleResult(
            triggered=True,
            signal=SignalType.BUY,
            confidence=confidence,
            reasoning=reasoning,
            contributing_factors={
                "MACD": macd,
                "MACD_SIGNAL": signal,
                "MACD_HISTOGRAM": histogram,
            }
        )


class MACDBearishCrossoverRule(Rule):
    """
    Natural Language: "Warning when MACD crosses below signal line"

    Indicates bearish momentum - potential exit or caution.
    """

    def __init__(self, histogram_threshold: float = 0.1):
        self.histogram_threshold = histogram_threshold

    @property
    def name(self) -> str:
        return "MACD Bearish Crossover"

    @property
    def description(self) -> str:
        return "Warning when MACD crosses below signal line (bearish momentum)"

    @property
    def required_indicators(self) -> list:
        return ["MACD", "MACD_SIGNAL"]

    def evaluate(self, context: SymbolContext) -> RuleResult:
        macd = context.get_indicator("MACD")
        signal = context.get_indicator("MACD_SIGNAL")
        histogram = context.indicators.get("MACD_HISTOGRAM", macd - signal)

        # MACD below signal indicates bearish
        if macd >= signal:
            return RuleResult(
                triggered=False,
                reasoning=f"MACD ({macd:.3f}) above signal ({signal:.3f})"
            )

        # Check for fresh crossover
        if -self.histogram_threshold < histogram < 0:
            confidence = min(0.6 + abs(histogram) * 3, 0.8)
            reasoning = f"Fresh bearish crossover: MACD ({macd:.3f}) just crossed below signal ({signal:.3f})"
            signal_type = SignalType.WATCH  # Warning, not immediate sell
        elif histogram <= -self.histogram_threshold:
            confidence = 0.5
            reasoning = f"MACD ({macd:.3f}) below signal ({signal:.3f}), bearish momentum confirmed"
            signal_type = SignalType.WATCH
        else:
            return RuleResult(triggered=False, reasoning="No bearish crossover detected")

        return RuleResult(
            triggered=True,
            signal=signal_type,
            confidence=confidence,
            reasoning=reasoning,
            contributing_factors={
                "MACD": macd,
                "MACD_SIGNAL": signal,
                "MACD_HISTOGRAM": histogram,
            }
        )


class MACDMomentumRule(Rule):
    """
    Natural Language: "Confirm momentum when MACD histogram is growing"

    A supporting indicator that confirms trend strength.
    """

    @property
    def name(self) -> str:
        return "MACD Momentum"

    @property
    def description(self) -> str:
        return "Confirm bullish momentum when MACD histogram is positive and growing"

    @property
    def required_indicators(self) -> list:
        return ["MACD", "MACD_SIGNAL"]

    def evaluate(self, context: SymbolContext) -> RuleResult:
        macd = context.get_indicator("MACD")
        signal = context.get_indicator("MACD_SIGNAL")
        histogram = macd - signal

        if histogram > 0.05:
            # Strong positive momentum
            confidence = min(0.4 + histogram * 2, 0.7)
            return RuleResult(
                triggered=True,
                signal=SignalType.BUY,
                confidence=confidence,
                reasoning=f"Strong bullish momentum: MACD histogram at {histogram:.3f}",
                contributing_factors={
                    "MACD": macd,
                    "MACD_SIGNAL": signal,
                    "MACD_HISTOGRAM": histogram,
                }
            )
        elif histogram < -0.05:
            # Strong negative momentum
            confidence = min(0.4 + abs(histogram) * 2, 0.7)
            return RuleResult(
                triggered=True,
                signal=SignalType.WATCH,
                confidence=confidence,
                reasoning=f"Bearish momentum: MACD histogram at {histogram:.3f}",
                contributing_factors={
                    "MACD": macd,
                    "MACD_SIGNAL": signal,
                    "MACD_HISTOGRAM": histogram,
                }
            )

        return RuleResult(
            triggered=False,
            reasoning=f"MACD histogram at {histogram:.3f}, no strong momentum signal"
        )
