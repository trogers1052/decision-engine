"""
Trend-based trading rules using SMA indicators.

These rules implement your core requirement:
- Weekly uptrend: SMA_20 > SMA_50
- Monthly uptrend: SMA_50 > SMA_200
"""

from .base import Rule, RuleResult, SignalType, SymbolContext


class WeeklyUptrendRule(Rule):
    """
    Natural Language: "Stock must be trending up over past week"

    This is YOUR REQUIRED condition - SMA_20 > SMA_50 indicates
    short-term momentum is positive.
    """

    @property
    def name(self) -> str:
        return "Weekly Uptrend"

    @property
    def description(self) -> str:
        return "Stock trending up over past week (SMA_20 > SMA_50)"

    @property
    def required_indicators(self) -> list:
        return ["SMA_20", "SMA_50"]

    def evaluate(self, context: SymbolContext) -> RuleResult:
        sma20 = context.get_indicator("SMA_20")
        sma50 = context.get_indicator("SMA_50")

        if sma20 <= sma50:
            spread_pct = (sma50 - sma20) / sma50 * 100 if sma50 > 0 else 0
            return RuleResult(
                triggered=False,
                reasoning=f"No weekly uptrend: SMA_20 ({sma20:.2f}) below SMA_50 ({sma50:.2f}), spread: -{spread_pct:.2f}%"
            )

        # Calculate trend strength based on spread
        spread_pct = (sma20 - sma50) / sma50 * 100 if sma50 > 0 else 0

        # Stronger spread = higher confidence
        if spread_pct >= 2.0:
            confidence = 0.85
            reasoning = f"Strong weekly uptrend: SMA_20 ({sma20:.2f}) well above SMA_50 ({sma50:.2f})"
        elif spread_pct >= 1.0:
            confidence = 0.7
            reasoning = f"Solid weekly uptrend: SMA_20 ({sma20:.2f}) above SMA_50 ({sma50:.2f})"
        else:
            confidence = 0.55
            reasoning = f"Weak weekly uptrend: SMA_20 ({sma20:.2f}) slightly above SMA_50 ({sma50:.2f})"

        return RuleResult(
            triggered=True,
            signal=SignalType.WATCH,  # Trend confirmation, not a buy signal alone
            confidence=confidence,
            reasoning=reasoning,
            contributing_factors={
                "SMA_20": sma20,
                "SMA_50": sma50,
                "spread_pct": round(spread_pct, 2),
            }
        )


class MonthlyUptrendRule(Rule):
    """
    Natural Language: "Stock trending up over past month (preferred, not required)"

    SMA_50 > SMA_200 indicates longer-term bullish trend.
    """

    @property
    def name(self) -> str:
        return "Monthly Uptrend"

    @property
    def description(self) -> str:
        return "Stock trending up over past month (SMA_50 > SMA_200)"

    @property
    def required_indicators(self) -> list:
        return ["SMA_50", "SMA_200"]

    def evaluate(self, context: SymbolContext) -> RuleResult:
        sma50 = context.get_indicator("SMA_50")
        sma200 = context.get_indicator("SMA_200")

        if sma50 <= sma200:
            spread_pct = (sma200 - sma50) / sma200 * 100 if sma200 > 0 else 0
            return RuleResult(
                triggered=False,
                reasoning=f"No monthly uptrend: SMA_50 ({sma50:.2f}) below SMA_200 ({sma200:.2f})"
            )

        spread_pct = (sma50 - sma200) / sma200 * 100 if sma200 > 0 else 0

        if spread_pct >= 5.0:
            confidence = 0.85
        elif spread_pct >= 2.0:
            confidence = 0.7
        else:
            confidence = 0.55

        return RuleResult(
            triggered=True,
            signal=SignalType.WATCH,
            confidence=confidence,
            reasoning=f"Monthly uptrend: SMA_50 ({sma50:.2f}) above SMA_200 ({sma200:.2f}), spread: {spread_pct:.2f}%",
            contributing_factors={
                "SMA_50": sma50,
                "SMA_200": sma200,
                "spread_pct": round(spread_pct, 2),
            }
        )


class FullTrendAlignmentRule(Rule):
    """
    Natural Language: "Strongest uptrend when SMA_20 > SMA_50 > SMA_200"

    Full alignment means all timeframes agree - very bullish.
    """

    @property
    def name(self) -> str:
        return "Full Trend Alignment"

    @property
    def description(self) -> str:
        return "All SMAs aligned bullishly: SMA_20 > SMA_50 > SMA_200"

    @property
    def required_indicators(self) -> list:
        return ["SMA_20", "SMA_50", "SMA_200"]

    def evaluate(self, context: SymbolContext) -> RuleResult:
        sma20 = context.get_indicator("SMA_20")
        sma50 = context.get_indicator("SMA_50")
        sma200 = context.get_indicator("SMA_200")

        if not (sma20 > sma50 > sma200):
            # Not fully aligned
            if sma20 > sma50:
                return RuleResult(
                    triggered=False,
                    reasoning=f"Partial alignment: weekly uptrend but SMA_50 ({sma50:.2f}) below SMA_200 ({sma200:.2f})"
                )
            return RuleResult(
                triggered=False,
                reasoning=f"No trend alignment: SMA_20 ({sma20:.2f}), SMA_50 ({sma50:.2f}), SMA_200 ({sma200:.2f})"
            )

        # Calculate overall trend strength
        spread_20_50 = (sma20 - sma50) / sma50 * 100 if sma50 > 0 else 0
        spread_50_200 = (sma50 - sma200) / sma200 * 100 if sma200 > 0 else 0
        total_spread = spread_20_50 + spread_50_200

        # More spread = stronger trend = higher confidence
        confidence = min(0.6 + total_spread / 30, 0.95)

        return RuleResult(
            triggered=True,
            signal=SignalType.BUY,
            confidence=confidence,
            reasoning=f"Full bullish alignment: SMA_20 ({sma20:.2f}) > SMA_50 ({sma50:.2f}) > SMA_200 ({sma200:.2f})",
            contributing_factors={
                "SMA_20": sma20,
                "SMA_50": sma50,
                "SMA_200": sma200,
                "spread_20_50": round(spread_20_50, 2),
                "spread_50_200": round(spread_50_200, 2),
            }
        )


class TrendBreakWarningRule(Rule):
    """
    Natural Language: "Alert when SMA_20 crosses below SMA_50 - uptrend ending"

    This is an EXIT signal - the weekly uptrend is breaking down.
    """

    @property
    def name(self) -> str:
        return "Trend Break Warning"

    @property
    def description(self) -> str:
        return "Warning: SMA_20 crossed below SMA_50 - uptrend may be ending"

    @property
    def required_indicators(self) -> list:
        return ["SMA_20", "SMA_50"]

    def evaluate(self, context: SymbolContext) -> RuleResult:
        sma20 = context.get_indicator("SMA_20")
        sma50 = context.get_indicator("SMA_50")

        if sma20 >= sma50:
            return RuleResult(
                triggered=False,
                reasoning=f"No trend break: SMA_20 ({sma20:.2f}) still above SMA_50 ({sma50:.2f})"
            )

        # SMA_20 is below SMA_50 - trend breaking
        spread_pct = (sma50 - sma20) / sma50 * 100 if sma50 > 0 else 0

        if spread_pct < 0.5:
            # Just crossed - fresh warning
            confidence = 0.7
            reasoning = f"Fresh trend break: SMA_20 ({sma20:.2f}) just crossed below SMA_50 ({sma50:.2f})"
        else:
            # Already broken for a while
            confidence = 0.6
            reasoning = f"Trend broken: SMA_20 ({sma20:.2f}) below SMA_50 ({sma50:.2f}) by {spread_pct:.2f}%"

        return RuleResult(
            triggered=True,
            signal=SignalType.SELL,
            confidence=confidence,
            reasoning=reasoning,
            contributing_factors={
                "SMA_20": sma20,
                "SMA_50": sma50,
                "spread_pct": round(-spread_pct, 2),
            }
        )


class GoldenCrossRule(Rule):
    """
    Natural Language: "Buy when 50-day SMA crosses above 200-day SMA"

    Classic long-term bullish signal.
    """

    @property
    def name(self) -> str:
        return "Golden Cross"

    @property
    def description(self) -> str:
        return "Buy when SMA_50 crosses above SMA_200 (long-term bullish)"

    @property
    def required_indicators(self) -> list:
        return ["SMA_50", "SMA_200"]

    def evaluate(self, context: SymbolContext) -> RuleResult:
        sma50 = context.get_indicator("SMA_50")
        sma200 = context.get_indicator("SMA_200")

        if sma50 <= sma200:
            return RuleResult(
                triggered=False,
                reasoning=f"No golden cross: SMA_50 ({sma50:.2f}) below SMA_200 ({sma200:.2f})"
            )

        spread_pct = (sma50 - sma200) / sma200 * 100 if sma200 > 0 else 0

        # Fresh cross = small spread
        if spread_pct < 1.0:
            confidence = 0.75
            reasoning = f"Golden cross forming: SMA_50 ({sma50:.2f}) just crossed above SMA_200 ({sma200:.2f})"
        else:
            confidence = 0.5
            reasoning = f"In golden cross territory: SMA_50 ({sma50:.2f}) above SMA_200 ({sma200:.2f})"

        return RuleResult(
            triggered=True,
            signal=SignalType.BUY,
            confidence=confidence,
            reasoning=reasoning,
            contributing_factors={
                "SMA_50": sma50,
                "SMA_200": sma200,
                "spread_pct": round(spread_pct, 2),
            }
        )


class DeathCrossRule(Rule):
    """
    Natural Language: "Warning when 50-day SMA crosses below 200-day SMA"

    Classic long-term bearish signal.
    """

    @property
    def name(self) -> str:
        return "Death Cross"

    @property
    def description(self) -> str:
        return "Warning when SMA_50 crosses below SMA_200 (long-term bearish)"

    @property
    def required_indicators(self) -> list:
        return ["SMA_50", "SMA_200"]

    def evaluate(self, context: SymbolContext) -> RuleResult:
        sma50 = context.get_indicator("SMA_50")
        sma200 = context.get_indicator("SMA_200")

        if sma50 >= sma200:
            return RuleResult(
                triggered=False,
                reasoning=f"No death cross: SMA_50 ({sma50:.2f}) above SMA_200 ({sma200:.2f})"
            )

        spread_pct = (sma200 - sma50) / sma200 * 100 if sma200 > 0 else 0

        if spread_pct < 1.0:
            confidence = 0.75
            reasoning = f"Death cross forming: SMA_50 ({sma50:.2f}) just crossed below SMA_200 ({sma200:.2f})"
        else:
            confidence = 0.6
            reasoning = f"In death cross territory: SMA_50 ({sma50:.2f}) below SMA_200 ({sma200:.2f})"

        return RuleResult(
            triggered=True,
            signal=SignalType.SELL,
            confidence=confidence,
            reasoning=reasoning,
            contributing_factors={
                "SMA_50": sma50,
                "SMA_200": sma200,
                "spread_pct": round(-spread_pct, 2),
            }
        )
