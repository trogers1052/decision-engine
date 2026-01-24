"""
Composite rules that combine multiple indicators.

These implement YOUR CORE STRATEGY:
"Buy dips in uptrending stocks, prioritize high win rate over big wins"
"""

from .base import Rule, RuleResult, SignalType, SymbolContext


class BuyDipInUptrendRule(Rule):
    """
    YOUR PRIMARY RULE:
    Natural Language: "Buy when RSI dips to 35-40 AND weekly uptrend is intact"

    This is the bread and butter of your strategy:
    - REQUIRED: SMA_20 > SMA_50 (weekly uptrend)
    - TRIGGER: RSI drops below threshold (dip opportunity)

    Higher confidence when:
    - RSI is lower (deeper dip)
    - Trend is stronger (bigger SMA spread)
    """

    def __init__(self, rsi_threshold: float = 40.0):
        self.rsi_threshold = rsi_threshold

    @property
    def name(self) -> str:
        return "Buy Dip in Uptrend"

    @property
    def description(self) -> str:
        return f"Buy when RSI < {self.rsi_threshold} AND weekly uptrend intact (SMA_20 > SMA_50)"

    @property
    def required_indicators(self) -> list:
        return ["RSI_14", "SMA_20", "SMA_50"]

    def evaluate(self, context: SymbolContext) -> RuleResult:
        rsi = context.get_indicator("RSI_14")
        sma20 = context.get_indicator("SMA_20")
        sma50 = context.get_indicator("SMA_50")

        # First check: Is the weekly uptrend intact?
        if sma20 <= sma50:
            return RuleResult(
                triggered=False,
                reasoning=f"No uptrend: SMA_20 ({sma20:.2f}) not above SMA_50 ({sma50:.2f}). RSI at {rsi:.1f}."
            )

        # Second check: Is there a dip?
        if rsi >= self.rsi_threshold:
            return RuleResult(
                triggered=False,
                reasoning=f"No dip: RSI at {rsi:.1f}, waiting for < {self.rsi_threshold}. Uptrend intact."
            )

        # We have both conditions! Calculate confidence
        trend_spread = (sma20 - sma50) / sma50 * 100 if sma50 > 0 else 0

        # Base confidence from RSI level
        if rsi < 30:
            rsi_confidence = 0.85
            dip_quality = "deep"
        elif rsi < 35:
            rsi_confidence = 0.7
            dip_quality = "solid"
        else:
            rsi_confidence = 0.55
            dip_quality = "shallow"

        # Boost for stronger trend
        if trend_spread >= 2.0:
            trend_bonus = 0.1
            trend_quality = "strong"
        elif trend_spread >= 1.0:
            trend_bonus = 0.05
            trend_quality = "solid"
        else:
            trend_bonus = 0.0
            trend_quality = "weak"

        confidence = min(rsi_confidence + trend_bonus, 0.95)

        return RuleResult(
            triggered=True,
            signal=SignalType.BUY,
            confidence=confidence,
            reasoning=f"BUY DIP: {dip_quality} dip (RSI: {rsi:.1f}) in {trend_quality} uptrend (spread: {trend_spread:.1f}%)",
            contributing_factors={
                "RSI_14": rsi,
                "SMA_20": sma20,
                "SMA_50": sma50,
                "trend_spread_pct": round(trend_spread, 2),
                "dip_quality": dip_quality,
                "trend_quality": trend_quality,
            }
        )


class StrongBuySignalRule(Rule):
    """
    YOUR BEST OPPORTUNITY:
    Natural Language: "Buy when RSI < 35 AND full trend alignment"

    This is the highest confidence signal:
    - RSI deeply oversold (< 35)
    - Full alignment: SMA_20 > SMA_50 > SMA_200

    These are your best entries - prioritize them!
    """

    def __init__(self, rsi_threshold: float = 35.0):
        self.rsi_threshold = rsi_threshold

    @property
    def name(self) -> str:
        return "Strong Buy Signal"

    @property
    def description(self) -> str:
        return f"Buy when RSI < {self.rsi_threshold} AND SMA_20 > SMA_50 > SMA_200"

    @property
    def required_indicators(self) -> list:
        return ["RSI_14", "SMA_20", "SMA_50", "SMA_200"]

    def evaluate(self, context: SymbolContext) -> RuleResult:
        rsi = context.get_indicator("RSI_14")
        sma20 = context.get_indicator("SMA_20")
        sma50 = context.get_indicator("SMA_50")
        sma200 = context.get_indicator("SMA_200")

        # Check full trend alignment
        if not (sma20 > sma50 > sma200):
            if sma20 > sma50:
                missing = "monthly trend (SMA_50 < SMA_200)"
            else:
                missing = "weekly trend (SMA_20 < SMA_50)"
            return RuleResult(
                triggered=False,
                reasoning=f"Missing {missing}. RSI at {rsi:.1f}."
            )

        # Check for deep dip
        if rsi >= self.rsi_threshold:
            return RuleResult(
                triggered=False,
                reasoning=f"RSI at {rsi:.1f}, waiting for < {self.rsi_threshold}. Full alignment present."
            )

        # Calculate confidence
        spread_20_50 = (sma20 - sma50) / sma50 * 100 if sma50 > 0 else 0
        spread_50_200 = (sma50 - sma200) / sma200 * 100 if sma200 > 0 else 0

        # Deep dip + full alignment = high confidence
        if rsi < 25:
            base_confidence = 0.9
        elif rsi < 30:
            base_confidence = 0.8
        else:
            base_confidence = 0.7

        # Trend strength bonus
        total_spread = spread_20_50 + spread_50_200
        trend_bonus = min(total_spread / 50, 0.1)

        confidence = min(base_confidence + trend_bonus, 0.98)

        return RuleResult(
            triggered=True,
            signal=SignalType.BUY,
            confidence=confidence,
            reasoning=f"STRONG BUY: Deep dip (RSI: {rsi:.1f}) with full trend alignment. Best opportunity!",
            contributing_factors={
                "RSI_14": rsi,
                "SMA_20": sma20,
                "SMA_50": sma50,
                "SMA_200": sma200,
                "spread_20_50": round(spread_20_50, 2),
                "spread_50_200": round(spread_50_200, 2),
            }
        )


class RSIAndMACDConfluenceRule(Rule):
    """
    Natural Language: "Buy when RSI oversold AND MACD showing bullish momentum"

    High-confidence entry when multiple indicators agree.
    """

    def __init__(self, rsi_threshold: float = 35.0):
        self.rsi_threshold = rsi_threshold

    @property
    def name(self) -> str:
        return "RSI + MACD Confluence"

    @property
    def description(self) -> str:
        return f"Buy when RSI < {self.rsi_threshold} AND MACD bullish"

    @property
    def required_indicators(self) -> list:
        return ["RSI_14", "MACD", "MACD_SIGNAL"]

    def evaluate(self, context: SymbolContext) -> RuleResult:
        rsi = context.get_indicator("RSI_14")
        macd = context.get_indicator("MACD")
        signal = context.get_indicator("MACD_SIGNAL")

        rsi_oversold = rsi < self.rsi_threshold
        macd_bullish = macd > signal

        if not rsi_oversold:
            return RuleResult(
                triggered=False,
                reasoning=f"RSI at {rsi:.1f}, not oversold. MACD bullish: {macd_bullish}"
            )

        if not macd_bullish:
            return RuleResult(
                triggered=False,
                reasoning=f"RSI oversold ({rsi:.1f}) but MACD bearish. Waiting for confirmation."
            )

        # Both conditions met
        base_confidence = 0.7

        # More oversold = higher confidence
        if rsi < 30:
            base_confidence += 0.15
        elif rsi < 33:
            base_confidence += 0.1

        # Stronger MACD = higher confidence
        histogram = macd - signal
        if histogram > 0.05:
            base_confidence += 0.05

        return RuleResult(
            triggered=True,
            signal=SignalType.BUY,
            confidence=min(base_confidence, 0.95),
            reasoning=f"Confluence: RSI oversold ({rsi:.1f}) with bullish MACD (histogram: {histogram:.3f})",
            contributing_factors={
                "RSI_14": rsi,
                "MACD": macd,
                "MACD_SIGNAL": signal,
                "MACD_HISTOGRAM": histogram,
            }
        )


class TrendDipRecoveryRule(Rule):
    """
    Natural Language: "Buy when RSI was oversold and is now recovering, in uptrend"

    Catches the early part of the bounce after a dip.
    RSI between 30-45, heading up, with uptrend intact.
    """

    def __init__(self, recovery_zone_low: float = 30.0, recovery_zone_high: float = 45.0):
        self.recovery_zone_low = recovery_zone_low
        self.recovery_zone_high = recovery_zone_high

    @property
    def name(self) -> str:
        return "Dip Recovery"

    @property
    def description(self) -> str:
        return f"Buy when RSI recovering ({self.recovery_zone_low}-{self.recovery_zone_high}) with uptrend intact"

    @property
    def required_indicators(self) -> list:
        return ["RSI_14", "SMA_20", "SMA_50"]

    def evaluate(self, context: SymbolContext) -> RuleResult:
        rsi = context.get_indicator("RSI_14")
        sma20 = context.get_indicator("SMA_20")
        sma50 = context.get_indicator("SMA_50")

        # Check uptrend
        if sma20 <= sma50:
            return RuleResult(
                triggered=False,
                reasoning=f"No uptrend for recovery. RSI at {rsi:.1f}."
            )

        # Check if in recovery zone
        if not (self.recovery_zone_low <= rsi <= self.recovery_zone_high):
            if rsi < self.recovery_zone_low:
                return RuleResult(
                    triggered=False,
                    reasoning=f"RSI at {rsi:.1f}, still deeply oversold. Waiting for recovery."
                )
            return RuleResult(
                triggered=False,
                reasoning=f"RSI at {rsi:.1f}, above recovery zone."
            )

        # In recovery zone with uptrend
        confidence = 0.55 + (self.recovery_zone_high - rsi) / 30  # Higher confidence closer to oversold

        return RuleResult(
            triggered=True,
            signal=SignalType.BUY,
            confidence=min(confidence, 0.75),
            reasoning=f"Dip recovery: RSI at {rsi:.1f} (recovering) with uptrend intact",
            contributing_factors={
                "RSI_14": rsi,
                "SMA_20": sma20,
                "SMA_50": sma50,
                "recovery_zone": f"{self.recovery_zone_low}-{self.recovery_zone_high}",
            }
        )
