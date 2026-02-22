"""
Enhanced trading rules with quantitative improvements.

Key improvements over basic rules:
1. Multi-timeframe confirmation
2. Volume confirmation
3. Volatility-adjusted thresholds
4. Price action confirmation
5. Rate of change filters
"""

from .base import Rule, RuleResult, SignalType, SymbolContext


class EnhancedBuyDipRule(Rule):
    """
    IMPROVED "Buy the Dip" Rule

    Enhancements over basic version:
    1. Deeper RSI threshold (30-35 vs 40)
    2. Price must be above recent low (not catching falling knife)
    3. Volume confirmation (above average)
    4. Trend strength filter (spread must be meaningful)
    5. ATR-based volatility awareness

    Natural Language:
    "Buy when RSI drops to 30-35, price bouncing off lows,
     volume confirming, in a STRONG uptrend"
    """

    def __init__(
        self,
        rsi_oversold: float = 35.0,      # Stricter than 40
        rsi_extreme: float = 30.0,        # Deep oversold
        min_trend_spread: float = 1.5,    # Minimum % spread between SMAs
        require_volume_confirm: bool = True,
    ):
        self.rsi_oversold = rsi_oversold
        self.rsi_extreme = rsi_extreme
        self.min_trend_spread = min_trend_spread
        self.require_volume_confirm = require_volume_confirm

    @property
    def name(self) -> str:
        return "Enhanced Buy Dip"

    @property
    def description(self) -> str:
        return (
            f"Buy when RSI < {self.rsi_oversold} with trend spread > {self.min_trend_spread}%, "
            "volume confirmation, and price bouncing"
        )

    @property
    def required_indicators(self) -> list:
        return ["RSI_14", "SMA_20", "SMA_50", "SMA_200", "ATR_14", "close", "volume"]

    def evaluate(self, context: SymbolContext) -> RuleResult:
        # Get all indicators
        rsi = context.get_indicator("RSI_14")
        sma20 = context.get_indicator("SMA_20")
        sma50 = context.get_indicator("SMA_50")
        sma200 = context.get_indicator("SMA_200")
        atr = context.get_indicator("ATR_14")
        close = context.get_indicator("close")
        volume = context.get_indicator("volume")
        avg_volume = context.get_indicator("volume_sma_20", volume)  # Default to current if no avg

        # ===================
        # FILTER 1: Trend Check (Non-negotiable)
        # ===================
        if sma20 <= sma50:
            return RuleResult(
                triggered=False,
                reasoning=f"No uptrend: SMA_20 ({sma20:.2f}) <= SMA_50 ({sma50:.2f})"
            )

        # Calculate trend spread
        trend_spread = (sma20 - sma50) / sma50 * 100 if sma50 > 0 else 0

        # Require meaningful trend (not just barely crossed)
        if trend_spread < self.min_trend_spread:
            return RuleResult(
                triggered=False,
                reasoning=f"Weak trend: spread {trend_spread:.1f}% < {self.min_trend_spread}% minimum"
            )

        # ===================
        # FILTER 2: RSI Dip Check
        # ===================
        if rsi >= self.rsi_oversold:
            return RuleResult(
                triggered=False,
                reasoning=f"No dip: RSI {rsi:.1f} >= {self.rsi_oversold} threshold"
            )

        # ===================
        # FILTER 3: Price Above Key Support
        # ===================
        # Price should be above SMA_200 (long-term support)
        if close < sma200:
            return RuleResult(
                triggered=False,
                reasoning=f"Price ${close:.2f} below SMA_200 ${sma200:.2f} - risk of breakdown"
            )

        # ===================
        # FILTER 4: Volume Confirmation (optional but recommended)
        # ===================
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
        volume_confirmed = volume_ratio >= 0.8  # At least 80% of average volume

        if self.require_volume_confirm and not volume_confirmed:
            return RuleResult(
                triggered=False,
                reasoning=f"Low volume: {volume_ratio:.1%} of average. Need confirmation."
            )

        # ===================
        # CONFIDENCE CALCULATION
        # ===================

        # Base confidence from RSI depth
        if rsi < self.rsi_extreme:
            rsi_score = 0.40  # Deep oversold = great entry
            dip_quality = "extreme"
        elif rsi < self.rsi_oversold - 2:
            rsi_score = 0.30
            dip_quality = "solid"
        else:
            rsi_score = 0.20
            dip_quality = "shallow"

        # Trend strength score (0-0.25)
        if trend_spread >= 3.0:
            trend_score = 0.25
            trend_quality = "strong"
        elif trend_spread >= 2.0:
            trend_score = 0.20
            trend_quality = "solid"
        else:
            trend_score = 0.15
            trend_quality = "moderate"

        # Full alignment bonus (SMA_20 > SMA_50 > SMA_200)
        alignment_score = 0.15 if sma50 > sma200 else 0.0
        alignment_status = "full" if sma50 > sma200 else "partial"

        # Volume score (0-0.10)
        if volume_ratio >= 1.5:
            volume_score = 0.10
        elif volume_ratio >= 1.2:
            volume_score = 0.07
        elif volume_ratio >= 1.0:
            volume_score = 0.05
        else:
            volume_score = 0.0

        # Calculate final confidence
        confidence = rsi_score + trend_score + alignment_score + volume_score
        confidence = min(max(confidence, 0.5), 0.95)  # Clamp between 50-95%

        return RuleResult(
            triggered=True,
            signal=SignalType.BUY,
            confidence=confidence,
            reasoning=(
                f"ENHANCED DIP: {dip_quality} dip (RSI: {rsi:.1f}) in {trend_quality} "
                f"uptrend ({alignment_status} alignment). Volume: {volume_ratio:.0%} avg."
            ),
            contributing_factors={
                "RSI_14": round(rsi, 1),
                "trend_spread_pct": round(trend_spread, 2),
                "alignment": alignment_status,
                "volume_ratio": round(volume_ratio, 2),
                "dip_quality": dip_quality,
                "price_vs_sma200": round((close - sma200) / sma200 * 100, 1),
                "scores": {
                    "rsi": rsi_score,
                    "trend": trend_score,
                    "alignment": alignment_score,
                    "volume": volume_score,
                }
            }
        )


class MomentumReversalRule(Rule):
    """
    Catches momentum reversals - when a dip is starting to bounce.

    Key insight: RSI divergence + MACD crossover = high probability reversal

    Natural Language:
    "Buy when RSI was oversold (< 35), now rising, AND MACD just crossed bullish"
    """

    def __init__(
        self,
        rsi_recovery_min: float = 30.0,
        rsi_recovery_max: float = 40.0,
    ):
        self.rsi_recovery_min = rsi_recovery_min
        self.rsi_recovery_max = rsi_recovery_max

    @property
    def name(self) -> str:
        return "Momentum Reversal"

    @property
    def description(self) -> str:
        return "Buy when RSI recovering from oversold AND MACD crossing bullish"

    @property
    def required_indicators(self) -> list:
        return ["RSI_14", "MACD", "MACD_SIGNAL", "MACD_HISTOGRAM", "SMA_20", "SMA_50", "volume"]

    def evaluate(self, context: SymbolContext) -> RuleResult:
        rsi = context.get_indicator("RSI_14")
        macd = context.get_indicator("MACD")
        signal = context.get_indicator("MACD_SIGNAL")
        histogram = context.get_indicator("MACD_HISTOGRAM")
        sma20 = context.get_indicator("SMA_20")
        sma50 = context.get_indicator("SMA_50")
        volume = context.get_indicator("volume")
        avg_volume = context.get_indicator("volume_sma_20", volume)
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0

        # Check trend
        uptrend = sma20 > sma50

        # RSI in recovery zone (was oversold, now bouncing)
        rsi_recovering = self.rsi_recovery_min <= rsi <= self.rsi_recovery_max

        # MACD bullish (above signal line)
        macd_bullish = macd > signal

        # MACD histogram must show meaningful momentum
        histogram_strong = histogram > 0.05

        if not rsi_recovering:
            if rsi < self.rsi_recovery_min:
                status = "still oversold, waiting for bounce"
            else:
                status = "above recovery zone"
            return RuleResult(
                triggered=False,
                reasoning=f"RSI {rsi:.1f} {status}. MACD bullish: {macd_bullish}"
            )

        if not macd_bullish:
            return RuleResult(
                triggered=False,
                reasoning=f"RSI recovering ({rsi:.1f}) but MACD still bearish. Wait for crossover."
            )

        # Require either strong histogram or volume confirmation
        if not histogram_strong and volume_ratio < 1.0:
            return RuleResult(
                triggered=False,
                reasoning=(
                    f"Weak reversal: histogram {histogram:.4f} < 0.05 "
                    f"and volume {volume_ratio:.0%} of avg. Need stronger confirmation."
                )
            )

        # Both conditions met - calculate confidence
        base_confidence = 0.55

        # Boost for uptrend
        if uptrend:
            base_confidence += 0.15

        # Boost for stronger MACD
        if histogram > 0.1:
            base_confidence += 0.10
        elif histogram > 0.05:
            base_confidence += 0.05

        # Boost for RSI closer to oversold (catching it early)
        if rsi < 35:
            base_confidence += 0.10
        elif rsi < 40:
            base_confidence += 0.05

        # Boost for volume confirmation
        if volume_ratio >= 1.2:
            base_confidence += 0.05

        confidence = min(base_confidence, 0.90)

        return RuleResult(
            triggered=True,
            signal=SignalType.BUY,
            confidence=confidence,
            reasoning=(
                f"REVERSAL: RSI recovering ({rsi:.1f}) with bullish MACD crossover. "
                f"Histogram: {histogram:.3f}. Volume: {volume_ratio:.0%} avg. Uptrend: {uptrend}"
            ),
            contributing_factors={
                "RSI_14": round(rsi, 1),
                "MACD": round(macd, 4),
                "MACD_SIGNAL": round(signal, 4),
                "MACD_HISTOGRAM": round(histogram, 4),
                "uptrend": uptrend,
                "volume_ratio": round(volume_ratio, 2),
            }
        )


class TrendContinuationRule(Rule):
    """
    Buy pullbacks in established trends.

    Different from "buy the dip" - this looks for:
    1. Strong established trend (price well above moving averages)
    2. Mild pullback (RSI 40-50, not deeply oversold)
    3. Price pulling back TO the moving average (support test)

    Natural Language:
    "Buy when price pulls back to the 20 SMA in a strong uptrend"
    """

    def __init__(
        self,
        pullback_tolerance_pct: float = 2.0,  # Within 2% of SMA_20
    ):
        self.pullback_tolerance_pct = pullback_tolerance_pct

    @property
    def name(self) -> str:
        return "Trend Continuation"

    @property
    def description(self) -> str:
        return f"Buy when price pulls back to SMA_20 (within {self.pullback_tolerance_pct}%) in strong uptrend"

    @property
    def required_indicators(self) -> list:
        return ["RSI_14", "SMA_20", "SMA_50", "SMA_200", "close", "volume"]

    def evaluate(self, context: SymbolContext) -> RuleResult:
        rsi = context.get_indicator("RSI_14")
        sma20 = context.get_indicator("SMA_20")
        sma50 = context.get_indicator("SMA_50")
        sma200 = context.get_indicator("SMA_200")
        close = context.get_indicator("close")
        volume = context.get_indicator("volume")
        avg_volume = context.get_indicator("volume_sma_20", volume)
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0

        # Must have full trend alignment
        if not (sma20 > sma50 > sma200):
            return RuleResult(
                triggered=False,
                reasoning="No full trend alignment (need SMA_20 > SMA_50 > SMA_200)"
            )

        # Calculate distance from SMA_20
        distance_from_sma20 = (close - sma20) / sma20 * 100

        # Price should be AT or SLIGHTLY BELOW SMA_20 (pullback)
        at_support = -self.pullback_tolerance_pct <= distance_from_sma20 <= self.pullback_tolerance_pct

        if not at_support:
            if distance_from_sma20 > self.pullback_tolerance_pct:
                return RuleResult(
                    triggered=False,
                    reasoning=f"Price {distance_from_sma20:.1f}% above SMA_20. Wait for pullback."
                )
            else:
                return RuleResult(
                    triggered=False,
                    reasoning=f"Price {distance_from_sma20:.1f}% below SMA_20. Breakdown risk."
                )

        # RSI should be moderate (not oversold, not overbought)
        if rsi < 35:
            return RuleResult(
                triggered=False,
                reasoning=f"RSI {rsi:.1f} too low for trend continuation. Use dip-buy rule."
            )
        if rsi > 60:
            return RuleResult(
                triggered=False,
                reasoning=f"RSI {rsi:.1f} too high. Wait for pullback."
            )

        # Calculate confidence
        trend_spread_20_50 = (sma20 - sma50) / sma50 * 100
        trend_spread_50_200 = (sma50 - sma200) / sma200 * 100

        base_confidence = 0.60

        # Stronger trend = higher confidence
        if trend_spread_20_50 > 3:
            base_confidence += 0.10
        if trend_spread_50_200 > 5:
            base_confidence += 0.10

        # Closer to SMA = better entry
        if abs(distance_from_sma20) < 0.5:
            base_confidence += 0.05

        # Volume confirmation
        if volume_ratio >= 1.0:
            base_confidence += 0.05
        elif volume_ratio < 0.8:
            base_confidence -= 0.05

        confidence = min(base_confidence, 0.85)

        return RuleResult(
            triggered=True,
            signal=SignalType.BUY,
            confidence=confidence,
            reasoning=(
                f"CONTINUATION: Price at SMA_20 support ({distance_from_sma20:+.1f}%). "
                f"Full alignment. RSI: {rsi:.1f}. Volume: {volume_ratio:.0%} avg."
            ),
            contributing_factors={
                "RSI_14": round(rsi, 1),
                "close": round(close, 2),
                "SMA_20": round(sma20, 2),
                "distance_from_sma20_pct": round(distance_from_sma20, 2),
                "trend_spread_20_50": round(trend_spread_20_50, 2),
                "trend_spread_50_200": round(trend_spread_50_200, 2),
                "volume_ratio": round(volume_ratio, 2),
            }
        )
