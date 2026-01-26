"""
Mining Stock Specific Rules

Specialized rules for trading mining stocks (gold, silver, uranium, rare earth, etc.)
based on well-known sector dynamics.

Key Insights:
1. Miners are leveraged plays on commodity prices (2-3x moves)
2. Inverse correlation with USD strength
3. Strong seasonality patterns
4. Volume confirmation is critical for breakouts
5. Mean reversion when miners lag the underlying metal
"""

from datetime import datetime
from typing import Optional

from .base import Rule, RuleResult, SignalType, SymbolContext


# =============================================================================
# Symbol to Commodity Mapping
# =============================================================================

# Map mining stocks to their underlying commodities
MINER_COMMODITY_MAP = {
    # Gold miners
    "GDX": "gold",
    "GDXJ": "gold",
    "GLD": "gold",
    "GOLD": "gold",
    "NEM": "gold",
    "RGLD": "gold",
    "WPM": "gold",  # Also silver
    "FNV": "gold",

    # Silver miners
    "SLV": "silver",
    "SIL": "silver",
    "SILJ": "silver",
    "AG": "silver",
    "PAAS": "silver",
    "HL": "silver",      # Hecla Mining
    "MAG": "silver",     # MAG Silver
    "CDE": "silver",     # Coeur Mining
    "EXK": "silver",     # Endeavour Silver
    "FSM": "silver",     # Fortuna Silver

    # Uranium miners
    "CCJ": "uranium",
    "URA": "uranium",
    "URNM": "uranium",
    "UUUU": "uranium",
    "DNN": "uranium",
    "UEC": "uranium",
    "NXE": "uranium",

    # Platinum/Palladium
    "PPLT": "platinum",
    "PALL": "palladium",

    # Copper miners
    "COPX": "copper",
    "FCX": "copper",
    "SCCO": "copper",

    # Rare earth / Other
    "MP": "rare_earth",
    "IAUM": "gold",  # iShares Gold Micro
    "CAT": "industrial",  # Mining equipment
    "ETN": "industrial",
    "AVAV": "industrial",
}

# Seasonal strength months for different commodities
SEASONAL_STRENGTH = {
    "gold": [1, 2, 8, 9, 11, 12],      # Strong: Jan-Feb, Aug-Sep, Nov-Dec
    "silver": [1, 2, 7, 8, 9],          # Strong: Jan-Feb, Jul-Sep
    "uranium": [1, 2, 3, 9, 10, 11],    # Strong: Q1 and Q4
    "platinum": [1, 4, 12],
    "copper": [1, 2, 3, 4],             # Strong: Q1
    "rare_earth": [1, 2, 3, 10, 11],
    "industrial": [1, 2, 10, 11, 12],
}

# Weak months (avoid or reduce position size)
SEASONAL_WEAKNESS = {
    "gold": [5, 6],      # "Sell in May"
    "silver": [5, 6],
    "uranium": [6, 7],
    "platinum": [5, 6],
    "copper": [5, 6, 7],
    "rare_earth": [5, 6],
    "industrial": [5, 6, 7],
}


class CommodityBreakoutRule(Rule):
    """
    Buy miners when the underlying commodity breaks above resistance.

    Mining stocks are leveraged plays on commodities - when gold/uranium/etc
    breaks out, miners typically move 2-3x as much.

    Detection:
    - Price above SMA_50 (intermediate trend)
    - Price breaks above SMA_20 from below (momentum)
    - RSI showing strength (45-70 range, not overbought)

    Natural Language:
    "Buy miners when their underlying commodity shows breakout momentum"
    """

    def __init__(
        self,
        breakout_threshold_pct: float = 2.0,  # % above SMA for breakout
        min_trend_strength: float = 1.0,      # Min % SMA_20 above SMA_50
    ):
        self.breakout_threshold_pct = breakout_threshold_pct
        self.min_trend_strength = min_trend_strength

    @property
    def name(self) -> str:
        return "Commodity Breakout"

    @property
    def description(self) -> str:
        return f"Buy miners on commodity breakout (price {self.breakout_threshold_pct}%+ above SMA_20)"

    @property
    def required_indicators(self) -> list:
        return ["RSI_14", "SMA_20", "SMA_50", "close", "volume"]

    def evaluate(self, context: SymbolContext) -> RuleResult:
        rsi = context.get_indicator("RSI_14")
        sma20 = context.get_indicator("SMA_20")
        sma50 = context.get_indicator("SMA_50")
        close = context.get_indicator("close")
        volume = context.get_indicator("volume")
        avg_volume = context.get_indicator("volume_sma_20", volume)

        # Check if this is a mining stock
        commodity = MINER_COMMODITY_MAP.get(context.symbol.upper())
        if not commodity:
            return RuleResult(
                triggered=False,
                reasoning=f"{context.symbol} not in mining stock list"
            )

        # Trend must be up (SMA_20 > SMA_50)
        if sma20 <= sma50:
            return RuleResult(
                triggered=False,
                reasoning=f"No uptrend: SMA_20 ({sma20:.2f}) <= SMA_50 ({sma50:.2f})"
            )

        trend_strength = (sma20 - sma50) / sma50 * 100
        if trend_strength < self.min_trend_strength:
            return RuleResult(
                triggered=False,
                reasoning=f"Weak trend: {trend_strength:.1f}% < {self.min_trend_strength}% minimum"
            )

        # Price must be breaking out above SMA_20
        breakout_pct = (close - sma20) / sma20 * 100
        if breakout_pct < self.breakout_threshold_pct:
            return RuleResult(
                triggered=False,
                reasoning=f"No breakout: price only {breakout_pct:.1f}% above SMA_20"
            )

        # RSI should show momentum but not overbought
        if rsi < 45:
            return RuleResult(
                triggered=False,
                reasoning=f"RSI {rsi:.1f} too weak for breakout confirmation"
            )
        if rsi > 75:
            return RuleResult(
                triggered=False,
                reasoning=f"RSI {rsi:.1f} overbought - breakout may be exhausted"
            )

        # Volume confirmation
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0

        # Calculate confidence
        base_confidence = 0.55

        # Stronger breakout = higher confidence
        if breakout_pct > 4.0:
            base_confidence += 0.15
        elif breakout_pct > 3.0:
            base_confidence += 0.10
        elif breakout_pct > 2.0:
            base_confidence += 0.05

        # Volume confirmation boost
        if volume_ratio > 1.5:
            base_confidence += 0.10
        elif volume_ratio > 1.2:
            base_confidence += 0.05

        # RSI sweet spot (55-65)
        if 55 <= rsi <= 65:
            base_confidence += 0.05

        confidence = min(base_confidence, 0.85)

        return RuleResult(
            triggered=True,
            signal=SignalType.BUY,
            confidence=confidence,
            reasoning=(
                f"BREAKOUT: {context.symbol} ({commodity}) breaking out "
                f"{breakout_pct:.1f}% above SMA_20. RSI: {rsi:.1f}, "
                f"Volume: {volume_ratio:.1f}x avg. Trend strength: {trend_strength:.1f}%"
            ),
            contributing_factors={
                "commodity": commodity,
                "breakout_pct": round(breakout_pct, 2),
                "trend_strength_pct": round(trend_strength, 2),
                "RSI_14": round(rsi, 1),
                "volume_ratio": round(volume_ratio, 2),
            }
        )


class MinerMetalRatioRule(Rule):
    """
    Mean reversion rule - buy miners when they underperform the metal.

    When the GDX/GLD ratio (or miner vs commodity) drops significantly,
    miners are "cheap" relative to the metal and tend to catch up.

    Detection (using price action as proxy):
    - Miner RSI oversold while in uptrend
    - Price near support (SMA_50 or SMA_200)
    - This indicates miners have pulled back more than typical

    Natural Language:
    "Buy miners when they've underperformed the metal and are due to catch up"
    """

    def __init__(
        self,
        rsi_oversold: float = 35.0,
        support_tolerance_pct: float = 3.0,
    ):
        self.rsi_oversold = rsi_oversold
        self.support_tolerance_pct = support_tolerance_pct

    @property
    def name(self) -> str:
        return "Miner Metal Ratio"

    @property
    def description(self) -> str:
        return "Buy miners when oversold near support (underperforming metal)"

    @property
    def required_indicators(self) -> list:
        return ["RSI_14", "SMA_20", "SMA_50", "SMA_200", "close"]

    def evaluate(self, context: SymbolContext) -> RuleResult:
        rsi = context.get_indicator("RSI_14")
        sma20 = context.get_indicator("SMA_20")
        sma50 = context.get_indicator("SMA_50")
        sma200 = context.get_indicator("SMA_200")
        close = context.get_indicator("close")

        # Check if this is a mining stock
        commodity = MINER_COMMODITY_MAP.get(context.symbol.upper())
        if not commodity:
            return RuleResult(
                triggered=False,
                reasoning=f"{context.symbol} not in mining stock list"
            )

        # Long-term trend must be intact (price > SMA_200 or SMA_50 > SMA_200)
        long_term_intact = close > sma200 or sma50 > sma200
        if not long_term_intact:
            return RuleResult(
                triggered=False,
                reasoning=f"Long-term trend broken: price ${close:.2f} < SMA_200 ${sma200:.2f}"
            )

        # RSI must be oversold (indicating underperformance)
        if rsi > self.rsi_oversold:
            return RuleResult(
                triggered=False,
                reasoning=f"RSI {rsi:.1f} not oversold (need < {self.rsi_oversold})"
            )

        # Price should be near support (SMA_50 or SMA_200)
        dist_to_sma50 = abs(close - sma50) / sma50 * 100
        dist_to_sma200 = abs(close - sma200) / sma200 * 100

        near_sma50 = dist_to_sma50 <= self.support_tolerance_pct
        near_sma200 = dist_to_sma200 <= self.support_tolerance_pct

        if not (near_sma50 or near_sma200):
            return RuleResult(
                triggered=False,
                reasoning=f"Not near support: {dist_to_sma50:.1f}% from SMA_50, {dist_to_sma200:.1f}% from SMA_200"
            )

        # Calculate confidence
        base_confidence = 0.60

        # Deeper oversold = better entry
        if rsi < 25:
            base_confidence += 0.15
        elif rsi < 30:
            base_confidence += 0.10

        # Near SMA_200 is stronger support
        if near_sma200:
            base_confidence += 0.10
            support_level = "SMA_200"
        else:
            support_level = "SMA_50"

        # Full alignment bonus
        if sma20 > sma50 > sma200:
            base_confidence += 0.05

        confidence = min(base_confidence, 0.85)

        return RuleResult(
            triggered=True,
            signal=SignalType.BUY,
            confidence=confidence,
            reasoning=(
                f"MEAN REVERSION: {context.symbol} ({commodity}) oversold "
                f"(RSI: {rsi:.1f}) at {support_level} support. "
                f"Miners likely to catch up to metal."
            ),
            contributing_factors={
                "commodity": commodity,
                "RSI_14": round(rsi, 1),
                "support_level": support_level,
                "dist_to_support_pct": round(min(dist_to_sma50, dist_to_sma200), 2),
                "long_term_trend": "intact",
            }
        )


class DollarWeaknessRule(Rule):
    """
    Buy miners when USD is showing weakness.

    Commodities (and miners) have strong inverse correlation with USD.
    When the dollar weakens, commodity prices rise, and miners benefit.

    Detection (proxy via miner price action):
    - Strong uptrend in miners (indicates dollar weakness)
    - Price well above moving averages
    - Momentum accelerating (MACD positive)

    Note: For best results, integrate DXY (Dollar Index) data directly.

    Natural Language:
    "Buy miners showing strength that indicates dollar weakness"
    """

    def __init__(
        self,
        min_trend_spread: float = 2.0,
        require_macd_positive: bool = True,
    ):
        self.min_trend_spread = min_trend_spread
        self.require_macd_positive = require_macd_positive

    @property
    def name(self) -> str:
        return "Dollar Weakness"

    @property
    def description(self) -> str:
        return "Buy miners showing strength (dollar weakness indicator)"

    @property
    def required_indicators(self) -> list:
        indicators = ["RSI_14", "SMA_20", "SMA_50", "SMA_200", "close"]
        if self.require_macd_positive:
            indicators.extend(["MACD", "MACD_SIGNAL"])
        return indicators

    def evaluate(self, context: SymbolContext) -> RuleResult:
        rsi = context.get_indicator("RSI_14")
        sma20 = context.get_indicator("SMA_20")
        sma50 = context.get_indicator("SMA_50")
        sma200 = context.get_indicator("SMA_200")
        close = context.get_indicator("close")

        # Check if this is a mining stock
        commodity = MINER_COMMODITY_MAP.get(context.symbol.upper())
        if not commodity:
            return RuleResult(
                triggered=False,
                reasoning=f"{context.symbol} not in mining stock list"
            )

        # Must have full trend alignment (strong uptrend = dollar weakness)
        if not (sma20 > sma50 > sma200):
            return RuleResult(
                triggered=False,
                reasoning="No full trend alignment (need SMA_20 > SMA_50 > SMA_200)"
            )

        # Check trend strength
        trend_spread_20_50 = (sma20 - sma50) / sma50 * 100
        trend_spread_50_200 = (sma50 - sma200) / sma200 * 100

        if trend_spread_20_50 < self.min_trend_spread:
            return RuleResult(
                triggered=False,
                reasoning=f"Trend not strong enough: {trend_spread_20_50:.1f}% < {self.min_trend_spread}%"
            )

        # Check MACD if required
        if self.require_macd_positive:
            macd = context.get_indicator("MACD")
            signal = context.get_indicator("MACD_SIGNAL")
            if macd <= signal:
                return RuleResult(
                    triggered=False,
                    reasoning="MACD not bullish (below signal line)"
                )

        # RSI should show momentum (50-70)
        if rsi < 50:
            return RuleResult(
                triggered=False,
                reasoning=f"RSI {rsi:.1f} too weak for momentum play"
            )
        if rsi > 75:
            return RuleResult(
                triggered=False,
                reasoning=f"RSI {rsi:.1f} overbought"
            )

        # Price should be in upper half of range (above SMA_20)
        price_position = (close - sma20) / sma20 * 100

        # Calculate confidence
        base_confidence = 0.55

        # Stronger trend = higher confidence
        if trend_spread_20_50 > 4.0:
            base_confidence += 0.10
        elif trend_spread_20_50 > 3.0:
            base_confidence += 0.05

        if trend_spread_50_200 > 6.0:
            base_confidence += 0.10
        elif trend_spread_50_200 > 4.0:
            base_confidence += 0.05

        # RSI sweet spot
        if 55 <= rsi <= 65:
            base_confidence += 0.05

        confidence = min(base_confidence, 0.85)

        return RuleResult(
            triggered=True,
            signal=SignalType.BUY,
            confidence=confidence,
            reasoning=(
                f"DOLLAR WEAKNESS: {context.symbol} ({commodity}) in strong uptrend "
                f"(full alignment). Trend spread: {trend_spread_20_50:.1f}% / {trend_spread_50_200:.1f}%. "
                f"RSI: {rsi:.1f}. Strong momentum suggests USD weakness."
            ),
            contributing_factors={
                "commodity": commodity,
                "RSI_14": round(rsi, 1),
                "trend_spread_20_50": round(trend_spread_20_50, 2),
                "trend_spread_50_200": round(trend_spread_50_200, 2),
                "price_vs_sma20_pct": round(price_position, 2),
            }
        )


class SeasonalityRule(Rule):
    """
    Adjust signals based on seasonal patterns in mining stocks.

    Gold/silver miners have well-documented seasonal patterns:
    - Strong: January-February, August-September
    - Weak: May-June ("Sell in May" affects miners heavily)

    This rule boosts confidence during strong months and reduces
    it during weak months.

    Natural Language:
    "Boost buy confidence in seasonally strong months, reduce in weak months"
    """

    def __init__(
        self,
        strong_month_boost: float = 0.10,
        weak_month_penalty: float = 0.15,
        require_base_signal: bool = True,  # Only apply to existing buy signals
    ):
        self.strong_month_boost = strong_month_boost
        self.weak_month_penalty = weak_month_penalty
        self.require_base_signal = require_base_signal

    @property
    def name(self) -> str:
        return "Seasonality Filter"

    @property
    def description(self) -> str:
        return "Adjust confidence based on seasonal patterns"

    @property
    def required_indicators(self) -> list:
        return ["RSI_14", "SMA_20", "SMA_50", "close"]

    def evaluate(self, context: SymbolContext) -> RuleResult:
        rsi = context.get_indicator("RSI_14")
        sma20 = context.get_indicator("SMA_20")
        sma50 = context.get_indicator("SMA_50")
        close = context.get_indicator("close")

        # Check if this is a mining stock
        commodity = MINER_COMMODITY_MAP.get(context.symbol.upper())
        if not commodity:
            return RuleResult(
                triggered=False,
                reasoning=f"{context.symbol} not in mining stock list"
            )

        # Get current month
        current_month = context.timestamp.month

        # Check seasonal strength/weakness
        strong_months = SEASONAL_STRENGTH.get(commodity, [])
        weak_months = SEASONAL_WEAKNESS.get(commodity, [])

        is_strong_month = current_month in strong_months
        is_weak_month = current_month in weak_months

        # Base signal check - need uptrend and reasonable RSI
        uptrend = sma20 > sma50
        reasonable_rsi = 30 <= rsi <= 65

        if not uptrend:
            return RuleResult(
                triggered=False,
                reasoning="No uptrend - seasonality rule requires base trend"
            )

        if not reasonable_rsi:
            if rsi < 30:
                reason = f"RSI {rsi:.1f} deeply oversold - use dip-buying rule instead"
            else:
                reason = f"RSI {rsi:.1f} overbought"
            return RuleResult(triggered=False, reasoning=reason)

        # Calculate confidence based on seasonality
        base_confidence = 0.55

        if is_strong_month:
            base_confidence += self.strong_month_boost
            seasonal_status = "STRONG"
            month_names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                         7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
            note = f"{month_names[current_month]} is historically strong for {commodity}"
        elif is_weak_month:
            base_confidence -= self.weak_month_penalty
            seasonal_status = "WEAK"
            month_names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                         7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
            note = f"{month_names[current_month]} is historically weak for {commodity} - caution"
        else:
            seasonal_status = "NEUTRAL"
            note = "Neutral seasonal period"

        # Don't trigger on weak months unless already in position
        if is_weak_month and context.current_position != "long":
            return RuleResult(
                triggered=False,
                reasoning=f"Weak seasonal month for {commodity}. Avoid new entries in {current_month}."
            )

        # Trend strength bonus
        trend_spread = (sma20 - sma50) / sma50 * 100
        if trend_spread > 2.0:
            base_confidence += 0.05

        confidence = max(min(base_confidence, 0.85), 0.40)

        return RuleResult(
            triggered=True,
            signal=SignalType.BUY,
            confidence=confidence,
            reasoning=(
                f"SEASONALITY: {context.symbol} ({commodity}) - {seasonal_status} month. "
                f"{note}. RSI: {rsi:.1f}, Trend: +{trend_spread:.1f}%"
            ),
            contributing_factors={
                "commodity": commodity,
                "month": current_month,
                "seasonal_status": seasonal_status,
                "RSI_14": round(rsi, 1),
                "trend_spread_pct": round(trend_spread, 2),
            }
        )


class VolumeBreakoutRule(Rule):
    """
    Buy mining stocks on high-volume breakouts.

    Mining stock breakouts without volume often fail. This rule
    requires significant volume confirmation (1.5x+ average) for
    breakout signals.

    Natural Language:
    "Buy miners breaking out with 1.5x+ average volume"
    """

    def __init__(
        self,
        min_volume_ratio: float = 1.5,
        breakout_threshold_pct: float = 2.0,
        rsi_min: float = 50.0,
        rsi_max: float = 70.0,
    ):
        self.min_volume_ratio = min_volume_ratio
        self.breakout_threshold_pct = breakout_threshold_pct
        self.rsi_min = rsi_min
        self.rsi_max = rsi_max

    @property
    def name(self) -> str:
        return "Volume Breakout"

    @property
    def description(self) -> str:
        return f"Buy miners on breakout with {self.min_volume_ratio}x+ volume"

    @property
    def required_indicators(self) -> list:
        return ["RSI_14", "SMA_20", "SMA_50", "close", "volume"]

    def evaluate(self, context: SymbolContext) -> RuleResult:
        rsi = context.get_indicator("RSI_14")
        sma20 = context.get_indicator("SMA_20")
        sma50 = context.get_indicator("SMA_50")
        close = context.get_indicator("close")
        volume = context.get_indicator("volume")
        avg_volume = context.get_indicator("volume_sma_20", volume)

        # Check if this is a mining stock
        commodity = MINER_COMMODITY_MAP.get(context.symbol.upper())
        if not commodity:
            return RuleResult(
                triggered=False,
                reasoning=f"{context.symbol} not in mining stock list"
            )

        # Must be in uptrend
        if sma20 <= sma50:
            return RuleResult(
                triggered=False,
                reasoning=f"No uptrend: SMA_20 <= SMA_50"
            )

        # Check volume ratio
        volume_ratio = volume / avg_volume if avg_volume > 0 else 0
        if volume_ratio < self.min_volume_ratio:
            return RuleResult(
                triggered=False,
                reasoning=f"Volume {volume_ratio:.1f}x below {self.min_volume_ratio}x threshold"
            )

        # Check breakout
        breakout_pct = (close - sma20) / sma20 * 100
        if breakout_pct < self.breakout_threshold_pct:
            return RuleResult(
                triggered=False,
                reasoning=f"No breakout: only {breakout_pct:.1f}% above SMA_20"
            )

        # RSI should show momentum but not overbought
        if rsi < self.rsi_min:
            return RuleResult(
                triggered=False,
                reasoning=f"RSI {rsi:.1f} too low for breakout confirmation"
            )
        if rsi > self.rsi_max:
            return RuleResult(
                triggered=False,
                reasoning=f"RSI {rsi:.1f} overbought - breakout may fail"
            )

        # Calculate confidence
        base_confidence = 0.60

        # Higher volume = higher confidence
        if volume_ratio > 2.5:
            base_confidence += 0.15
        elif volume_ratio > 2.0:
            base_confidence += 0.10
        elif volume_ratio > 1.5:
            base_confidence += 0.05

        # Stronger breakout = higher confidence
        if breakout_pct > 4.0:
            base_confidence += 0.10
        elif breakout_pct > 3.0:
            base_confidence += 0.05

        confidence = min(base_confidence, 0.90)

        return RuleResult(
            triggered=True,
            signal=SignalType.BUY,
            confidence=confidence,
            reasoning=(
                f"VOLUME BREAKOUT: {context.symbol} ({commodity}) breaking out "
                f"{breakout_pct:.1f}% with {volume_ratio:.1f}x volume! "
                f"RSI: {rsi:.1f}. High-conviction signal."
            ),
            contributing_factors={
                "commodity": commodity,
                "volume_ratio": round(volume_ratio, 2),
                "breakout_pct": round(breakout_pct, 2),
                "RSI_14": round(rsi, 1),
            }
        )
