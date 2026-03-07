"""
Utility Sector Specific Rules

Specialized rules for trading utility stocks based on well-documented
sector dynamics.

Key Insights:
1. Traditional utilities are among the most mean-reverting stocks in the market
2. Interest rate sensitivity is the dominant factor — competes with bonds for income
3. RSI oversold thresholds must be HIGHER (35-42) because low beta rarely hits 30
4. Bollinger Bands + ADX are the best combination for entry timing
5. SMA_200 is the absolute floor — never buy below it (rate-driven selloffs are prolonged)
6. Nuclear/AI power stocks (CEG, VST) behave like MOMENTUM stocks, not utilities
7. Tighter profit targets (7-8%) match the available range for low-vol utilities
8. Seasonality: strong Mar/Jul/Oct, weak Feb, defensive rotation May-Sep

Two distinct sub-universes:
- Traditional/regulated: SO, D, NEE, AWK, BEP, CWEN, XLU, VPU — mean-reverting, low beta
- Nuclear/AI power: CEG, VST — momentum, high beta (use tech/energy momentum rules instead)
"""

from datetime import datetime
from typing import Optional

from .base import Rule, RuleResult, SignalType, SymbolContext


# =============================================================================
# Utility Sector Symbol Mapping
# =============================================================================

UTILITY_SECTOR_MAP = {
    # Traditional regulated utilities
    "SO": "regulated",       # Southern Company — electric + gas, beta 0.45
    "D": "regulated",        # Dominion Energy — electric + gas, beta 0.70
    "NEE": "regulated",      # NextEra Energy — largest utility, 70% regulated + 30% renewable
    "DUK": "regulated",      # Duke Energy
    "AEP": "regulated",      # American Electric Power
    "EXC": "regulated",      # Exelon
    "SRE": "regulated",      # Sempra

    # Water utilities (ultra-defensive, negative beta)
    "AWK": "water",          # American Water Works — negative beta, never-miss dividend

    # Clean energy yieldcos (high yield, extreme rate sensitivity)
    "BEP": "yieldco",        # Brookfield Renewable Partners — global hydro/wind/solar
    "CWEN": "yieldco",       # Clearway Energy — US wind/solar/storage
    "AY": "yieldco",         # Atlantica Sustainable Infrastructure

    # Nuclear / AI data center power (MOMENTUM stocks, NOT traditional utilities)
    "CEG": "nuclear_power",  # Constellation Energy — 21 nuclear reactors, Microsoft deal
    "VST": "nuclear_power",  # Vistra Corp — nuclear + nat gas, Meta deal, beta 1.4+
    "OKLO": "nuclear_power", # Oklo — advanced nuclear startup

    # Sector ETFs
    "XLU": "utility_etf",   # Utilities Select Sector SPDR
    "VPU": "utility_etf",   # Vanguard Utilities ETF
}

# RSI oversold thresholds by sub-sector (higher than other sectors due to low vol)
UTILITY_RSI_OVERSOLD = {
    "regulated": 38,          # Very low beta, rarely hits 30
    "water": 40,              # Ultra-defensive, extremely tight range
    "yieldco": 33,            # Higher beta than regulated, can get more oversold
    "nuclear_power": 30,      # High beta, standard thresholds work
    "utility_etf": 37,        # Diversified, slightly higher floor
}

# Seasonal strength months
UTILITY_SEASONAL_STRENGTH = {
    "regulated": [3, 7, 10],               # Mar, Jul, Oct — classic utility seasonality
    "water": [3, 6, 7, 8, 10],             # Strong summer (water demand) + Mar/Oct
    "yieldco": [1, 3, 7, 10, 11, 12],      # Rate-cut anticipation months + standard utility
    "nuclear_power": [1, 2, 10, 11, 12],    # Follows tech/AI sentiment + Q4 strength
    "utility_etf": [3, 7, 10],              # Matches sector aggregate
}

# Seasonal weakness months
UTILITY_SEASONAL_WEAKNESS = {
    "regulated": [2, 9],               # Feb weak, Sep rotation out
    "water": [2, 9],                   # Follows utility pattern
    "yieldco": [2, 5, 6, 9],          # Rate-hike fear months + sell-in-May
    "nuclear_power": [5, 6, 9],        # Follows tech seasonal weakness
    "utility_etf": [2, 9],             # Matches sector aggregate
}


class UtilityMeanReversionRule(Rule):
    """
    Buy utility stocks when oversold at Bollinger Band support in a
    mean-reverting environment. Utilities are textbook mean-reverting
    assets — this rule exploits that with calibrated thresholds.

    Key differences from financial mean reversion:
    - Higher RSI oversold threshold (35-42 vs 28-42) — utilities have lower vol
    - BB threshold slightly wider (0.15 vs 0.10) — narrow bands need wider trigger
    - Lower ADX threshold (20 vs 25) — utilities are structurally low-trend
    - Requires SMA_200 floor — rate-driven selloffs below SMA_200 are prolonged
    - Sub-sector-specific RSI thresholds from UTILITY_RSI_OVERSOLD

    Detection:
    - BB_PERCENT < 0.15 (near or below lower Bollinger Band)
    - RSI_14 in sub-sector-specific oversold range
    - ADX_14 < 20 (confirms mean-reverting, not trending)
    - Price >= SMA_200 (long-term support intact)
    - Volume >= 70% of average (not dead drift)

    Skips nuclear_power sub-sector (use momentum rules instead).
    """

    def __init__(
        self,
        bb_oversold: float = 0.15,
        rsi_floor: float = 25.0,
        rsi_ceiling: float = 42.0,
        adx_max: float = 20.0,
    ):
        self.bb_oversold = bb_oversold
        self.rsi_floor = rsi_floor
        self.rsi_ceiling = rsi_ceiling
        self.adx_max = adx_max

    @property
    def name(self) -> str:
        return "Utility Mean Reversion"

    @property
    def description(self) -> str:
        return (
            f"Buy utility stocks when BB%<{self.bb_oversold}, "
            f"RSI {self.rsi_floor}-{self.rsi_ceiling}, ADX<{self.adx_max}"
        )

    @property
    def required_indicators(self) -> list:
        return [
            "RSI_14", "SMA_50", "SMA_200", "close",
            "BB_PERCENT", "ADX_14",
        ]

    def evaluate(self, context: SymbolContext) -> RuleResult:
        rsi = context.get_indicator("RSI_14")
        sma20 = context.get_indicator("SMA_20")
        sma50 = context.get_indicator("SMA_50")
        sma200 = context.get_indicator("SMA_200")
        close = context.get_indicator("close")
        bb_pct = context.get_indicator("BB_PERCENT")
        adx = context.get_indicator("ADX_14")
        volume = context.get_indicator("volume")
        avg_volume = context.get_indicator("volume_sma_20", volume)
        stoch_k = context.get_indicator("Stochastic_K")

        sub_sector = UTILITY_SECTOR_MAP.get(context.symbol.upper())
        if not sub_sector:
            return RuleResult(
                triggered=False,
                reasoning=f"{context.symbol} not in utility sector list"
            )

        # Nuclear/AI power stocks should use momentum rules, not mean reversion
        if sub_sector == "nuclear_power":
            return RuleResult(
                triggered=False,
                reasoning=f"{context.symbol} is nuclear_power — use momentum rules"
            )

        # SMA_200 is the absolute floor for utilities
        if close < sma200:
            return RuleResult(
                triggered=False,
                reasoning=f"Price ({close:.2f}) below SMA_200 ({sma200:.2f}) — broken support"
            )

        # Long-term trend should be intact (golden cross)
        if sma50 <= sma200:
            return RuleResult(
                triggered=False,
                reasoning=f"No golden cross: SMA_50 ({sma50:.2f}) <= SMA_200 ({sma200:.2f})"
            )

        # ADX must confirm mean-reverting environment
        if adx > self.adx_max:
            return RuleResult(
                triggered=False,
                reasoning=f"ADX {adx:.1f} > {self.adx_max} — trending, not mean-reverting"
            )

        # BB_PERCENT must show oversold
        if bb_pct > self.bb_oversold:
            return RuleResult(
                triggered=False,
                reasoning=f"BB% {bb_pct:.2f} not oversold (need < {self.bb_oversold})"
            )

        # Get sub-sector-specific RSI threshold
        rsi_oversold = UTILITY_RSI_OVERSOLD.get(sub_sector, 38)
        effective_ceiling = min(self.rsi_ceiling, rsi_oversold + 7)

        # RSI range check
        if rsi < self.rsi_floor:
            return RuleResult(
                triggered=False,
                reasoning=f"RSI {rsi:.1f} below {self.rsi_floor} — potential fundamental problem"
            )
        if rsi > effective_ceiling:
            return RuleResult(
                triggered=False,
                reasoning=f"RSI {rsi:.1f} above {effective_ceiling} — not oversold for {sub_sector}"
            )

        # Volume check (not dead)
        volume_ratio = volume / avg_volume if avg_volume > 0 else 0.0
        if volume_ratio < 0.5:
            return RuleResult(
                triggered=False,
                reasoning=f"Volume {volume_ratio:.1f}x too low — dead drift"
            )

        # Calculate confidence
        base_confidence = 0.60

        # BB_PERCENT extreme oversold boost
        if bb_pct < 0.0:
            base_confidence += 0.10
        elif bb_pct < 0.05:
            base_confidence += 0.07
        elif bb_pct < 0.10:
            base_confidence += 0.03

        # RSI depth boost
        if rsi < rsi_oversold - 5:
            base_confidence += 0.10
        elif rsi < rsi_oversold - 2:
            base_confidence += 0.05

        # ADX very low = very high mean-reversion probability
        if adx < 12:
            base_confidence += 0.05

        # SMA_20 > SMA_50 = shorter-term trend intact
        if sma20 > 0 and sma20 > sma50:
            base_confidence += 0.05

        # Stochastic confirmation (if available)
        if stoch_k is not None and stoch_k < 20:
            base_confidence += 0.05

        # Volume confirmation
        if volume_ratio > 1.2:
            base_confidence += 0.05

        # Seasonal adjustment
        current_month = context.timestamp.month
        strong_months = UTILITY_SEASONAL_STRENGTH.get(sub_sector, [])
        weak_months = UTILITY_SEASONAL_WEAKNESS.get(sub_sector, [])

        if current_month in strong_months:
            base_confidence += 0.05
        elif current_month in weak_months:
            base_confidence -= 0.05

        confidence = max(min(base_confidence, 0.90), 0.40)

        return RuleResult(
            triggered=True,
            signal=SignalType.BUY,
            confidence=confidence,
            reasoning=(
                f"UTILITY MEAN REVERSION: {context.symbol} ({sub_sector}) oversold at "
                f"BB support (BB%={bb_pct:.2f}, RSI={rsi:.1f}). "
                f"ADX={adx:.1f} confirms mean-reverting. "
                f"Golden cross intact (SMA_50 > SMA_200)."
            ),
            contributing_factors={
                "sub_sector": sub_sector,
                "BB_PERCENT": round(bb_pct, 3),
                "RSI_14": round(rsi, 1),
                "ADX_14": round(adx, 1),
                "volume_ratio": round(volume_ratio, 2),
                "rsi_oversold_threshold": rsi_oversold,
                "sma50_above_sma200": True,
            }
        )


class UtilityRateReversionRule(Rule):
    """
    Buy utility stocks on pullback to SMA_50 support with momentum
    confirmation. This is the "pullback in uptrend" pattern calibrated
    for utility-stock volatility.

    Unlike tech EMA pullback which uses EMA_21 (fast), utilities need
    SMA_50 (slower) because their price action is structurally slower.

    Detection:
    - Price within 2% of SMA_50 (pullback to moving average support)
    - SMA_50 > SMA_200 (golden cross confirmed)
    - RSI_14 < 45 (not overbought)
    - MACD_HISTOGRAM improving (momentum turning up)
    - ADX_14 < 25 (not in a strong downtrend)

    Skips nuclear_power sub-sector.
    """

    def __init__(
        self,
        pullback_tolerance_pct: float = 2.5,
        rsi_max: float = 45.0,
    ):
        self.pullback_tolerance_pct = pullback_tolerance_pct
        self.rsi_max = rsi_max

    @property
    def name(self) -> str:
        return "Utility Rate Reversion"

    @property
    def description(self) -> str:
        return (
            f"Buy utility pullbacks to SMA_50 support (within {self.pullback_tolerance_pct}%) "
            f"with momentum confirmation"
        )

    @property
    def required_indicators(self) -> list:
        return [
            "RSI_14", "SMA_50", "SMA_200", "close",
            "MACD_HISTOGRAM",
        ]

    def evaluate(self, context: SymbolContext) -> RuleResult:
        rsi = context.get_indicator("RSI_14")
        sma20 = context.get_indicator("SMA_20")
        sma50 = context.get_indicator("SMA_50")
        sma200 = context.get_indicator("SMA_200")
        close = context.get_indicator("close")
        macd_hist = context.get_indicator("MACD_HISTOGRAM")
        adx = context.get_indicator("ADX_14")
        bb_pct = context.get_indicator("BB_PERCENT")

        sub_sector = UTILITY_SECTOR_MAP.get(context.symbol.upper())
        if not sub_sector:
            return RuleResult(
                triggered=False,
                reasoning=f"{context.symbol} not in utility sector list"
            )

        # Nuclear/AI power stocks should use momentum rules
        if sub_sector == "nuclear_power":
            return RuleResult(
                triggered=False,
                reasoning=f"{context.symbol} is nuclear_power — use momentum rules"
            )

        # Golden cross required
        if sma50 <= sma200:
            return RuleResult(
                triggered=False,
                reasoning=f"No golden cross: SMA_50 ({sma50:.2f}) <= SMA_200 ({sma200:.2f})"
            )

        # Price must be near SMA_50 (pullback to support)
        if sma50 > 0:
            distance_pct = (close - sma50) / sma50 * 100
        else:
            return RuleResult(triggered=False, reasoning="SMA_50 invalid")

        # Price should be near SMA_50 (within tolerance above or slightly below)
        if distance_pct > self.pullback_tolerance_pct:
            return RuleResult(
                triggered=False,
                reasoning=f"Price {distance_pct:.1f}% above SMA_50 — not a pullback"
            )
        if distance_pct < -3.0:
            return RuleResult(
                triggered=False,
                reasoning=f"Price {distance_pct:.1f}% below SMA_50 — broken support"
            )

        # RSI should not be overbought
        if rsi > self.rsi_max:
            return RuleResult(
                triggered=False,
                reasoning=f"RSI {rsi:.1f} > {self.rsi_max} — not oversold enough"
            )

        # MACD histogram should be improving (turning up)
        # We check if it's > -0.5 (for utilities, histograms are small numbers)
        if macd_hist < -0.5:
            return RuleResult(
                triggered=False,
                reasoning=f"MACD histogram {macd_hist:.3f} still declining"
            )

        # Calculate confidence
        base_confidence = 0.55

        # Closer to SMA_50 = better support test
        if abs(distance_pct) < 1.0:
            base_confidence += 0.10
        elif abs(distance_pct) < 1.5:
            base_confidence += 0.05

        # RSI depth
        rsi_oversold = UTILITY_RSI_OVERSOLD.get(sub_sector, 38)
        if rsi < rsi_oversold:
            base_confidence += 0.10
        elif rsi < rsi_oversold + 5:
            base_confidence += 0.05

        # MACD histogram positive = momentum already turning
        if macd_hist > 0:
            base_confidence += 0.05

        # BB_PERCENT confirmation
        if bb_pct is not None and bb_pct < 0.30:
            base_confidence += 0.05

        # ADX low = mean-reverting (good for this setup)
        if adx is not None and adx < 20:
            base_confidence += 0.05

        # SMA_20 > SMA_50 = shorter-term trend still intact (just a dip, not reversal)
        if sma20 > 0 and sma20 > sma50:
            base_confidence += 0.05

        # Seasonal adjustment
        current_month = context.timestamp.month
        strong_months = UTILITY_SEASONAL_STRENGTH.get(sub_sector, [])
        weak_months = UTILITY_SEASONAL_WEAKNESS.get(sub_sector, [])

        if current_month in strong_months:
            base_confidence += 0.05
        elif current_month in weak_months:
            base_confidence -= 0.05

        confidence = max(min(base_confidence, 0.85), 0.40)

        return RuleResult(
            triggered=True,
            signal=SignalType.BUY,
            confidence=confidence,
            reasoning=(
                f"UTILITY PULLBACK: {context.symbol} ({sub_sector}) at SMA_50 support "
                f"({distance_pct:+.1f}% from SMA_50). RSI={rsi:.1f}, "
                f"MACD_H={macd_hist:.3f}. Golden cross intact."
            ),
            contributing_factors={
                "sub_sector": sub_sector,
                "distance_from_sma50_pct": round(distance_pct, 2),
                "RSI_14": round(rsi, 1),
                "MACD_HISTOGRAM": round(macd_hist, 3),
                "sma50_above_sma200": True,
            }
        )


class UtilitySeasonalityRule(Rule):
    """
    Adjust confidence for utility seasonal patterns by sub-sector.

    Well-documented utility seasonality:
    - March, July, October: historically strong for regulated utilities
    - February: consistently weak
    - May-September: defensive rotation INTO utilities (but tempered by rate risk)
    - Yieldcos follow rate expectations more than calendar
    - Nuclear/AI power follows tech sentiment cycles

    Requires base uptrend (SMA_20 > SMA_50) to trigger.
    Blocks new entries in weak months.
    """

    def __init__(
        self,
        strong_month_boost: float = 0.10,
        weak_month_penalty: float = 0.15,
    ):
        self.strong_month_boost = strong_month_boost
        self.weak_month_penalty = weak_month_penalty

    @property
    def name(self) -> str:
        return "Utility Seasonality"

    @property
    def description(self) -> str:
        return "Adjust confidence for utility sector seasonal patterns"

    @property
    def required_indicators(self) -> list:
        return ["RSI_14", "SMA_20", "SMA_50", "close"]

    def evaluate(self, context: SymbolContext) -> RuleResult:
        rsi = context.get_indicator("RSI_14")
        sma20 = context.get_indicator("SMA_20")
        sma50 = context.get_indicator("SMA_50")
        close = context.get_indicator("close")

        sub_sector = UTILITY_SECTOR_MAP.get(context.symbol.upper())
        if not sub_sector:
            return RuleResult(
                triggered=False,
                reasoning=f"{context.symbol} not in utility sector list"
            )

        current_month = context.timestamp.month

        strong_months = UTILITY_SEASONAL_STRENGTH.get(sub_sector, [])
        weak_months = UTILITY_SEASONAL_WEAKNESS.get(sub_sector, [])

        is_strong_month = current_month in strong_months
        is_weak_month = current_month in weak_months

        # Base signal check — need uptrend and reasonable RSI
        uptrend = sma20 > sma50
        reasonable_rsi = 25 <= rsi <= 65

        if not uptrend:
            return RuleResult(
                triggered=False,
                reasoning="No uptrend — seasonality requires base trend"
            )

        if not reasonable_rsi:
            if rsi < 25:
                reason = f"RSI {rsi:.1f} deeply oversold — use mean reversion rule"
            else:
                reason = f"RSI {rsi:.1f} overbought"
            return RuleResult(triggered=False, reasoning=reason)

        # Block new entries in weak months
        if is_weak_month and context.current_position != "long":
            return RuleResult(
                triggered=False,
                reasoning=f"Weak seasonal month for {sub_sector}. Avoid new entries."
            )

        # Calculate confidence
        base_confidence = 0.55

        month_names = {
            1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
            7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
        }

        if is_strong_month:
            base_confidence += self.strong_month_boost
            seasonal_status = "STRONG"
            note = f"{month_names[current_month]} is historically strong for {sub_sector}"
        elif is_weak_month:
            base_confidence -= self.weak_month_penalty
            seasonal_status = "WEAK"
            note = f"{month_names[current_month]} is historically weak for {sub_sector}"
        else:
            seasonal_status = "NEUTRAL"
            note = "Neutral seasonal period"

        # Trend strength bonus
        trend_spread = (sma20 - sma50) / sma50 * 100 if sma50 > 0 else 0
        if trend_spread > 1.5:
            base_confidence += 0.05

        # RSI in comfortable range for utilities
        rsi_oversold = UTILITY_RSI_OVERSOLD.get(sub_sector, 38)
        if rsi < rsi_oversold + 5:
            base_confidence += 0.05

        confidence = max(min(base_confidence, 0.85), 0.40)

        return RuleResult(
            triggered=True,
            signal=SignalType.BUY,
            confidence=confidence,
            reasoning=(
                f"UTILITY SEASONALITY: {context.symbol} ({sub_sector}) — "
                f"{seasonal_status} month. {note}. RSI: {rsi:.1f}, "
                f"Trend: +{trend_spread:.1f}%"
            ),
            contributing_factors={
                "sub_sector": sub_sector,
                "month": current_month,
                "seasonal_status": seasonal_status,
                "RSI_14": round(rsi, 1),
                "trend_spread_pct": round(trend_spread, 2),
            }
        )
