"""
Industrial Sector Specific Rules

Specialized rules for trading industrial stocks based on
well-documented sector dynamics.

Key Insights:
1. Industrials are cyclical — earnings track ISM/PMI and capex cycles
2. Beta typically 0.9-1.3, higher than staples/utilities but lower than tech
3. Mean-reversion works for diversified conglomerates (HON, MMM, ITW)
4. Momentum/trend works for heavy equipment (CAT, DE) tied to capex cycles
5. ADX < 22 confirms mean-reverting regime (between utilities' 20 and financials' 25)
6. Seasonality: Q4/Q1 strong (capex budgets flush), summer weak (seasonal slowdown)
7. Rate sensitivity is moderate — capex decisions lag rate changes by 6-12 months
8. Backlog visibility gives industrials more predictable earnings than most cyclicals
9. Book-to-bill ratio > 1.0 is bullish for equipment makers
10. Infrastructure spending (IIJA) provides multi-year tailwind through 2030
"""

from datetime import datetime
from typing import Optional

from .base import Rule, RuleResult, SignalType, SymbolContext


# =============================================================================
# Industrial Sector Symbol Mapping
# =============================================================================

INDUSTRIAL_SECTOR_MAP = {
    # Heavy equipment / Ag & Construction (cyclical, capex-driven)
    "CAT": "heavy_equipment",   # Caterpillar — beta 1.0, construction/mining capex
    "DE": "heavy_equipment",    # Deere — beta 1.0, ag + construction capex
    "PCAR": "heavy_equipment",  # PACCAR — beta 0.9, truck manufacturing

    # Diversified conglomerate (lower beta, mean-reverting)
    "HON": "conglomerate",      # Honeywell — beta 1.0, aerospace/automation/materials
    "MMM": "conglomerate",      # 3M — beta 1.0, diversified industrial (litigation risk)
    "ITW": "conglomerate",      # Illinois Tool Works — beta 1.0, 80/20 model, premium margins

    # Electrical / Power management (infrastructure + data center tailwind)
    "ETN": "electrical",        # Eaton — beta 1.1, power management, data center exposure
    "EMR": "electrical",        # Emerson Electric — beta 1.1, automation/climate tech
    "ROK": "electrical",        # Rockwell Automation — beta 1.2, factory automation

    # Aerospace industrial (defense-adjacent, long-cycle)
    "GE": "aerospace",          # GE Aerospace — beta 1.2, jet engines, aftermarket
    "TXT": "aerospace",         # Textron — beta 1.1, aviation/defense/industrial

    # Infrastructure engineering
    "WLDN": "electrical",       # Willdan Group — beta ~1.1, infrastructure engineering + energy services

    # Sector ETFs
    "XLI": "industrial_etf",    # Industrial Select Sector SPDR
}

# RSI oversold thresholds by sub-sector
INDUSTRIAL_RSI_OVERSOLD = {
    "heavy_equipment": 33,       # Higher beta, deeper RSI dips than staples
    "conglomerate": 36,          # Moderate beta, mean-reverts reliably
    "electrical": 34,            # Moderate-high beta, infrastructure demand floor
    "aerospace": 33,             # Higher beta, long-cycle backlog support
    "industrial_etf": 35,        # Diversified, dampened volatility
}

# Seasonal strength months (capex budget cycles)
INDUSTRIAL_SEASONAL_STRENGTH = {
    "heavy_equipment": [1, 2, 3, 10, 11, 12],   # Q1 capex deployment + Q4 budget flush
    "conglomerate": [1, 2, 11, 12],              # Q1 orders + Q4 budget flush
    "electrical": [1, 2, 3, 10, 11, 12],         # Infrastructure spending front-loaded
    "aerospace": [1, 2, 3, 11, 12],              # Defense budget cycle + airline orders
    "industrial_etf": [1, 2, 3, 10, 11, 12],     # Broad capex cycle
}

# Seasonal weakness months
INDUSTRIAL_SEASONAL_WEAKNESS = {
    "heavy_equipment": [6, 7, 8, 9],             # Summer slowdown, ag harvest uncertainty
    "conglomerate": [6, 7, 9],                    # Summer doldrums
    "electrical": [7, 8, 9],                      # Project delays in summer
    "aerospace": [7, 8, 9],                       # Airline seasonality noise
    "industrial_etf": [6, 7, 8, 9],              # Broad summer weakness
}


class IndustrialMeanReversionRule(Rule):
    """
    Buy industrial stocks when oversold at Bollinger Band support
    in a mean-reverting environment.

    Industrials mean-revert because:
    - Diversified revenue streams dampen single-segment shocks
    - Long backlogs provide earnings visibility (ITW, HON)
    - Institutional rebalancing forces sector mean-reversion
    - Infrastructure spending provides multi-year demand floor

    Key differences from other sectors:
    - ADX threshold 22 (between staples' 18 and financials' 25)
    - RSI thresholds 33-36 (deeper dips than staples/utilities)
    - Excludes heavy_equipment in trending markets (use momentum)

    Detection:
    - BB_PERCENT < 0.15 (near or below lower Bollinger Band)
    - RSI_14 in sub-sector-specific oversold range
    - ADX_14 < 22 (confirms mean-reverting, not trending)
    - Price >= SMA_200 (long-term support intact)
    - Volume >= 50% of average (not dead drift)
    """

    def __init__(
        self,
        bb_oversold: float = 0.15,
        rsi_floor: float = 22.0,
        rsi_ceiling: float = 42.0,
        adx_max: float = 22.0,
    ):
        self.bb_oversold = bb_oversold
        self.rsi_floor = rsi_floor
        self.rsi_ceiling = rsi_ceiling
        self.adx_max = adx_max

    @property
    def name(self) -> str:
        return "Industrial Mean Reversion"

    @property
    def description(self) -> str:
        return (
            f"Buy industrials when BB%<{self.bb_oversold}, "
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
        avg_volume = context.get_indicator("volume_sma_20")
        stoch_k = context.get_indicator("Stochastic_K")

        sub_sector = INDUSTRIAL_SECTOR_MAP.get(context.symbol.upper())
        if not sub_sector:
            return RuleResult(
                triggered=False,
                reasoning=f"{context.symbol} not in industrial sector list"
            )

        # SMA_200 is the absolute floor
        if close < sma200:
            return RuleResult(
                triggered=False,
                reasoning=f"Price ({close:.2f}) below SMA_200 ({sma200:.2f}) — broken support"
            )

        # Golden cross required
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

        # Sub-sector-specific RSI threshold
        rsi_oversold = INDUSTRIAL_RSI_OVERSOLD.get(sub_sector, 35)
        effective_ceiling = min(self.rsi_ceiling, rsi_oversold + 7)

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

        # Volume check
        volume_ratio = volume / avg_volume if avg_volume > 0 else 0.0
        if volume_ratio < 0.5:
            return RuleResult(
                triggered=False,
                reasoning=f"Volume {volume_ratio:.1f}x too low — dead drift"
            )

        # Calculate confidence
        base_confidence = 0.58

        # BB extreme oversold boost
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

        # ADX very low = strong mean-reversion
        if adx < 15:
            base_confidence += 0.05

        # SMA_20 > SMA_50 = shorter-term trend intact
        if sma20 > 0 and sma20 > sma50:
            base_confidence += 0.05

        # Stochastic confirmation
        if stoch_k is not None and stoch_k < 20:
            base_confidence += 0.05

        # Volume confirmation
        if volume_ratio > 1.2:
            base_confidence += 0.05

        # Seasonal adjustment
        current_month = context.timestamp.month
        strong_months = INDUSTRIAL_SEASONAL_STRENGTH.get(sub_sector, [])
        weak_months = INDUSTRIAL_SEASONAL_WEAKNESS.get(sub_sector, [])

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
                f"INDUSTRIAL MEAN REVERSION: {context.symbol} ({sub_sector}) oversold at "
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


class IndustrialPullbackRule(Rule):
    """
    Buy industrial stocks on pullback to SMA_50 support with momentum
    confirmation.

    Industrials pull back to SMA_50 during normal capex cycle fluctuations.
    The SMA_50 acts as dynamic support because institutional buyers
    accumulate on dips to this level.
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
        return "Industrial Pullback"

    @property
    def description(self) -> str:
        return (
            f"Buy industrial pullbacks to SMA_50 (within {self.pullback_tolerance_pct}%) "
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

        sub_sector = INDUSTRIAL_SECTOR_MAP.get(context.symbol.upper())
        if not sub_sector:
            return RuleResult(
                triggered=False,
                reasoning=f"{context.symbol} not in industrial sector list"
            )

        # Golden cross required
        if sma50 <= sma200:
            return RuleResult(
                triggered=False,
                reasoning=f"No golden cross: SMA_50 ({sma50:.2f}) <= SMA_200 ({sma200:.2f})"
            )

        # Price must be near SMA_50
        if sma50 > 0:
            distance_pct = (close - sma50) / sma50 * 100
        else:
            return RuleResult(triggered=False, reasoning="SMA_50 invalid")

        if distance_pct > self.pullback_tolerance_pct:
            return RuleResult(
                triggered=False,
                reasoning=f"Price {distance_pct:.1f}% above SMA_50 — not a pullback"
            )
        if distance_pct < -4.0:
            return RuleResult(
                triggered=False,
                reasoning=f"Price {distance_pct:.1f}% below SMA_50 — broken support"
            )

        # RSI not overbought
        if rsi > self.rsi_max:
            return RuleResult(
                triggered=False,
                reasoning=f"RSI {rsi:.1f} > {self.rsi_max} — not oversold enough"
            )

        # MACD histogram should be improving
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
        elif abs(distance_pct) < 2.0:
            base_confidence += 0.05

        # RSI depth
        rsi_oversold = INDUSTRIAL_RSI_OVERSOLD.get(sub_sector, 35)
        if rsi < rsi_oversold:
            base_confidence += 0.10
        elif rsi < rsi_oversold + 5:
            base_confidence += 0.05

        # MACD positive = momentum turning
        if macd_hist > 0:
            base_confidence += 0.05

        # BB confirmation
        if bb_pct is not None and bb_pct < 0.30:
            base_confidence += 0.05

        # ADX moderate = not trending hard against us
        if adx is not None and adx < 22:
            base_confidence += 0.05

        # SMA_20 > SMA_50 still intact
        if sma20 > 0 and sma20 > sma50:
            base_confidence += 0.05

        # Seasonal adjustment
        current_month = context.timestamp.month
        strong_months = INDUSTRIAL_SEASONAL_STRENGTH.get(sub_sector, [])
        weak_months = INDUSTRIAL_SEASONAL_WEAKNESS.get(sub_sector, [])

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
                f"INDUSTRIAL PULLBACK: {context.symbol} ({sub_sector}) at SMA_50 support "
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


class IndustrialSeasonalityRule(Rule):
    """
    Adjust confidence for industrial seasonal patterns.

    Industrial seasonality is driven by capital expenditure cycles:
    - Q4/Q1: Budget flush (Oct-Dec) + new year capex deployment (Jan-Mar)
    - Summer: Seasonal slowdown, project delays, vacation-driven decision lag
    - Sep: Universally weak across all sub-sectors

    Heavy equipment has additional ag harvest uncertainty in summer.
    Aerospace follows airline order cycles (Jan-Mar ordering season).
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
        return "Industrial Seasonality"

    @property
    def description(self) -> str:
        return "Adjust confidence for industrial seasonal patterns (Q4/Q1 strong, summer weak)"

    @property
    def required_indicators(self) -> list:
        return ["RSI_14", "SMA_20", "SMA_50", "close"]

    def evaluate(self, context: SymbolContext) -> RuleResult:
        rsi = context.get_indicator("RSI_14")
        sma20 = context.get_indicator("SMA_20")
        sma50 = context.get_indicator("SMA_50")
        close = context.get_indicator("close")

        sub_sector = INDUSTRIAL_SECTOR_MAP.get(context.symbol.upper())
        if not sub_sector:
            return RuleResult(
                triggered=False,
                reasoning=f"{context.symbol} not in industrial sector list"
            )

        current_month = context.timestamp.month

        strong_months = INDUSTRIAL_SEASONAL_STRENGTH.get(sub_sector, [])
        weak_months = INDUSTRIAL_SEASONAL_WEAKNESS.get(sub_sector, [])

        is_strong_month = current_month in strong_months
        is_weak_month = current_month in weak_months

        # Base signal check
        uptrend = sma20 > sma50
        reasonable_rsi = 22 <= rsi <= 65

        if not uptrend:
            return RuleResult(
                triggered=False,
                reasoning="No uptrend — seasonality requires base trend"
            )

        if not reasonable_rsi:
            if rsi < 22:
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
        if trend_spread > 2.0:
            base_confidence += 0.05

        # RSI in comfortable range
        rsi_oversold = INDUSTRIAL_RSI_OVERSOLD.get(sub_sector, 35)
        if rsi < rsi_oversold + 5:
            base_confidence += 0.05

        confidence = max(min(base_confidence, 0.85), 0.40)

        return RuleResult(
            triggered=True,
            signal=SignalType.BUY,
            confidence=confidence,
            reasoning=(
                f"INDUSTRIAL SEASONALITY: {context.symbol} ({sub_sector}) — "
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
