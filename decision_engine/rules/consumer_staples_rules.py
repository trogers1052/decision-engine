"""
Consumer Staples Sector Specific Rules

Specialized rules for trading consumer staples stocks based on
well-documented sector dynamics.

Key Insights:
1. Consumer staples are among the lowest-beta stocks (0.03-0.36 for household/beverages)
2. Textbook mean-reverting: tight ranges, Bollinger Bands + ADX are primary entry tools
3. RSI oversold thresholds must be HIGHER (40-42) — even higher than utilities
4. ADX < 18 confirms mean-reverting regime (tighter than utilities' 20)
5. SMA_200 is the absolute floor — same as utilities
6. COST is a MOMENTUM stock (beta ~1.0) — exclude from mean-reversion rules
7. Tighter profit targets (6-7%) match the available range for ultra-low-vol names
8. Seasonality: Q1 strong, summer defensive rotation, Sep weak, Q4 holiday strength
9. Rate sensitivity is moderate (less than utilities, but still a factor)
10. Staples outperform in late-cycle and recession — current environment is favorable
"""

from datetime import datetime
from typing import Optional

from .base import Rule, RuleResult, SignalType, SymbolContext


# =============================================================================
# Consumer Staples Sector Symbol Mapping
# =============================================================================

CONSUMER_STAPLES_SECTOR_MAP = {
    # Beverages (ultra-defensive, Dividend Kings)
    "KO": "beverages",       # Coca-Cola — beta 0.11-0.36, yield 2.6%, 62yr div streak
    "PEP": "beverages",      # PepsiCo — similar profile to KO

    # Household products (ultra-low beta, textbook mean-reversion)
    "PG": "household",       # Procter & Gamble — beta 0.15, consumer staple bellwether
    "CL": "household",       # Colgate-Palmolive — beta 0.03-0.30, negative skew
    "CHD": "household",      # Church & Dwight — beta 0.02-0.47, Arm & Hammer
    "KMB": "household",      # Kimberly-Clark — beta 0.08-0.31, yield ~5%, Dividend King

    # Mass retail (higher beta, hybrid defensive/growth)
    "WMT": "mass_retail",    # Walmart — beta 0.26-0.66, e-commerce growth
    "TGT": "mass_retail",    # Target — if added later

    # Warehouse/growth retail (MOMENTUM — exclude from mean-reversion)
    "COST": "retail_growth", # Costco — beta ~1.0, P/E ~50, membership model

    # Sector ETFs
    "XLP": "staples_etf",    # Consumer Staples Select Sector SPDR
    "VDC": "staples_etf",    # Vanguard Consumer Staples ETF
}

# RSI oversold thresholds by sub-sector
STAPLES_RSI_OVERSOLD = {
    "beverages": 40,          # Ultra-low beta, RSI rarely dips below 38
    "household": 40,          # Ultra-low beta, similar to beverages
    "mass_retail": 36,        # Higher beta allows deeper RSI dips
    "retail_growth": 30,      # Standard thresholds (high beta, like tech)
    "staples_etf": 37,        # Diversified, dampened volatility
}

# Seasonal strength months
STAPLES_SEASONAL_STRENGTH = {
    "beverages": [1, 2, 3, 6, 7, 8],         # Q1 + summer (beverage demand)
    "household": [1, 2, 3, 10, 11, 12],       # Q1 + Q4 (holiday consumer spending)
    "mass_retail": [1, 2, 10, 11, 12],        # Holiday quarter dominant
    "retail_growth": [10, 11, 12, 1],          # Holiday + membership renewals
    "staples_etf": [1, 2, 3, 6, 7, 8],        # Q1 + defensive rotation summer
}

# Seasonal weakness months
STAPLES_SEASONAL_WEAKNESS = {
    "beverages": [9],                          # Sep rotation out
    "household": [5, 6, 9],                    # Sell in May + Sep
    "mass_retail": [5, 6, 9],                  # Sell in May + Sep
    "retail_growth": [5, 6, 9],                # Follows tech seasonal weakness
    "staples_etf": [9],                        # Sep weakness
}


class ConsumerStaplesMeanReversionRule(Rule):
    """
    Buy consumer staples stocks when oversold at Bollinger Band support
    in a mean-reverting environment.

    Consumer staples are the lowest-beta stocks in the market. The
    mean-reversion signal is extremely reliable because:
    - Demand for essentials is inelastic
    - High dividend yields create natural price floors
    - Institutional rebalancing forces mean-reversion

    Key differences from utility mean reversion:
    - Even higher RSI thresholds (40-42 vs 38 for utilities)
    - Tighter ADX threshold (18 vs 20) — staples are even lower-trend
    - Excludes retail_growth (COST) — use momentum rules instead

    Detection:
    - BB_PERCENT < 0.15 (near or below lower Bollinger Band)
    - RSI_14 in sub-sector-specific oversold range
    - ADX_14 < 18 (confirms mean-reverting, not trending)
    - Price >= SMA_200 (long-term support intact)
    - Volume >= 50% of average (not dead drift)
    """

    def __init__(
        self,
        bb_oversold: float = 0.15,
        rsi_floor: float = 25.0,
        rsi_ceiling: float = 45.0,
        adx_max: float = 18.0,
    ):
        self.bb_oversold = bb_oversold
        self.rsi_floor = rsi_floor
        self.rsi_ceiling = rsi_ceiling
        self.adx_max = adx_max

    @property
    def name(self) -> str:
        return "Consumer Staples Mean Reversion"

    @property
    def description(self) -> str:
        return (
            f"Buy consumer staples when BB%<{self.bb_oversold}, "
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

        sub_sector = CONSUMER_STAPLES_SECTOR_MAP.get(context.symbol.upper())
        if not sub_sector:
            return RuleResult(
                triggered=False,
                reasoning=f"{context.symbol} not in consumer staples sector list"
            )

        # Retail growth (COST) should use momentum rules, not mean reversion
        if sub_sector == "retail_growth":
            return RuleResult(
                triggered=False,
                reasoning=f"{context.symbol} is retail_growth — use momentum rules"
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
        rsi_oversold = STAPLES_RSI_OVERSOLD.get(sub_sector, 40)
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
        base_confidence = 0.60

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
        if adx < 12:
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
        strong_months = STAPLES_SEASONAL_STRENGTH.get(sub_sector, [])
        weak_months = STAPLES_SEASONAL_WEAKNESS.get(sub_sector, [])

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
                f"STAPLES MEAN REVERSION: {context.symbol} ({sub_sector}) oversold at "
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


class ConsumerStaplesPullbackRule(Rule):
    """
    Buy consumer staples on pullback to SMA_50 support with momentum
    confirmation.

    Uses SMA_50 (not EMA_21) because staples price action is
    structurally slow — fast moving averages generate noise.

    Excludes retail_growth (COST).
    """

    def __init__(
        self,
        pullback_tolerance_pct: float = 2.0,
        rsi_max: float = 45.0,
    ):
        self.pullback_tolerance_pct = pullback_tolerance_pct
        self.rsi_max = rsi_max

    @property
    def name(self) -> str:
        return "Consumer Staples Pullback"

    @property
    def description(self) -> str:
        return (
            f"Buy consumer staples pullbacks to SMA_50 (within {self.pullback_tolerance_pct}%) "
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

        sub_sector = CONSUMER_STAPLES_SECTOR_MAP.get(context.symbol.upper())
        if not sub_sector:
            return RuleResult(
                triggered=False,
                reasoning=f"{context.symbol} not in consumer staples sector list"
            )

        if sub_sector == "retail_growth":
            return RuleResult(
                triggered=False,
                reasoning=f"{context.symbol} is retail_growth — use momentum rules"
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
        if distance_pct < -3.0:
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
        if macd_hist < -0.3:
            return RuleResult(
                triggered=False,
                reasoning=f"MACD histogram {macd_hist:.3f} still declining"
            )

        # Calculate confidence
        base_confidence = 0.55

        # Closer to SMA_50 = better support test
        if abs(distance_pct) < 0.8:
            base_confidence += 0.10
        elif abs(distance_pct) < 1.5:
            base_confidence += 0.05

        # RSI depth
        rsi_oversold = STAPLES_RSI_OVERSOLD.get(sub_sector, 40)
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

        # ADX low = mean-reverting
        if adx is not None and adx < 18:
            base_confidence += 0.05

        # SMA_20 > SMA_50 still intact
        if sma20 > 0 and sma20 > sma50:
            base_confidence += 0.05

        # Seasonal adjustment
        current_month = context.timestamp.month
        strong_months = STAPLES_SEASONAL_STRENGTH.get(sub_sector, [])
        weak_months = STAPLES_SEASONAL_WEAKNESS.get(sub_sector, [])

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
                f"STAPLES PULLBACK: {context.symbol} ({sub_sector}) at SMA_50 support "
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


class ConsumerStaplesSeasonalityRule(Rule):
    """
    Adjust confidence for consumer staples seasonal patterns.

    Staples seasonality:
    - Beverages: Q1 + summer (demand peak) strong, Sep weak
    - Household: Q1 + Q4 (holiday spending) strong, May/Jun/Sep weak
    - Mass retail: holiday quarter dominant, sell-in-May pattern
    - ETFs: Q1 + defensive rotation summer, Sep weak

    Requires base uptrend (SMA_20 > SMA_50). Blocks new entries in weak months.
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
        return "Consumer Staples Seasonality"

    @property
    def description(self) -> str:
        return "Adjust confidence for consumer staples seasonal patterns"

    @property
    def required_indicators(self) -> list:
        return ["RSI_14", "SMA_20", "SMA_50", "close"]

    def evaluate(self, context: SymbolContext) -> RuleResult:
        rsi = context.get_indicator("RSI_14")
        sma20 = context.get_indicator("SMA_20")
        sma50 = context.get_indicator("SMA_50")
        close = context.get_indicator("close")

        sub_sector = CONSUMER_STAPLES_SECTOR_MAP.get(context.symbol.upper())
        if not sub_sector:
            return RuleResult(
                triggered=False,
                reasoning=f"{context.symbol} not in consumer staples sector list"
            )

        current_month = context.timestamp.month

        strong_months = STAPLES_SEASONAL_STRENGTH.get(sub_sector, [])
        weak_months = STAPLES_SEASONAL_WEAKNESS.get(sub_sector, [])

        is_strong_month = current_month in strong_months
        is_weak_month = current_month in weak_months

        # Base signal check
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

        # RSI in comfortable range
        rsi_oversold = STAPLES_RSI_OVERSOLD.get(sub_sector, 40)
        if rsi < rsi_oversold + 5:
            base_confidence += 0.05

        confidence = max(min(base_confidence, 0.85), 0.40)

        return RuleResult(
            triggered=True,
            signal=SignalType.BUY,
            confidence=confidence,
            reasoning=(
                f"STAPLES SEASONALITY: {context.symbol} ({sub_sector}) — "
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
