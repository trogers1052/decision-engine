"""
Financial Sector Specific Rules

Specialized rules for trading financial stocks (banks, insurance, fintech, REITs)
based on well-documented sector dynamics.

Key Insights:
1. Financials are predominantly mean-reverting, not momentum-driven
2. Banks borrow short, lend long — yield curve drives profitability
3. Credit spreads are a leading indicator for financial stock health
4. Bollinger Bands are more effective than RSI alone for financials (adaptive to own volatility)
5. ADX < 20 confirms mean-reverting regime (best for this strategy)
6. RSI oversold levels are shallower for financials (35-42 vs 25-35 for tech)
7. Strong Q1 and Q4 seasonality across most financial sub-sectors
"""

from datetime import datetime
from typing import Optional

from .base import Rule, RuleResult, SignalType, SymbolContext


# =============================================================================
# Financial Sector Symbol Mapping
# =============================================================================

FINANCIAL_SECTOR_MAP = {
    # Large-cap diversified banks
    "JPM": "bank",
    "BAC": "bank",
    "WFC": "bank",
    "C": "bank",
    "USB": "bank",
    "PNC": "bank",
    "TFC": "bank",

    # Investment banks
    "GS": "investment_bank",
    "MS": "investment_bank",

    # Insurance / Conglomerates
    "BRK.B": "insurance",
    "CB": "insurance",
    "PGR": "insurance",
    "ALL": "insurance",
    "MET": "insurance",
    "AIG": "insurance",

    # Payment networks / Fintech (mature)
    "V": "payments",
    "MA": "payments",
    "PYPL": "payments",

    # Financial data / Ratings
    "SPGI": "financial_data",
    "MCO": "financial_data",

    # Fintech (growth) — behaves more like tech, use with caution
    "SOFI": "fintech",
    "NU": "fintech",
    "SQ": "fintech",

    # REITs
    "O": "reit",
    "AMT": "reit",
    "SPG": "reit",
    "PLD": "reit",

    # Sector ETFs
    "XLF": "sector_etf",
    "KRE": "regional_bank_etf",
}

# Seasonal strength months for financial sub-sectors
FINANCIAL_SEASONAL_STRENGTH = {
    "bank": [1, 2, 3, 10, 11, 12],            # Q4 earnings run-up, Q1 strength
    "investment_bank": [1, 2, 3, 10, 11, 12],
    "insurance": [1, 2, 11, 12],
    "payments": [1, 2, 10, 11, 12],            # Holiday spending season
    "financial_data": [1, 2, 3, 10, 11, 12],
    "fintech": [1, 11, 12],                    # More like tech
    "reit": [1, 2, 3, 10, 11, 12],
    "sector_etf": [1, 2, 3, 10, 11, 12],
    "regional_bank_etf": [1, 2, 3, 10, 11, 12],
}

# Weak months (avoid or reduce position size)
FINANCIAL_SEASONAL_WEAKNESS = {
    "bank": [5, 6, 8, 9],
    "investment_bank": [5, 6, 8, 9],
    "insurance": [5, 6, 8],
    "payments": [5, 6, 9],
    "financial_data": [5, 6, 8, 9],
    "fintech": [5, 6, 9],
    "reit": [5, 6, 8, 9],
    "sector_etf": [5, 6, 8, 9],
    "regional_bank_etf": [5, 6, 8, 9],
}


class FinancialMeanReversionRule(Rule):
    """
    Buy financial stocks when oversold near Bollinger Band support in a
    low-trend-strength (mean-reverting) environment.

    Financials are predominantly mean-reverting. This rule exploits that
    behavior using Bollinger Bands (adaptive to each stock's volatility)
    combined with ADX to confirm mean-reverting conditions.

    Detection:
    - BB_PERCENT < 0.05 (price near or below lower Bollinger Band)
    - RSI_14 between 30 and 42 (oversold for financials — shallower than tech)
    - ADX_14 < 25 (confirming mean-reverting, NOT trending regime)
    - SMA_50 > SMA_200 (long-term uptrend intact — golden cross)
    - Volume not dead (volume_ratio >= 0.7)

    Natural Language:
    "Buy financial stocks when they're oversold at Bollinger Band support
    in a mean-reverting environment with the long-term trend intact"
    """

    def __init__(
        self,
        bb_oversold: float = 0.10,          # BB_PERCENT threshold for entry
        rsi_floor: float = 28.0,            # Below this, something is fundamentally wrong
        rsi_ceiling: float = 42.0,          # Above this, not oversold enough for financials
        adx_max: float = 25.0,              # Above this, market is trending, not mean-reverting
        support_tolerance_pct: float = 3.0,  # Max distance below SMA_200 allowed
    ):
        self.bb_oversold = bb_oversold
        self.rsi_floor = rsi_floor
        self.rsi_ceiling = rsi_ceiling
        self.adx_max = adx_max
        self.support_tolerance_pct = support_tolerance_pct

    @property
    def name(self) -> str:
        return "Financial Mean Reversion"

    @property
    def description(self) -> str:
        return (
            f"Buy financial stocks when BB%<{self.bb_oversold}, "
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

        # Check if this is a financial stock
        sub_sector = FINANCIAL_SECTOR_MAP.get(context.symbol.upper())
        if not sub_sector:
            return RuleResult(
                triggered=False,
                reasoning=f"{context.symbol} not in financial sector list"
            )

        # Long-term trend must be intact (golden cross)
        if sma50 <= sma200:
            return RuleResult(
                triggered=False,
                reasoning=f"No golden cross: SMA_50 ({sma50:.2f}) <= SMA_200 ({sma200:.2f})"
            )

        # Price shouldn't be too far below SMA_200 (broken support)
        if sma200 > 0:
            dist_below_200 = (sma200 - close) / sma200 * 100
            if dist_below_200 > self.support_tolerance_pct:
                return RuleResult(
                    triggered=False,
                    reasoning=f"Price {dist_below_200:.1f}% below SMA_200 — support broken"
                )

        # ADX must confirm mean-reverting environment
        if adx > self.adx_max:
            return RuleResult(
                triggered=False,
                reasoning=f"ADX {adx:.1f} > {self.adx_max} — market is trending, not mean-reverting"
            )

        # BB_PERCENT must show oversold (near or below lower band)
        if bb_pct > self.bb_oversold:
            return RuleResult(
                triggered=False,
                reasoning=f"BB% {bb_pct:.2f} not oversold (need < {self.bb_oversold})"
            )

        # RSI must be in the financial oversold range
        if rsi < self.rsi_floor:
            return RuleResult(
                triggered=False,
                reasoning=f"RSI {rsi:.1f} below {self.rsi_floor} — potential fundamental problem"
            )
        if rsi > self.rsi_ceiling:
            return RuleResult(
                triggered=False,
                reasoning=f"RSI {rsi:.1f} above {self.rsi_ceiling} — not oversold enough"
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

        # BB_PERCENT below 0 = extreme oversold (below lower band)
        if bb_pct < 0.0:
            base_confidence += 0.10
        elif bb_pct < 0.05:
            base_confidence += 0.05

        # Deeper RSI oversold = higher confidence
        if rsi < 32:
            base_confidence += 0.10
        elif rsi < 35:
            base_confidence += 0.05

        # ADX very low = very high mean-reversion probability
        if adx < 15:
            base_confidence += 0.05

        # Shorter-term uptrend intact (SMA_20 > SMA_50)
        if sma20 > 0 and sma20 > sma50:
            base_confidence += 0.05

        # Volume confirmation boost
        if volume_ratio > 1.2:
            base_confidence += 0.05

        # Seasonal adjustment
        current_month = context.timestamp.month
        strong_months = FINANCIAL_SEASONAL_STRENGTH.get(sub_sector, [])
        weak_months = FINANCIAL_SEASONAL_WEAKNESS.get(sub_sector, [])

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
                f"MEAN REVERSION: {context.symbol} ({sub_sector}) oversold at "
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
                "sma50_above_sma200": sma50 > sma200,
            }
        )


class FinancialPullbackRule(Rule):
    """
    Buy financial stocks on pullback to SMA_50 support in a confirmed uptrend.

    Financials (especially banks and payment networks) pull back to SMA_50
    reliably in bull markets. Banks are interest-rate sensitive — when the
    yield curve steepens, pullbacks to support are high-conviction entries.
    Payment networks (V, MA) are secular growth — they pull back to shorter
    MAs (EMA_21) like tech stocks.

    Key insights:
    1. Banks (JPM, GS) pull back to SMA_50 in 2-4 week cycles
    2. Payment networks (V, MA) pull back to EMA_21 in shorter cycles
    3. Investment banks (GS, MS) have higher vol — use wider tolerance
    4. Insurance (BRK.B, CB, PGR) is low-beta — tight pullbacks to SMA_20

    Detection:
    - Price within 2-3% of SMA_50 (or EMA_21 for payments)
    - SMA_50 > SMA_200 (golden cross)
    - RSI 35-55 (pulling back, not collapsing)
    - MACD histogram improving (momentum turning)
    - Volume >= 60% of average (institutional activity)

    Skips fintech (SOFI, NU, SQ) — too volatile for pullback plays.
    """

    def __init__(
        self,
        pullback_tolerance_pct: float = 3.0,
        min_volume_ratio: float = 0.6,
    ):
        self.pullback_tolerance_pct = pullback_tolerance_pct
        self.min_volume_ratio = min_volume_ratio

    @property
    def name(self) -> str:
        return "Financial Pullback"

    @property
    def description(self) -> str:
        return "Buy financial stocks on pullback to SMA_50 support in uptrend"

    @property
    def required_indicators(self) -> list:
        return [
            "RSI_14", "EMA_21", "SMA_20", "SMA_50", "SMA_200", "close",
            "volume", "MACD_HISTOGRAM",
        ]

    def evaluate(self, context: SymbolContext) -> RuleResult:
        sub_sector = FINANCIAL_SECTOR_MAP.get(context.symbol.upper())
        if not sub_sector:
            return RuleResult(
                triggered=False,
                reasoning=f"{context.symbol} not in financial sector list"
            )

        # Skip fintech — too volatile for pullback strategy
        if sub_sector in ("fintech",):
            return RuleResult(
                triggered=False,
                reasoning=f"{context.symbol} is fintech — too volatile for pullback"
            )

        rsi = context.get_indicator("RSI_14")
        ema21 = context.get_indicator("EMA_21")
        sma20 = context.get_indicator("SMA_20")
        sma50 = context.get_indicator("SMA_50")
        sma200 = context.get_indicator("SMA_200")
        close = context.get_indicator("close")
        volume = context.get_indicator("volume")
        avg_volume = context.get_indicator("volume_sma_20")
        macd_hist = context.get_indicator("MACD_HISTOGRAM")

        # Golden cross required
        if sma50 <= sma200:
            return RuleResult(
                triggered=False,
                reasoning=f"No golden cross: SMA_50={sma50:.2f} <= SMA_200={sma200:.2f}"
            )

        # Determine support level by sub-sector
        # Payments use EMA_21 (faster, like tech growth)
        # Banks/insurance use SMA_50 (slower, more cyclical)
        if sub_sector in ("payments", "financial_data"):
            support_ma = ema21
            support_name = "EMA_21"
            tolerance = 2.0  # Tighter for low-vol payments
        elif sub_sector in ("investment_bank",):
            support_ma = sma50
            support_name = "SMA_50"
            tolerance = self.pullback_tolerance_pct + 1.0  # Wider for volatile I-banks
        else:
            support_ma = sma50
            support_name = "SMA_50"
            tolerance = self.pullback_tolerance_pct

        # Price must be near support MA
        if support_ma > 0:
            dist_to_support = (close - support_ma) / support_ma * 100
        else:
            return RuleResult(triggered=False, reasoning=f"{support_name} invalid")

        if dist_to_support > tolerance:
            return RuleResult(
                triggered=False,
                reasoning=f"Price {dist_to_support:+.1f}% above {support_name} — not a pullback"
            )
        if dist_to_support < -(tolerance + 1.0):
            return RuleResult(
                triggered=False,
                reasoning=f"Price {dist_to_support:+.1f}% below {support_name} — broken support"
            )

        # Price must be above SMA_200
        if close < sma200:
            return RuleResult(
                triggered=False,
                reasoning=f"Price ${close:.2f} below SMA_200 ${sma200:.2f} — trend broken"
            )

        # RSI range (pulling back, not collapsing or overbought)
        if rsi < 30:
            return RuleResult(
                triggered=False,
                reasoning=f"RSI {rsi:.1f} < 30 — too deep, use mean reversion rule"
            )
        if rsi > 60:
            return RuleResult(
                triggered=False,
                reasoning=f"RSI {rsi:.1f} > 60 — not a pullback"
            )

        # MACD histogram should be improving
        if macd_hist < -0.5:
            return RuleResult(
                triggered=False,
                reasoning=f"MACD histogram {macd_hist:.3f} — momentum still declining"
            )

        # Volume check
        volume_ratio = volume / avg_volume if avg_volume > 0 else 0.0
        if volume_ratio < self.min_volume_ratio:
            return RuleResult(
                triggered=False,
                reasoning=f"Volume {volume_ratio:.1f}x too low (need {self.min_volume_ratio}x)"
            )

        # Calculate confidence
        base_confidence = 0.55

        # Proximity to support
        if abs(dist_to_support) < 1.0:
            base_confidence += 0.10
        elif abs(dist_to_support) < 2.0:
            base_confidence += 0.05

        # RSI sweet spot (40-50 = controlled pullback)
        if 38 <= rsi <= 50:
            base_confidence += 0.05

        # MACD turning positive
        if macd_hist > 0:
            base_confidence += 0.05

        # Volume surge on pullback (institutional buying)
        if volume_ratio > 1.3:
            base_confidence += 0.05

        # SMA_20 still above SMA_50 (short-term trend intact)
        if sma20 > sma50:
            base_confidence += 0.05

        # Sub-sector adjustments
        if sub_sector in ("payments", "financial_data"):
            base_confidence += 0.05  # Higher conviction for secular growth
        elif sub_sector == "investment_bank":
            base_confidence -= 0.03  # Slight penalty for vol

        # Seasonal boost
        current_month = context.timestamp.month
        strong_months = FINANCIAL_SEASONAL_STRENGTH.get(sub_sector, [])
        if current_month in strong_months:
            base_confidence += 0.05

        confidence = max(min(base_confidence, 0.85), 0.40)

        return RuleResult(
            triggered=True,
            signal=SignalType.BUY,
            confidence=confidence,
            reasoning=(
                f"FINANCIAL PULLBACK: {context.symbol} ({sub_sector}) at {support_name} "
                f"support ({dist_to_support:+.1f}%). RSI={rsi:.1f}, "
                f"MACD_H={macd_hist:.3f}. Golden cross intact."
            ),
            contributing_factors={
                "sub_sector": sub_sector,
                "support_level": support_name,
                "distance_from_support_pct": round(dist_to_support, 2),
                "RSI_14": round(rsi, 1),
                "MACD_HISTOGRAM": round(macd_hist, 3),
                "volume_ratio": round(volume_ratio, 2),
            }
        )


class FinancialSeasonalityRule(Rule):
    """
    Adjust signals based on seasonal patterns in financial stocks.

    Banks have well-documented seasonal strength in Q1 and Q4:
    - Strong: January-March (post-earnings, new year allocation)
    - Strong: October-December (Q4 run-up, tax-loss harvesting recovery)
    - Weak: May-June ("Sell in May"), August-September

    Natural Language:
    "Boost buy confidence in seasonally strong months for financials"
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
        return "Financial Seasonality"

    @property
    def description(self) -> str:
        return "Adjust confidence based on financial sector seasonal patterns"

    @property
    def required_indicators(self) -> list:
        return ["RSI_14", "SMA_20", "SMA_50", "close"]

    def evaluate(self, context: SymbolContext) -> RuleResult:
        rsi = context.get_indicator("RSI_14")
        sma20 = context.get_indicator("SMA_20")
        sma50 = context.get_indicator("SMA_50")
        close = context.get_indicator("close")

        # Check if this is a financial stock
        sub_sector = FINANCIAL_SECTOR_MAP.get(context.symbol.upper())
        if not sub_sector:
            return RuleResult(
                triggered=False,
                reasoning=f"{context.symbol} not in financial sector list"
            )

        # Get current month
        current_month = context.timestamp.month

        # Check seasonal strength/weakness
        strong_months = FINANCIAL_SEASONAL_STRENGTH.get(sub_sector, [])
        weak_months = FINANCIAL_SEASONAL_WEAKNESS.get(sub_sector, [])

        is_strong_month = current_month in strong_months
        is_weak_month = current_month in weak_months

        # Base signal check — need uptrend and reasonable RSI
        uptrend = sma20 > sma50
        reasonable_rsi = 30 <= rsi <= 65

        if not uptrend:
            return RuleResult(
                triggered=False,
                reasoning="No uptrend — seasonality rule requires base trend"
            )

        if not reasonable_rsi:
            if rsi < 30:
                reason = f"RSI {rsi:.1f} deeply oversold — use mean reversion rule instead"
            else:
                reason = f"RSI {rsi:.1f} overbought"
            return RuleResult(triggered=False, reasoning=reason)

        # Don't trigger on weak months unless already in position
        if is_weak_month and context.current_position != "long":
            return RuleResult(
                triggered=False,
                reasoning=f"Weak seasonal month for {sub_sector}. Avoid new entries."
            )

        # Calculate confidence based on seasonality
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
        trend_spread = (sma20 - sma50) / sma50 * 100
        if trend_spread > 2.0:
            base_confidence += 0.05

        confidence = max(min(base_confidence, 0.85), 0.40)

        return RuleResult(
            triggered=True,
            signal=SignalType.BUY,
            confidence=confidence,
            reasoning=(
                f"FINANCIAL SEASONALITY: {context.symbol} ({sub_sector}) — "
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
