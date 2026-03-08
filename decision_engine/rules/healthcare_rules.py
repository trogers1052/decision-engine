"""
Healthcare Sector Specific Rules

Specialized rules for trading healthcare stocks based on
well-documented sector dynamics.

Key Insights:
1. Healthcare spans 5+ sub-sectors with dramatically different profiles
2. Large-cap pharma (ABBV) and diversified (JNJ) are textbook mean-reverters (beta 0.14-0.36)
3. Managed care (UNH) mean-reverts BUT with rare violent policy/regulatory gaps
4. Large-cap biotech (VRTX) is unusually low-beta for biotech — hybrid approach
5. Medical devices (SYK) and med-tech growth (TMDX) are momentum/growth — exclude from mean-reversion
6. Clinical-stage biotech (RXRX, VKTX) have BINARY trial risk — EXCLUDE from ALL rules
7. Seasonality: Nov-Dec strongest (75% win rate), Sep weakest (-1.2%, 42% win rate)
8. Q1 pharma weakness from insurance deductible resets
9. Hospital budget cycle creates Q4 device spending boost
10. ADX < 20 for mean-reversion confirmation (pharma/JNJ behave like utilities)
"""

from datetime import datetime
from typing import Optional

from .base import Rule, RuleResult, SignalType, SymbolContext


# =============================================================================
# Healthcare Sector Symbol Mapping
# =============================================================================

HEALTHCARE_SECTOR_MAP = {
    # Large-cap pharma (low beta, dividend, mean-reverting)
    "ABBV": "pharma",          # AbbVie — beta 0.32-0.36, yield ~3%, 12yr div growth
    "LLY": "pharma",           # Eli Lilly — if added later
    "MRK": "pharma",           # Merck — if added later
    "PFE": "pharma",           # Pfizer — if added later

    # Diversified healthcare (ultra-low beta, Dividend King)
    "JNJ": "diversified",      # Johnson & Johnson — beta 0.14-0.35, 54yr div streak

    # Managed care / insurance (low beta BUT political gap risk)
    "UNH": "managed_care",     # UnitedHealth — beta 0.42, can gap 10-15% on policy news

    # Large-cap biotech (unusually low beta for biotech, recurring revenue)
    "VRTX": "large_biotech",   # Vertex Pharma — beta 0.23-0.49, $12B CF revenue

    # Medical devices / growth (momentum stocks, NOT mean-reverting)
    "SYK": "med_devices",      # Stryker — beta 0.69-0.87, growth compounder
    "TMDX": "med_growth",      # TransMedics — beta 1.1-2.05, high-growth organ transplant

    # Clinical-stage biotech (BINARY RISK — EXCLUDE from all rules)
    "RXRX": "clinical_biotech",  # Recursion — beta 3.21, pre-revenue, AI drug discovery
    "VKTX": "clinical_biotech",  # Viking Therapeutics — Phase 3 obesity trials, 50-70% gap risk

    # Sector ETFs
    "XLV": "healthcare_etf",   # Health Care Select Sector SPDR
    "XBI": "healthcare_etf",   # SPDR S&P Biotech ETF — if added later
}

# RSI oversold thresholds by sub-sector
HEALTHCARE_RSI_OVERSOLD = {
    "pharma": 38,              # Low beta, rarely dips below 35
    "diversified": 40,         # Ultra-low beta, like consumer staples (PG/KMB)
    "managed_care": 30,        # Standard — UNH has enough beta to reach true oversold
    "large_biotech": 35,       # Lower than pharma (growth biotech can dip harder)
    "med_devices": 30,         # Standard (near-market beta, momentum stock)
    "med_growth": 28,          # High-vol growth, can sustain lower RSI
    "clinical_biotech": 25,    # N/A — excluded from rules, but included for completeness
    "healthcare_etf": 37,      # ETF dampening prevents extreme readings
}

# Seasonal strength months (based on 25yr XLV data)
HEALTHCARE_SEASONAL_STRENGTH = {
    "pharma": [3, 4, 5, 7, 10, 11, 12],        # Mar-May + Jul + Q4 strong
    "diversified": [1, 3, 4, 5, 7, 10, 11, 12], # Broad strength, defensive rotation
    "managed_care": [4, 5, 7, 10, 11, 12],      # Q4 strong; Q1 weak (deductible reset)
    "large_biotech": [1, 5, 7, 10, 11, 12],     # JPM conference Jan, Nov-Dec strongest
    "med_devices": [4, 7, 10, 11, 12],           # Q4 hospital budget deployment
    "med_growth": [4, 7, 10, 11, 12],            # Follows device seasonality
    "clinical_biotech": [],                       # No meaningful seasonality
    "healthcare_etf": [1, 3, 4, 5, 7, 10, 11, 12],  # Broad sector pattern
}

# Seasonal weakness months
HEALTHCARE_SEASONAL_WEAKNESS = {
    "pharma": [2, 8, 9],                         # Feb (Q1 Rx reset), Aug-Sep
    "diversified": [8, 9],                        # Aug-Sep rotation
    "managed_care": [1, 2, 9],                    # Q1 deductible reset + Sep
    "large_biotech": [8, 9],                      # Summer lull
    "med_devices": [2, 8, 9],                     # Feb + Aug-Sep
    "med_growth": [2, 8, 9],                      # Follows broader pattern
    "clinical_biotech": [9],                       # Sep universal weakness
    "healthcare_etf": [2, 8, 9],                  # Broad sector pattern
}


class HealthcareMeanReversionRule(Rule):
    """
    Buy healthcare stocks when oversold at Bollinger Band support
    in a mean-reverting environment.

    Applies to: pharma (ABBV), diversified (JNJ), managed_care (UNH),
                large_biotech (VRTX), healthcare_etf (XLV)

    Excludes: med_devices (SYK), med_growth (TMDX) — use momentum rules
    Excludes: clinical_biotech (RXRX, VKTX) — binary trial risk

    Key differences from consumer staples mean reversion:
    - UNH gets wider RSI range (standard 30 vs raised thresholds)
    - VRTX treated as low-beta biotech (RSI 35, slightly wider)
    - ADX < 20 (same as utilities, looser than staples' 18)

    Detection:
    - BB_PERCENT < 0.15 (near or below lower Bollinger Band)
    - RSI_14 in sub-sector-specific oversold range
    - ADX_14 < 20 (confirms mean-reverting, not trending)
    - Price >= SMA_200 (long-term support intact)
    - Volume >= 50% of average (not dead drift)
    """

    def __init__(
        self,
        bb_oversold: float = 0.15,
        rsi_floor: float = 22.0,
        rsi_ceiling: float = 45.0,
        adx_max: float = 20.0,
    ):
        self.bb_oversold = bb_oversold
        self.rsi_floor = rsi_floor
        self.rsi_ceiling = rsi_ceiling
        self.adx_max = adx_max

    @property
    def name(self) -> str:
        return "Healthcare Mean Reversion"

    @property
    def description(self) -> str:
        return (
            f"Buy healthcare stocks when BB%<{self.bb_oversold}, "
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

        sub_sector = HEALTHCARE_SECTOR_MAP.get(context.symbol.upper())
        if not sub_sector:
            return RuleResult(
                triggered=False,
                reasoning=f"{context.symbol} not in healthcare sector list"
            )

        # Clinical-stage biotech — EXCLUDE from ALL rules
        if sub_sector == "clinical_biotech":
            return RuleResult(
                triggered=False,
                reasoning=f"{context.symbol} is clinical_biotech — binary trial risk, excluded from all rules"
            )

        # Medical devices/growth — use momentum rules, not mean reversion
        if sub_sector in ("med_devices", "med_growth"):
            return RuleResult(
                triggered=False,
                reasoning=f"{context.symbol} is {sub_sector} — use momentum rules"
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
        rsi_oversold = HEALTHCARE_RSI_OVERSOLD.get(sub_sector, 38)
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
        strong_months = HEALTHCARE_SEASONAL_STRENGTH.get(sub_sector, [])
        weak_months = HEALTHCARE_SEASONAL_WEAKNESS.get(sub_sector, [])

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
                f"HEALTHCARE MEAN REVERSION: {context.symbol} ({sub_sector}) oversold at "
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


class HealthcarePullbackRule(Rule):
    """
    Buy healthcare stocks on pullback to SMA_50 support with momentum
    confirmation.

    Applies to all mean-reverting healthcare sub-sectors.
    Excludes med_devices, med_growth (use momentum), clinical_biotech (binary risk).
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
        return "Healthcare Pullback"

    @property
    def description(self) -> str:
        return (
            f"Buy healthcare pullbacks to SMA_50 (within {self.pullback_tolerance_pct}%) "
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

        sub_sector = HEALTHCARE_SECTOR_MAP.get(context.symbol.upper())
        if not sub_sector:
            return RuleResult(
                triggered=False,
                reasoning=f"{context.symbol} not in healthcare sector list"
            )

        if sub_sector == "clinical_biotech":
            return RuleResult(
                triggered=False,
                reasoning=f"{context.symbol} is clinical_biotech — binary trial risk, excluded"
            )

        if sub_sector in ("med_devices", "med_growth"):
            return RuleResult(
                triggered=False,
                reasoning=f"{context.symbol} is {sub_sector} — use momentum rules"
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
        if macd_hist < -0.5:
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
        rsi_oversold = HEALTHCARE_RSI_OVERSOLD.get(sub_sector, 38)
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
        if adx is not None and adx < 20:
            base_confidence += 0.05

        # SMA_20 > SMA_50 still intact
        if sma20 > 0 and sma20 > sma50:
            base_confidence += 0.05

        # Seasonal adjustment
        current_month = context.timestamp.month
        strong_months = HEALTHCARE_SEASONAL_STRENGTH.get(sub_sector, [])
        weak_months = HEALTHCARE_SEASONAL_WEAKNESS.get(sub_sector, [])

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
                f"HEALTHCARE PULLBACK: {context.symbol} ({sub_sector}) at SMA_50 support "
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


class HealthcareSeasonalityRule(Rule):
    """
    Adjust confidence for healthcare seasonal patterns.

    Healthcare seasonality (25yr XLV data):
    - Strongest: Nov (+2.0%, 75% WR), Dec (+2.1%, 75% WR)
    - Weakest: Sep (-1.2%, 42% WR)
    - Q1 pharma weakness from insurance deductible resets
    - Q4 device spending (hospital budget deployment)

    Requires base uptrend (SMA_20 > SMA_50). Blocks new entries in weak months.
    Excludes clinical_biotech (no meaningful seasonality).
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
        return "Healthcare Seasonality"

    @property
    def description(self) -> str:
        return "Adjust confidence for healthcare seasonal patterns (Nov-Dec strong, Sep weak)"

    @property
    def required_indicators(self) -> list:
        return ["RSI_14", "SMA_20", "SMA_50", "close"]

    def evaluate(self, context: SymbolContext) -> RuleResult:
        rsi = context.get_indicator("RSI_14")
        sma20 = context.get_indicator("SMA_20")
        sma50 = context.get_indicator("SMA_50")
        close = context.get_indicator("close")

        sub_sector = HEALTHCARE_SECTOR_MAP.get(context.symbol.upper())
        if not sub_sector:
            return RuleResult(
                triggered=False,
                reasoning=f"{context.symbol} not in healthcare sector list"
            )

        # Clinical-stage biotech excluded
        if sub_sector == "clinical_biotech":
            return RuleResult(
                triggered=False,
                reasoning=f"{context.symbol} is clinical_biotech — no meaningful seasonality"
            )

        current_month = context.timestamp.month

        strong_months = HEALTHCARE_SEASONAL_STRENGTH.get(sub_sector, [])
        weak_months = HEALTHCARE_SEASONAL_WEAKNESS.get(sub_sector, [])

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
        if trend_spread > 1.5:
            base_confidence += 0.05

        # RSI in comfortable range
        rsi_oversold = HEALTHCARE_RSI_OVERSOLD.get(sub_sector, 38)
        if rsi < rsi_oversold + 5:
            base_confidence += 0.05

        confidence = max(min(base_confidence, 0.85), 0.40)

        return RuleResult(
            triggered=True,
            signal=SignalType.BUY,
            confidence=confidence,
            reasoning=(
                f"HEALTHCARE SEASONALITY: {context.symbol} ({sub_sector}) — "
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
