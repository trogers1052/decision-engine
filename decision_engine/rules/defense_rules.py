"""
Defense/Aerospace Sector Specific Rules

Specialized rules for trading defense contractor stocks and ETFs based on
well-documented sector dynamics unique to government-contracted defense companies.

Key Insights:
1. Defense primes have ultra-low betas (LMT 0.20, NOC 0.01, RTX 0.41, GD 0.37)
2. Counter-cyclical: outperformed SPY by +24.6% in 2022 bear market
3. Geopolitical spikes mean-revert in ~28 days (NBER: 20 events studied)
4. Strong seasonality tied to DoD fiscal year (Oct 1 start):
   - Jul-Sep: $260B+ "use-it-or-lose-it" spending surge (30% of annual contracts)
   - Oct-Dec: NDAA passage, new fiscal year contract awards
   - Feb: President's budget request catalyst
   - May: Armed Services Committee markups
5. Multi-year backlog visibility (RTX 3.0x, LMT 2.6x revenue) creates price floor
6. Dividend aristocrats provide downside support
7. CRs/shutdowns barely dent large primes but hammer mid-cap defense tech
8. "Rearmament supercycle" (2022+): structural, not cyclical — trend-follow, don't fade

Academic References:
- NBER WP 29837: Stock volatility 25% lower during wartime (defense spending reduces uncertainty)
- PMC 11700249: 81.4% of defense companies impacted by Ukraine invasion, non-normal returns
- Fisher Investments: "Acting after fighting starts is acting on old news"
- Quantpedia: Military spending predictive only for US market (Pax Americana effect)
"""

from datetime import datetime
from typing import Optional

from .base import Rule, RuleResult, SignalType, SymbolContext


# =============================================================================
# Defense Sector Symbol Mapping
# =============================================================================

DEFENSE_SECTOR_MAP = {
    # Prime contractors (ultra-low beta, dividend aristocrats, multi-year backlogs)
    "LMT": "prime",       # Lockheed Martin — beta 0.20, F-35, missiles, space
    "RTX": "prime",        # RTX Corp — beta 0.41, Pratt & Whitney, missiles, radar
    "NOC": "prime",        # Northrop Grumman — beta 0.01, B-21 bomber, space, cyber
    "GD": "prime",         # General Dynamics — beta 0.37, Gulfstream, submarines, IT

    # Mid-cap defense tech (higher beta, growth-oriented, no dividends)
    "AVAV": "defense_tech",  # AeroVironment — beta 1.26, Switchblade drones, loitering munitions
    "PLTR": "defense_tech",  # Palantir — beta ~1.5, AI/data analytics for defense
    "KTOS": "defense_tech",  # Kratos — beta ~1.3, drones, hypersonics, satellite
    "RKLB": "defense_tech",  # Rocket Lab — beta ~1.5, small launch, space

    # Defense IT / services (moderate beta, steady contract revenue)
    "BAH": "defense_services",  # Booz Allen Hamilton — beta ~0.7, consulting, cyber
    "LDOS": "defense_services", # Leidos — beta ~0.5, IT modernization, health
    "CACI": "defense_services", # CACI International — beta ~0.6, signals intel, IT
    "SAIC": "defense_services", # Science Applications — beta ~0.6, IT, engineering

    # Defense ETFs (tradeable at $1K account size)
    "ITA": "defense_etf",    # iShares US Aerospace & Defense — market-cap weighted
    "XAR": "defense_etf",    # SPDR S&P Aerospace & Defense — equal-weighted
    "PPA": "defense_etf",    # Invesco Aerospace & Defense — modified market-cap
    "SHLD": "defense_etf",   # Global X Defense Tech ETF — defense tech focus

    # Dual-use aerospace (NOT pure defense — commercial exposure dominates)
    "BA": "dual_use",        # Boeing — defense backlog $85B but MAX issues dominate
    "GE": "dual_use",        # GE Aerospace — commercial engine orders drive stock
    "HII": "prime",          # Huntington Ingalls — beta ~0.4, naval shipbuilding
    "LHX": "prime",          # L3Harris Technologies — beta ~0.5, comms, sensors
}

# DoD fiscal year seasonality (Oct 1 = new FY)
# Strong: budget request (Feb), committee markups (May), fiscal Q4 spending rush (Jul-Sep), NDAA passage (Oct-Dec)
# Weak: shoulder months (Mar-Apr), CR/shutdown risk (Oct if Congress misses deadline)
DEFENSE_SEASONAL_STRENGTH = {
    "prime": [1, 2, 5, 7, 8, 9, 11, 12],          # Budget request + spending rush + NDAA
    "defense_tech": [1, 2, 5, 7, 8, 9, 11, 12],    # Same cycle but more volatile
    "defense_services": [1, 2, 7, 8, 9, 10, 11, 12],  # Services benefit from new FY (Oct)
    "defense_etf": [1, 2, 5, 7, 8, 9, 11, 12],     # Broad sector follows budget cycle
    "dual_use": [1, 2, 5, 6, 7, 8, 9, 11, 12],     # Commercial aerospace adds summer strength
}

DEFENSE_SEASONAL_WEAKNESS = {
    "prime": [3, 4, 6],           # Shoulder months, summer lull before Jul spending
    "defense_tech": [3, 4, 6],    # Same but sharper drops
    "defense_services": [3, 4, 5, 6],  # Longer weak window — services lag hardware
    "defense_etf": [3, 4, 6],     # Broad weakness follows primes
    "dual_use": [3, 4, 10],       # Oct shutdown risk hits commercial aero harder
}

# RSI oversold thresholds by sub-sector
# Primes have ultra-low beta — they don't dip as deep as other sectors
DEFENSE_RSI_OVERSOLD = {
    "prime": 38,              # Very low beta — RSI rarely goes below 35
    "defense_tech": 30,       # High beta — deeper RSI dips are normal
    "defense_services": 35,   # Moderate beta, steady revenue
    "defense_etf": 35,        # Diversified, dampened volatility
    "dual_use": 33,           # Commercial exposure adds volatility
}


class DefenseBudgetCycleRule(Rule):
    """
    Buy defense stocks aligned with DoD fiscal year spending catalysts.

    The US defense budget follows a predictable annual cycle:
    - Feb: President submits budget request (sets topline expectations)
    - May-Jun: Armed Services Committee markups (program-specific catalysts)
    - Jul-Sep: Fiscal Q4 "use-it-or-lose-it" rush — Pentagon spent $93.4B
      in Sep 2025 alone, $50.1B in the last 5 working days
    - Oct 1: New fiscal year starts (if appropriations passed)
    - Oct-Dec: NDAA conference/passage (certainty catalyst)

    The Jul-Sep spending surge is the most tradeable pattern: 30% of all
    federal contracting dollars are obligated in these three months.

    Detection:
    - Current month in budget catalyst window (Feb, May, Jul-Sep, Nov-Dec)
    - Price above SMA_50 (base uptrend intact)
    - SMA_50 > SMA_200 (golden cross)
    - RSI in reasonable range (not overbought)
    - Volume at least 50% of average
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
        return "Defense Budget Cycle"

    @property
    def description(self) -> str:
        return "Align entries with DoD fiscal year spending catalysts (Jul-Sep rush, NDAA passage)"

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
        avg_volume = context.get_indicator("volume_sma_20")

        sub_sector = DEFENSE_SECTOR_MAP.get(context.symbol.upper())
        if not sub_sector:
            return RuleResult(
                triggered=False,
                reasoning=f"{context.symbol} not in defense sector list"
            )

        # Dual-use aerospace: budget cycle matters less (commercial dominates)
        if sub_sector == "dual_use":
            return RuleResult(
                triggered=False,
                reasoning="Dual-use aerospace — budget cycle is secondary to commercial dynamics"
            )

        # Base trend required
        if sma50 <= sma200:
            return RuleResult(
                triggered=False,
                reasoning=f"No golden cross: SMA_50 ({sma50:.2f}) <= SMA_200 ({sma200:.2f})"
            )

        # Price above SMA_200 (structural support)
        if close < sma200:
            return RuleResult(
                triggered=False,
                reasoning=f"Price ({close:.2f}) below SMA_200 ({sma200:.2f}) — support broken"
            )

        # RSI check — not overbought
        rsi_oversold = DEFENSE_RSI_OVERSOLD.get(sub_sector, 35)
        if rsi > 72:
            return RuleResult(
                triggered=False,
                reasoning=f"RSI {rsi:.1f} overbought"
            )
        if rsi < 20:
            return RuleResult(
                triggered=False,
                reasoning=f"RSI {rsi:.1f} extremely oversold — potential structural break"
            )

        # Volume check
        volume_ratio = volume / avg_volume if avg_volume > 0 else 0.0
        if volume_ratio < 0.5:
            return RuleResult(
                triggered=False,
                reasoning=f"Volume {volume_ratio:.1f}x too low"
            )

        # Seasonal assessment
        current_month = context.timestamp.month
        strong_months = DEFENSE_SEASONAL_STRENGTH.get(sub_sector, [])
        weak_months = DEFENSE_SEASONAL_WEAKNESS.get(sub_sector, [])

        is_strong = current_month in strong_months
        is_weak = current_month in weak_months

        # Block new entries in weak months
        if is_weak and context.current_position != "long":
            return RuleResult(
                triggered=False,
                reasoning=f"Weak seasonal month for defense {sub_sector}. Avoid new entries."
            )

        # Need uptrend
        uptrend = sma20 > sma50
        if not uptrend:
            return RuleResult(
                triggered=False,
                reasoning="SMA_20 below SMA_50 — no short-term uptrend"
            )

        # Calculate confidence
        base_confidence = 0.55

        month_names = {
            1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
            7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
        }

        if is_strong:
            base_confidence += self.strong_month_boost
            seasonal_status = "STRONG"
        elif is_weak:
            base_confidence -= self.weak_month_penalty
            seasonal_status = "WEAK"
        else:
            seasonal_status = "NEUTRAL"

        # Peak spending rush bonus (Jul-Sep = DoD fiscal Q4)
        if current_month in [7, 8, 9]:
            base_confidence += 0.05
            note = f"DoD fiscal Q4 spending rush — 30% of annual contracts obligated"
        elif current_month == 2:
            note = f"President's budget request month — topline catalyst"
        elif current_month in [11, 12]:
            note = f"NDAA passage window — certainty catalyst"
        elif current_month == 5:
            note = f"Armed Services Committee markups — program catalysts"
        else:
            note = f"{month_names[current_month]} — {seasonal_status.lower()} period"

        # RSI in favorable zone
        if rsi < rsi_oversold + 5:
            base_confidence += 0.05

        # Trend strength bonus
        trend_spread = (sma20 - sma50) / sma50 * 100 if sma50 > 0 else 0
        if trend_spread > 2.0:
            base_confidence += 0.05

        # Volume confirmation
        if volume_ratio > 1.2:
            base_confidence += 0.05

        # ETF stability bonus (diversified = more reliable seasonality)
        if sub_sector == "defense_etf":
            base_confidence += 0.03

        confidence = max(min(base_confidence, 0.85), 0.40)

        return RuleResult(
            triggered=True,
            signal=SignalType.BUY,
            confidence=confidence,
            reasoning=(
                f"DEFENSE BUDGET CYCLE: {context.symbol} ({sub_sector}) — "
                f"{seasonal_status} month. {note}. RSI: {rsi:.1f}, "
                f"Trend: +{trend_spread:.1f}%"
            ),
            contributing_factors={
                "sub_sector": sub_sector,
                "month": current_month,
                "seasonal_status": seasonal_status,
                "RSI_14": round(rsi, 1),
                "trend_spread_pct": round(trend_spread, 2),
                "volume_ratio": round(volume_ratio, 2),
            }
        )


class DefenseMeanReversionRule(Rule):
    """
    Buy defense stocks on oversold pullbacks with structural support.

    Defense primes mean-revert reliably because:
    - Multi-year backlogs guarantee 2-3 years of revenue (price floor)
    - Dividend yields expand on dips, attracting income buyers
    - Ultra-low betas mean SPY selloffs barely touch defense
    - Government revenue is non-cyclical (budgets are sticky)

    Geopolitical spikes mean-revert in ~28 days on average (NBER data,
    20 events). This rule catches the OTHER side: oversold pullbacks
    that revert upward due to fundamental backlog support.

    Key difference from industrial mean-reversion:
    - Lower RSI thresholds (primes rarely dip deep — beta 0.01-0.41)
    - ADX threshold lower (defense trends gently, not violently)
    - No cyclical capex exposure (government revenue ≠ economic cycle)

    Detection:
    - Price above SMA_200 (backlog-supported floor intact)
    - RSI in sub-sector-specific oversold range
    - BB_PERCENT < 0.20 (near lower Bollinger Band)
    - ADX_14 < 25 (not in strong trend against us)
    - Volume >= 50% of average
    """

    def __init__(
        self,
        bb_oversold: float = 0.20,
        rsi_floor: float = 20.0,
        rsi_ceiling: float = 45.0,
        adx_max: float = 25.0,
    ):
        self.bb_oversold = bb_oversold
        self.rsi_floor = rsi_floor
        self.rsi_ceiling = rsi_ceiling
        self.adx_max = adx_max

    @property
    def name(self) -> str:
        return "Defense Mean Reversion"

    @property
    def description(self) -> str:
        return (
            f"Buy defense stocks on oversold pullbacks (BB%<{self.bb_oversold}, "
            f"RSI {self.rsi_floor}-{self.rsi_ceiling}) with backlog support"
        )

    @property
    def required_indicators(self) -> list:
        return [
            "RSI_14", "SMA_20", "SMA_50", "SMA_200", "close",
            "BB_PERCENT", "ADX_14", "volume",
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

        sub_sector = DEFENSE_SECTOR_MAP.get(context.symbol.upper())
        if not sub_sector:
            return RuleResult(
                triggered=False,
                reasoning=f"{context.symbol} not in defense sector list"
            )

        # Defense tech is momentum-driven, not mean-reverting
        if sub_sector == "defense_tech":
            return RuleResult(
                triggered=False,
                reasoning="Defense tech is momentum-driven — use momentum rule instead"
            )

        # SMA_200 is the backlog-supported floor
        if close < sma200:
            return RuleResult(
                triggered=False,
                reasoning=f"Price ({close:.2f}) below SMA_200 ({sma200:.2f}) — floor broken"
            )

        # Golden cross preferred but not required for mean-reversion
        golden_cross = sma50 > sma200

        # ADX must confirm non-trending (mean-reverting) environment
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
        rsi_oversold = DEFENSE_RSI_OVERSOLD.get(sub_sector, 35)
        effective_ceiling = min(self.rsi_ceiling, rsi_oversold + 8)

        if rsi < self.rsi_floor:
            return RuleResult(
                triggered=False,
                reasoning=f"RSI {rsi:.1f} below {self.rsi_floor} — potential structural break"
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
                reasoning=f"Volume {volume_ratio:.1f}x too low"
            )

        # Calculate confidence
        base_confidence = 0.58

        # BB extreme oversold boost
        if bb_pct < 0.0:
            base_confidence += 0.10
        elif bb_pct < 0.05:
            base_confidence += 0.07
        elif bb_pct < 0.10:
            base_confidence += 0.04

        # RSI depth boost
        if rsi < rsi_oversold - 5:
            base_confidence += 0.10
        elif rsi < rsi_oversold - 2:
            base_confidence += 0.05

        # ADX very low = strong mean-reversion environment
        if adx < 15:
            base_confidence += 0.05

        # Golden cross bonus
        if golden_cross:
            base_confidence += 0.05

        # SMA_20 > SMA_50 = short-term trend recovering
        if sma20 > 0 and sma20 > sma50:
            base_confidence += 0.05

        # Stochastic confirmation
        if stoch_k is not None and stoch_k < 20:
            base_confidence += 0.05

        # Volume surge on selloff = capitulation (bullish for reversion)
        if volume_ratio > 1.5:
            base_confidence += 0.05
        elif volume_ratio > 1.2:
            base_confidence += 0.03

        # Prime contractor reliability bonus (backlogs = floor)
        if sub_sector == "prime":
            base_confidence += 0.05

        # ETF diversification bonus
        if sub_sector == "defense_etf":
            base_confidence += 0.03

        # Seasonal adjustment
        current_month = context.timestamp.month
        strong_months = DEFENSE_SEASONAL_STRENGTH.get(sub_sector, [])
        weak_months = DEFENSE_SEASONAL_WEAKNESS.get(sub_sector, [])

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
                f"DEFENSE MEAN REVERSION: {context.symbol} ({sub_sector}) oversold at "
                f"BB support (BB%={bb_pct:.2f}, RSI={rsi:.1f}). "
                f"ADX={adx:.1f} confirms mean-reverting. "
                f"{'Golden cross intact. ' if golden_cross else ''}"
                f"Backlog-supported price floor at SMA_200 ({sma200:.2f})."
            ),
            contributing_factors={
                "sub_sector": sub_sector,
                "BB_PERCENT": round(bb_pct, 3),
                "RSI_14": round(rsi, 1),
                "ADX_14": round(adx, 1),
                "volume_ratio": round(volume_ratio, 2),
                "rsi_oversold_threshold": rsi_oversold,
                "golden_cross": golden_cross,
            }
        )


class DefenseMomentumRule(Rule):
    """
    Buy defense stocks in confirmed uptrends during rearmament/spending cycles.

    Defense stocks trend strongly during structural spending increases:
    - NATO rearmament (2022+): EU defense +19% in 2024, +15% in 2025
    - NDAA topline growth: FY2026 authorized $924.7B
    - Global spending: $2.72T in 2024, growing 8-9% CAGR to 2028

    This is NOT for chasing geopolitical spikes (those mean-revert in 28 days).
    This is for riding the multi-year structural trend.

    Best for: Defense ETFs (ITA, XAR), defense tech (AVAV, KTOS), services (BAH)
    Also works for primes, but their ultra-low beta means momentum signals are weaker.

    Detection:
    - ADX_14 > 20 (lower than energy's 25 — defense trends gently)
    - Price above SMA_50 > SMA_200 (golden cross)
    - MACD histogram positive (momentum intact)
    - RSI 40-72 (defense doesn't get as overbought as tech)
    - Volume at least 70% of average
    """

    def __init__(
        self,
        adx_min: float = 20.0,
        rsi_max: float = 72.0,
        min_volume_ratio: float = 0.7,
    ):
        self.adx_min = adx_min
        self.rsi_max = rsi_max
        self.min_volume_ratio = min_volume_ratio

    @property
    def name(self) -> str:
        return "Defense Momentum"

    @property
    def description(self) -> str:
        return (
            f"Buy defense stocks in confirmed uptrend (ADX>{self.adx_min}, "
            f"golden cross, MACD positive) during spending cycles"
        )

    @property
    def required_indicators(self) -> list:
        return [
            "RSI_14", "SMA_20", "SMA_50", "SMA_200", "close",
            "ADX_14", "MACD_HIST", "EMA_9", "EMA_21", "volume",
        ]

    def evaluate(self, context: SymbolContext) -> RuleResult:
        rsi = context.get_indicator("RSI_14")
        sma20 = context.get_indicator("SMA_20")
        sma50 = context.get_indicator("SMA_50")
        sma200 = context.get_indicator("SMA_200")
        close = context.get_indicator("close")
        adx = context.get_indicator("ADX_14")
        macd_hist = context.get_indicator("MACD_HIST")
        ema9 = context.get_indicator("EMA_9")
        ema21 = context.get_indicator("EMA_21")
        volume = context.get_indicator("volume")
        avg_volume = context.get_indicator("volume_sma_20")

        sub_sector = DEFENSE_SECTOR_MAP.get(context.symbol.upper())
        if not sub_sector:
            return RuleResult(
                triggered=False,
                reasoning=f"{context.symbol} not in defense sector list"
            )

        # Dual-use: commercial dynamics dominate, defense momentum is secondary
        if sub_sector == "dual_use":
            return RuleResult(
                triggered=False,
                reasoning="Dual-use aerospace — commercial dynamics dominate momentum"
            )

        # Golden cross required
        if sma50 <= sma200:
            return RuleResult(
                triggered=False,
                reasoning=f"No golden cross: SMA_50 ({sma50:.2f}) <= SMA_200 ({sma200:.2f})"
            )

        # Price above SMA_50
        if close < sma50:
            return RuleResult(
                triggered=False,
                reasoning=f"Price ({close:.2f}) below SMA_50 ({sma50:.2f})"
            )

        # ADX confirms trend — lower threshold for defense (gentler trends)
        if adx < self.adx_min:
            return RuleResult(
                triggered=False,
                reasoning=f"ADX {adx:.1f} < {self.adx_min} — trend not confirmed"
            )

        # MACD histogram positive
        if macd_hist <= 0:
            return RuleResult(
                triggered=False,
                reasoning=f"MACD histogram {macd_hist:.3f} not positive — momentum lost"
            )

        # RSI: momentum zone but not overbought
        if rsi < 40:
            return RuleResult(
                triggered=False,
                reasoning=f"RSI {rsi:.1f} too weak for momentum confirmation"
            )
        if rsi > self.rsi_max:
            return RuleResult(
                triggered=False,
                reasoning=f"RSI {rsi:.1f} overbought"
            )

        # Volume check
        volume_ratio = volume / avg_volume if avg_volume > 0 else 0.0
        if volume_ratio < self.min_volume_ratio:
            return RuleResult(
                triggered=False,
                reasoning=f"Volume {volume_ratio:.1f}x below {self.min_volume_ratio}x minimum"
            )

        # EMA alignment for entry timing
        ema_aligned = ema9 > ema21

        # Pullback entry: price near EMA_21 support (within 2%)
        pullback_pct = (close - ema21) / ema21 * 100 if ema21 > 0 else 0
        near_support = pullback_pct < 2.0

        # Calculate confidence
        base_confidence = 0.55

        # ADX strength tiers (defense trends more gently)
        if adx > 30:
            base_confidence += 0.10
        elif adx > 25:
            base_confidence += 0.05

        # Volume confirmation
        if volume_ratio > 1.5:
            base_confidence += 0.10
        elif volume_ratio > 1.2:
            base_confidence += 0.05

        # EMA alignment bonus
        if ema_aligned:
            base_confidence += 0.05

        # Pullback to support = better entry
        if near_support and ema_aligned:
            base_confidence += 0.05

        # RSI sweet spot (45-60 for defense)
        if 45 <= rsi <= 60:
            base_confidence += 0.05

        # Defense tech gets momentum bonus (higher beta = stronger trends)
        if sub_sector == "defense_tech":
            base_confidence += 0.05

        # Seasonal adjustment
        current_month = context.timestamp.month
        strong_months = DEFENSE_SEASONAL_STRENGTH.get(sub_sector, [])
        weak_months = DEFENSE_SEASONAL_WEAKNESS.get(sub_sector, [])
        if current_month in strong_months:
            base_confidence += 0.05
        elif current_month in weak_months:
            base_confidence -= 0.05

        confidence = max(min(base_confidence, 0.85), 0.40)

        trend_spread = (sma50 - sma200) / sma200 * 100 if sma200 > 0 else 0

        return RuleResult(
            triggered=True,
            signal=SignalType.BUY,
            confidence=confidence,
            reasoning=(
                f"DEFENSE MOMENTUM: {context.symbol} ({sub_sector}) in confirmed uptrend. "
                f"ADX={adx:.1f}, MACD_hist={macd_hist:.3f}, RSI={rsi:.1f}. "
                f"Golden cross spread: {trend_spread:.1f}%. "
                f"Volume: {volume_ratio:.1f}x avg."
            ),
            contributing_factors={
                "sub_sector": sub_sector,
                "ADX_14": round(adx, 1),
                "MACD_HIST": round(macd_hist, 3),
                "RSI_14": round(rsi, 1),
                "volume_ratio": round(volume_ratio, 2),
                "ema_aligned": ema_aligned,
                "golden_cross_spread_pct": round(trend_spread, 2),
            }
        )


class DefenseCounterCyclicalRule(Rule):
    """
    Buy defense stocks when the broader market enters bearish regime.

    Defense is one of the strongest counter-cyclical sectors:
    - 2022: ITA +9.97% vs SPY -14.67% (+24.6% relative outperformance)
    - Ultra-low betas (0.01-0.41) provide natural protection
    - Government revenue is budget-driven, not economically cyclical
    - Flight-to-quality: institutional money rotates into defense in risk-off

    This rule fires when SPY shows bear signals but the defense stock
    maintains relative strength — a regime rotation signal.

    Detection:
    - Defense stock above SMA_50 (its own trend intact)
    - Defense stock RSI > broad market RSI (relative strength)
    - SMA_20 > SMA_50 on defense stock (local uptrend)
    - Not overbought (RSI < 70)
    """

    def __init__(
        self,
        rsi_relative_min: float = 5.0,    # Defense RSI must be this much above floor
    ):
        self.rsi_relative_min = rsi_relative_min

    @property
    def name(self) -> str:
        return "Defense Counter-Cyclical"

    @property
    def description(self) -> str:
        return "Buy defense stocks showing relative strength vs weakening broad market"

    @property
    def required_indicators(self) -> list:
        return [
            "RSI_14", "SMA_20", "SMA_50", "SMA_200", "close",
            "ADX_14", "BB_PERCENT", "volume",
        ]

    def evaluate(self, context: SymbolContext) -> RuleResult:
        rsi = context.get_indicator("RSI_14")
        sma20 = context.get_indicator("SMA_20")
        sma50 = context.get_indicator("SMA_50")
        sma200 = context.get_indicator("SMA_200")
        close = context.get_indicator("close")
        adx = context.get_indicator("ADX_14")
        bb_pct = context.get_indicator("BB_PERCENT")
        volume = context.get_indicator("volume")
        avg_volume = context.get_indicator("volume_sma_20")

        sub_sector = DEFENSE_SECTOR_MAP.get(context.symbol.upper())
        if not sub_sector:
            return RuleResult(
                triggered=False,
                reasoning=f"{context.symbol} not in defense sector list"
            )

        # Defense tech is high-beta — it sells off WITH the market, not counter-cyclical
        if sub_sector == "defense_tech":
            return RuleResult(
                triggered=False,
                reasoning="Defense tech (high beta) sells off with market — not counter-cyclical"
            )

        # Dual-use has commercial exposure — not purely counter-cyclical
        if sub_sector == "dual_use":
            return RuleResult(
                triggered=False,
                reasoning="Dual-use has commercial exposure — not purely counter-cyclical"
            )

        # Defense stock must be in its own uptrend
        if close < sma50:
            return RuleResult(
                triggered=False,
                reasoning=f"Price ({close:.2f}) below SMA_50 ({sma50:.2f}) — no relative strength"
            )

        # Short-term uptrend
        if sma20 <= sma50:
            return RuleResult(
                triggered=False,
                reasoning="SMA_20 below SMA_50 — no short-term uptrend"
            )

        # RSI must show strength (above 40 = stock not weak)
        if rsi < 40:
            return RuleResult(
                triggered=False,
                reasoning=f"RSI {rsi:.1f} too weak — defense stock also selling off"
            )

        if rsi > 70:
            return RuleResult(
                triggered=False,
                reasoning=f"RSI {rsi:.1f} overbought"
            )

        # Volume check
        volume_ratio = volume / avg_volume if avg_volume > 0 else 0.0
        if volume_ratio < 0.5:
            return RuleResult(
                triggered=False,
                reasoning=f"Volume {volume_ratio:.1f}x too low"
            )

        # Calculate confidence based on relative strength signals
        base_confidence = 0.55

        # Price well above SMA_200 = strong relative strength
        if sma200 > 0:
            dist_above_200 = (close - sma200) / sma200 * 100
            if dist_above_200 > 10:
                base_confidence += 0.10
            elif dist_above_200 > 5:
                base_confidence += 0.05

        # RSI in sweet spot (45-60 = steady strength, not overbought)
        if 45 <= rsi <= 60:
            base_confidence += 0.05

        # Trend spread (wider = stronger relative trend)
        trend_spread = (sma20 - sma50) / sma50 * 100 if sma50 > 0 else 0
        if trend_spread > 2.0:
            base_confidence += 0.05

        # BB in middle range (not stretched) = sustainable
        if bb_pct is not None and 0.3 <= bb_pct <= 0.7:
            base_confidence += 0.03

        # ADX confirming trend
        if adx > 20:
            base_confidence += 0.05

        # Volume rising = institutional rotation into defense
        if volume_ratio > 1.3:
            base_confidence += 0.07
        elif volume_ratio > 1.1:
            base_confidence += 0.03

        # Prime contractor bonus (most counter-cyclical sub-sector)
        if sub_sector == "prime":
            base_confidence += 0.05

        # ETF bonus (diversified, lower idiosyncratic risk)
        if sub_sector == "defense_etf":
            base_confidence += 0.03

        confidence = max(min(base_confidence, 0.85), 0.40)

        return RuleResult(
            triggered=True,
            signal=SignalType.BUY,
            confidence=confidence,
            reasoning=(
                f"DEFENSE COUNTER-CYCLICAL: {context.symbol} ({sub_sector}) showing "
                f"relative strength. RSI={rsi:.1f}, price {((close-sma50)/sma50*100):.1f}% "
                f"above SMA_50. Trend spread: {trend_spread:.1f}%. "
                f"Volume: {volume_ratio:.1f}x avg."
            ),
            contributing_factors={
                "sub_sector": sub_sector,
                "RSI_14": round(rsi, 1),
                "trend_spread_pct": round(trend_spread, 2),
                "volume_ratio": round(volume_ratio, 2),
                "ADX_14": round(adx, 1),
                "price_above_sma200_pct": round(
                    (close - sma200) / sma200 * 100 if sma200 > 0 else 0, 2
                ),
            }
        )
