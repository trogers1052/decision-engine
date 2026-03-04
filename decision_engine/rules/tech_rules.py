"""
Big Tech / Growth Stock Specific Rules

Specialized rules for trading large-cap tech, SaaS, semiconductor, and growth stocks
based on well-documented sector dynamics.

Key Insights:
1. Mega-cap tech (AMZN, GOOGL) mean-revert reliably at moving averages (institutional support)
2. SaaS stocks (CRM, NOW, CRWD) are momentum-dominant — trend-following works, fading is risky
3. Semi equipment (LRCX, AMAT) are cyclical — only buy oversold in uptrends (above 200-SMA)
4. Memory (MU) is the most cyclical — extreme RSI oversold (25) produces reliable bounces
5. Low-beta tech (APH, QQQ) mean-revert with tight parameters
6. Growth stocks pull back to EMA_21 in uptrends — the most common institutional entry pattern
7. Tech seasonality: strong Nov-Jan, weak May-Jun, volatile Sep-Oct
8. RSI thresholds vary by sub-sector: mega-cap 33, SaaS 30, semi 30, memory 25, low-beta 35

Academic References:
- Ball & Brown (1968): Post-earnings announcement drift
- Jegadeesh & Titman (1993): Momentum profits in growth stocks
- Connors & Alvarez (2009): Short-term RSI mean-reversion on ETFs
- DeBondt & Thaler (1985): Overreaction in high-beta sectors
"""

from datetime import datetime
from typing import Optional

from .base import Rule, RuleResult, SignalType, SymbolContext


# =============================================================================
# Tech Sector Symbol Mapping
# =============================================================================

TECH_SECTOR_MAP = {
    # Mega-cap tech (quality growth, strong balance sheets, mean-revert at support)
    "AMZN": "mega_cap",
    "GOOGL": "mega_cap",
    "AAPL": "mega_cap",

    # Enterprise SaaS (high growth, momentum-dominant, earnings-sensitive)
    "CRM": "saas",
    "NOW": "saas",
    "CRWD": "saas",
    "NET": "saas",
    "PANW": "saas",      # Cybersecurity SaaS
    "SOFI": "fintech",   # Fintech, not traditional SaaS

    # Semiconductor equipment (cyclical growth, semi capex cycle)
    "LRCX": "semi_equip",
    "AMAT": "semi_equip",

    # Memory chips (most cyclical, extreme mean-reversion at cycle turns)
    "MU": "memory",

    # Electronic components (low-beta, consistent compounder, mean-reverts)
    "APH": "low_beta_tech",

    # AI/Data center speculative (very high beta, narrative-driven)
    "ALAB": "ai_speculative",

    # Tech ETFs (lower vol, well-documented mean-reversion strategies)
    "QQQ": "tech_etf",
    "XLK": "tech_etf",
}

# Tech seasonal patterns
# Nov-Jan: strong (holiday spending, year-end window dressing)
# May-Jun: weak (sell in May)
# Sep-Oct: volatile (historically crash-prone months)
TECH_SEASONAL_STRENGTH = {
    "mega_cap": [1, 2, 7, 11, 12],           # Jan-Feb, Jul (Prime Day), Nov-Dec
    "saas": [1, 2, 9, 11, 12],               # Q1 + Dreamforce(Sep) + Q4
    "semi_equip": [1, 2, 3, 10, 11, 12],     # Q1 capex orders + Q4
    "memory": [1, 2, 3, 10, 11],             # Cycle restocking + Q4
    "low_beta_tech": [1, 2, 10, 11, 12],     # Q1 + Q4 (industrial cycle)
    "ai_speculative": [1, 2, 11, 12],        # Q1 + Q4 (AI conference season)
    "fintech": [1, 2, 11, 12],
    "tech_etf": [1, 2, 4, 11, 12],           # Q1 + April recovery + Q4
}

TECH_SEASONAL_WEAKNESS = {
    "mega_cap": [5, 6, 9],                   # Sell in May + Sep volatility
    "saas": [5, 6, 8],                       # Summer doldrums
    "semi_equip": [5, 6, 7],                 # Summer weakness
    "memory": [5, 6, 7],                     # Summer cycle trough risk
    "low_beta_tech": [5, 6, 7],
    "ai_speculative": [5, 6, 7, 8, 9],       # Long weak period (narrative vacuum)
    "fintech": [5, 6, 7],
    "tech_etf": [5, 6, 9],                   # Sell in May + Sep
}

# RSI oversold thresholds by sub-sector (tech stocks have higher floors than commodities)
TECH_RSI_OVERSOLD = {
    "mega_cap": 33.0,          # Institutional support prevents deep oversold
    "saas": 30.0,              # Higher vol, can get genuinely oversold
    "semi_equip": 30.0,        # Cyclical — gets oversold in downturns
    "memory": 25.0,            # Most cyclical — genuine extremes
    "low_beta_tech": 35.0,     # Rarely deeply oversold
    "ai_speculative": 22.0,    # Extreme stock, extreme thresholds
    "fintech": 28.0,           # High beta fintech
    "tech_etf": 30.0,          # Standard
}


class TechEMAPullbackRule(Rule):
    """
    Buy growth stocks on controlled pullbacks to the EMA_21 in confirmed uptrends.

    This is the most common institutional 'add to winners' pattern for growth stocks.
    Works best for consistent trend-followers: NOW, AMZN, APH, CRM.
    Less effective for highly volatile stocks: MU, ALAB, CRWD.

    Detection:
    - Price within proximity_pct of EMA_21 (touching or slightly below)
    - Stacked moving averages: EMA_21 > SMA_50 > SMA_200 (confirmed uptrend)
    - RSI in the 40-65 range (pulling back, not collapsing)
    - Not in freefall: price above SMA_50 (pullback, not breakdown)

    Academic basis: Jegadeesh & Titman (1993) momentum, institutional rebalancing
    """

    def __init__(
        self,
        proximity_pct: float = 1.5,    # Within 1.5% of EMA_21
        min_rsi: float = 38.0,
        max_rsi: float = 65.0,
        min_volume_ratio: float = 0.7,  # At least 70% of avg volume
    ):
        self.proximity_pct = proximity_pct
        self.min_rsi = min_rsi
        self.max_rsi = max_rsi
        self.min_volume_ratio = min_volume_ratio

    @property
    def name(self) -> str:
        return "Tech EMA Pullback"

    @property
    def description(self) -> str:
        return "Buy growth stocks on pullback to EMA_21 in confirmed uptrend"

    @property
    def required_indicators(self) -> list:
        return ["RSI_14", "EMA_21", "SMA_50", "SMA_200", "close", "volume"]

    def evaluate(self, context: SymbolContext) -> RuleResult:
        sub_sector = TECH_SECTOR_MAP.get(context.symbol.upper())
        if not sub_sector:
            return RuleResult(
                triggered=False,
                reasoning=f"{context.symbol} not in tech stock list"
            )

        # Skip highly volatile stocks where pullbacks overshoot EMA
        if sub_sector in ("ai_speculative",):
            return RuleResult(
                triggered=False,
                reasoning=f"{context.symbol} too volatile for EMA pullback (use momentum rules)"
            )

        rsi = context.get_indicator("RSI_14")
        ema21 = context.get_indicator("EMA_21")
        sma50 = context.get_indicator("SMA_50")
        sma200 = context.get_indicator("SMA_200")
        close = context.get_indicator("close")
        volume = context.get_indicator("volume")
        avg_volume = context.get_indicator("volume_sma_20", volume)

        # Must have stacked moving averages (confirmed uptrend)
        if not (ema21 > sma50 > sma200):
            return RuleResult(
                triggered=False,
                reasoning=f"No stacked MAs: EMA_21={ema21:.2f}, SMA_50={sma50:.2f}, SMA_200={sma200:.2f}"
            )

        # Price must be near EMA_21 (within proximity_pct above or below)
        dist_to_ema = (close - ema21) / ema21 * 100
        if dist_to_ema > self.proximity_pct:
            return RuleResult(
                triggered=False,
                reasoning=f"Price {dist_to_ema:+.1f}% from EMA_21 — too far above (not a pullback)"
            )
        if dist_to_ema < -self.proximity_pct * 2:
            return RuleResult(
                triggered=False,
                reasoning=f"Price {dist_to_ema:+.1f}% from EMA_21 — too far below (breakdown, not pullback)"
            )

        # Price must still be above SMA_50 (pullback, not breakdown)
        if close < sma50:
            return RuleResult(
                triggered=False,
                reasoning=f"Price ${close:.2f} below SMA_50 ${sma50:.2f} — this is a breakdown, not a pullback"
            )

        # RSI must be in pullback range (not collapsing, not overbought)
        if rsi < self.min_rsi:
            return RuleResult(
                triggered=False,
                reasoning=f"RSI {rsi:.1f} too low — momentum broken"
            )
        if rsi > self.max_rsi:
            return RuleResult(
                triggered=False,
                reasoning=f"RSI {rsi:.1f} too high — not a pullback entry"
            )

        # Volume check (not required to be high — pullbacks often have lower volume)
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
        if volume_ratio < self.min_volume_ratio:
            return RuleResult(
                triggered=False,
                reasoning=f"Volume {volume_ratio:.1f}x too low (need {self.min_volume_ratio}x)"
            )

        # Calculate confidence
        base_confidence = 0.55

        # Closer to EMA = higher confidence (better entry)
        if abs(dist_to_ema) < 0.5:
            base_confidence += 0.15
        elif abs(dist_to_ema) < 1.0:
            base_confidence += 0.10
        elif abs(dist_to_ema) < 1.5:
            base_confidence += 0.05

        # RSI sweet spot for pullback (45-55 = controlled pullback)
        if 45 <= rsi <= 55:
            base_confidence += 0.05

        # Volume uptick on pullback (possible buying)
        if volume_ratio > 1.2:
            base_confidence += 0.05

        # Trend strength bonus
        trend_spread = (ema21 - sma200) / sma200 * 100
        if trend_spread > 10.0:
            base_confidence += 0.05

        confidence = min(base_confidence, 0.85)

        return RuleResult(
            triggered=True,
            signal=SignalType.BUY,
            confidence=confidence,
            reasoning=(
                f"EMA PULLBACK: {context.symbol} ({sub_sector}) pulling back to EMA_21 "
                f"({dist_to_ema:+.1f}%) in confirmed uptrend. RSI: {rsi:.1f}, "
                f"Volume: {volume_ratio:.1f}x. Trend spread: {trend_spread:.1f}%"
            ),
            contributing_factors={
                "sub_sector": sub_sector,
                "dist_to_ema21_pct": round(dist_to_ema, 2),
                "RSI_14": round(rsi, 1),
                "volume_ratio": round(volume_ratio, 2),
                "trend_spread_pct": round(trend_spread, 2),
            }
        )


class TechMeanReversionRule(Rule):
    """
    Buy tech stocks at triple-oversold confluence (RSI + BB + Stochastic) near support.

    Tech stocks (especially mega-caps and low-beta tech) have strong institutional
    support at key levels. When RSI, Bollinger Bands, and Stochastic all signal
    oversold simultaneously near the SMA_200, it's a high-conviction mean-reversion entry.

    Works best for: GOOGL, AMZN, APH, QQQ (strong institutional buy-the-dip)
    Less reliable for: CRWD, ALAB (event-driven selloffs can continue)

    Detection:
    - RSI_14 below sub-sector-specific oversold threshold
    - BB_PERCENT < bb_oversold (near lower Bollinger Band)
    - Stochastic K < stoch_oversold
    - Price near SMA_200 or SMA_50 support
    - Reversal evidence: stochastic cross or RSI turning up

    Academic basis: DeBondt & Thaler (1985) overreaction hypothesis
    """

    def __init__(
        self,
        bb_oversold: float = 0.10,
        stoch_oversold: float = 20.0,
        support_tolerance_pct: float = 5.0,
    ):
        self.bb_oversold = bb_oversold
        self.stoch_oversold = stoch_oversold
        self.support_tolerance_pct = support_tolerance_pct

    @property
    def name(self) -> str:
        return "Tech Mean Reversion"

    @property
    def description(self) -> str:
        return "Buy tech stocks at triple-oversold (RSI+BB+Stochastic) near support"

    @property
    def required_indicators(self) -> list:
        return [
            "RSI_14", "BB_PERCENT", "STOCH_K", "STOCH_D",
            "SMA_50", "SMA_200", "MACD", "MACD_SIGNAL", "close",
        ]

    def evaluate(self, context: SymbolContext) -> RuleResult:
        sub_sector = TECH_SECTOR_MAP.get(context.symbol.upper())
        if not sub_sector:
            return RuleResult(
                triggered=False,
                reasoning=f"{context.symbol} not in tech stock list"
            )

        # Skip speculative stocks (event-driven selloffs continue)
        if sub_sector in ("ai_speculative",):
            return RuleResult(
                triggered=False,
                reasoning=f"{context.symbol} too speculative for mean-reversion"
            )

        rsi = context.get_indicator("RSI_14")
        bb_pct = context.get_indicator("BB_PERCENT")
        stoch_k = context.get_indicator("STOCH_K")
        stoch_d = context.get_indicator("STOCH_D")
        sma50 = context.get_indicator("SMA_50")
        sma200 = context.get_indicator("SMA_200")
        macd = context.get_indicator("MACD")
        macd_signal = context.get_indicator("MACD_SIGNAL")
        close = context.get_indicator("close")

        # Get sub-sector-specific RSI oversold threshold
        rsi_threshold = TECH_RSI_OVERSOLD.get(sub_sector, 30.0)

        # RSI must be oversold (sub-sector-specific threshold)
        if rsi > rsi_threshold:
            return RuleResult(
                triggered=False,
                reasoning=f"RSI {rsi:.1f} not oversold for {sub_sector} (need < {rsi_threshold})"
            )

        # Reject extreme freefall (likely structural issue, not a dip)
        rsi_floor = max(rsi_threshold - 15, 10.0)
        if rsi < rsi_floor:
            return RuleResult(
                triggered=False,
                reasoning=f"RSI {rsi:.1f} in freefall — too risky for mean-reversion"
            )

        # Need at least 2 of 3 oversold confirmations
        oversold_count = 0
        if rsi <= rsi_threshold:
            oversold_count += 1
        if bb_pct <= self.bb_oversold:
            oversold_count += 1
        if stoch_k <= self.stoch_oversold:
            oversold_count += 1

        if oversold_count < 2:
            return RuleResult(
                triggered=False,
                reasoning=f"Only {oversold_count}/3 oversold signals (need 2+)"
            )

        # Price must be near support (SMA_50 or SMA_200)
        dist_to_sma50 = abs(close - sma50) / sma50 * 100
        dist_to_sma200 = abs(close - sma200) / sma200 * 100
        near_sma50 = dist_to_sma50 <= self.support_tolerance_pct
        near_sma200 = dist_to_sma200 <= self.support_tolerance_pct

        if not (near_sma50 or near_sma200):
            return RuleResult(
                triggered=False,
                reasoning=f"Not near support: {dist_to_sma50:.1f}% from SMA_50, {dist_to_sma200:.1f}% from SMA_200"
            )

        # Check for reversal evidence
        has_reversal = False
        reversal_type = "none"

        # Stochastic cross (K crossing above D)
        if stoch_k > stoch_d:
            has_reversal = True
            reversal_type = "stoch_cross"

        # MACD improving (histogram getting less negative)
        macd_hist = macd - macd_signal
        if macd_hist > -0.5:  # MACD close to crossing
            has_reversal = True
            reversal_type = "macd_improving"

        # For strong triple-oversold, waive the reversal requirement
        if oversold_count >= 3:
            has_reversal = True
            reversal_type = "triple_oversold"

        if not has_reversal:
            return RuleResult(
                triggered=False,
                reasoning="No reversal evidence (need stoch cross or MACD improving)"
            )

        # Calculate confidence
        base_confidence = 0.55

        # More oversold confirmations = higher confidence
        if oversold_count >= 3:
            base_confidence += 0.15
        elif oversold_count >= 2:
            base_confidence += 0.10

        # Near SMA_200 = stronger support
        if near_sma200:
            base_confidence += 0.10
            support_level = "SMA_200"
        else:
            support_level = "SMA_50"

        # Mega-cap/low-beta bonus (stronger institutional support)
        if sub_sector in ("mega_cap", "low_beta_tech", "tech_etf"):
            base_confidence += 0.05

        # Stochastic cross confirms reversal
        if stoch_k > stoch_d and stoch_k < 30:
            base_confidence += 0.05

        confidence = min(base_confidence, 0.85)

        return RuleResult(
            triggered=True,
            signal=SignalType.BUY,
            confidence=confidence,
            reasoning=(
                f"TECH MEAN REVERSION: {context.symbol} ({sub_sector}) "
                f"{oversold_count}/3 oversold at {support_level}. "
                f"RSI: {rsi:.1f}, BB%: {bb_pct:.2f}, Stoch: {stoch_k:.1f}. "
                f"Reversal: {reversal_type}"
            ),
            contributing_factors={
                "sub_sector": sub_sector,
                "oversold_count": oversold_count,
                "support_level": support_level,
                "RSI_14": round(rsi, 1),
                "BB_PERCENT": round(bb_pct, 3),
                "STOCH_K": round(stoch_k, 1),
                "reversal_type": reversal_type,
            }
        )


class TechSeasonalityRule(Rule):
    """
    Adjust confidence for tech-specific seasonal patterns.

    Tech stocks have well-documented seasonality:
    - Strong: Nov-Jan (holiday spending, year-end window dressing)
    - Weak: May-Jun (sell in May), Sep (crash-prone month)
    - Sub-sector specific: semi equipment strongest Q1, SaaS strongest Q4

    This rule boosts confidence in strong months and penalizes weak months.
    Requires base uptrend to trigger (SMA_20 > SMA_50).
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
        return "Tech Seasonality"

    @property
    def description(self) -> str:
        return "Adjust confidence for tech-specific seasonal patterns"

    @property
    def required_indicators(self) -> list:
        return ["RSI_14", "SMA_20", "SMA_50", "close"]

    def evaluate(self, context: SymbolContext) -> RuleResult:
        sub_sector = TECH_SECTOR_MAP.get(context.symbol.upper())
        if not sub_sector:
            return RuleResult(
                triggered=False,
                reasoning=f"{context.symbol} not in tech stock list"
            )

        rsi = context.get_indicator("RSI_14")
        sma20 = context.get_indicator("SMA_20")
        sma50 = context.get_indicator("SMA_50")
        close = context.get_indicator("close")

        # Need uptrend
        if sma20 <= sma50:
            return RuleResult(
                triggered=False,
                reasoning="No uptrend — tech seasonality requires base trend"
            )

        # RSI check
        if rsi < 30 or rsi > 70:
            return RuleResult(
                triggered=False,
                reasoning=f"RSI {rsi:.1f} outside tradeable range"
            )

        current_month = context.timestamp.month

        strong_months = TECH_SEASONAL_STRENGTH.get(sub_sector, [])
        weak_months = TECH_SEASONAL_WEAKNESS.get(sub_sector, [])

        is_strong = current_month in strong_months
        is_weak = current_month in weak_months

        if is_weak and context.current_position != "long":
            return RuleResult(
                triggered=False,
                reasoning=f"Weak seasonal month for {sub_sector}. Avoid new entries."
            )

        base_confidence = 0.55
        if is_strong:
            base_confidence += self.strong_month_boost
            seasonal_status = "STRONG"
        elif is_weak:
            base_confidence -= self.weak_month_penalty
            seasonal_status = "WEAK"
        else:
            seasonal_status = "NEUTRAL"

        # Trend strength bonus
        trend_spread = (sma20 - sma50) / sma50 * 100
        if trend_spread > 2.0:
            base_confidence += 0.05

        confidence = max(min(base_confidence, 0.85), 0.40)

        month_names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                       7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}

        return RuleResult(
            triggered=True,
            signal=SignalType.BUY,
            confidence=confidence,
            reasoning=(
                f"TECH SEASONALITY: {context.symbol} ({sub_sector}) — "
                f"{seasonal_status} month ({month_names[current_month]}). "
                f"RSI: {rsi:.1f}, Trend: +{trend_spread:.1f}%"
            ),
            contributing_factors={
                "sub_sector": sub_sector,
                "month": current_month,
                "seasonal_status": seasonal_status,
                "RSI_14": round(rsi, 1),
                "trend_spread_pct": round(trend_spread, 2),
            }
        )
