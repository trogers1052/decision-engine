"""
Energy Sector Specific Rules

Specialized rules for trading energy stocks (integrated majors, E&P, midstream MLPs,
energy ETFs) based on well-documented sector dynamics.

Key Insights:
1. Energy stocks are commodity-driven — oil/gas prices dominate fundamentals
2. Upstream E&P are momentum-driven with violent reversals (ADX is the regime switch)
3. Integrated majors (XOM, CVX) mean-revert more reliably (downstream hedges upstream)
4. Midstream MLPs (EPD, ET) are yield-anchored mean-reverters (fee-based cash flows)
5. RSI oversold thresholds should be WIDER for energy (25 not 30) due to extreme moves
6. ATR-based stops essential — fixed % stops hit by normal energy volatility
7. Volume confirmation critical — false breakouts common due to institutional algo trading
8. Strong seasonality: driving season (May-Sep), winter heating (Nov-Feb)

Academic References:
- Moskowitz, Ooi, Pedersen (2012): Time-series momentum in commodity-linked equities
- DeBondt & Thaler (1985): Overreaction hypothesis (stronger in high-beta sectors)
- Karpoff (1987): Volume-price relationship, breakout validation
- Driesprong et al. (2008): Oil price changes predict energy stock returns
"""

from datetime import datetime
from typing import Optional

from .base import Rule, RuleResult, SignalType, SymbolContext


# =============================================================================
# Energy Sector Symbol Mapping
# =============================================================================

ENERGY_SECTOR_MAP = {
    # Integrated majors (upstream + downstream + chemicals)
    "XOM": "integrated",
    "CVX": "integrated",
    "EQNR": "integrated",

    # Upstream E&P (direct commodity exposure, high beta)
    "COP": "upstream",
    "EOG": "upstream",
    "OXY": "upstream",

    # Midstream MLPs (fee-based, high yield, low commodity sensitivity)
    "EPD": "midstream",
    "ET": "midstream",

    # Energy ETFs
    "XLE": "energy_etf",
    "GUSH": "leveraged_etf",

    # Canadian E&P
    "CNQ": "upstream",

    # Oilfield services (capex cycle-driven, not commodity-direct)
    "FET": "oilfield_services",
}

# Energy seasonal patterns
# Driving season = gasoline demand peaks May-Sep
# Winter heating = natural gas demand Nov-Feb
# Tax-loss selling recovery = January effect
ENERGY_SEASONAL_STRENGTH = {
    "integrated": [1, 2, 5, 6, 7, 8, 9, 11, 12],    # Benefit from both driving + winter
    "upstream": [1, 2, 5, 6, 7, 8, 9, 11, 12],       # Same but more volatile
    "midstream": [1, 2, 3, 10, 11, 12],               # Q1 + Q4 (steady throughput)
    "energy_etf": [1, 2, 5, 6, 7, 8, 9, 11, 12],
    "leveraged_etf": [1, 2],                          # Only January effect
    "oilfield_services": [1, 2, 5, 6, 7, 8, 9, 11, 12],  # Follows upstream + capex cycle
}

ENERGY_SEASONAL_WEAKNESS = {
    "integrated": [3, 4, 10],         # Shoulder months (between seasons)
    "upstream": [3, 4, 10],
    "midstream": [5, 6, 8, 9],        # Summer doldrums for midstream
    "energy_etf": [3, 4, 10],
    "leveraged_etf": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],  # Most months (decay)
    "oilfield_services": [3, 4, 10],                       # Shoulder months
}


class EnergyMomentumRule(Rule):
    """
    Buy upstream energy stocks in established uptrends with momentum confirmation.

    Energy E&P stocks trend strongly during commodity upcycles (3-6 months).
    Uses ADX to confirm trend strength, MACD histogram for momentum timing,
    and EMA alignment for entry timing.

    Best for: COP, EOG, OXY, XLE (high-beta commodity-linked names)

    Detection:
    - ADX_14 > 25 (confirmed trend, not choppy)
    - Price above SMA_50 > SMA_200 (golden cross)
    - MACD histogram positive (momentum intact)
    - RSI 45-75 (wider upper band for energy momentum)
    - Volume at least average (institutional participation)

    Natural Language:
    "Buy energy stocks pulling back to EMA support in a confirmed uptrend"
    """

    def __init__(
        self,
        adx_min: float = 25.0,         # Minimum ADX for trend confirmation
        rsi_max: float = 75.0,         # Wider than 70 for energy momentum
        min_volume_ratio: float = 0.8,  # At least 80% of average volume
    ):
        self.adx_min = adx_min
        self.rsi_max = rsi_max
        self.min_volume_ratio = min_volume_ratio

    @property
    def name(self) -> str:
        return "Energy Momentum"

    @property
    def description(self) -> str:
        return (
            f"Buy energy stocks in confirmed uptrend (ADX>{self.adx_min}, "
            f"golden cross, MACD positive)"
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

        # Check if this is an energy stock
        sub_sector = ENERGY_SECTOR_MAP.get(context.symbol.upper())
        if not sub_sector:
            return RuleResult(
                triggered=False,
                reasoning=f"{context.symbol} not in energy sector list"
            )

        # Leveraged ETFs: don't use momentum rule (decay kills)
        if sub_sector == "leveraged_etf":
            return RuleResult(
                triggered=False,
                reasoning="Leveraged ETF — momentum rule inappropriate (volatility decay)"
            )

        # Trend filters
        golden_cross = sma50 > sma200
        if not golden_cross:
            return RuleResult(
                triggered=False,
                reasoning=f"No golden cross: SMA_50 ({sma50:.2f}) <= SMA_200 ({sma200:.2f})"
            )

        above_sma50 = close > sma50
        if not above_sma50:
            return RuleResult(
                triggered=False,
                reasoning=f"Price ({close:.2f}) below SMA_50 ({sma50:.2f})"
            )

        # ADX confirms trend strength
        if adx < self.adx_min:
            return RuleResult(
                triggered=False,
                reasoning=f"ADX {adx:.1f} < {self.adx_min} — trend not confirmed"
            )

        # MACD histogram must be positive (momentum intact)
        if macd_hist <= 0:
            return RuleResult(
                triggered=False,
                reasoning=f"MACD histogram {macd_hist:.3f} not positive — momentum lost"
            )

        # RSI: momentum zone but not overbought (wider for energy)
        if rsi < 45:
            return RuleResult(
                triggered=False,
                reasoning=f"RSI {rsi:.1f} too weak for momentum confirmation"
            )
        if rsi > self.rsi_max:
            return RuleResult(
                triggered=False,
                reasoning=f"RSI {rsi:.1f} overbought (energy threshold {self.rsi_max})"
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

        # ADX strength tiers
        if adx > 35:
            base_confidence += 0.10
        elif adx > 30:
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

        # RSI sweet spot (50-65)
        if 50 <= rsi <= 65:
            base_confidence += 0.05

        # Seasonal adjustment
        current_month = context.timestamp.month
        strong_months = ENERGY_SEASONAL_STRENGTH.get(sub_sector, [])
        weak_months = ENERGY_SEASONAL_WEAKNESS.get(sub_sector, [])
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
                f"ENERGY MOMENTUM: {context.symbol} ({sub_sector}) in confirmed uptrend. "
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


class EnergyMeanReversionRule(Rule):
    """
    Buy integrated energy majors on oversold extremes when structural support holds.

    XOM and CVX mean-revert reliably because downstream operations hedge upstream
    exposure, and dividend yield creates a price floor. Uses triple-oversold
    confluence (RSI + BB + Stochastic) to avoid catching falling knives.

    Best for: XOM, CVX, EQNR (integrated majors with stable dividends)
    Also works for: XLE, EPD, ET (with lower threshold adjustments)

    Detection:
    - Price above SMA_200 (structural support intact)
    - At least 2 of 3 oversold indicators triggered:
      * RSI_14 < 30 (deep oversold for energy)
      * BB_PERCENT < 0.10 (near lower Bollinger Band)
      * Stochastic_K < 20 AND Stochastic_D < 20
    - ADX_14 < 30 or price above SMA_50 (not in confirmed downtrend)
    - Reversal evidence: Stochastic K crossing D, or MACD histogram improving

    Natural Language:
    "Buy energy majors at triple-oversold extremes when SMA_200 support holds"
    """

    def __init__(
        self,
        rsi_oversold: float = 30.0,       # RSI threshold (30 for energy, not 35)
        rsi_extreme_floor: float = 15.0,   # Below this = potential structural break
        bb_oversold: float = 0.10,         # BB_PERCENT threshold
        stoch_oversold: float = 20.0,      # Stochastic threshold
        support_tolerance_pct: float = 3.0, # Max % below SMA_200 allowed
    ):
        self.rsi_oversold = rsi_oversold
        self.rsi_extreme_floor = rsi_extreme_floor
        self.bb_oversold = bb_oversold
        self.stoch_oversold = stoch_oversold
        self.support_tolerance_pct = support_tolerance_pct

    @property
    def name(self) -> str:
        return "Energy Mean Reversion"

    @property
    def description(self) -> str:
        return (
            f"Buy energy majors at triple-oversold (RSI<{self.rsi_oversold}, "
            f"BB%<{self.bb_oversold}, Stoch<{self.stoch_oversold}) with SMA_200 support"
        )

    @property
    def required_indicators(self) -> list:
        return [
            "RSI_14", "SMA_20", "SMA_50", "SMA_200", "close",
            "BB_PERCENT", "Stochastic_K", "Stochastic_D",
            "ADX_14", "MACD_HIST", "volume",
        ]

    def evaluate(self, context: SymbolContext) -> RuleResult:
        rsi = context.get_indicator("RSI_14")
        sma20 = context.get_indicator("SMA_20")
        sma50 = context.get_indicator("SMA_50")
        sma200 = context.get_indicator("SMA_200")
        close = context.get_indicator("close")
        bb_pct = context.get_indicator("BB_PERCENT")
        stoch_k = context.get_indicator("Stochastic_K")
        stoch_d = context.get_indicator("Stochastic_D")
        adx = context.get_indicator("ADX_14")
        macd_hist = context.get_indicator("MACD_HIST")
        volume = context.get_indicator("volume")
        avg_volume = context.get_indicator("volume_sma_20")

        # Check if this is an energy stock
        sub_sector = ENERGY_SECTOR_MAP.get(context.symbol.upper())
        if not sub_sector:
            return RuleResult(
                triggered=False,
                reasoning=f"{context.symbol} not in energy sector list"
            )

        # Leveraged ETFs: skip (mean-reversion doesn't work with daily rebalancing)
        if sub_sector == "leveraged_etf":
            return RuleResult(
                triggered=False,
                reasoning="Leveraged ETF — mean-reversion inappropriate (rebalancing decay)"
            )

        # Structural support: price above SMA_200 or within tolerance
        if sma200 > 0:
            dist_below_200 = (sma200 - close) / sma200 * 100
            if dist_below_200 > self.support_tolerance_pct:
                return RuleResult(
                    triggered=False,
                    reasoning=f"Price {dist_below_200:.1f}% below SMA_200 — support broken"
                )

        # RSI extreme floor check (below this = structural break, not mean-reversion)
        if rsi < self.rsi_extreme_floor:
            return RuleResult(
                triggered=False,
                reasoning=f"RSI {rsi:.1f} below {self.rsi_extreme_floor} — potential structural break"
            )

        # Triple-oversold check (need at least 2 of 3)
        rsi_os = rsi < self.rsi_oversold
        bb_os = bb_pct < self.bb_oversold
        stoch_os = stoch_k < self.stoch_oversold and stoch_d < self.stoch_oversold

        oversold_count = sum([rsi_os, bb_os, stoch_os])
        if oversold_count < 2:
            return RuleResult(
                triggered=False,
                reasoning=(
                    f"Not enough oversold signals ({oversold_count}/3): "
                    f"RSI={rsi:.1f}{'✓' if rsi_os else '✗'}, "
                    f"BB%={bb_pct:.2f}{'✓' if bb_os else '✗'}, "
                    f"StochK={stoch_k:.1f}{'✓' if stoch_os else '✗'}"
                )
            )

        # Not in confirmed strong downtrend
        strong_downtrend = adx > 30 and close < sma50 and macd_hist < 0
        if strong_downtrend:
            return RuleResult(
                triggered=False,
                reasoning=(
                    f"Strong downtrend: ADX={adx:.1f}, price below SMA_50, "
                    f"MACD negative — not safe for mean-reversion"
                )
            )

        # Reversal evidence (at least one of)
        stoch_cross = stoch_k > stoch_d
        macd_improving = macd_hist > -0.1  # Histogram approaching zero
        has_reversal = stoch_cross or macd_improving

        if not has_reversal:
            return RuleResult(
                triggered=False,
                reasoning=(
                    f"No reversal evidence: Stoch K({stoch_k:.1f})<D({stoch_d:.1f}), "
                    f"MACD hist({macd_hist:.3f}) still falling"
                )
            )

        # Volume check — fail-closed on missing data
        volume_ratio = volume / avg_volume if avg_volume > 0 else 0.0
        if volume_ratio < 0.5:
            return RuleResult(
                triggered=False,
                reasoning=f"Insufficient volume: {volume_ratio:.1f}x of average (need 0.5x minimum)"
            )

        # Calculate confidence
        base_confidence = 0.55

        # Triple oversold = highest conviction
        if oversold_count == 3:
            base_confidence += 0.10
        else:
            base_confidence += 0.05

        # Volume spike on selloff = capitulation (bullish for reversion)
        if volume_ratio > 1.5:
            base_confidence += 0.10
        elif volume_ratio > 1.2:
            base_confidence += 0.05

        # Both reversal signals = stronger
        if stoch_cross and macd_improving:
            base_confidence += 0.05

        # Price above SMA_200 (support holding) vs below
        if close > sma200:
            base_confidence += 0.05

        # Deeper RSI oversold (within safe range)
        if rsi < 22:
            base_confidence += 0.05
        elif rsi < 25:
            base_confidence += 0.03

        # Sub-sector adjustment: midstream = more reliable mean-reversion
        if sub_sector == "midstream":
            base_confidence += 0.05

        # Seasonal adjustment
        current_month = context.timestamp.month
        strong_months = ENERGY_SEASONAL_STRENGTH.get(sub_sector, [])
        weak_months = ENERGY_SEASONAL_WEAKNESS.get(sub_sector, [])
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
                f"ENERGY MEAN REVERSION: {context.symbol} ({sub_sector}) triple-oversold "
                f"({oversold_count}/3). RSI={rsi:.1f}, BB%={bb_pct:.2f}, "
                f"StochK={stoch_k:.1f}. "
                f"{'Stoch K>D bullish cross. ' if stoch_cross else ''}"
                f"{'MACD improving. ' if macd_improving else ''}"
                f"SMA_200 support at {sma200:.2f}."
            ),
            contributing_factors={
                "sub_sector": sub_sector,
                "oversold_count": oversold_count,
                "RSI_14": round(rsi, 1),
                "BB_PERCENT": round(bb_pct, 3),
                "Stochastic_K": round(stoch_k, 1),
                "Stochastic_D": round(stoch_d, 1),
                "ADX_14": round(adx, 1),
                "volume_ratio": round(volume_ratio, 2),
                "stoch_cross": stoch_cross,
                "macd_improving": macd_improving,
            }
        )


class EnergySeasonalityRule(Rule):
    """
    Adjust signals based on energy sector seasonal patterns.

    Well-documented energy seasonality:
    - Driving season (May-Sep): gasoline demand peaks, refining margins widen
    - Winter heating (Nov-Feb): natural gas demand peaks
    - January effect: tax-loss selling recovery
    - Shoulder months (Mar-Apr, Oct): transitional, often weaker

    Key difference from generic seasonality: energy is one of few sectors
    that OUTPERFORMS in summer (driving season), contra "sell in May".

    Natural Language:
    "Boost buy confidence in energy's strong seasonal months, reduce in weak"
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
        return "Energy Seasonality"

    @property
    def description(self) -> str:
        return "Adjust confidence based on energy sector seasonal patterns"

    @property
    def required_indicators(self) -> list:
        return ["RSI_14", "SMA_20", "SMA_50", "close"]

    def evaluate(self, context: SymbolContext) -> RuleResult:
        rsi = context.get_indicator("RSI_14")
        sma20 = context.get_indicator("SMA_20")
        sma50 = context.get_indicator("SMA_50")
        close = context.get_indicator("close")

        # Check if this is an energy stock
        sub_sector = ENERGY_SECTOR_MAP.get(context.symbol.upper())
        if not sub_sector:
            return RuleResult(
                triggered=False,
                reasoning=f"{context.symbol} not in energy sector list"
            )

        # Leveraged ETFs: seasonal pattern exists but decay dominates
        if sub_sector == "leveraged_etf":
            return RuleResult(
                triggered=False,
                reasoning="Leveraged ETF — seasonality dominated by daily rebalancing decay"
            )

        current_month = context.timestamp.month

        strong_months = ENERGY_SEASONAL_STRENGTH.get(sub_sector, [])
        weak_months = ENERGY_SEASONAL_WEAKNESS.get(sub_sector, [])

        is_strong = current_month in strong_months
        is_weak = current_month in weak_months

        # Base signal: need uptrend and reasonable RSI
        uptrend = sma20 > sma50
        reasonable_rsi = 25 <= rsi <= 70  # Wider for energy

        if not uptrend:
            return RuleResult(
                triggered=False,
                reasoning="No uptrend — seasonality rule requires base trend"
            )

        if not reasonable_rsi:
            if rsi < 25:
                reason = f"RSI {rsi:.1f} deeply oversold — use mean-reversion rule"
            else:
                reason = f"RSI {rsi:.1f} overbought"
            return RuleResult(triggered=False, reasoning=reason)

        # Don't trigger on weak months for new entries
        if is_weak and context.current_position != "long":
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

        if is_strong:
            base_confidence += self.strong_month_boost
            seasonal_status = "STRONG"
            note = f"{month_names[current_month]} is historically strong for {sub_sector}"
        elif is_weak:
            base_confidence -= self.weak_month_penalty
            seasonal_status = "WEAK"
            note = f"{month_names[current_month]} is historically weak for {sub_sector}"
        else:
            seasonal_status = "NEUTRAL"
            note = "Neutral seasonal period"

        # Driving season bonus for upstream/integrated
        if current_month in [6, 7, 8] and sub_sector in ("upstream", "integrated", "energy_etf"):
            base_confidence += 0.05
            note += " (peak driving season)"

        # Trend strength bonus
        trend_spread = (sma20 - sma50) / sma50 * 100 if sma50 > 0 else 0
        if trend_spread > 2.0:
            base_confidence += 0.05

        confidence = max(min(base_confidence, 0.85), 0.40)

        return RuleResult(
            triggered=True,
            signal=SignalType.BUY,
            confidence=confidence,
            reasoning=(
                f"ENERGY SEASONALITY: {context.symbol} ({sub_sector}) — "
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


class MidstreamYieldReversionRule(Rule):
    """
    Buy midstream MLPs when price drops push implied yield to extremes.

    EPD and ET have remarkably stable fee-based cash flows. When prices drop
    5%+ below SMA_200, dividend yields expand enough to attract institutional
    income buyers, creating a reliable mean-reversion floor.

    Uses distance from SMA_200 as a yield expansion proxy (no direct dividend
    data needed). When price falls far below SMA_200, the yield has expanded
    proportionally.

    Best for: EPD, ET (also applicable to integrated majors with stable divs)

    Detection:
    - Price 5%+ below SMA_200 (yield expansion proxy)
    - RSI < 35 or BB_PERCENT < 0.15 (oversold)
    - Not in freefall (ADX < 30 or price above SMA_50)
    - Price stabilizing (within 3% of SMA_20)

    Natural Language:
    "Buy midstream MLPs at expanded yields when price stabilizes below SMA_200"
    """

    def __init__(
        self,
        min_discount_pct: float = 5.0,    # Min % below SMA_200 for yield expansion
        rsi_oversold: float = 35.0,        # Slightly higher for low-vol MLPs
        bb_oversold: float = 0.15,         # BB threshold
    ):
        self.min_discount_pct = min_discount_pct
        self.rsi_oversold = rsi_oversold
        self.bb_oversold = bb_oversold

    @property
    def name(self) -> str:
        return "Midstream Yield Reversion"

    @property
    def description(self) -> str:
        return (
            f"Buy midstream MLPs at {self.min_discount_pct}%+ discount to SMA_200 "
            f"(yield expansion signal)"
        )

    @property
    def required_indicators(self) -> list:
        return [
            "RSI_14", "SMA_20", "SMA_50", "SMA_200", "close",
            "BB_PERCENT", "ADX_14", "ATR_14", "volume",
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

        # Check if this is an energy stock (works best for midstream but not exclusive)
        sub_sector = ENERGY_SECTOR_MAP.get(context.symbol.upper())
        if not sub_sector:
            return RuleResult(
                triggered=False,
                reasoning=f"{context.symbol} not in energy sector list"
            )

        # Best for midstream, also applicable to integrated (with dividends)
        if sub_sector not in ("midstream", "integrated"):
            return RuleResult(
                triggered=False,
                reasoning=f"{sub_sector} is not yield-driven — use momentum rule instead"
            )

        # Yield expansion proxy: price below SMA_200
        if sma200 <= 0:
            return RuleResult(triggered=False, reasoning="SMA_200 not available")

        discount_pct = (sma200 - close) / sma200 * 100
        if discount_pct < self.min_discount_pct:
            return RuleResult(
                triggered=False,
                reasoning=(
                    f"Price only {discount_pct:.1f}% below SMA_200 "
                    f"(need {self.min_discount_pct}%+ for yield expansion signal)"
                )
            )

        # Oversold check (RSI or BB)
        rsi_os = rsi < self.rsi_oversold
        bb_os = bb_pct < self.bb_oversold
        if not (rsi_os or bb_os):
            return RuleResult(
                triggered=False,
                reasoning=(
                    f"Not oversold: RSI={rsi:.1f} (need <{self.rsi_oversold}), "
                    f"BB%={bb_pct:.2f} (need <{self.bb_oversold})"
                )
            )

        # Not in freefall
        freefall = adx > 30 and close < sma50
        if freefall:
            return RuleResult(
                triggered=False,
                reasoning=f"In freefall: ADX={adx:.1f}, price below SMA_50 — wait for stabilization"
            )

        # Price stabilizing: within 3% of SMA_20
        dist_to_sma20 = abs(close - sma20) / sma20 * 100 if sma20 > 0 else 99
        stabilizing = dist_to_sma20 < 3.0
        if not stabilizing:
            return RuleResult(
                triggered=False,
                reasoning=f"Price {dist_to_sma20:.1f}% from SMA_20 — not stabilized yet"
            )

        # Volume check — fail-closed on missing data
        volume_ratio = volume / avg_volume if avg_volume > 0 else 0.0
        if volume_ratio < 0.5:
            return RuleResult(
                triggered=False,
                reasoning=f"Insufficient volume: {volume_ratio:.1f}x of average (need 0.5x minimum)"
            )

        # Calculate confidence
        base_confidence = 0.55

        # Deeper discount = higher yield = more income buyer interest
        if discount_pct > 10.0:
            base_confidence += 0.15
        elif discount_pct > 8.0:
            base_confidence += 0.10
        elif discount_pct > 6.0:
            base_confidence += 0.05

        # Both RSI and BB oversold = stronger
        if rsi_os and bb_os:
            base_confidence += 0.05

        # Extreme RSI
        if rsi < 25:
            base_confidence += 0.05

        # Volume capitulation
        if volume_ratio > 1.5:
            base_confidence += 0.05

        # Midstream gets reliability bonus
        if sub_sector == "midstream":
            base_confidence += 0.05

        # Seasonal adjustment
        current_month = context.timestamp.month
        strong_months = ENERGY_SEASONAL_STRENGTH.get(sub_sector, [])
        if current_month in strong_months:
            base_confidence += 0.05

        confidence = max(min(base_confidence, 0.85), 0.40)

        return RuleResult(
            triggered=True,
            signal=SignalType.BUY,
            confidence=confidence,
            reasoning=(
                f"YIELD REVERSION: {context.symbol} ({sub_sector}) trading "
                f"{discount_pct:.1f}% below SMA_200 (implied yield expansion). "
                f"RSI={rsi:.1f}, BB%={bb_pct:.2f}. "
                f"ADX={adx:.1f} (not in freefall). "
                f"Target: return to SMA_50 ({sma50:.2f})."
            ),
            contributing_factors={
                "sub_sector": sub_sector,
                "discount_pct": round(discount_pct, 1),
                "RSI_14": round(rsi, 1),
                "BB_PERCENT": round(bb_pct, 3),
                "ADX_14": round(adx, 1),
                "volume_ratio": round(volume_ratio, 2),
                "sma200": round(sma200, 2),
            }
        )
