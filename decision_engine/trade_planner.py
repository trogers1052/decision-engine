"""
Trade Plan Engine — deterministic entry/stop/target/R:R calculation.

Reads account balance from Redis (robinhood:portfolio → total_equity),
falls back to settings.default_account_balance if Redis is unavailable.

All math is deterministic. $0 LLM cost.
"""

import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Optional

from .models.signals import AggregatedSignal
from .models.trade_plan import SetupType, TradePlan

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Attempt to import PositionSizer from stop-loss-guardian package.
# The package may be installed as stop_loss_guardian (pip install -e .)
# from the sibling directory, or not installed at all (tests mock it).
# -----------------------------------------------------------------------
try:
    from stop_loss_guardian.position_sizer import PositionSizer

    _HAS_POSITION_SIZER = True
except ImportError:
    _HAS_POSITION_SIZER = False
    PositionSizer = None  # type: ignore

# -----------------------------------------------------------------------
# Attempt to import Redis client
# -----------------------------------------------------------------------
try:
    import redis as _redis_lib

    _HAS_REDIS = True
except ImportError:
    _HAS_REDIS = False
    _redis_lib = None  # type: ignore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _end_of_trading_day() -> datetime:
    """Return today's 21:00 UTC (4 PM ET close + buffer) as a UTC datetime."""
    now = datetime.now(timezone.utc)
    eod = now.replace(hour=21, minute=0, second=0, microsecond=0)
    if eod <= now:
        eod += timedelta(days=1)
    return eod


def _trading_days_from_now(days: int) -> datetime:
    """Rough approximation: add calendar days (weekends excluded best-effort)."""
    now = datetime.now(timezone.utc)
    added = 0
    dt = now
    while added < days:
        dt += timedelta(days=1)
        if dt.weekday() < 5:  # Mon–Fri
            added += 1
    return dt


# ---------------------------------------------------------------------------
# TradePlanEngine
# ---------------------------------------------------------------------------

class TradePlanEngine:
    """
    Generates a complete TradePlan from an AggregatedSignal + indicator snapshot.

    Constructor params (all sourced from rules.yaml trade_plan_engine section):
        atr_multiplier          — default 2.0
        min_rr_ratio            — default 2.0  (plan_valid=False if below this)
        stop_min_pct            — default 3.0  (widen stop if tighter)
        stop_max_pct            — default 15.0 (cap stop if wider)
        default_account_balance — fallback when Redis unavailable
        account_balance_redis_key — Redis hash key for portfolio data
        symbol_exit_strategies  — dict of symbol → exit_strategy from rules.yaml
        redis_host / redis_port / redis_password / redis_db — Redis connection params
    """

    # Cache account balance for 60 seconds to avoid Redis overhead
    _BALANCE_CACHE_TTL = 60

    def __init__(
        self,
        atr_multiplier: float = 2.0,
        min_rr_ratio: float = 2.0,
        stop_min_pct: float = 3.0,
        stop_max_pct: float = 15.0,
        default_account_balance: float = 888.80,
        account_balance_redis_key: str = "robinhood:portfolio",
        symbol_exit_strategies: Optional[dict] = None,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_password: str = "",
        redis_db: int = 1,
    ):
        self.atr_multiplier = atr_multiplier
        self.min_rr_ratio = min_rr_ratio
        self.stop_min_pct = stop_min_pct
        self.stop_max_pct = stop_max_pct
        self.default_account_balance = default_account_balance
        self.account_balance_redis_key = account_balance_redis_key
        self.symbol_exit_strategies: dict = symbol_exit_strategies or {}

        # Redis connection params (lazy connect)
        self._redis_host = redis_host
        self._redis_port = redis_port
        self._redis_password = redis_password
        self._redis_db = redis_db
        self._redis_client = None

        # Account balance cache
        self._cached_balance: Optional[float] = None
        self._balance_cached_at: Optional[datetime] = None
        self._redis_failure_until: Optional[float] = None

        # PositionSizer — max 2% risk, max 20% position
        if _HAS_POSITION_SIZER:
            self._sizer = PositionSizer(
                max_risk_pct=Decimal("2.0"),
                max_position_pct=Decimal("20.0"),
            )
        else:
            self._sizer = None
            logger.warning(
                "stop_loss_guardian.position_sizer not available — "
                "position sizing will be estimated from 2% risk rule"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        signal: AggregatedSignal,
        indicators: dict,
    ) -> TradePlan:
        """
        Generate a complete TradePlan for the given signal + indicators snapshot.

        Required indicator keys: close, ATR_14
        Optional:               SMA_20, SMA_50, BB_UPPER
        """
        symbol = signal.symbol
        rules_triggered = signal.contributing_signals
        rule_names = [s.rule_name for s in rules_triggered]
        rules_contributed = rule_names

        warnings: list[str] = []

        # ----------------------------------------------------------------
        # Extract indicators (with safe fallbacks)
        # ----------------------------------------------------------------
        close = float(indicators.get("close", 0.0))
        atr = float(indicators.get("ATR_14", 0.0))
        sma_20 = float(indicators.get("SMA_20", close))
        sma_50 = float(indicators.get("SMA_50", close * 0.95))
        bb_upper = indicators.get("BB_UPPER")
        if bb_upper is not None:
            bb_upper = float(bb_upper)

        if close <= 0:
            raise ValueError(f"Invalid close price for {symbol}: {close}")
        if atr <= 0:
            raise ValueError(f"Invalid ATR_14 for {symbol}: {atr}")

        # ----------------------------------------------------------------
        # Step 1: Classify setup
        # ----------------------------------------------------------------
        setup_type = self._classify_setup(rule_names)

        # ----------------------------------------------------------------
        # Step 2: Entry zone
        # ----------------------------------------------------------------
        entry_price, entry_zone_low, entry_zone_high, valid_until = (
            self._entry_zone(setup_type, close, sma_20)
        )

        # ----------------------------------------------------------------
        # Step 3: Stop loss (per-stock % + support snapping)
        # ----------------------------------------------------------------
        raw_stop, stop_method, stop_pct_val, support_level_used = (
            self._calculate_stop(symbol, entry_price, atr, indicators, warnings)
        )

        # ----------------------------------------------------------------
        # Step 4: Invalidation price
        # ----------------------------------------------------------------
        invalidation = self._invalidation_price(
            setup_type, raw_stop, sma_50, close, atr
        )

        # ----------------------------------------------------------------
        # Step 5: Targets (per-stock profit target from rules.yaml)
        # ----------------------------------------------------------------
        risk_per_share = entry_price - raw_stop
        if risk_per_share <= 0:
            raise ValueError(
                f"Stop price {raw_stop:.4f} >= entry price {entry_price:.4f} for {symbol}"
            )

        symbol_target_pct: Optional[float] = (
            self.symbol_exit_strategies.get(symbol, {}).get("profit_target")
        )

        if symbol_target_pct is not None:
            # Per-stock targets from backtested exit strategy
            target_1 = entry_price * (1 + symbol_target_pct)
            target_2 = entry_price * (1 + symbol_target_pct * 1.5)
        else:
            # Fallback: mechanical 2:1 / 3:1 for unknown symbols
            target_1 = entry_price + (risk_per_share * 2.0)
            target_2 = entry_price + (risk_per_share * 3.0)

        rr_ratio = (target_1 - entry_price) / risk_per_share

        # BB_UPPER resistance is noted but does NOT affect targets or R:R
        resistance_note: Optional[str] = None
        if bb_upper is not None and bb_upper > entry_price and bb_upper < target_1:
            resistance_note = (
                f"BB_UPPER ${bb_upper:.2f} is nearby resistance — "
                f"watch for stall before target ${target_1:.2f}"
            )

        # ----------------------------------------------------------------
        # Step 5b: Target probability, timeframe, and price context
        # ----------------------------------------------------------------
        t1_prob = self._estimate_probability(
            entry_price, target_1, raw_stop, indicators
        )
        t2_prob = self._estimate_probability(
            entry_price, target_2, raw_stop, indicators
        )
        t1_days = self._estimate_days(entry_price, target_1, atr, indicators)
        t2_days = self._estimate_days(entry_price, target_2, atr, indicators)
        price_context = self._build_price_context(close, indicators)

        # ----------------------------------------------------------------
        # Step 6: Position sizing
        # ----------------------------------------------------------------
        account_balance = self._get_account_balance()
        shares, dollar_risk, risk_pct, position_value, sizing_warnings = (
            self._size_position(symbol, entry_price, raw_stop, account_balance, target_1)
        )
        warnings.extend(sizing_warnings)

        # ----------------------------------------------------------------
        # Step 6b: Goal projection
        # ----------------------------------------------------------------
        goal_years, expected_annual_return = self._calculate_goal_projection(
            symbol, account_balance
        )

        # ----------------------------------------------------------------
        # Step 7: Validate and flag
        # ----------------------------------------------------------------
        # Configured stocks always pass — their PT/SL ratio IS the
        # backtested edge.  Unknown symbols still require min R:R.
        has_config = symbol in self.symbol_exit_strategies
        plan_valid = has_config or rr_ratio >= self.min_rr_ratio
        rr_warning: Optional[str] = None
        if not plan_valid:
            rr_warning = (
                f"R:R {rr_ratio:.1f}:1 is below minimum "
                f"{self.min_rr_ratio:.0f}:1 — consider skipping"
            )

        # Reject if position can't be sized (stock too expensive for account)
        if shares == 0:
            plan_valid = False
            rr_warning = (
                f"Cannot size position: stock at ${entry_price:.2f} "
                f"exceeds 2% risk budget on ${account_balance:.0f} account"
            )

        return TradePlan(
            setup_type=setup_type,
            rules_contributed=rules_contributed,
            entry_price=round(entry_price, 2),
            entry_zone_low=round(entry_zone_low, 2),
            entry_zone_high=round(entry_zone_high, 2),
            valid_until=valid_until,
            stop_price=round(raw_stop, 2),
            stop_method=stop_method,
            stop_pct=round(stop_pct_val, 2),
            support_level_used=support_level_used,
            target_1=round(target_1, 2),
            target_2=round(target_2, 2),
            symbol_target_pct=symbol_target_pct,
            resistance_note=resistance_note,
            target_1_probability=round(t1_prob, 2),
            target_1_est_days=t1_days,
            target_2_probability=round(t2_prob, 2),
            target_2_est_days=t2_days,
            price_context=price_context,
            risk_reward_ratio=round(rr_ratio, 2),
            shares=shares,
            dollar_risk=round(dollar_risk, 2),
            risk_pct=round(risk_pct, 2),
            position_value=round(position_value, 2),
            goal_years=goal_years,
            expected_annual_return=expected_annual_return,
            invalidation_price=round(invalidation, 2),
            plan_valid=plan_valid,
            rr_warning=rr_warning,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Step 1: Setup classification
    # ------------------------------------------------------------------

    def _classify_setup(self, rule_names: list[str]) -> SetupType:
        rule_set = set(rule_names)
        if rule_set & {
            "Enhanced Buy Dip",
            "Momentum Reversal",
            "Buy Dip in Uptrend",
            "Strong Buy Signal",
            "RSI + MACD Confluence",
            "Dip Recovery",
        }:
            return SetupType.OVERSOLD_BOUNCE
        if "Trend Continuation" in rule_set:
            return SetupType.PULLBACK_TO_SUPPORT
        if "Commodity Breakout" in rule_set or "Volume Breakout" in rule_set:
            return SetupType.BREAKOUT
        return SetupType.SIGNAL

    # ------------------------------------------------------------------
    # Step 2: Entry zone
    # ------------------------------------------------------------------

    def _entry_zone(
        self,
        setup_type: SetupType,
        close: float,
        sma_20: float,
    ) -> tuple[float, float, float, datetime]:
        if setup_type == SetupType.OVERSOLD_BOUNCE:
            entry = close
            low = close * 0.997
            high = close * 1.005
            valid_until = _end_of_trading_day()

        elif setup_type == SetupType.PULLBACK_TO_SUPPORT:
            entry = sma_20
            low = sma_20 * 0.995
            high = sma_20 * 1.005
            valid_until = _trading_days_from_now(2)

        elif setup_type == SetupType.BREAKOUT:
            entry = close * 1.001
            low = close
            high = close * 1.01
            # Breakouts go or fail fast — 2 hours
            valid_until = datetime.now(timezone.utc) + timedelta(hours=2)

        else:  # SIGNAL
            entry = close
            low = close * 0.998
            high = close * 1.005
            valid_until = _end_of_trading_day()

        return entry, low, high, valid_until

    # ------------------------------------------------------------------
    # Step 3: Stop loss
    # ------------------------------------------------------------------

    # Proximity threshold for support snapping (% of entry price)
    _SNAP_PROXIMITY_PCT = 1.5
    # Buffer below support level (0.5% below)
    _SUPPORT_BUFFER = 0.005

    def _calculate_stop(
        self,
        symbol: str,
        entry_price: float,
        atr: float,
        indicators: dict,
        warnings: list[str],
    ) -> tuple[float, str, float, Optional[str]]:
        """Return (stop_price, stop_method, stop_pct, support_level_used)."""

        # 1. Per-stock percentage stop from rules.yaml exit_strategy
        symbol_stop_pct = self.symbol_exit_strategies.get(symbol, {}).get("stop_loss")

        if symbol_stop_pct is not None:
            raw_stop = entry_price * (1 - symbol_stop_pct)
            stop_pct = symbol_stop_pct * 100
            stop_method = f"config_{stop_pct:.0f}pct"
        else:
            # Fallback: ATR-based (existing behavior for unknown symbols)
            raw_stop = entry_price - (atr * self.atr_multiplier)
            stop_pct = (entry_price - raw_stop) / entry_price * 100

            if stop_pct < self.stop_min_pct:
                floor_pct = self.stop_min_pct + 1.0
                raw_stop = entry_price * (1 - floor_pct / 100)
                stop_method = f"percentage_{floor_pct:.0f}pct"
                stop_pct = floor_pct
                warnings.append(
                    f"ATR-based stop was <{self.stop_min_pct:.0f}% — widened to {floor_pct:.0f}%"
                )
            elif stop_pct > self.stop_max_pct:
                raw_stop = entry_price * 0.90
                stop_method = "percentage_10pct"
                stop_pct = 10.0
                warnings.append("ATR-based stop was >15% — capped at 10%")
            else:
                stop_method = "atr_2x"

        # 2. Support snapping: if a support level is close to the stop and
        #    slightly above it, place the stop just below that support instead.
        support_level_used: Optional[str] = None

        support_levels: dict[str, float] = {}
        for key in ("SMA_20", "SMA_50", "SMA_200", "BB_LOWER"):
            val = indicators.get(key)
            if val is not None:
                fval = float(val)
                if 0 < fval < entry_price:
                    support_levels[key] = fval

        if support_levels:
            best_name: Optional[str] = None
            best_price: Optional[float] = None

            for name, price in support_levels.items():
                dist_pct = abs(price - raw_stop) / entry_price * 100
                if dist_pct <= self._SNAP_PROXIMITY_PCT and price > raw_stop:
                    if best_price is None or price < best_price:
                        best_name = name
                        best_price = price

            if best_name is not None and best_price is not None:
                snapped_stop = best_price * (1 - self._SUPPORT_BUFFER)
                stop_pct = (entry_price - snapped_stop) / entry_price * 100
                support_level_used = f"{best_name} ${best_price:.2f}"
                raw_stop = snapped_stop
                stop_method = f"support_{best_name.lower()}"
                warnings.append(
                    f"Stop snapped below {best_name} "
                    f"(${best_price:.2f}) → ${snapped_stop:.2f}"
                )

        return raw_stop, stop_method, stop_pct, support_level_used

    # ------------------------------------------------------------------
    # Goal projection
    # ------------------------------------------------------------------

    def _calculate_goal_projection(
        self,
        symbol: str,
        account_balance: float,
        monthly_contribution: float = 50.0,
        goal_amount: float = 1_000_000.0,
        max_years: int = 50,
    ) -> tuple[Optional[float], Optional[float]]:
        """
        Calculate years to reach goal based on this stock's backtested performance.

        Returns (goal_years, expected_annual_return_pct) or (None, None) if
        the required fields (win_rate, profit_target, stop_loss, trades_per_year)
        are not configured for this symbol.
        """
        exit_strat = self.symbol_exit_strategies.get(symbol, {})
        win_rate = exit_strat.get("win_rate")
        profit_target = exit_strat.get("profit_target")
        stop_loss = exit_strat.get("stop_loss")
        trades_per_year = exit_strat.get("trades_per_year")

        if any(v is None for v in [win_rate, profit_target, stop_loss, trades_per_year]):
            return None, None

        # Expected return per trade
        e_per_trade = win_rate * profit_target - (1 - win_rate) * stop_loss
        if e_per_trade <= 0:
            return None, None  # negative expectancy — never reaches goal

        # Annual return via geometric compounding of per-trade returns
        annual_return = (1 + e_per_trade) ** trades_per_year - 1
        if annual_return <= 0:
            return None, None

        # Compound year-by-year with monthly contributions
        balance = account_balance
        annual_contribution = monthly_contribution * 12
        for year in range(1, max_years + 1):
            balance = balance * (1 + annual_return) + annual_contribution
            if balance >= goal_amount:
                return float(year), round(annual_return * 100, 1)

        return float(max_years), round(annual_return * 100, 1)

    # ------------------------------------------------------------------
    # Step 4: Invalidation
    # ------------------------------------------------------------------

    def _invalidation_price(
        self,
        setup_type: SetupType,
        raw_stop: float,
        sma_50: float,
        close: float,
        atr: float,
    ) -> float:
        if setup_type == SetupType.OVERSOLD_BOUNCE:
            return min(raw_stop * 0.99, sma_50 * 0.99)
        elif setup_type == SetupType.PULLBACK_TO_SUPPORT:
            return sma_50 * 0.99
        elif setup_type == SetupType.BREAKOUT:
            return close - (atr * 0.5)
        else:  # SIGNAL
            return raw_stop * 0.99

    # ------------------------------------------------------------------
    # Step 5b: Target probability, timeframe, and price context
    # ------------------------------------------------------------------

    def _estimate_probability(
        self,
        entry: float,
        target: float,
        stop: float,
        indicators: dict,
    ) -> float:
        """
        Heuristic probability estimate for reaching a target price.

        Uses indicator-based adjustments around base probabilities derived
        from the R:R ratio.  Higher R:R targets are harder to reach.
        """
        risk = entry - stop
        reward = target - entry
        rr = reward / risk if risk > 0 else 0.0

        # Base probability inversely related to R:R (empirical swing-trade heuristics)
        if rr <= 1.5:
            base = 0.55
        elif rr <= 2.0:
            base = 0.45
        elif rr <= 2.5:
            base = 0.38
        elif rr <= 3.0:
            base = 0.30
        else:
            base = 0.22

        adj = 0.0

        # --- RSI: oversold = bullish, overbought = bearish ---
        rsi = indicators.get("RSI_14")
        if rsi is not None:
            rsi = float(rsi)
            if rsi < 30:
                adj += 0.12
            elif rsi < 40:
                adj += 0.08
            elif rsi > 80:
                adj -= 0.12
            elif rsi > 70:
                adj -= 0.08

        # --- MACD momentum direction ---
        macd = indicators.get("MACD")
        macd_sig = indicators.get("MACD_SIGNAL")
        if macd is not None and macd_sig is not None:
            if float(macd) > float(macd_sig):
                adj += 0.05  # bullish momentum
            else:
                adj -= 0.05  # bearish momentum

        # --- ADX trend strength ---
        adx = indicators.get("ADX_14")
        if adx is not None:
            adx = float(adx)
            if adx > 30:
                adj += 0.08  # strong trend
            elif adx > 20:
                adj += 0.03  # moderate trend
            elif adx < 15:
                adj -= 0.05  # no trend, choppy

        # --- Bollinger Band position: target above BB_UPPER = harder ---
        bb_upper = indicators.get("BB_UPPER")
        bb_lower = indicators.get("BB_LOWER")
        close = float(indicators.get("close", entry))
        if bb_upper is not None and target > float(bb_upper):
            adj -= 0.08  # target above upper band
        if bb_lower is not None and close <= float(bb_lower) * 1.02:
            adj += 0.05  # price near lower band, room to expand

        # --- SMA alignment: full uptrend favors targets ---
        sma_20 = indicators.get("SMA_20")
        sma_50 = indicators.get("SMA_50")
        sma_200 = indicators.get("SMA_200")
        if sma_20 is not None and sma_50 is not None and sma_200 is not None:
            s20, s50, s200 = float(sma_20), float(sma_50), float(sma_200)
            if close > s20 > s50 > s200:
                adj += 0.05  # full alignment
            elif close < s20 and close < s50:
                adj -= 0.05  # below key averages

        # --- Extension penalty: far above SMA_20 = harder ---
        if sma_20 is not None:
            pct_above_sma20 = (close - float(sma_20)) / float(sma_20) * 100
            if pct_above_sma20 > 5.0:
                adj -= 0.08  # extended
            elif pct_above_sma20 > 3.0:
                adj -= 0.04  # moderately extended

        # --- Volume confirmation ---
        vol = indicators.get("volume")
        vol_avg = indicators.get("volume_sma_20")
        if vol is not None and vol_avg is not None and float(vol_avg) > 0:
            if float(vol) > float(vol_avg) * 1.5:
                adj += 0.05  # strong volume
            elif float(vol) < float(vol_avg) * 0.5:
                adj -= 0.03  # low volume, weak conviction

        return max(0.15, min(0.80, base + adj))

    def _estimate_days(
        self,
        entry: float,
        target: float,
        atr: float,
        indicators: dict,
    ) -> int:
        """
        Rough estimate of trading days to reach target based on ATR.

        Assumes price advances ~0.5 ATR per day in the desired direction
        on average, adjusted for trend strength (ADX).
        """
        distance = target - entry
        if distance <= 0 or atr <= 0:
            return 1

        # Daily progress factor: fraction of ATR the stock moves favorably
        factor = 0.5  # base: half ATR per day

        adx = indicators.get("ADX_14")
        if adx is not None:
            adx = float(adx)
            if adx > 30:
                factor = 0.7  # trending strongly, faster
            elif adx > 20:
                factor = 0.55
            elif adx < 15:
                factor = 0.3  # choppy, slow

        est = distance / (atr * factor)
        return max(1, min(60, round(est)))  # cap 1–60 days

    def _build_price_context(self, close: float, indicators: dict) -> str:
        """
        Generate a short human-readable context about where price sits
        relative to key technical levels.
        """
        notes: list[str] = []

        # BB position — where is price within the Bollinger range?
        bb_upper = indicators.get("BB_UPPER")
        bb_lower = indicators.get("BB_LOWER")
        if bb_upper is not None and bb_lower is not None:
            bb_u, bb_l = float(bb_upper), float(bb_lower)
            bb_range = bb_u - bb_l
            if bb_range > 0:
                bb_pos = (close - bb_l) / bb_range
                if bb_pos > 0.90:
                    notes.append("At upper Bollinger Band — extended")
                elif bb_pos > 0.75:
                    notes.append("Upper half of BB range")
                elif bb_pos < 0.10:
                    notes.append("At lower Bollinger Band — oversold")
                elif bb_pos < 0.25:
                    notes.append("Lower half of BB range")

        # SMA relationship
        sma_20 = indicators.get("SMA_20")
        sma_50 = indicators.get("SMA_50")
        sma_200 = indicators.get("SMA_200")

        if sma_20 is not None:
            pct_from_sma20 = (close - float(sma_20)) / float(sma_20) * 100
            if pct_from_sma20 > 5.0:
                notes.append(f"{pct_from_sma20:.1f}% above SMA20 — stretched")
            elif abs(pct_from_sma20) <= 1.0:
                notes.append("At SMA20 support")

        if sma_200 is not None:
            if close < float(sma_200):
                notes.append("Below SMA200 — long-term downtrend")

        # Trend alignment
        if sma_20 is not None and sma_50 is not None and sma_200 is not None:
            s20, s50, s200 = float(sma_20), float(sma_50), float(sma_200)
            if close > s20 > s50 > s200:
                notes.append("Full trend alignment (bullish)")
            elif close < s20 < s50:
                notes.append("Below key averages")

        # RSI context
        rsi = indicators.get("RSI_14")
        if rsi is not None:
            rsi = float(rsi)
            if rsi > 75:
                notes.append(f"RSI {rsi:.0f} — overbought, reversal risk")
            elif rsi < 30:
                notes.append(f"RSI {rsi:.0f} — deeply oversold")

        if not notes:
            notes.append("Mid-range, no extremes")

        return ". ".join(notes)

    # ------------------------------------------------------------------
    # Step 6: Position sizing
    # ------------------------------------------------------------------

    def _size_position(
        self,
        symbol: str,
        entry_price: float,
        stop_price: float,
        account_balance: float,
        target_1: float,
    ) -> tuple[int, float, float, float, list[str]]:
        """Return (shares, dollar_risk, risk_pct, position_value, warnings)."""
        warnings: list[str] = []

        if self._sizer is not None:
            try:
                result = self._sizer.calculate(
                    symbol=symbol,
                    entry_price=Decimal(str(round(entry_price, 4))),
                    stop_price=Decimal(str(round(stop_price, 4))),
                    account_balance=Decimal(str(round(account_balance, 2))),
                    target_price=Decimal(str(round(target_1, 4))),
                )
                warnings.extend(result.warnings)
                return (
                    result.max_shares,
                    float(result.dollar_risk),
                    float(result.risk_pct),
                    float(result.position_value),
                    warnings,
                )
            except Exception as exc:
                logger.warning(f"PositionSizer error for {symbol}: {exc} — using fallback")

        # Fallback: 2% risk rule
        max_dollar_risk = account_balance * 0.02
        risk_per_share = entry_price - stop_price
        if risk_per_share <= 0:
            return 0, 0.0, 0.0, 0.0, warnings
        shares = int(max_dollar_risk / risk_per_share)
        if shares == 0:
            risk_needed = risk_per_share / account_balance * 100 if account_balance > 0 else 0.0
            warnings.append(
                f"Stock too expensive: 1 share of ${entry_price:.2f} risks "
                f"${risk_per_share:.2f} ({risk_needed:.1f}% of account) — "
                f"exceeds 2% max"
            )
            return 0, 0.0, 0.0, 0.0, warnings
        dollar_risk = shares * risk_per_share
        risk_pct = (dollar_risk / account_balance) * 100 if account_balance > 0 else 0.0
        position_value = shares * entry_price
        return shares, dollar_risk, risk_pct, position_value, warnings

    # ------------------------------------------------------------------
    # Account balance (Redis + cache)
    # ------------------------------------------------------------------

    def _get_account_balance(self) -> float:
        """Read account balance from Redis, use cache if fresh, fall back to default."""
        now = datetime.now(timezone.utc)

        # Return cached value if still fresh
        if (
            self._cached_balance is not None
            and self._balance_cached_at is not None
            and (now - self._balance_cached_at).total_seconds() < self._BALANCE_CACHE_TTL
        ):
            return self._cached_balance

        # Try Redis
        balance = self._fetch_balance_from_redis()
        if balance is not None:
            self._cached_balance = balance
            self._balance_cached_at = now
            return balance

        # Fallback
        logger.debug(
            f"Using default_account_balance={self.default_account_balance} "
            "(Redis unavailable or key missing)"
        )
        return self.default_account_balance

    def _fetch_balance_from_redis(self) -> Optional[float]:
        """Try to read total_equity from robinhood:portfolio in Redis."""
        if not _HAS_REDIS:
            return None

        import time
        if self._redis_failure_until and time.time() < self._redis_failure_until:
            return None

        try:
            if self._redis_client is None:
                kwargs: dict = {
                    "host": self._redis_host,
                    "port": self._redis_port,
                    "db": self._redis_db,
                    "decode_responses": True,
                    "socket_connect_timeout": 2,
                    "socket_timeout": 2,
                }
                if self._redis_password:
                    kwargs["password"] = self._redis_password
                self._redis_client = _redis_lib.Redis(**kwargs)

            raw = self._redis_client.hget(self.account_balance_redis_key, "total_equity")
            if raw is None:
                return None
            return float(raw)

        except Exception as exc:
            logger.warning(f"Redis unavailable for balance fetch: {exc}")
            self._redis_client = None
            # Short backoff — position sizing accuracy depends on fresh balance.
            # 60s is too long; stale balance causes over/under-sizing.
            self._redis_failure_until = time.time() + 15
            return None

    # ------------------------------------------------------------------
    # Class method: build from rules.yaml config dict
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: dict,
        symbol_exit_strategies: Optional[dict] = None,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_password: str = "",
        redis_db: int = 1,
    ) -> "TradePlanEngine":
        """Construct engine from the full rules.yaml config dict."""
        tpe_cfg = config.get("trade_plan_engine", {})
        return cls(
            atr_multiplier=float(tpe_cfg.get("atr_multiplier", 2.0)),
            min_rr_ratio=float(tpe_cfg.get("min_rr_ratio", 2.0)),
            stop_min_pct=float(tpe_cfg.get("stop_min_pct", 3.0)),
            stop_max_pct=float(tpe_cfg.get("stop_max_pct", 15.0)),
            default_account_balance=float(
                tpe_cfg.get("default_account_balance", 888.80)
            ),
            account_balance_redis_key=tpe_cfg.get(
                "account_balance_redis_key", "robinhood:portfolio"
            ),
            symbol_exit_strategies=symbol_exit_strategies or {},
            redis_host=redis_host,
            redis_port=redis_port,
            redis_password=redis_password,
            redis_db=redis_db,
        )
