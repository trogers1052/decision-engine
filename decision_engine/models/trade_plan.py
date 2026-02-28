"""
Trade plan model — deterministic entry/stop/target calculation from signal indicators.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel
from datetime import datetime


class SetupType(str, Enum):
    OVERSOLD_BOUNCE = "oversold_bounce"         # RSI < 35, dip in uptrend
    PULLBACK_TO_SUPPORT = "pullback_to_support"  # Pullback to SMA_20 in uptrend
    BREAKOUT = "breakout"                        # Price/volume breakout above SMA_20
    SIGNAL = "signal"                            # Generic / multiple rules


class TradePlan(BaseModel):
    # Setup classification
    setup_type: SetupType
    rules_contributed: list[str]                # Rule names that contributed

    # Entry
    entry_price: float                          # Exact suggested entry (close at signal time)
    entry_zone_low: float                       # Valid entry range lower bound
    entry_zone_high: float                      # Valid entry range upper bound
    valid_until: datetime                       # When this plan expires

    # Stop loss
    stop_price: float                           # Exact stop price
    stop_method: str                            # "atr_2x" | "config_Xpct" | "support_sma_50" etc.
    stop_pct: float                             # % distance from entry to stop
    support_level_used: Optional[str] = None    # e.g. "SMA_50 $43.25" if stop snapped to support

    # Targets
    target_1: float                             # 2:1 R:R target
    target_2: float                             # 3:1 R:R target
    symbol_target_pct: Optional[float]          # Symbol-specific % target from rules.yaml
    resistance_note: Optional[str]              # e.g. "BB_UPPER $48.90 may resist"

    # Target context — probability, timeframe, and price positioning
    target_1_probability: Optional[float] = None  # 0.0–1.0 estimated hit probability
    target_1_est_days: Optional[int] = None       # estimated trading days to reach target 1
    target_2_probability: Optional[float] = None  # 0.0–1.0 estimated hit probability
    target_2_est_days: Optional[int] = None       # estimated trading days to reach target 2
    price_context: Optional[str] = None           # e.g. "Near 20-day high — upside limited"

    # Risk metrics
    risk_reward_ratio: float                    # (target_1 - entry) / (entry - stop)
    shares: int
    dollar_risk: float
    risk_pct: float
    position_value: float

    # Goal projection
    goal_years: Optional[float] = None             # Years to reach $1M at this stock's perf
    expected_annual_return: Optional[float] = None  # Expected annual return % from this stock

    # Invalidation
    invalidation_price: float                   # Below this = setup is dead

    # Validity
    plan_valid: bool                            # True if R:R >= min_rr_ratio
    rr_warning: Optional[str]                   # Set if R:R < min_rr_ratio
    warnings: list[str]                         # Non-blocking warnings
