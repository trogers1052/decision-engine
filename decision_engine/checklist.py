"""
Pre-trade checklist evaluator.

Evaluates five gating checks before a BUY signal is published:

  1. stop_loss_defined       — trade plan has a stop price set
  2. position_sized_correctly — risk_pct ≤ 2% of account
  3. rr_ratio_acceptable     — risk/reward ≥ 2:1 AND plan_valid=True
  4. no_earnings_imminent    — no earnings report within 5 trading days
  5. regime_compatible       — market regime is not outright BEAR

Status logic:
  BLOCKED  — earnings within 5 days (hard gate; the MOH lesson)
           OR position sizing > 5% (double the normal 2% limit)
  REVIEW   — any check failed but not BLOCKED
  GO       — all five checks passed

Only BUY signals are checked.  SELL / WATCH bypass the checklist entirely
(multiplier already handled by the regime layer).

Earnings data is read from Redis key ``robinhood:earnings:{SYMBOL}``
written by robinhood-sync.  Key absent = no known upcoming earnings (safe).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import redis

from .models.trade_plan import TradePlan

logger = logging.getLogger(__name__)

# Hard gates — trigger BLOCKED status
EARNINGS_HARD_GATE_DAYS = 5    # earnings within this many days → BLOCKED
MAX_RISK_PCT_BLOCKED = 5.0     # risk > 5% of account → BLOCKED
MAX_RISK_PCT_REVIEW = 2.0      # risk > 2% but ≤ 5% → REVIEW (positioned_sized_correctly=False)

MIN_RR_RATIO = 2.0             # minimum acceptable R:R
BEAR_REGIMES = {"BEAR"}        # regimes considered regime-incompatible
EARNINGS_STALENESS_HOURS = 24  # reject earnings data older than this


@dataclass
class ChecklistResult:
    """
    Result of the pre-trade checklist evaluation.

    All five bool fields are set regardless of BLOCKED/REVIEW/GO so the
    alert message can show exactly which checks failed.
    """

    # Individual checks
    stop_loss_defined: bool = False
    position_sized_correctly: bool = False     # risk_pct ≤ 2%
    rr_ratio_acceptable: bool = False          # R:R ≥ 2:1
    no_earnings_imminent: bool = False         # no earnings within 5 days
    regime_compatible: bool = False            # regime is not BEAR

    # Aggregated status
    all_checks_passed: bool = False
    status: str = "REVIEW"                    # "GO" | "REVIEW" | "BLOCKED"

    # Optional detail fields for display
    earnings_date: Optional[str] = None       # e.g. "2026-03-12"
    earnings_days_away: Optional[int] = None
    earnings_verified: Optional[bool] = None
    regime_id: str = "UNKNOWN"
    risk_pct: float = 0.0
    rr_ratio: float = 0.0

    def to_dict(self) -> dict:
        return {
            "stop_loss_defined": self.stop_loss_defined,
            "position_sized_correctly": self.position_sized_correctly,
            "rr_ratio_acceptable": self.rr_ratio_acceptable,
            "no_earnings_imminent": self.no_earnings_imminent,
            "regime_compatible": self.regime_compatible,
            "all_checks_passed": self.all_checks_passed,
            "status": self.status,
            "earnings_date": self.earnings_date,
            "earnings_days_away": self.earnings_days_away,
            "earnings_verified": self.earnings_verified,
            "regime_id": self.regime_id,
            "risk_pct": round(self.risk_pct, 2),
            "rr_ratio": round(self.rr_ratio, 2),
        }


class ChecklistEvaluator:
    """
    Evaluates the pre-trade checklist for BUY signals.

    Usage::

        evaluator = ChecklistEvaluator(
            redis_host="redis", redis_port=6379, redis_db=0
        )
        result = evaluator.evaluate(trade_plan, regime_id="BULL", symbol="WPM")
    """

    EARNINGS_KEY_PREFIX = "robinhood:earnings"

    def __init__(
        self,
        redis_host: str,
        redis_port: int,
        redis_db: int,
        redis_password: str = "",
    ):
        self._host = redis_host
        self._port = redis_port
        self._db = redis_db
        self._password = redis_password
        self._client: Optional[redis.Redis] = None

    def connect(self) -> bool:
        """Connect to Redis (best-effort; checklist degrades gracefully if unavailable)."""
        try:
            self._client = redis.Redis(
                host=self._host,
                port=self._port,
                db=self._db,
                password=self._password or None,
                decode_responses=True,
                socket_connect_timeout=3,
                socket_timeout=3,
            )
            self._client.ping()
            logger.info(
                f"ChecklistEvaluator connected to Redis {self._host}:{self._port}"
            )
            return True
        except redis.RedisError as exc:
            logger.warning(
                f"ChecklistEvaluator could not connect to Redis: {exc}. "
                "Earnings check will default to no_earnings_imminent=True (permissive)."
            )
            self._client = None
            return False

    def close(self) -> None:
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        trade_plan: Optional[TradePlan],
        regime_id: str,
        symbol: str,
    ) -> ChecklistResult:
        """
        Run all five checks and return a ChecklistResult.

        trade_plan may be None if plan generation failed or the trade plan
        engine is disabled.  In that case checks 1-3 (stop, sizing, R:R)
        default to False conservatively — you can't verify them, so you
        shouldn't pretend they passed.  Checks 4 (earnings) and 5 (regime)
        are fully independent and always run.

        This ensures the earnings hard gate fires even when trade plan
        generation throws an exception.

        Args:
            trade_plan: The generated TradePlan, or None if unavailable.
            regime_id:  Current market regime (BULL / SIDEWAYS / BEAR / UNKNOWN).
            symbol:     Ticker symbol (used to look up earnings in Redis).
        """
        result = ChecklistResult(regime_id=regime_id)

        if trade_plan is not None:
            # 1. Stop loss defined
            result.stop_loss_defined = trade_plan.stop_price > 0

            # 2. Position sizing
            result.risk_pct = trade_plan.risk_pct
            result.position_sized_correctly = trade_plan.risk_pct <= MAX_RISK_PCT_REVIEW

            # 3. R:R ratio
            result.rr_ratio = trade_plan.risk_reward_ratio
            result.rr_ratio_acceptable = (
                trade_plan.plan_valid and trade_plan.risk_reward_ratio >= MIN_RR_RATIO
            )
        # else: checks 1-3 remain False (conservative default)

        # 4. Earnings imminence (read from Redis — always runs)
        earnings = self._get_earnings(symbol)
        if earnings is None and self._client is None:
            # Redis is DOWN — we can't verify earnings. MOH lesson:
            # assume NOT safe (conservative). Status becomes REVIEW so
            # the trader sees the failed check and can verify manually.
            # A Redis outage must never silently let a trade through
            # earnings unprotected.
            logger.warning(
                f"Earnings check unavailable for {symbol} (Redis down) — "
                f"marking no_earnings_imminent=False (conservative)"
            )
            result.no_earnings_imminent = False
        elif earnings is None:
            # Redis is up but no earnings key → no known upcoming earnings → safe
            result.no_earnings_imminent = True
        else:
            days_away = earnings.get("days_away", 999)
            result.earnings_date = earnings.get("date")
            result.earnings_days_away = days_away
            result.earnings_verified = earnings.get("verified")
            result.no_earnings_imminent = days_away > EARNINGS_HARD_GATE_DAYS

        # 5. Regime compatibility (always runs)
        result.regime_compatible = regime_id not in BEAR_REGIMES

        # Aggregate
        result.all_checks_passed = (
            result.stop_loss_defined
            and result.position_sized_correctly
            and result.rr_ratio_acceptable
            and result.no_earnings_imminent
            and result.regime_compatible
        )

        # Status — hard gates first
        earnings_days = result.earnings_days_away
        is_earnings_blocked = (
            earnings_days is not None
            and earnings_days <= EARNINGS_HARD_GATE_DAYS
        )
        # Size block only applies when we have plan data
        is_size_blocked = (
            trade_plan is not None and trade_plan.risk_pct > MAX_RISK_PCT_BLOCKED
        )

        if is_earnings_blocked or is_size_blocked:
            result.status = "BLOCKED"
        elif result.all_checks_passed:
            result.status = "GO"
        else:
            result.status = "REVIEW"

        logger.info(
            f"Checklist {symbol}: {result.status} — "
            f"stop={result.stop_loss_defined}, "
            f"size={result.position_sized_correctly} ({result.risk_pct:.1f}%), "
            f"rr={result.rr_ratio_acceptable} ({result.rr_ratio:.1f}:1), "
            f"earnings={result.no_earnings_imminent} "
            f"({'n/a' if earnings_days is None else f'{earnings_days}d'}), "
            f"regime={result.regime_compatible} ({regime_id})"
        )

        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_earnings(self, symbol: str) -> Optional[dict]:
        """Read earnings data from Redis.  Returns None if key absent or Redis down."""
        if not self._client:
            return None
        try:
            key = f"{self.EARNINGS_KEY_PREFIX}:{symbol}"
            raw = self._client.get(key)
            if not raw:
                return None
            data = json.loads(raw)

            # Reject stale earnings data — if robinhood-sync hasn't refreshed
            # the key in 24h the data may be outdated and dangerous to trust.
            updated_at = data.get("updated_at")
            if updated_at:
                try:
                    age_hours = (time.time() - float(updated_at)) / 3600
                    if age_hours > EARNINGS_STALENESS_HOURS:
                        logger.warning(
                            f"Earnings data for {symbol} is {age_hours:.1f}h old "
                            f"(limit {EARNINGS_STALENESS_HOURS}h) — treating as absent"
                        )
                        return None
                except (ValueError, TypeError):
                    pass  # updated_at not a valid timestamp — use data as-is

            return data
        except (redis.RedisError, json.JSONDecodeError) as exc:
            logger.debug(f"Could not read earnings for {symbol}: {exc}")
            return None
