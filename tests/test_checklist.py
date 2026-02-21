"""
Tests for ChecklistEvaluator.

All tests use unittest.mock to avoid real Redis connections.
"""

import json
import unittest
from datetime import date, datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

from decision_engine.checklist import (
    ChecklistEvaluator,
    ChecklistResult,
    EARNINGS_HARD_GATE_DAYS,
    MAX_RISK_PCT_BLOCKED,
    MAX_RISK_PCT_REVIEW,
    MIN_RR_RATIO,
)
from decision_engine.models.trade_plan import TradePlan, SetupType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_plan(
    stop_price: float = 10.0,
    risk_pct: float = 1.5,
    rr_ratio: float = 2.5,
    plan_valid: bool = True,
) -> TradePlan:
    """Build a minimal TradePlan for checklist tests."""
    entry = 20.0
    stop = entry - stop_price  # ignored; we set stop_price directly
    return TradePlan(
        setup_type=SetupType.SIGNAL,
        rules_contributed=["rule_a"],
        entry_price=entry,
        entry_zone_low=entry * 0.99,
        entry_zone_high=entry * 1.01,
        valid_until=datetime.now(timezone.utc) + timedelta(hours=8),
        stop_price=stop_price,
        stop_method="atr_2x",
        stop_pct=abs(entry - stop_price) / entry * 100,
        target_1=entry * 1.1,
        target_2=entry * 1.15,
        symbol_target_pct=None,
        resistance_note=None,
        risk_reward_ratio=rr_ratio,
        shares=10,
        dollar_risk=15.0,
        risk_pct=risk_pct,
        position_value=200.0,
        invalidation_price=9.0,
        plan_valid=plan_valid,
        rr_warning=None if plan_valid else "R:R below 2:1",
        warnings=[],
    )


def _make_evaluator(earnings_payload=None) -> ChecklistEvaluator:
    """Return a ChecklistEvaluator with a mocked Redis client."""
    evaluator = ChecklistEvaluator(
        redis_host="localhost", redis_port=6379, redis_db=0
    )
    mock_redis = MagicMock()
    if earnings_payload is None:
        mock_redis.get.return_value = None
    else:
        mock_redis.get.return_value = json.dumps(earnings_payload)
    evaluator._client = mock_redis
    return evaluator


def _earnings(days_away: int, verified: bool = True) -> dict:
    """Build an earnings payload with the given days_away."""
    report_date = (date.today() + timedelta(days=days_away)).isoformat()
    return {
        "date": report_date,
        "timing": "pm",
        "verified": verified,
        "days_away": days_away,
    }


# ---------------------------------------------------------------------------
# ChecklistResult.to_dict
# ---------------------------------------------------------------------------

class TestChecklistResultToDict(unittest.TestCase):

    def test_to_dict_contains_all_keys(self):
        result = ChecklistResult(
            stop_loss_defined=True,
            position_sized_correctly=True,
            rr_ratio_acceptable=True,
            no_earnings_imminent=True,
            regime_compatible=True,
            all_checks_passed=True,
            status="GO",
            regime_id="BULL",
            risk_pct=1.5,
            rr_ratio=2.5,
        )
        d = result.to_dict()
        for key in (
            "stop_loss_defined", "position_sized_correctly", "rr_ratio_acceptable",
            "no_earnings_imminent", "regime_compatible", "all_checks_passed",
            "status", "earnings_date", "earnings_days_away", "earnings_verified",
            "regime_id", "risk_pct", "rr_ratio",
        ):
            self.assertIn(key, d)

    def test_go_status_serializes(self):
        result = ChecklistResult(status="GO", all_checks_passed=True)
        self.assertEqual(result.to_dict()["status"], "GO")


# ---------------------------------------------------------------------------
# Stop loss check
# ---------------------------------------------------------------------------

class TestStopLossCheck(unittest.TestCase):

    def test_positive_stop_price_passes(self):
        ev = _make_evaluator()
        result = ev.evaluate(_make_plan(stop_price=9.5), "BULL", "WPM")
        self.assertTrue(result.stop_loss_defined)

    def test_zero_stop_price_fails(self):
        ev = _make_evaluator()
        result = ev.evaluate(_make_plan(stop_price=0.0), "BULL", "WPM")
        self.assertFalse(result.stop_loss_defined)


# ---------------------------------------------------------------------------
# Position sizing check
# ---------------------------------------------------------------------------

class TestPositionSizingCheck(unittest.TestCase):

    def test_at_2pct_passes(self):
        ev = _make_evaluator()
        result = ev.evaluate(_make_plan(risk_pct=2.0), "BULL", "WPM")
        self.assertTrue(result.position_sized_correctly)

    def test_below_2pct_passes(self):
        ev = _make_evaluator()
        result = ev.evaluate(_make_plan(risk_pct=1.0), "BULL", "WPM")
        self.assertTrue(result.position_sized_correctly)

    def test_above_2pct_fails(self):
        ev = _make_evaluator()
        result = ev.evaluate(_make_plan(risk_pct=2.1), "BULL", "WPM")
        self.assertFalse(result.position_sized_correctly)

    def test_above_5pct_triggers_blocked(self):
        ev = _make_evaluator()
        result = ev.evaluate(_make_plan(risk_pct=5.1), "BULL", "WPM")
        self.assertFalse(result.position_sized_correctly)
        self.assertEqual(result.status, "BLOCKED")


# ---------------------------------------------------------------------------
# R:R check
# ---------------------------------------------------------------------------

class TestRRRatioCheck(unittest.TestCase):

    def test_rr_at_2_passes(self):
        ev = _make_evaluator()
        result = ev.evaluate(_make_plan(rr_ratio=2.0), "BULL", "WPM")
        self.assertTrue(result.rr_ratio_acceptable)

    def test_rr_above_2_passes(self):
        ev = _make_evaluator()
        result = ev.evaluate(_make_plan(rr_ratio=3.5), "BULL", "WPM")
        self.assertTrue(result.rr_ratio_acceptable)

    def test_rr_below_2_fails(self):
        ev = _make_evaluator()
        result = ev.evaluate(_make_plan(rr_ratio=1.9), "BULL", "WPM")
        self.assertFalse(result.rr_ratio_acceptable)

    def test_plan_invalid_fails_rr_even_if_ratio_ok(self):
        # plan_valid=False overrides the ratio check
        ev = _make_evaluator()
        result = ev.evaluate(_make_plan(rr_ratio=2.5, plan_valid=False), "BULL", "WPM")
        self.assertFalse(result.rr_ratio_acceptable)


# ---------------------------------------------------------------------------
# Earnings imminence check
# ---------------------------------------------------------------------------

class TestEarningsImminenceCheck(unittest.TestCase):

    def test_no_redis_key_passes(self):
        """Absent key → no known earnings → safe."""
        ev = _make_evaluator(earnings_payload=None)
        result = ev.evaluate(_make_plan(), "BULL", "SPY")
        self.assertTrue(result.no_earnings_imminent)
        self.assertIsNone(result.earnings_date)

    def test_earnings_far_away_passes(self):
        ev = _make_evaluator(_earnings(days_away=30))
        result = ev.evaluate(_make_plan(), "BULL", "WPM")
        self.assertTrue(result.no_earnings_imminent)
        self.assertEqual(result.earnings_days_away, 30)

    def test_earnings_exactly_at_threshold_fails(self):
        ev = _make_evaluator(_earnings(days_away=EARNINGS_HARD_GATE_DAYS))
        result = ev.evaluate(_make_plan(), "BULL", "WPM")
        self.assertFalse(result.no_earnings_imminent)

    def test_earnings_one_day_away_triggers_blocked(self):
        ev = _make_evaluator(_earnings(days_away=1))
        result = ev.evaluate(_make_plan(), "BULL", "WPM")
        self.assertFalse(result.no_earnings_imminent)
        self.assertEqual(result.status, "BLOCKED")

    def test_earnings_tomorrow_triggers_blocked(self):
        ev = _make_evaluator(_earnings(days_away=2))
        result = ev.evaluate(_make_plan(), "BULL", "WPM")
        self.assertEqual(result.status, "BLOCKED")

    def test_earnings_6_days_away_passes(self):
        ev = _make_evaluator(_earnings(days_away=6))
        result = ev.evaluate(_make_plan(), "BULL", "WPM")
        self.assertTrue(result.no_earnings_imminent)
        self.assertNotEqual(result.status, "BLOCKED")

    def test_earnings_verified_flag_stored(self):
        ev = _make_evaluator(_earnings(days_away=20, verified=True))
        result = ev.evaluate(_make_plan(), "BULL", "WPM")
        self.assertTrue(result.earnings_verified)

    def test_earnings_unverified_flag_stored(self):
        ev = _make_evaluator(_earnings(days_away=20, verified=False))
        result = ev.evaluate(_make_plan(), "BULL", "WPM")
        self.assertFalse(result.earnings_verified)


# ---------------------------------------------------------------------------
# Regime compatibility check
# ---------------------------------------------------------------------------

class TestRegimeCompatibilityCheck(unittest.TestCase):

    def test_bull_regime_passes(self):
        ev = _make_evaluator()
        result = ev.evaluate(_make_plan(), "BULL", "WPM")
        self.assertTrue(result.regime_compatible)

    def test_sideways_regime_passes(self):
        ev = _make_evaluator()
        result = ev.evaluate(_make_plan(), "SIDEWAYS", "WPM")
        self.assertTrue(result.regime_compatible)

    def test_unknown_regime_passes(self):
        ev = _make_evaluator()
        result = ev.evaluate(_make_plan(), "UNKNOWN", "WPM")
        self.assertTrue(result.regime_compatible)

    def test_bear_regime_fails(self):
        ev = _make_evaluator()
        result = ev.evaluate(_make_plan(), "BEAR", "WPM")
        self.assertFalse(result.regime_compatible)

    def test_bear_regime_does_not_trigger_blocked(self):
        """BEAR regime is a soft failure (REVIEW), not a hard gate (BLOCKED)."""
        ev = _make_evaluator()
        result = ev.evaluate(_make_plan(), "BEAR", "WPM")
        self.assertNotEqual(result.status, "BLOCKED")
        self.assertEqual(result.status, "REVIEW")


# ---------------------------------------------------------------------------
# Status aggregation
# ---------------------------------------------------------------------------

class TestStatusAggregation(unittest.TestCase):

    def test_all_pass_is_go(self):
        ev = _make_evaluator(earnings_payload=None)
        result = ev.evaluate(_make_plan(risk_pct=1.5, rr_ratio=2.5), "BULL", "WPM")
        self.assertTrue(result.all_checks_passed)
        self.assertEqual(result.status, "GO")

    def test_one_soft_failure_is_review(self):
        # BEAR regime = soft fail
        ev = _make_evaluator(earnings_payload=None)
        result = ev.evaluate(_make_plan(risk_pct=1.5, rr_ratio=2.5), "BEAR", "WPM")
        self.assertFalse(result.all_checks_passed)
        self.assertEqual(result.status, "REVIEW")

    def test_earnings_within_5_days_is_blocked(self):
        ev = _make_evaluator(_earnings(days_away=3))
        result = ev.evaluate(_make_plan(), "BULL", "WPM")
        self.assertEqual(result.status, "BLOCKED")

    def test_risk_above_5pct_is_blocked(self):
        ev = _make_evaluator(earnings_payload=None)
        result = ev.evaluate(_make_plan(risk_pct=6.0), "BULL", "WPM")
        self.assertEqual(result.status, "BLOCKED")

    def test_rr_below_2_is_review_not_blocked(self):
        ev = _make_evaluator(earnings_payload=None)
        result = ev.evaluate(_make_plan(rr_ratio=1.5), "BULL", "WPM")
        self.assertFalse(result.rr_ratio_acceptable)
        self.assertEqual(result.status, "REVIEW")
        self.assertNotEqual(result.status, "BLOCKED")


# ---------------------------------------------------------------------------
# No trade plan (plan engine disabled or threw)
# ---------------------------------------------------------------------------

class TestNoPlanPath(unittest.TestCase):
    """trade_plan=None → checks 1-3 are False; checks 4 and 5 still run."""

    def test_no_plan_checks_1_3_are_false(self):
        ev = _make_evaluator(earnings_payload=None)
        result = ev.evaluate(None, "BULL", "WPM")
        self.assertFalse(result.stop_loss_defined)
        self.assertFalse(result.position_sized_correctly)
        self.assertFalse(result.rr_ratio_acceptable)

    def test_no_plan_earnings_check_still_runs_safe(self):
        ev = _make_evaluator(earnings_payload=None)
        result = ev.evaluate(None, "BULL", "WPM")
        self.assertTrue(result.no_earnings_imminent)

    def test_no_plan_regime_check_still_runs(self):
        ev = _make_evaluator(earnings_payload=None)
        result = ev.evaluate(None, "BULL", "WPM")
        self.assertTrue(result.regime_compatible)

    def test_no_plan_status_is_review_when_no_earnings(self):
        """No plan + no earnings → can't pass checks 1-3 → REVIEW."""
        ev = _make_evaluator(earnings_payload=None)
        result = ev.evaluate(None, "BULL", "WPM")
        self.assertEqual(result.status, "REVIEW")

    def test_no_plan_earnings_imminent_still_blocks(self):
        """The whole point: earnings hard gate fires even with no trade plan."""
        ev = _make_evaluator(_earnings(days_away=2))
        result = ev.evaluate(None, "BULL", "WPM")
        self.assertFalse(result.no_earnings_imminent)
        self.assertEqual(result.status, "BLOCKED")

    def test_no_plan_size_blocked_does_not_fire(self):
        """Size block requires plan data — with no plan it can't trigger."""
        ev = _make_evaluator(earnings_payload=None)
        result = ev.evaluate(None, "BULL", "WPM")
        # risk_pct is 0.0 (default) — should not be BLOCKED on size
        self.assertNotEqual(result.status, "BLOCKED")

    def test_no_plan_risk_pct_defaults_to_zero(self):
        ev = _make_evaluator(earnings_payload=None)
        result = ev.evaluate(None, "BULL", "WPM")
        self.assertEqual(result.risk_pct, 0.0)

    def test_no_plan_rr_ratio_defaults_to_zero(self):
        ev = _make_evaluator(earnings_payload=None)
        result = ev.evaluate(None, "BULL", "WPM")
        self.assertEqual(result.rr_ratio, 0.0)


# ---------------------------------------------------------------------------
# Redis degradation
# ---------------------------------------------------------------------------

class TestRedisUnavailable(unittest.TestCase):

    def test_no_client_earnings_defaults_to_safe(self):
        """If Redis is unavailable, earnings check defaults to no_earnings_imminent=True."""
        evaluator = ChecklistEvaluator(
            redis_host="bad-host", redis_port=6379, redis_db=0
        )
        # _client intentionally None (connect never called)
        result = evaluator.evaluate(_make_plan(risk_pct=1.5, rr_ratio=2.5), "BULL", "WPM")
        self.assertTrue(result.no_earnings_imminent)

    def test_redis_error_during_get_defaults_to_safe(self):
        """RedisError during GET must default to no_earnings_imminent=True."""
        import redis as _redis
        ev = ChecklistEvaluator(redis_host="localhost", redis_port=6379, redis_db=0)
        mock_redis = MagicMock()
        mock_redis.get.side_effect = _redis.RedisError("connection refused")
        ev._client = mock_redis
        result = ev.evaluate(_make_plan(risk_pct=1.5, rr_ratio=2.5), "BULL", "WPM")
        self.assertTrue(result.no_earnings_imminent)


# ---------------------------------------------------------------------------
# connect() / close() lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle(unittest.TestCase):

    def test_connect_success(self):
        import redis as _redis
        with patch("decision_engine.checklist.redis.Redis") as mock_cls:
            instance = MagicMock()
            instance.ping.return_value = True
            mock_cls.return_value = instance
            ev = ChecklistEvaluator("localhost", 6379, 0)
            result = ev.connect()
            self.assertTrue(result)
            self.assertIsNotNone(ev._client)

    def test_connect_failure_sets_no_client(self):
        import redis as _redis
        with patch("decision_engine.checklist.redis.Redis") as mock_cls:
            instance = MagicMock()
            instance.ping.side_effect = _redis.RedisError("refused")
            mock_cls.return_value = instance
            ev = ChecklistEvaluator("bad-host", 6379, 0)
            result = ev.connect()
            self.assertFalse(result)
            self.assertIsNone(ev._client)

    def test_close_without_connect_does_not_raise(self):
        ev = ChecklistEvaluator("localhost", 6379, 0)
        ev.close()  # must not raise


if __name__ == "__main__":
    unittest.main()
