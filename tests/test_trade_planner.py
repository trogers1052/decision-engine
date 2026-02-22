"""
Unit tests for TradePlanEngine.

Run with: pytest decision-engine/tests/test_trade_planner.py -v
(from the repo root, or from within decision-engine/ with pytest tests/test_trade_planner.py -v)
"""

import sys
import types
import unittest
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Stub out stop_loss_guardian if not installed, so tests run standalone
# ---------------------------------------------------------------------------
if "stop_loss_guardian" not in sys.modules:
    slg = types.ModuleType("stop_loss_guardian")
    slg_ps = types.ModuleType("stop_loss_guardian.position_sizer")

    class _FakeResult:
        def __init__(self, entry, stop, balance):
            risk_per_share = entry - stop
            max_dollar_risk = balance * 0.02
            self.max_shares = max(0, int(max_dollar_risk / risk_per_share))
            self.dollar_risk = Decimal(str(round(self.max_shares * risk_per_share, 2)))
            self.risk_pct = Decimal(str(round(float(self.dollar_risk) / balance * 100, 4)))
            self.position_value = Decimal(str(round(self.max_shares * entry, 2)))
            self.warnings = []
            self.is_valid = True
            self.blocked_reason = None

    class _FakePositionSizer:
        def __init__(self, **kwargs):
            pass

        def calculate(self, symbol, entry_price, stop_price, account_balance, target_price=None):
            return _FakeResult(float(entry_price), float(stop_price), float(account_balance))

    slg_ps.PositionSizer = _FakePositionSizer
    slg.position_sizer = slg_ps
    sys.modules["stop_loss_guardian"] = slg
    sys.modules["stop_loss_guardian.position_sizer"] = slg_ps

from decision_engine.models.signals import AggregatedSignal, Signal
from decision_engine.models.trade_plan import SetupType, TradePlan
from decision_engine.rules.base import SignalType
from decision_engine.trade_planner import TradePlanEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_signal(rule_name: str, confidence: float = 0.80) -> Signal:
    return Signal(
        rule_name=rule_name,
        rule_description="",
        signal_type=SignalType.BUY,
        confidence=confidence,
        reasoning="test",
        contributing_factors={},
        timestamp=datetime.now(timezone.utc),
    )


def _make_aggregated(symbol: str, rule_names: list[str]) -> AggregatedSignal:
    signals = [_make_signal(n) for n in rule_names]
    return AggregatedSignal(
        symbol=symbol,
        signal_type=SignalType.BUY,
        aggregate_confidence=0.82,
        primary_reasoning="test reasoning",
        contributing_signals=signals,
        timestamp=datetime.now(timezone.utc),
        rules_triggered=len(signals),
        rules_evaluated=len(signals),
    )


BASE_INDICATORS = {
    "close": 45.67,
    "ATR_14": 1.20,
    "SMA_20": 44.50,
    "SMA_50": 43.00,
    "BB_UPPER": 52.00,  # must stay above target_1 (45.67 + 2.40*2 = 50.47) to avoid resistance cap
}

DEFAULT_BALANCE = 888.80


def _engine(**overrides) -> TradePlanEngine:
    kwargs = dict(
        default_account_balance=DEFAULT_BALANCE,
        atr_multiplier=2.0,
        min_rr_ratio=2.0,
        stop_min_pct=3.0,
        stop_max_pct=15.0,
    )
    kwargs.update(overrides)
    return TradePlanEngine(**kwargs)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSetupClassification(unittest.TestCase):
    """Step 1 — setup type from triggered rule names."""

    def test_oversold_bounce_from_enhanced_buy_dip(self):
        engine = _engine()
        result = engine._classify_setup(["Enhanced Buy Dip"])
        self.assertEqual(result, SetupType.OVERSOLD_BOUNCE)

    def test_oversold_bounce_from_momentum_reversal(self):
        engine = _engine()
        result = engine._classify_setup(["Momentum Reversal"])
        self.assertEqual(result, SetupType.OVERSOLD_BOUNCE)

    def test_pullback_to_support(self):
        engine = _engine()
        result = engine._classify_setup(["Trend Continuation"])
        self.assertEqual(result, SetupType.PULLBACK_TO_SUPPORT)

    def test_breakout_commodity(self):
        engine = _engine()
        result = engine._classify_setup(["Commodity Breakout"])
        self.assertEqual(result, SetupType.BREAKOUT)

    def test_breakout_volume(self):
        engine = _engine()
        result = engine._classify_setup(["Volume Breakout"])
        self.assertEqual(result, SetupType.BREAKOUT)

    def test_oversold_bounce_from_buy_dip_in_uptrend(self):
        engine = _engine()
        result = engine._classify_setup(["Buy Dip in Uptrend"])
        self.assertEqual(result, SetupType.OVERSOLD_BOUNCE)

    def test_oversold_bounce_from_strong_buy_signal(self):
        engine = _engine()
        result = engine._classify_setup(["Strong Buy Signal"])
        self.assertEqual(result, SetupType.OVERSOLD_BOUNCE)

    def test_oversold_bounce_from_rsi_macd_confluence(self):
        engine = _engine()
        result = engine._classify_setup(["RSI + MACD Confluence"])
        self.assertEqual(result, SetupType.OVERSOLD_BOUNCE)

    def test_oversold_bounce_from_dip_recovery(self):
        engine = _engine()
        result = engine._classify_setup(["Dip Recovery"])
        self.assertEqual(result, SetupType.OVERSOLD_BOUNCE)

    def test_oversold_bounce_when_primary_and_enhanced_mix(self):
        # Real-world: multiple rules fire together
        engine = _engine()
        result = engine._classify_setup(["Buy Dip in Uptrend", "Strong Buy Signal"])
        self.assertEqual(result, SetupType.OVERSOLD_BOUNCE)

    def test_signal_fallback(self):
        engine = _engine()
        result = engine._classify_setup(["Some Unknown Rule"])
        self.assertEqual(result, SetupType.SIGNAL)


class TestOversoldBounce(unittest.TestCase):
    """Full plan generation for OVERSOLD_BOUNCE setup."""

    def setUp(self):
        self.engine = _engine()
        self.signal = _make_aggregated("CCJ", ["Enhanced Buy Dip"])
        self.plan = self.engine.generate(self.signal, BASE_INDICATORS)

    def test_setup_type(self):
        self.assertEqual(self.plan.setup_type, SetupType.OVERSOLD_BOUNCE)

    def test_entry_price_is_close(self):
        self.assertAlmostEqual(self.plan.entry_price, 45.67, places=1)

    def test_entry_zone(self):
        self.assertLess(self.plan.entry_zone_low, self.plan.entry_price)
        self.assertGreater(self.plan.entry_zone_high, self.plan.entry_price)

    def test_stop_uses_atr(self):
        # ATR_14=1.20, multiplier=2.0 → stop ~45.67 - 2.40 = 43.27
        expected_stop = 45.67 - 1.20 * 2.0
        self.assertAlmostEqual(self.plan.stop_price, expected_stop, places=1)
        self.assertEqual(self.plan.stop_method, "atr_2x")

    def test_targets_at_2_and_3_rr(self):
        risk = self.plan.entry_price - self.plan.stop_price
        expected_t1 = self.plan.entry_price + risk * 2.0
        expected_t2 = self.plan.entry_price + risk * 3.0
        self.assertAlmostEqual(self.plan.target_1, expected_t1, places=1)
        self.assertAlmostEqual(self.plan.target_2, expected_t2, places=1)

    def test_rr_ratio_is_2(self):
        self.assertAlmostEqual(self.plan.risk_reward_ratio, 2.0, places=1)

    def test_plan_valid(self):
        self.assertTrue(self.plan.plan_valid)
        self.assertIsNone(self.plan.rr_warning)

    def test_shares_positive(self):
        self.assertGreater(self.plan.shares, 0)

    def test_dollar_risk_within_2pct(self):
        self.assertLessEqual(self.plan.dollar_risk, DEFAULT_BALANCE * 0.02 + 0.01)

    def test_invalidation_below_stop(self):
        self.assertLess(self.plan.invalidation_price, self.plan.stop_price)

    def test_valid_until_is_future(self):
        self.assertGreater(self.plan.valid_until, datetime.now(timezone.utc))


class TestPullbackToSupport(unittest.TestCase):
    """PULLBACK_TO_SUPPORT — entry at SMA_20, invalidation at SMA_50."""

    def setUp(self):
        self.engine = _engine()
        self.signal = _make_aggregated("CAT", ["Trend Continuation"])
        self.plan = self.engine.generate(self.signal, BASE_INDICATORS)

    def test_setup_type(self):
        self.assertEqual(self.plan.setup_type, SetupType.PULLBACK_TO_SUPPORT)

    def test_entry_at_sma20(self):
        # Entry should be SMA_20 (44.50)
        self.assertAlmostEqual(self.plan.entry_price, BASE_INDICATORS["SMA_20"], places=1)

    def test_invalidation_near_sma50(self):
        expected = BASE_INDICATORS["SMA_50"] * 0.99
        self.assertAlmostEqual(self.plan.invalidation_price, expected, places=1)

    def test_valid_until_multi_day(self):
        # Should be ~2 trading days out, not just EOD today
        delta = self.plan.valid_until - datetime.now(timezone.utc)
        self.assertGreater(delta.total_seconds(), 60 * 60 * 24)  # > 1 day


class TestBreakout(unittest.TestCase):
    """BREAKOUT setup — entry above close, short validity window."""

    def setUp(self):
        self.engine = _engine()
        self.signal = _make_aggregated("WPM", ["Commodity Breakout"])
        self.plan = self.engine.generate(self.signal, BASE_INDICATORS)

    def test_setup_type(self):
        self.assertEqual(self.plan.setup_type, SetupType.BREAKOUT)

    def test_entry_above_close(self):
        # Entry = close * 1.001
        self.assertGreater(self.plan.entry_price, BASE_INDICATORS["close"])

    def test_valid_until_is_within_2_hours(self):
        delta = self.plan.valid_until - datetime.now(timezone.utc)
        self.assertLessEqual(delta.total_seconds(), 2 * 3600 + 10)  # ~2 hours
        self.assertGreater(delta.total_seconds(), 0)

    def test_invalidation_near_close(self):
        atr = BASE_INDICATORS["ATR_14"]
        expected = BASE_INDICATORS["close"] - atr * 0.5
        self.assertAlmostEqual(self.plan.invalidation_price, expected, places=1)


class TestATRTooTight(unittest.TestCase):
    """When ATR is very small → stop is widened to 4%."""

    def setUp(self):
        indicators = dict(BASE_INDICATORS)
        indicators["ATR_14"] = 0.20  # Very small — stop would be ~0.4% = <3%
        self.engine = _engine()
        self.signal = _make_aggregated("CCJ", ["Enhanced Buy Dip"])
        self.plan = self.engine.generate(self.signal, indicators)

    def test_stop_method_is_4pct(self):
        self.assertEqual(self.plan.stop_method, "percentage_4pct")

    def test_stop_pct_is_4(self):
        self.assertAlmostEqual(self.plan.stop_pct, 4.0, places=1)

    def test_warning_added(self):
        self.assertTrue(
            any("widened to 4%" in w for w in self.plan.warnings),
            f"Expected widened warning in {self.plan.warnings}",
        )


class TestATRFloorIsConfigDriven(unittest.TestCase):
    """Stop floor respects stop_min_pct, not a hardcoded 4%."""

    def test_custom_stop_min_pct_changes_floor(self):
        # stop_min_pct=5.0 → floor should be 6%
        indicators = dict(BASE_INDICATORS)
        indicators["ATR_14"] = 0.10  # tiny ATR, stop would be ~0.4%
        engine = _engine(stop_min_pct=5.0)
        signal = _make_aggregated("CCJ", ["Enhanced Buy Dip"])
        plan = engine.generate(signal, indicators)
        self.assertEqual(plan.stop_method, "percentage_6pct")
        self.assertAlmostEqual(plan.stop_pct, 6.0, places=1)
        self.assertTrue(any("widened to 6%" in w for w in plan.warnings))


class TestATRTooWide(unittest.TestCase):
    """When ATR is huge → stop is capped at 10%."""

    def setUp(self):
        indicators = dict(BASE_INDICATORS)
        indicators["ATR_14"] = 10.0  # Very large — stop would be >20%
        self.engine = _engine()
        self.signal = _make_aggregated("CCJ", ["Enhanced Buy Dip"])
        self.plan = self.engine.generate(self.signal, indicators)

    def test_stop_method_is_10pct(self):
        self.assertEqual(self.plan.stop_method, "percentage_10pct")

    def test_stop_pct_is_10(self):
        self.assertAlmostEqual(self.plan.stop_pct, 10.0, places=1)

    def test_warning_added(self):
        self.assertTrue(
            any("capped at 10%" in w for w in self.plan.warnings),
            f"Expected capped warning in {self.plan.warnings}",
        )


class TestLowRR(unittest.TestCase):
    """When R:R < 2:1, plan_valid=False and rr_warning is populated."""

    def setUp(self):
        # Force a tight stop so R:R is well below 2:1 by lowering min_rr_ratio
        # Actually easier: just set min_rr_ratio above 2.0 to force flag
        self.engine = _engine(min_rr_ratio=3.0)  # plan requires 3:1 minimum
        self.signal = _make_aggregated("CCJ", ["Enhanced Buy Dip"])
        self.plan = self.engine.generate(self.signal, BASE_INDICATORS)

    def test_plan_invalid(self):
        # rr_ratio will be 2.0, but min is 3.0 → invalid
        self.assertFalse(self.plan.plan_valid)

    def test_rr_warning_populated(self):
        self.assertIsNotNone(self.plan.rr_warning)
        self.assertIn("below minimum", self.plan.rr_warning)


class TestResistanceNote(unittest.TestCase):
    """BB_UPPER below target_1 triggers a resistance note."""

    def test_resistance_note_when_bb_upper_below_target(self):
        # With close=45.67, ATR=1.20, target_1 ≈ 45.67 + 2.4 = 48.07
        # Set BB_UPPER to 47.00 — below target_1
        indicators = dict(BASE_INDICATORS)
        indicators["BB_UPPER"] = 47.00
        engine = _engine()
        signal = _make_aggregated("CCJ", ["Enhanced Buy Dip"])
        plan = engine.generate(signal, indicators)
        self.assertIsNotNone(plan.resistance_note)
        self.assertIn("BB_UPPER", plan.resistance_note)
        self.assertIn("47.00", plan.resistance_note)

    def test_no_resistance_note_when_bb_upper_above_target(self):
        # BB_UPPER = 55.00 — well above target_1
        indicators = dict(BASE_INDICATORS)
        indicators["BB_UPPER"] = 55.00
        engine = _engine()
        signal = _make_aggregated("CCJ", ["Enhanced Buy Dip"])
        plan = engine.generate(signal, indicators)
        self.assertIsNone(plan.resistance_note)

    def test_no_resistance_note_when_bb_upper_missing(self):
        indicators = {k: v for k, v in BASE_INDICATORS.items() if k != "BB_UPPER"}
        engine = _engine()
        signal = _make_aggregated("CCJ", ["Enhanced Buy Dip"])
        plan = engine.generate(signal, indicators)
        self.assertIsNone(plan.resistance_note)


class TestSymbolTargetPct(unittest.TestCase):
    """Symbol-specific profit target is included when configured."""

    def test_symbol_target_pct_present(self):
        engine = _engine(
            symbol_exit_strategies={"CCJ": {"profit_target": 0.10, "stop_loss": 0.05}}
        )
        signal = _make_aggregated("CCJ", ["Enhanced Buy Dip"])
        plan = engine.generate(signal, BASE_INDICATORS)
        self.assertAlmostEqual(plan.symbol_target_pct, 0.10)

    def test_symbol_target_pct_absent(self):
        engine = _engine(symbol_exit_strategies={})
        signal = _make_aggregated("UNKNOWN", ["Enhanced Buy Dip"])
        plan = engine.generate(signal, BASE_INDICATORS)
        self.assertIsNone(plan.symbol_target_pct)


class TestRulesContributed(unittest.TestCase):
    """rules_contributed reflects contributing signal rule names."""

    def test_rules_contributed_populated(self):
        engine = _engine()
        signal = _make_aggregated("CCJ", ["Enhanced Buy Dip", "Momentum Reversal"])
        plan = engine.generate(signal, BASE_INDICATORS)
        self.assertIn("Enhanced Buy Dip", plan.rules_contributed)
        self.assertIn("Momentum Reversal", plan.rules_contributed)


class TestRedisUnavailableFallback(unittest.TestCase):
    """When Redis is unavailable, falls back to default_account_balance."""

    def test_fallback_to_default_balance(self):
        engine = _engine(default_account_balance=1234.56)
        # Force _fetch_balance_from_redis to return None
        engine._fetch_balance_from_redis = lambda: None
        # Invalidate cache
        engine._cached_balance = None
        engine._balance_cached_at = None

        balance = engine._get_account_balance()
        self.assertAlmostEqual(balance, 1234.56)

    def test_redis_balance_used_when_available(self):
        engine = _engine(default_account_balance=888.80)
        engine._fetch_balance_from_redis = lambda: 999.99
        engine._cached_balance = None
        engine._balance_cached_at = None

        balance = engine._get_account_balance()
        self.assertAlmostEqual(balance, 999.99)

    def test_cached_balance_returned_without_redis_call(self):
        engine = _engine(default_account_balance=888.80)
        engine._cached_balance = 777.77
        engine._balance_cached_at = datetime.now(timezone.utc)
        # Redis should NOT be called since cache is fresh
        engine._fetch_balance_from_redis = lambda: (_ for _ in ()).throw(
            RuntimeError("Redis should not be called")
        )
        balance = engine._get_account_balance()
        self.assertAlmostEqual(balance, 777.77)


class TestFromConfig(unittest.TestCase):
    """TradePlanEngine.from_config() reads trade_plan_engine section correctly."""

    def test_from_config_defaults_when_section_missing(self):
        engine = TradePlanEngine.from_config({})
        self.assertAlmostEqual(engine.atr_multiplier, 2.0)
        self.assertAlmostEqual(engine.min_rr_ratio, 2.0)
        self.assertAlmostEqual(engine.default_account_balance, 888.80)

    def test_from_config_reads_values(self):
        config = {
            "trade_plan_engine": {
                "atr_multiplier": 2.5,
                "min_rr_ratio": 1.5,
                "stop_min_pct": 2.0,
                "stop_max_pct": 12.0,
                "default_account_balance": 1500.00,
                "account_balance_redis_key": "my:portfolio",
            }
        }
        engine = TradePlanEngine.from_config(config)
        self.assertAlmostEqual(engine.atr_multiplier, 2.5)
        self.assertAlmostEqual(engine.min_rr_ratio, 1.5)
        self.assertAlmostEqual(engine.stop_min_pct, 2.0)
        self.assertAlmostEqual(engine.stop_max_pct, 12.0)
        self.assertAlmostEqual(engine.default_account_balance, 1500.00)
        self.assertEqual(engine.account_balance_redis_key, "my:portfolio")


class TestTradePlanModel(unittest.TestCase):
    """TradePlan Pydantic model serializes and validates correctly."""

    def test_plan_serializes_to_dict(self):
        engine = _engine()
        signal = _make_aggregated("CCJ", ["Enhanced Buy Dip"])
        plan = engine.generate(signal, BASE_INDICATORS)
        d = plan.model_dump()
        self.assertIn("setup_type", d)
        self.assertIn("entry_price", d)
        self.assertIn("stop_price", d)
        self.assertIn("target_1", d)
        self.assertIn("risk_reward_ratio", d)
        self.assertIn("plan_valid", d)

    def test_setup_type_is_string_in_dict(self):
        engine = _engine()
        signal = _make_aggregated("CCJ", ["Enhanced Buy Dip"])
        plan = engine.generate(signal, BASE_INDICATORS)
        d = plan.model_dump()
        self.assertIsInstance(d["setup_type"], str)


if __name__ == "__main__":
    unittest.main()
