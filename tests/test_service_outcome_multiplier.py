"""
Tests for Stage 5 outcome quality multiplier in _aggregate_signals.

Verifies that the decision engine service applies the outcome quality
multiplier from FeedbackAccuracyReader to BUY signal confidence.
"""

import sys
import types
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

# Stub risk_engine before importing service (only available on Pi)
if "risk_engine" not in sys.modules:
    _stub = types.ModuleType("risk_engine")
    _stub.RiskAdapter = MagicMock
    sys.modules["risk_engine"] = _stub

from decision_engine.models.signals import Signal
from decision_engine.rules.base import SignalType
from decision_engine.service import DecisionEngineService


def _make_signal(rule_name: str, confidence: float = 0.7) -> Signal:
    """Create a minimal Signal for testing aggregation."""
    return Signal(
        rule_name=rule_name,
        rule_description=f"{rule_name} test",
        signal_type=SignalType.BUY,
        confidence=confidence,
        reasoning="test",
        contributing_factors={},
        timestamp=datetime.utcnow(),
    )


def _make_service(feedback_reader=None) -> DecisionEngineService:
    """Create a service with mocked dependencies for unit-testing _aggregate_signals."""
    settings = MagicMock()
    svc = DecisionEngineService.__new__(DecisionEngineService)
    svc.settings = settings
    svc.market_context_reader = None
    svc.tier_reader = None
    svc.feedback_reader = feedback_reader
    return svc


class TestStage5OutcomeMultiplier(unittest.TestCase):

    def test_outcome_multiplier_dampens_confidence(self):
        """Low win-rate rules should reduce confidence."""
        reader = MagicMock()
        reader.get_aggregate_multiplier.return_value = 1.0  # Stage 4 neutral
        reader.get_aggregate_outcome_multiplier.return_value = 0.6  # bad win rate
        svc = _make_service(feedback_reader=reader)

        signals = [_make_signal("BadRule", confidence=0.80)]
        result = svc._aggregate_signals(
            "AAPL", SignalType.BUY, signals, 5, datetime.utcnow()
        )

        # 0.80 * 0.6 = 0.48
        self.assertAlmostEqual(result.aggregate_confidence, 0.48, places=2)

    def test_outcome_multiplier_boosts_confidence(self):
        """High win-rate rules should boost confidence."""
        reader = MagicMock()
        reader.get_aggregate_multiplier.return_value = 1.0
        reader.get_aggregate_outcome_multiplier.return_value = 1.4
        svc = _make_service(feedback_reader=reader)

        signals = [_make_signal("GoodRule", confidence=0.70)]
        result = svc._aggregate_signals(
            "AAPL", SignalType.BUY, signals, 5, datetime.utcnow()
        )

        # 0.70 * 1.4 = 0.98
        self.assertAlmostEqual(result.aggregate_confidence, 0.98, places=2)

    def test_outcome_multiplier_capped_at_1(self):
        """Confidence must never exceed 1.0."""
        reader = MagicMock()
        reader.get_aggregate_multiplier.return_value = 1.0
        reader.get_aggregate_outcome_multiplier.return_value = 1.5
        svc = _make_service(feedback_reader=reader)

        signals = [_make_signal("GoodRule", confidence=0.90)]
        result = svc._aggregate_signals(
            "AAPL", SignalType.BUY, signals, 5, datetime.utcnow()
        )

        # 0.90 * 1.5 = 1.35 → capped at 1.0
        self.assertAlmostEqual(result.aggregate_confidence, 1.0, places=2)

    def test_outcome_neutral_no_change(self):
        """outcome_mult=1.0 should leave confidence unchanged."""
        reader = MagicMock()
        reader.get_aggregate_multiplier.return_value = 1.0
        reader.get_aggregate_outcome_multiplier.return_value = 1.0
        svc = _make_service(feedback_reader=reader)

        signals = [_make_signal("NeutralRule", confidence=0.75)]
        result = svc._aggregate_signals(
            "AAPL", SignalType.BUY, signals, 5, datetime.utcnow()
        )

        # consensus_boost for single signal = 0.75 (no boost)
        self.assertAlmostEqual(result.aggregate_confidence, 0.75, places=2)

    def test_no_feedback_reader_no_change(self):
        """Without a feedback reader, Stage 4 and 5 are skipped."""
        svc = _make_service(feedback_reader=None)

        signals = [_make_signal("AnyRule", confidence=0.80)]
        result = svc._aggregate_signals(
            "AAPL", SignalType.BUY, signals, 5, datetime.utcnow()
        )

        self.assertAlmostEqual(result.aggregate_confidence, 0.80, places=2)

    def test_sell_signal_skips_outcome_multiplier(self):
        """Stage 5 only applies to BUY signals."""
        reader = MagicMock()
        svc = _make_service(feedback_reader=reader)

        signals = [_make_signal("SellRule", confidence=0.80)]
        signals[0] = Signal(
            rule_name="SellRule",
            rule_description="test",
            signal_type=SignalType.SELL,
            confidence=0.80,
            reasoning="test",
            contributing_factors={},
            timestamp=datetime.utcnow(),
        )
        result = svc._aggregate_signals(
            "AAPL", SignalType.SELL, signals, 5, datetime.utcnow()
        )

        # Reader should not be called for SELL
        reader.get_aggregate_outcome_multiplier.assert_not_called()
        self.assertAlmostEqual(result.aggregate_confidence, 0.80, places=2)

    def test_both_stage4_and_stage5_applied(self):
        """Stage 4 and Stage 5 multiply sequentially."""
        reader = MagicMock()
        reader.get_aggregate_multiplier.return_value = 0.8     # Stage 4: dampened
        reader.get_aggregate_outcome_multiplier.return_value = 1.3  # Stage 5: boosted
        svc = _make_service(feedback_reader=reader)

        signals = [_make_signal("MixedRule", confidence=0.70)]
        result = svc._aggregate_signals(
            "AAPL", SignalType.BUY, signals, 5, datetime.utcnow()
        )

        # 0.70 * 0.8 = 0.56, then 0.56 * 1.3 = 0.728
        self.assertAlmostEqual(result.aggregate_confidence, 0.728, places=2)

    def test_outcome_multiplier_passes_correct_rule_names(self):
        """Verify correct rule names and regime_id are passed to the reader."""
        reader = MagicMock()
        reader.get_aggregate_multiplier.return_value = 1.0
        reader.get_aggregate_outcome_multiplier.return_value = 1.0
        svc = _make_service(feedback_reader=reader)

        # Also set up market_context_reader to provide a regime
        ctx_reader = MagicMock()
        ctx_reader.get_regime.return_value = "BULL"
        ctx_reader.get_regime_confidence.return_value = 0.9
        ctx_reader.get_multiplier.return_value = 1.0
        svc.market_context_reader = ctx_reader

        signals = [
            _make_signal("MomentumReversal", confidence=0.70),
            _make_signal("RSIOversold", confidence=0.65),
        ]
        svc._aggregate_signals("AAPL", SignalType.BUY, signals, 5, datetime.utcnow())

        reader.get_aggregate_outcome_multiplier.assert_called_once_with(
            ["MomentumReversal", "RSIOversold"], "BULL"
        )


if __name__ == "__main__":
    unittest.main()
