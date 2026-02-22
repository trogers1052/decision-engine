"""Tests for signal models, RuleResult validation, SymbolContext, and ConfidenceAggregator."""

import pytest
from datetime import datetime, timezone

from decision_engine.rules.base import RuleResult, SignalType, SymbolContext
from decision_engine.models.signals import Signal, AggregatedSignal, ConfidenceAggregator


# ---------------------------------------------------------------------------
# RuleResult validation
# ---------------------------------------------------------------------------

class TestRuleResult:
    def test_not_triggered_no_signal(self):
        r = RuleResult(triggered=False, reasoning="nothing here")
        assert r.triggered is False
        assert r.signal is None
        assert r.confidence == 0.0

    def test_triggered_with_signal(self):
        r = RuleResult(triggered=True, signal=SignalType.BUY, confidence=0.8, reasoning="buy")
        assert r.triggered is True
        assert r.signal == SignalType.BUY
        assert r.confidence == 0.8

    def test_triggered_without_signal_raises(self):
        with pytest.raises(ValueError, match="must have a signal"):
            RuleResult(triggered=True, confidence=0.5, reasoning="oops")

    def test_confidence_above_1_raises(self):
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            RuleResult(triggered=True, signal=SignalType.BUY, confidence=1.5)

    def test_confidence_below_0_raises(self):
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            RuleResult(triggered=True, signal=SignalType.BUY, confidence=-0.1)

    def test_confidence_boundary_0(self):
        r = RuleResult(triggered=True, signal=SignalType.WATCH, confidence=0.0)
        assert r.confidence == 0.0

    def test_confidence_boundary_1(self):
        r = RuleResult(triggered=True, signal=SignalType.SELL, confidence=1.0)
        assert r.confidence == 1.0

    def test_contributing_factors_default_empty(self):
        r = RuleResult(triggered=False)
        assert r.contributing_factors == {}


# ---------------------------------------------------------------------------
# SymbolContext
# ---------------------------------------------------------------------------

class TestSymbolContext:
    def _ctx(self, **indicators):
        return SymbolContext(
            symbol="AAPL",
            indicators=indicators,
            timestamp=datetime(2026, 2, 20, 14, 30, tzinfo=timezone.utc),
        )

    def test_get_indicator_present(self):
        ctx = self._ctx(RSI_14=45.0)
        assert ctx.get_indicator("RSI_14") == 45.0

    def test_get_indicator_missing_returns_default(self):
        ctx = self._ctx()
        assert ctx.get_indicator("RSI_14") == 0.0
        assert ctx.get_indicator("RSI_14", 50.0) == 50.0

    def test_has_indicators_all_present(self):
        ctx = self._ctx(RSI_14=45.0, SMA_20=100.0, SMA_50=98.0)
        assert ctx.has_indicators("RSI_14", "SMA_20", "SMA_50") is True

    def test_has_indicators_missing_one(self):
        ctx = self._ctx(RSI_14=45.0)
        assert ctx.has_indicators("RSI_14", "SMA_20") is False

    def test_has_indicators_none_value(self):
        ctx = self._ctx(RSI_14=None)
        assert ctx.has_indicators("RSI_14") is False

    def test_has_indicators_empty(self):
        ctx = self._ctx()
        assert ctx.has_indicators() is True  # Vacuously true


# ---------------------------------------------------------------------------
# SignalType enum
# ---------------------------------------------------------------------------

class TestSignalType:
    def test_values(self):
        assert SignalType.BUY.value == "BUY"
        assert SignalType.SELL.value == "SELL"
        assert SignalType.WATCH.value == "WATCH"


# ---------------------------------------------------------------------------
# Signal dataclass
# ---------------------------------------------------------------------------

class TestSignal:
    def _make_signal(self, **overrides):
        defaults = dict(
            rule_name="RSI Oversold",
            rule_description="Buy when RSI < 30",
            signal_type=SignalType.BUY,
            confidence=0.8,
            reasoning="RSI at 25",
            contributing_factors={"RSI_14": 25.0},
            timestamp=datetime(2026, 2, 20, 14, 30, tzinfo=timezone.utc),
        )
        defaults.update(overrides)
        return Signal(**defaults)

    def test_to_dict(self):
        s = self._make_signal()
        d = s.to_dict()
        assert d["rule_name"] == "RSI Oversold"
        assert d["signal_type"] == "BUY"
        assert d["confidence"] == 0.8
        assert d["contributing_factors"]["RSI_14"] == 25.0
        assert d["timestamp"].endswith("Z")

    def test_different_signal_types(self):
        for st in SignalType:
            s = self._make_signal(signal_type=st)
            assert s.to_dict()["signal_type"] == st.value


# ---------------------------------------------------------------------------
# AggregatedSignal dataclass
# ---------------------------------------------------------------------------

class TestAggregatedSignal:
    def test_to_dict(self):
        sig = Signal(
            rule_name="R", rule_description="D", signal_type=SignalType.BUY,
            confidence=0.7, reasoning="test", contributing_factors={},
            timestamp=datetime(2026, 2, 20, tzinfo=timezone.utc),
        )
        agg = AggregatedSignal(
            symbol="GOOG",
            signal_type=SignalType.BUY,
            aggregate_confidence=0.75,
            primary_reasoning="combined",
            contributing_signals=[sig],
            timestamp=datetime(2026, 2, 20, tzinfo=timezone.utc),
            rules_triggered=1,
            rules_evaluated=3,
            regime_id="BULL",
            regime_confidence=0.9,
        )
        d = agg.to_dict()
        assert d["symbol"] == "GOOG"
        assert d["signal_type"] == "BUY"
        assert d["aggregate_confidence"] == 0.75
        assert d["rules_triggered"] == 1
        assert d["regime_id"] == "BULL"
        assert len(d["contributing_signals"]) == 1


# ---------------------------------------------------------------------------
# ConfidenceAggregator
# ---------------------------------------------------------------------------

class TestConfidenceAggregator:
    def _make_signals(self, confidences):
        ts = datetime(2026, 2, 20, tzinfo=timezone.utc)
        return [
            Signal(
                rule_name=f"Rule_{i}",
                rule_description="",
                signal_type=SignalType.BUY,
                confidence=c,
                reasoning="",
                contributing_factors={},
                timestamp=ts,
            )
            for i, c in enumerate(confidences)
        ]

    def test_weighted_average_uniform(self):
        sigs = self._make_signals([0.6, 0.8])
        assert ConfidenceAggregator.weighted_average(sigs) == pytest.approx(0.7)

    def test_weighted_average_with_weights(self):
        sigs = self._make_signals([0.6, 0.8])
        weights = {"Rule_0": 2.0, "Rule_1": 1.0}
        result = ConfidenceAggregator.weighted_average(sigs, weights)
        expected = (0.6 * 2 + 0.8 * 1) / 3
        assert result == pytest.approx(expected)

    def test_weighted_average_empty(self):
        assert ConfidenceAggregator.weighted_average([]) == 0.0

    def test_weighted_average_single(self):
        sigs = self._make_signals([0.9])
        assert ConfidenceAggregator.weighted_average(sigs) == pytest.approx(0.9)

    def test_highest_returns_max(self):
        sigs = self._make_signals([0.3, 0.9, 0.6])
        assert ConfidenceAggregator.highest(sigs) == 0.9

    def test_highest_empty(self):
        assert ConfidenceAggregator.highest([]) == 0.0

    def test_consensus_boost_single_signal(self):
        sigs = self._make_signals([0.7])
        result = ConfidenceAggregator.consensus_boost(sigs)
        # 1 signal: base=0.7, boost=(1-1)*0.05=0
        assert result == pytest.approx(0.7)

    def test_consensus_boost_two_signals(self):
        sigs = self._make_signals([0.6, 0.8])
        result = ConfidenceAggregator.consensus_boost(sigs)
        base = 0.7  # avg
        boost = 0.05  # (2-1)*0.05
        assert result == pytest.approx(base + boost)

    def test_consensus_boost_caps_at_15pct(self):
        sigs = self._make_signals([0.7, 0.7, 0.7, 0.7, 0.7])
        result = ConfidenceAggregator.consensus_boost(sigs)
        # base = 0.7, boost = min((5-1)*0.05, 0.15) = 0.15
        assert result == pytest.approx(0.85)

    def test_consensus_boost_caps_at_1(self):
        sigs = self._make_signals([0.95, 0.95, 0.95, 0.95])
        result = ConfidenceAggregator.consensus_boost(sigs)
        assert result == 1.0

    def test_consensus_boost_empty(self):
        assert ConfidenceAggregator.consensus_boost([]) == 0.0
