"""Tests for enhanced trading rules (EnhancedBuyDip, MomentumReversal, TrendContinuation)."""

import pytest
from datetime import datetime, timezone

from decision_engine.rules.base import RuleResult, SignalType, SymbolContext
from decision_engine.rules.enhanced_rules import (
    EnhancedBuyDipRule,
    MomentumReversalRule,
    TrendContinuationRule,
)

TS = datetime(2026, 2, 20, 14, 30, tzinfo=timezone.utc)


def ctx(**indicators):
    return SymbolContext(symbol="TEST", indicators=indicators, timestamp=TS)


# ---------------------------------------------------------------------------
# EnhancedBuyDipRule
# ---------------------------------------------------------------------------

class TestEnhancedBuyDipRule:
    """Covers the multi-filter buy-dip logic."""

    def _base_indicators(self, **overrides):
        """Return a complete set of indicators that should trigger the rule."""
        d = dict(
            RSI_14=28.0,        # Below oversold (35)
            SMA_20=105.0,       # Uptrend: SMA_20 > SMA_50
            SMA_50=103.0,       # Spread = 1.94% (above 1.5% minimum)
            SMA_200=90.0,       # Close > SMA_200
            ATR_14=2.0,
            close=95.0,         # Above SMA_200
            volume=1200,
            volume_sma_20=1000, # Ratio 1.2 (> 0.8)
        )
        d.update(overrides)
        return d

    def test_triggered_full_setup(self):
        r = EnhancedBuyDipRule()
        result = r.evaluate(ctx(**self._base_indicators()))
        assert result.triggered is True
        assert result.signal == SignalType.BUY
        assert result.confidence >= 0.5

    def test_filter_no_uptrend(self):
        r = EnhancedBuyDipRule()
        result = r.evaluate(ctx(**self._base_indicators(SMA_20=100.0, SMA_50=105.0)))
        assert result.triggered is False
        assert "uptrend" in result.reasoning.lower()

    def test_filter_weak_trend(self):
        r = EnhancedBuyDipRule()
        # Spread = (100.5 - 100.0)/100.0 * 100 = 0.5% < 1.5%
        result = r.evaluate(ctx(**self._base_indicators(SMA_20=100.5, SMA_50=100.0)))
        assert result.triggered is False
        assert "weak trend" in result.reasoning.lower() or "spread" in result.reasoning.lower()

    def test_filter_rsi_not_oversold(self):
        r = EnhancedBuyDipRule()
        result = r.evaluate(ctx(**self._base_indicators(RSI_14=40.0)))
        assert result.triggered is False
        assert "dip" in result.reasoning.lower() or "RSI" in result.reasoning

    def test_filter_price_below_sma200(self):
        r = EnhancedBuyDipRule()
        result = r.evaluate(ctx(**self._base_indicators(close=85.0, SMA_200=90.0)))
        assert result.triggered is False
        assert "SMA_200" in result.reasoning or "breakdown" in result.reasoning.lower()

    def test_filter_low_volume(self):
        r = EnhancedBuyDipRule(require_volume_confirm=True)
        # volume_ratio = 500/1000 = 0.5 < 0.8
        result = r.evaluate(ctx(**self._base_indicators(volume=500, volume_sma_20=1000)))
        assert result.triggered is False
        assert "volume" in result.reasoning.lower()

    def test_volume_confirm_disabled(self):
        r = EnhancedBuyDipRule(require_volume_confirm=False)
        # Low volume but volume confirm is off — should still trigger
        result = r.evaluate(ctx(**self._base_indicators(volume=500, volume_sma_20=1000)))
        assert result.triggered is True

    def test_extreme_oversold_higher_confidence(self):
        r = EnhancedBuyDipRule()
        # RSI 28 < extreme (30) → rsi_score = 0.40
        deep = r.evaluate(ctx(**self._base_indicators(RSI_14=28.0)))
        # RSI 34 → rsi_score = 0.20
        shallow = r.evaluate(ctx(**self._base_indicators(RSI_14=34.0)))
        assert deep.confidence > shallow.confidence

    def test_alignment_bonus(self):
        r = EnhancedBuyDipRule()
        # SMA_50 > SMA_200 → full alignment bonus (+0.15)
        full = r.evaluate(ctx(**self._base_indicators(SMA_50=103.0, SMA_200=90.0)))
        # SMA_50 < SMA_200 → no bonus
        partial = r.evaluate(ctx(**self._base_indicators(SMA_50=103.0, SMA_200=110.0, close=111.0)))
        assert full.confidence > partial.confidence

    def test_confidence_clamped(self):
        r = EnhancedBuyDipRule()
        result = r.evaluate(ctx(**self._base_indicators()))
        assert 0.5 <= result.confidence <= 0.95

    def test_contributing_factors_present(self):
        r = EnhancedBuyDipRule()
        result = r.evaluate(ctx(**self._base_indicators()))
        cf = result.contributing_factors
        assert "RSI_14" in cf
        assert "dip_quality" in cf
        assert "volume_ratio" in cf
        assert "scores" in cf

    def test_custom_thresholds(self):
        r = EnhancedBuyDipRule(rsi_oversold=40.0, rsi_extreme=35.0, min_trend_spread=1.0)
        # RSI 38 < 40 → oversold with custom threshold
        result = r.evaluate(ctx(**self._base_indicators(RSI_14=38.0)))
        assert result.triggered is True


# ---------------------------------------------------------------------------
# MomentumReversalRule
# ---------------------------------------------------------------------------

class TestMomentumReversalRule:
    """Catches RSI recovery + MACD crossover patterns."""

    def _base_indicators(self, **overrides):
        d = dict(
            RSI_14=35.0,            # In recovery zone [30, 40]
            MACD=0.5,               # Above signal → bullish
            MACD_SIGNAL=0.3,
            MACD_HISTOGRAM=0.2,     # > 0.05 → strong
            SMA_20=105.0,
            SMA_50=100.0,           # Uptrend
            volume=1200,
            volume_sma_20=1000,
        )
        d.update(overrides)
        return d

    def test_triggered_full_setup(self):
        r = MomentumReversalRule()
        result = r.evaluate(ctx(**self._base_indicators()))
        assert result.triggered is True
        assert result.signal == SignalType.BUY

    def test_rsi_below_recovery(self):
        r = MomentumReversalRule()
        result = r.evaluate(ctx(**self._base_indicators(RSI_14=25.0)))
        assert result.triggered is False
        assert "still oversold" in result.reasoning.lower()

    def test_rsi_above_recovery(self):
        r = MomentumReversalRule()
        result = r.evaluate(ctx(**self._base_indicators(RSI_14=45.0)))
        assert result.triggered is False
        assert "above recovery zone" in result.reasoning.lower()

    def test_macd_bearish(self):
        r = MomentumReversalRule()
        result = r.evaluate(ctx(**self._base_indicators(MACD=0.2, MACD_SIGNAL=0.5)))
        assert result.triggered is False
        assert "bearish" in result.reasoning.lower()

    def test_weak_histogram_and_low_volume_rejected(self):
        r = MomentumReversalRule()
        # histogram < 0.05 AND volume_ratio < 1.0
        result = r.evaluate(ctx(**self._base_indicators(
            MACD_HISTOGRAM=0.03, volume=800, volume_sma_20=1000
        )))
        assert result.triggered is False
        assert "weak reversal" in result.reasoning.lower()

    def test_weak_histogram_but_good_volume_passes(self):
        r = MomentumReversalRule()
        # histogram < 0.05 but volume_ratio >= 1.0
        result = r.evaluate(ctx(**self._base_indicators(
            MACD_HISTOGRAM=0.03, volume=1000, volume_sma_20=1000
        )))
        assert result.triggered is True

    def test_uptrend_boost(self):
        r = MomentumReversalRule()
        # Uptrend: SMA_20 > SMA_50 → +0.15
        up = r.evaluate(ctx(**self._base_indicators(SMA_20=105.0, SMA_50=100.0)))
        # No uptrend: SMA_20 < SMA_50
        down = r.evaluate(ctx(**self._base_indicators(SMA_20=98.0, SMA_50=100.0)))
        assert up.confidence > down.confidence

    def test_confidence_capped_at_90(self):
        r = MomentumReversalRule()
        result = r.evaluate(ctx(**self._base_indicators(
            RSI_14=33.0,          # +0.10 (RSI < 35)
            MACD_HISTOGRAM=0.15,  # +0.10
            volume=1500,          # +0.05
            volume_sma_20=1000,
        )))
        assert result.confidence <= 0.90

    def test_custom_recovery_zone(self):
        r = MomentumReversalRule(rsi_recovery_min=25.0, rsi_recovery_max=35.0)
        # RSI 28 is in custom zone
        result = r.evaluate(ctx(**self._base_indicators(RSI_14=28.0)))
        assert result.triggered is True
        # RSI 38 is outside custom zone
        result2 = r.evaluate(ctx(**self._base_indicators(RSI_14=38.0)))
        assert result2.triggered is False


# ---------------------------------------------------------------------------
# TrendContinuationRule
# ---------------------------------------------------------------------------

class TestTrendContinuationRule:
    """Buys pullbacks in strong trends."""

    def _base_indicators(self, **overrides):
        d = dict(
            RSI_14=45.0,
            SMA_20=100.0,
            SMA_50=95.0,       # Full alignment: SMA_20 > SMA_50 > SMA_200
            SMA_200=85.0,
            close=100.5,       # Within 2% of SMA_20 (0.5%)
            volume=1000,
            volume_sma_20=1000,
        )
        d.update(overrides)
        return d

    def test_triggered_at_sma20_support(self):
        r = TrendContinuationRule()
        result = r.evaluate(ctx(**self._base_indicators()))
        assert result.triggered is True
        assert result.signal == SignalType.BUY

    def test_no_full_alignment(self):
        r = TrendContinuationRule()
        # SMA_50 < SMA_200 → not full alignment
        result = r.evaluate(ctx(**self._base_indicators(SMA_50=80.0, SMA_200=85.0)))
        assert result.triggered is False
        assert "alignment" in result.reasoning.lower()

    def test_price_too_far_above_sma20(self):
        r = TrendContinuationRule()
        # close = 110 → distance = 10% > 2%
        result = r.evaluate(ctx(**self._base_indicators(close=110.0)))
        assert result.triggered is False
        assert "above SMA_20" in result.reasoning

    def test_price_too_far_below_sma20(self):
        r = TrendContinuationRule()
        # close = 95 → distance = -5% < -2%
        result = r.evaluate(ctx(**self._base_indicators(close=95.0)))
        assert result.triggered is False
        assert "below SMA_20" in result.reasoning.lower() or "breakdown" in result.reasoning.lower()

    def test_rsi_too_low(self):
        r = TrendContinuationRule()
        result = r.evaluate(ctx(**self._base_indicators(RSI_14=30.0)))
        assert result.triggered is False
        assert "too low" in result.reasoning.lower() or "dip-buy" in result.reasoning.lower()

    def test_rsi_too_high(self):
        r = TrendContinuationRule()
        result = r.evaluate(ctx(**self._base_indicators(RSI_14=65.0)))
        assert result.triggered is False
        assert "too high" in result.reasoning.lower()

    def test_stronger_trend_higher_confidence(self):
        r = TrendContinuationRule()
        # Big spreads → +0.10 each
        strong = r.evaluate(ctx(**self._base_indicators(
            SMA_20=110.0, SMA_50=103.0, SMA_200=85.0, close=110.3
        )))
        # Tight spreads
        weak = r.evaluate(ctx(**self._base_indicators(
            SMA_20=100.0, SMA_50=99.0, SMA_200=97.0, close=100.3
        )))
        assert strong.confidence > weak.confidence

    def test_custom_pullback_tolerance(self):
        r = TrendContinuationRule(pullback_tolerance_pct=5.0)
        # close = 104 → 4% above SMA_20 — within 5% tolerance
        result = r.evaluate(ctx(**self._base_indicators(close=104.0)))
        assert result.triggered is True

    def test_confidence_capped(self):
        r = TrendContinuationRule()
        result = r.evaluate(ctx(**self._base_indicators()))
        assert result.confidence <= 0.85

    def test_volume_boost(self):
        r = TrendContinuationRule()
        high_vol = r.evaluate(ctx(**self._base_indicators(volume=1200, volume_sma_20=1000)))
        low_vol = r.evaluate(ctx(**self._base_indicators(volume=700, volume_sma_20=1000)))
        assert high_vol.confidence > low_vol.confidence
