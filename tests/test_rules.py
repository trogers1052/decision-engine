"""Tests for all trading rule classes — RSI, MACD, Trend, Composite."""

from datetime import datetime, timezone
from decision_engine.rules.base import SignalType, SymbolContext, RuleResult
from decision_engine.rules.rsi_rules import (
    RSIOversoldRule, RSIOverboughtRule, RSIApproachingOversoldRule,
)
from decision_engine.rules.macd_rules import (
    MACDBullishCrossoverRule, MACDBearishCrossoverRule, MACDMomentumRule,
)
from decision_engine.rules.trend_rules import (
    WeeklyUptrendRule, MonthlyUptrendRule, FullTrendAlignmentRule,
    TrendBreakWarningRule, GoldenCrossRule, DeathCrossRule,
)
from decision_engine.rules.composite_rules import (
    BuyDipInUptrendRule, StrongBuySignalRule, RSIAndMACDConfluenceRule,
    TrendDipRecoveryRule,
)

TS = datetime(2026, 2, 20, 14, 30, tzinfo=timezone.utc)


def ctx(**indicators):
    return SymbolContext(symbol="TEST", indicators=indicators, timestamp=TS)


# ---------------------------------------------------------------------------
# RSI Rules
# ---------------------------------------------------------------------------

class TestRSIOversoldRule:
    def test_not_triggered_above_threshold(self):
        r = RSIOversoldRule(threshold=30)
        result = r.evaluate(ctx(RSI_14=45.0))
        assert result.triggered is False

    def test_at_threshold_not_triggered(self):
        r = RSIOversoldRule(threshold=30)
        result = r.evaluate(ctx(RSI_14=30.0))
        assert result.triggered is False

    def test_triggered_below_threshold(self):
        r = RSIOversoldRule(threshold=30)
        result = r.evaluate(ctx(RSI_14=28.0))
        assert result.triggered is True
        assert result.signal == SignalType.BUY
        assert 0.5 <= result.confidence <= 1.0

    def test_extreme_oversold_high_confidence(self):
        r = RSIOversoldRule(threshold=30, extreme_threshold=20)
        result = r.evaluate(ctx(RSI_14=15.0))
        assert result.confidence == 0.9

    def test_contributing_factors(self):
        r = RSIOversoldRule()
        result = r.evaluate(ctx(RSI_14=25.0))
        assert result.contributing_factors["RSI_14"] == 25.0

    def test_required_indicators(self):
        assert RSIOversoldRule().required_indicators == ["RSI_14"]

    def test_can_evaluate_missing(self):
        r = RSIOversoldRule()
        assert r.can_evaluate(ctx()) is False
        assert r.can_evaluate(ctx(RSI_14=40.0)) is True

    def test_custom_thresholds(self):
        r = RSIOversoldRule(threshold=40, extreme_threshold=25)
        result = r.evaluate(ctx(RSI_14=35.0))
        assert result.triggered is True


class TestRSIOverboughtRule:
    def test_not_triggered_below_threshold(self):
        r = RSIOverboughtRule(threshold=70)
        result = r.evaluate(ctx(RSI_14=65.0))
        assert result.triggered is False

    def test_watch_signal_between_thresholds(self):
        r = RSIOverboughtRule(threshold=70, extreme_threshold=80)
        result = r.evaluate(ctx(RSI_14=75.0))
        assert result.triggered is True
        assert result.signal == SignalType.WATCH

    def test_sell_signal_at_extreme(self):
        r = RSIOverboughtRule(threshold=70, extreme_threshold=80)
        result = r.evaluate(ctx(RSI_14=85.0))
        assert result.triggered is True
        assert result.signal == SignalType.SELL
        assert result.confidence == 0.85


class TestRSIApproachingOversoldRule:
    def test_in_watch_zone(self):
        r = RSIApproachingOversoldRule(watch_threshold=40, buy_threshold=30)
        result = r.evaluate(ctx(RSI_14=35.0))
        assert result.triggered is True
        assert result.signal == SignalType.WATCH

    def test_below_buy_threshold(self):
        r = RSIApproachingOversoldRule(watch_threshold=40, buy_threshold=30)
        result = r.evaluate(ctx(RSI_14=25.0))
        assert result.triggered is False

    def test_above_watch_threshold(self):
        r = RSIApproachingOversoldRule(watch_threshold=40, buy_threshold=30)
        result = r.evaluate(ctx(RSI_14=50.0))
        assert result.triggered is False

    def test_boundary_at_buy_threshold(self):
        r = RSIApproachingOversoldRule(watch_threshold=40, buy_threshold=30)
        result = r.evaluate(ctx(RSI_14=30.0))
        assert result.triggered is True  # 30 <= 30 <= 40


# ---------------------------------------------------------------------------
# MACD Rules
# ---------------------------------------------------------------------------

class TestMACDBullishCrossoverRule:
    def test_not_triggered_macd_below_signal(self):
        r = MACDBullishCrossoverRule()
        result = r.evaluate(ctx(MACD=-0.5, MACD_SIGNAL=0.0))
        assert result.triggered is False

    def test_fresh_crossover(self):
        r = MACDBullishCrossoverRule(histogram_threshold=0.1)
        result = r.evaluate(ctx(MACD=0.05, MACD_SIGNAL=0.0, MACD_HISTOGRAM=0.05))
        assert result.triggered is True
        assert result.signal == SignalType.BUY
        assert result.confidence >= 0.65

    def test_old_crossover_lower_confidence(self):
        r = MACDBullishCrossoverRule(histogram_threshold=0.1)
        result = r.evaluate(ctx(MACD=0.5, MACD_SIGNAL=0.0, MACD_HISTOGRAM=0.5))
        assert result.triggered is True
        assert result.confidence == 0.5

    def test_required_indicators(self):
        assert "MACD" in MACDBullishCrossoverRule().required_indicators
        assert "MACD_SIGNAL" in MACDBullishCrossoverRule().required_indicators


class TestMACDBearishCrossoverRule:
    def test_not_triggered_macd_above_signal(self):
        r = MACDBearishCrossoverRule()
        result = r.evaluate(ctx(MACD=0.5, MACD_SIGNAL=0.0))
        assert result.triggered is False

    def test_fresh_bearish_crossover(self):
        r = MACDBearishCrossoverRule(histogram_threshold=0.1)
        result = r.evaluate(ctx(MACD=-0.05, MACD_SIGNAL=0.0, MACD_HISTOGRAM=-0.05))
        assert result.triggered is True
        assert result.signal == SignalType.WATCH

    def test_confirmed_bearish(self):
        r = MACDBearishCrossoverRule(histogram_threshold=0.1)
        result = r.evaluate(ctx(MACD=-0.5, MACD_SIGNAL=0.0, MACD_HISTOGRAM=-0.5))
        assert result.triggered is True
        assert result.confidence == 0.5


class TestMACDMomentumRule:
    def test_strong_bullish_momentum(self):
        r = MACDMomentumRule()
        result = r.evaluate(ctx(MACD=0.2, MACD_SIGNAL=0.0))
        assert result.triggered is True
        assert result.signal == SignalType.BUY

    def test_strong_bearish_momentum(self):
        r = MACDMomentumRule()
        result = r.evaluate(ctx(MACD=-0.2, MACD_SIGNAL=0.0))
        assert result.triggered is True
        assert result.signal == SignalType.WATCH

    def test_no_momentum_near_zero(self):
        r = MACDMomentumRule()
        result = r.evaluate(ctx(MACD=0.02, MACD_SIGNAL=0.0))
        assert result.triggered is False

    def test_confidence_capped(self):
        r = MACDMomentumRule()
        result = r.evaluate(ctx(MACD=100, MACD_SIGNAL=0.0))
        assert result.confidence <= 0.7


# ---------------------------------------------------------------------------
# Trend Rules
# ---------------------------------------------------------------------------

class TestWeeklyUptrendRule:
    def test_uptrend_detected(self):
        r = WeeklyUptrendRule()
        result = r.evaluate(ctx(SMA_20=105.0, SMA_50=100.0))
        assert result.triggered is True
        assert result.signal == SignalType.WATCH

    def test_no_uptrend(self):
        r = WeeklyUptrendRule()
        result = r.evaluate(ctx(SMA_20=95.0, SMA_50=100.0))
        assert result.triggered is False

    def test_strong_uptrend_high_confidence(self):
        r = WeeklyUptrendRule()
        result = r.evaluate(ctx(SMA_20=103.0, SMA_50=100.0))
        assert result.confidence == 0.85

    def test_weak_uptrend_low_confidence(self):
        r = WeeklyUptrendRule()
        result = r.evaluate(ctx(SMA_20=100.5, SMA_50=100.0))
        assert result.confidence == 0.55


class TestMonthlyUptrendRule:
    def test_uptrend(self):
        r = MonthlyUptrendRule()
        result = r.evaluate(ctx(SMA_50=110.0, SMA_200=100.0))
        assert result.triggered is True
        assert result.signal == SignalType.WATCH

    def test_no_uptrend(self):
        r = MonthlyUptrendRule()
        result = r.evaluate(ctx(SMA_50=90.0, SMA_200=100.0))
        assert result.triggered is False

    def test_strong_spread_high_confidence(self):
        r = MonthlyUptrendRule()
        result = r.evaluate(ctx(SMA_50=106.0, SMA_200=100.0))
        assert result.confidence == 0.85


class TestFullTrendAlignmentRule:
    def test_full_alignment(self):
        r = FullTrendAlignmentRule()
        result = r.evaluate(ctx(SMA_20=110.0, SMA_50=105.0, SMA_200=100.0))
        assert result.triggered is True
        assert result.signal == SignalType.BUY

    def test_partial_alignment_not_triggered(self):
        r = FullTrendAlignmentRule()
        result = r.evaluate(ctx(SMA_20=110.0, SMA_50=95.0, SMA_200=100.0))
        assert result.triggered is False

    def test_no_alignment(self):
        r = FullTrendAlignmentRule()
        result = r.evaluate(ctx(SMA_20=90.0, SMA_50=95.0, SMA_200=100.0))
        assert result.triggered is False

    def test_confidence_scales_with_spread(self):
        r = FullTrendAlignmentRule()
        weak = r.evaluate(ctx(SMA_20=101.0, SMA_50=100.5, SMA_200=100.0))
        strong = r.evaluate(ctx(SMA_20=120.0, SMA_50=110.0, SMA_200=100.0))
        assert strong.confidence > weak.confidence


class TestTrendBreakWarningRule:
    def test_no_break(self):
        r = TrendBreakWarningRule()
        result = r.evaluate(ctx(SMA_20=105.0, SMA_50=100.0))
        assert result.triggered is False

    def test_fresh_break(self):
        r = TrendBreakWarningRule()
        result = r.evaluate(ctx(SMA_20=99.8, SMA_50=100.0))
        assert result.triggered is True
        assert result.signal == SignalType.SELL
        assert result.confidence == 0.7

    def test_old_break(self):
        r = TrendBreakWarningRule()
        result = r.evaluate(ctx(SMA_20=95.0, SMA_50=100.0))
        assert result.triggered is True
        assert result.confidence == 0.6


class TestGoldenCrossRule:
    def test_fresh_golden_cross(self):
        r = GoldenCrossRule()
        # spread_pct < 1.0 → fresh cross, confidence 0.75
        result = r.evaluate(ctx(SMA_50=100.5, SMA_200=100.0))
        assert result.triggered is True
        assert result.signal == SignalType.BUY
        assert result.confidence == 0.75

    def test_established_golden_cross(self):
        r = GoldenCrossRule()
        # spread_pct >= 1.0 → established cross, confidence 0.5
        result = r.evaluate(ctx(SMA_50=101.0, SMA_200=100.0))
        assert result.triggered is True
        assert result.confidence == 0.5

    def test_no_golden_cross(self):
        r = GoldenCrossRule()
        result = r.evaluate(ctx(SMA_50=95.0, SMA_200=100.0))
        assert result.triggered is False

    def test_old_golden_cross(self):
        r = GoldenCrossRule()
        result = r.evaluate(ctx(SMA_50=110.0, SMA_200=100.0))
        assert result.confidence == 0.5  # Already crossed


class TestDeathCrossRule:
    def test_fresh_death_cross(self):
        r = DeathCrossRule()
        # spread_pct < 1.0 → fresh cross, confidence 0.75
        result = r.evaluate(ctx(SMA_50=99.5, SMA_200=100.0))
        assert result.triggered is True
        assert result.signal == SignalType.SELL
        assert result.confidence == 0.75

    def test_established_death_cross(self):
        r = DeathCrossRule()
        # spread_pct >= 1.0 → established, confidence 0.6
        result = r.evaluate(ctx(SMA_50=99.0, SMA_200=100.0))
        assert result.triggered is True
        assert result.confidence == 0.6

    def test_no_death_cross(self):
        r = DeathCrossRule()
        result = r.evaluate(ctx(SMA_50=105.0, SMA_200=100.0))
        assert result.triggered is False


# ---------------------------------------------------------------------------
# Composite Rules
# ---------------------------------------------------------------------------

class TestBuyDipInUptrendRule:
    def test_dip_in_uptrend(self):
        r = BuyDipInUptrendRule(rsi_threshold=40)
        result = r.evaluate(ctx(RSI_14=35.0, SMA_20=105.0, SMA_50=100.0))
        assert result.triggered is True
        assert result.signal == SignalType.BUY

    def test_no_uptrend(self):
        r = BuyDipInUptrendRule(rsi_threshold=40)
        result = r.evaluate(ctx(RSI_14=25.0, SMA_20=95.0, SMA_50=100.0))
        assert result.triggered is False

    def test_no_dip(self):
        r = BuyDipInUptrendRule(rsi_threshold=40)
        result = r.evaluate(ctx(RSI_14=55.0, SMA_20=105.0, SMA_50=100.0))
        assert result.triggered is False

    def test_deep_dip_higher_confidence(self):
        r = BuyDipInUptrendRule(rsi_threshold=40)
        shallow = r.evaluate(ctx(RSI_14=38.0, SMA_20=105.0, SMA_50=100.0))
        deep = r.evaluate(ctx(RSI_14=25.0, SMA_20=105.0, SMA_50=100.0))
        assert deep.confidence > shallow.confidence

    def test_strong_trend_bonus(self):
        r = BuyDipInUptrendRule(rsi_threshold=40)
        weak = r.evaluate(ctx(RSI_14=35.0, SMA_20=100.5, SMA_50=100.0))
        strong = r.evaluate(ctx(RSI_14=35.0, SMA_20=103.0, SMA_50=100.0))
        assert strong.confidence > weak.confidence

    def test_contributing_factors_include_quality(self):
        r = BuyDipInUptrendRule(rsi_threshold=40)
        result = r.evaluate(ctx(RSI_14=25.0, SMA_20=105.0, SMA_50=100.0))
        assert "dip_quality" in result.contributing_factors
        assert result.contributing_factors["dip_quality"] == "deep"


class TestStrongBuySignalRule:
    def test_strong_buy(self):
        r = StrongBuySignalRule(rsi_threshold=35)
        result = r.evaluate(ctx(
            RSI_14=28.0, SMA_20=110.0, SMA_50=105.0, SMA_200=100.0,
        ))
        assert result.triggered is True
        assert result.confidence >= 0.7

    def test_missing_monthly_trend(self):
        r = StrongBuySignalRule(rsi_threshold=35)
        result = r.evaluate(ctx(
            RSI_14=28.0, SMA_20=110.0, SMA_50=95.0, SMA_200=100.0,
        ))
        assert result.triggered is False

    def test_rsi_not_deep_enough(self):
        r = StrongBuySignalRule(rsi_threshold=35)
        result = r.evaluate(ctx(
            RSI_14=40.0, SMA_20=110.0, SMA_50=105.0, SMA_200=100.0,
        ))
        assert result.triggered is False


class TestRSIAndMACDConfluenceRule:
    def test_confluence_buy(self):
        r = RSIAndMACDConfluenceRule(rsi_threshold=35)
        result = r.evaluate(ctx(RSI_14=28.0, MACD=0.5, MACD_SIGNAL=0.2))
        assert result.triggered is True
        assert result.signal == SignalType.BUY

    def test_rsi_not_oversold(self):
        r = RSIAndMACDConfluenceRule(rsi_threshold=35)
        result = r.evaluate(ctx(RSI_14=50.0, MACD=0.5, MACD_SIGNAL=0.2))
        assert result.triggered is False

    def test_macd_bearish(self):
        r = RSIAndMACDConfluenceRule(rsi_threshold=35)
        result = r.evaluate(ctx(RSI_14=28.0, MACD=-0.5, MACD_SIGNAL=0.2))
        assert result.triggered is False

    def test_very_oversold_higher_confidence(self):
        r = RSIAndMACDConfluenceRule(rsi_threshold=35)
        mild = r.evaluate(ctx(RSI_14=33.0, MACD=0.5, MACD_SIGNAL=0.2))
        deep = r.evaluate(ctx(RSI_14=25.0, MACD=0.5, MACD_SIGNAL=0.2))
        assert deep.confidence > mild.confidence


class TestTrendDipRecoveryRule:
    def test_recovery_in_uptrend(self):
        r = TrendDipRecoveryRule()
        result = r.evaluate(ctx(RSI_14=35.0, SMA_20=105.0, SMA_50=100.0))
        assert result.triggered is True
        assert result.signal == SignalType.BUY

    def test_no_uptrend(self):
        r = TrendDipRecoveryRule()
        result = r.evaluate(ctx(RSI_14=35.0, SMA_20=95.0, SMA_50=100.0))
        assert result.triggered is False

    def test_rsi_too_low(self):
        r = TrendDipRecoveryRule()
        result = r.evaluate(ctx(RSI_14=25.0, SMA_20=105.0, SMA_50=100.0))
        assert result.triggered is False

    def test_rsi_above_recovery_zone(self):
        r = TrendDipRecoveryRule()
        result = r.evaluate(ctx(RSI_14=50.0, SMA_20=105.0, SMA_50=100.0))
        assert result.triggered is False

    def test_confidence_capped(self):
        r = TrendDipRecoveryRule()
        result = r.evaluate(ctx(RSI_14=30.0, SMA_20=105.0, SMA_50=100.0))
        assert result.confidence <= 0.75
