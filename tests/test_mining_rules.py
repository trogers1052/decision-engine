"""Tests for mining stock rules (Commodity Breakout, Miner Metal Ratio, Dollar Weakness, Seasonality, Volume Breakout)."""

import pytest
from datetime import datetime, timezone

from decision_engine.rules.base import RuleResult, SignalType, SymbolContext
from decision_engine.rules.mining_rules import (
    CommodityBreakoutRule,
    MinerMetalRatioRule,
    DollarWeaknessRule,
    SeasonalityRule,
    VolumeBreakoutRule,
    MINER_COMMODITY_MAP,
    SEASONAL_STRENGTH,
    SEASONAL_WEAKNESS,
)


def mining_ctx(symbol="GDX", month=1, **indicators):
    """Helper — builds a SymbolContext for a mining stock."""
    return SymbolContext(
        symbol=symbol,
        indicators=indicators,
        timestamp=datetime(2026, month, 15, 14, 30, tzinfo=timezone.utc),
    )


# ---------------------------------------------------------------------------
# CommodityBreakoutRule
# ---------------------------------------------------------------------------

class TestCommodityBreakoutRule:
    def _base(self, **overrides):
        d = dict(
            RSI_14=60.0,         # In momentum range [45, 75]
            SMA_20=50.0,
            SMA_50=48.0,         # SMA_20 > SMA_50, spread 4.17%
            close=52.0,          # 4% above SMA_20 → breakout
            volume=1500,
            volume_sma_20=1000,
        )
        d.update(overrides)
        return d

    def test_triggered_miner(self):
        r = CommodityBreakoutRule()
        result = r.evaluate(mining_ctx(symbol="GDX", **self._base()))
        assert result.triggered is True
        assert result.signal == SignalType.BUY
        assert "gold" in result.contributing_factors.get("commodity", "")

    def test_non_miner_rejected(self):
        r = CommodityBreakoutRule()
        result = r.evaluate(mining_ctx(symbol="AAPL", **self._base()))
        assert result.triggered is False
        assert "not in mining stock list" in result.reasoning

    def test_no_uptrend(self):
        r = CommodityBreakoutRule()
        result = r.evaluate(mining_ctx(symbol="GDX", **self._base(SMA_20=47.0, SMA_50=48.0)))
        assert result.triggered is False

    def test_no_breakout(self):
        r = CommodityBreakoutRule()
        # close only 0.5% above SMA_20
        result = r.evaluate(mining_ctx(symbol="GDX", **self._base(close=50.25)))
        assert result.triggered is False
        assert "breakout" in result.reasoning.lower()

    def test_rsi_too_low(self):
        r = CommodityBreakoutRule()
        result = r.evaluate(mining_ctx(symbol="GDX", **self._base(RSI_14=40.0)))
        assert result.triggered is False

    def test_rsi_overbought(self):
        r = CommodityBreakoutRule()
        result = r.evaluate(mining_ctx(symbol="GDX", **self._base(RSI_14=80.0)))
        assert result.triggered is False

    def test_stronger_breakout_higher_confidence(self):
        r = CommodityBreakoutRule()
        strong = r.evaluate(mining_ctx(symbol="GDX", **self._base(close=55.0)))  # 10%
        mild = r.evaluate(mining_ctx(symbol="GDX", **self._base(close=51.1)))    # 2.2%
        assert strong.confidence > mild.confidence

    def test_volume_boost(self):
        r = CommodityBreakoutRule()
        hi_vol = r.evaluate(mining_ctx(symbol="GDX", **self._base(volume=2000)))
        lo_vol = r.evaluate(mining_ctx(symbol="GDX", **self._base(volume=1100)))
        assert hi_vol.confidence >= lo_vol.confidence


# ---------------------------------------------------------------------------
# MinerMetalRatioRule
# ---------------------------------------------------------------------------

class TestMinerMetalRatioRule:
    def _base(self, **overrides):
        d = dict(
            RSI_14=30.0,
            SMA_20=52.0,
            SMA_50=50.0,     # Close near SMA_50 (within 3%)
            SMA_200=45.0,
            close=49.0,      # Above SMA_200 (long-term intact), near SMA_50
        )
        d.update(overrides)
        return d

    def test_triggered(self):
        r = MinerMetalRatioRule()
        result = r.evaluate(mining_ctx(symbol="NEM", **self._base()))
        assert result.triggered is True
        assert "MEAN REVERSION" in result.reasoning

    def test_non_miner_rejected(self):
        r = MinerMetalRatioRule()
        result = r.evaluate(mining_ctx(symbol="TSLA", **self._base()))
        assert result.triggered is False

    def test_rsi_not_oversold(self):
        r = MinerMetalRatioRule()
        result = r.evaluate(mining_ctx(symbol="NEM", **self._base(RSI_14=40.0)))
        assert result.triggered is False

    def test_long_term_trend_broken(self):
        r = MinerMetalRatioRule()
        # close < SMA_200 AND SMA_50 < SMA_200
        result = r.evaluate(mining_ctx(
            symbol="NEM", **self._base(close=40.0, SMA_50=43.0, SMA_200=45.0)
        ))
        assert result.triggered is False

    def test_not_near_support(self):
        r = MinerMetalRatioRule()
        # close far from both SMAs
        result = r.evaluate(mining_ctx(
            symbol="NEM", **self._base(close=55.0, SMA_50=50.0, SMA_200=45.0)
        ))
        assert result.triggered is False

    def test_deeper_oversold_higher_confidence(self):
        r = MinerMetalRatioRule()
        deep = r.evaluate(mining_ctx(symbol="NEM", **self._base(RSI_14=22.0)))
        mild = r.evaluate(mining_ctx(symbol="NEM", **self._base(RSI_14=33.0)))
        assert deep.confidence > mild.confidence

    def test_near_sma200_gets_bonus(self):
        r = MinerMetalRatioRule()
        # Close near SMA_200 → +0.10
        near200 = r.evaluate(mining_ctx(
            symbol="NEM", **self._base(close=45.5, SMA_50=52.0, SMA_200=45.0)
        ))
        assert "SMA_200" in near200.contributing_factors.get("support_level", "")


# ---------------------------------------------------------------------------
# DollarWeaknessRule
# ---------------------------------------------------------------------------

class TestDollarWeaknessRule:
    def _base(self, **overrides):
        d = dict(
            RSI_14=60.0,
            SMA_20=110.0,
            SMA_50=105.0,       # Full alignment
            SMA_200=90.0,
            close=112.0,
            MACD=0.5,
            MACD_SIGNAL=0.3,    # MACD bullish
        )
        d.update(overrides)
        return d

    def test_triggered(self):
        r = DollarWeaknessRule()
        result = r.evaluate(mining_ctx(symbol="GLD", **self._base()))
        assert result.triggered is True
        assert "DOLLAR WEAKNESS" in result.reasoning

    def test_non_miner_rejected(self):
        r = DollarWeaknessRule()
        result = r.evaluate(mining_ctx(symbol="MSFT", **self._base()))
        assert result.triggered is False

    def test_no_full_alignment(self):
        r = DollarWeaknessRule()
        result = r.evaluate(mining_ctx(
            symbol="GLD", **self._base(SMA_50=95.0, SMA_200=100.0)
        ))
        assert result.triggered is False

    def test_weak_trend(self):
        r = DollarWeaknessRule(min_trend_spread=3.0)
        # Spread = (101-100)/100 = 1.0% < 3.0%
        result = r.evaluate(mining_ctx(
            symbol="GLD", **self._base(SMA_20=101.0, SMA_50=100.0, SMA_200=90.0)
        ))
        assert result.triggered is False

    def test_macd_bearish(self):
        r = DollarWeaknessRule(require_macd_positive=True)
        result = r.evaluate(mining_ctx(
            symbol="GLD", **self._base(MACD=0.2, MACD_SIGNAL=0.5)
        ))
        assert result.triggered is False

    def test_macd_not_required(self):
        r = DollarWeaknessRule(require_macd_positive=False)
        result = r.evaluate(mining_ctx(
            symbol="GLD", **self._base(MACD=0.2, MACD_SIGNAL=0.5)
        ))
        # Should pass without MACD check
        assert result.triggered is True

    def test_rsi_too_low(self):
        r = DollarWeaknessRule()
        result = r.evaluate(mining_ctx(symbol="GLD", **self._base(RSI_14=45.0)))
        assert result.triggered is False

    def test_rsi_overbought(self):
        r = DollarWeaknessRule()
        result = r.evaluate(mining_ctx(symbol="GLD", **self._base(RSI_14=80.0)))
        assert result.triggered is False


# ---------------------------------------------------------------------------
# SeasonalityRule
# ---------------------------------------------------------------------------

class TestSeasonalityRule:
    def _base(self, **overrides):
        d = dict(
            RSI_14=50.0,
            SMA_20=55.0,    # Uptrend
            SMA_50=52.0,
            close=56.0,
        )
        d.update(overrides)
        return d

    def test_strong_month_triggered(self):
        r = SeasonalityRule()
        # January is strong for gold → GDX
        result = r.evaluate(mining_ctx(symbol="GDX", month=1, **self._base()))
        assert result.triggered is True
        assert "STRONG" in result.reasoning

    def test_weak_month_no_position_rejected(self):
        r = SeasonalityRule()
        # May is weak for gold, no current_position → rejected
        result = r.evaluate(mining_ctx(symbol="GDX", month=5, **self._base()))
        assert result.triggered is False
        assert "weak" in result.reasoning.lower() or "Weak" in result.reasoning

    def test_neutral_month(self):
        r = SeasonalityRule()
        # March is neutral for gold (not in strong or weak list)
        result = r.evaluate(mining_ctx(symbol="GDX", month=3, **self._base()))
        assert result.triggered is True
        assert "NEUTRAL" in result.reasoning

    def test_non_miner_rejected(self):
        r = SeasonalityRule()
        result = r.evaluate(mining_ctx(symbol="AAPL", month=1, **self._base()))
        assert result.triggered is False

    def test_no_uptrend(self):
        r = SeasonalityRule()
        result = r.evaluate(mining_ctx(
            symbol="GDX", month=1, **self._base(SMA_20=50.0, SMA_50=55.0)
        ))
        assert result.triggered is False

    def test_rsi_too_low(self):
        r = SeasonalityRule()
        result = r.evaluate(mining_ctx(symbol="GDX", month=1, **self._base(RSI_14=25.0)))
        assert result.triggered is False
        assert "oversold" in result.reasoning.lower()

    def test_rsi_too_high(self):
        r = SeasonalityRule()
        result = r.evaluate(mining_ctx(symbol="GDX", month=1, **self._base(RSI_14=70.0)))
        assert result.triggered is False

    def test_uranium_seasonal(self):
        # Uranium strong months: [1,2,3,9,10,11]
        r = SeasonalityRule()
        result = r.evaluate(mining_ctx(symbol="CCJ", month=9, **self._base()))
        assert result.triggered is True
        assert "STRONG" in result.reasoning

    def test_confidence_range(self):
        r = SeasonalityRule()
        result = r.evaluate(mining_ctx(symbol="GDX", month=1, **self._base()))
        assert 0.40 <= result.confidence <= 0.85


# ---------------------------------------------------------------------------
# VolumeBreakoutRule
# ---------------------------------------------------------------------------

class TestVolumeBreakoutRule:
    def _base(self, **overrides):
        d = dict(
            RSI_14=60.0,
            SMA_20=50.0,
            SMA_50=48.0,         # Uptrend
            close=52.0,          # 4% above SMA_20
            volume=2000,         # 2x avg → above 1.5x threshold
            volume_sma_20=1000,
        )
        d.update(overrides)
        return d

    def test_triggered(self):
        r = VolumeBreakoutRule()
        result = r.evaluate(mining_ctx(symbol="AG", **self._base()))
        assert result.triggered is True
        assert result.signal == SignalType.BUY
        assert "VOLUME BREAKOUT" in result.reasoning

    def test_non_miner_rejected(self):
        r = VolumeBreakoutRule()
        result = r.evaluate(mining_ctx(symbol="GOOG", **self._base()))
        assert result.triggered is False

    def test_no_uptrend(self):
        r = VolumeBreakoutRule()
        result = r.evaluate(mining_ctx(symbol="AG", **self._base(SMA_20=47.0, SMA_50=48.0)))
        assert result.triggered is False

    def test_low_volume(self):
        r = VolumeBreakoutRule()
        result = r.evaluate(mining_ctx(symbol="AG", **self._base(volume=1000)))  # ratio=1.0
        assert result.triggered is False

    def test_no_breakout(self):
        r = VolumeBreakoutRule()
        result = r.evaluate(mining_ctx(symbol="AG", **self._base(close=50.5)))  # only 1%
        assert result.triggered is False

    def test_rsi_too_low(self):
        r = VolumeBreakoutRule()
        result = r.evaluate(mining_ctx(symbol="AG", **self._base(RSI_14=40.0)))
        assert result.triggered is False

    def test_rsi_overbought(self):
        r = VolumeBreakoutRule()
        result = r.evaluate(mining_ctx(symbol="AG", **self._base(RSI_14=75.0)))
        assert result.triggered is False

    def test_higher_volume_higher_confidence(self):
        r = VolumeBreakoutRule()
        hi = r.evaluate(mining_ctx(symbol="AG", **self._base(volume=3000)))  # 3x
        lo = r.evaluate(mining_ctx(symbol="AG", **self._base(volume=1600)))  # 1.6x
        assert hi.confidence > lo.confidence

    def test_confidence_capped(self):
        r = VolumeBreakoutRule()
        result = r.evaluate(mining_ctx(symbol="AG", **self._base(volume=5000, close=55.0)))
        assert result.confidence <= 0.90


# ---------------------------------------------------------------------------
# Data maps smoke tests
# ---------------------------------------------------------------------------

class TestMiningDataMaps:
    def test_commodity_map_known_entries(self):
        assert MINER_COMMODITY_MAP["GDX"] == "gold"
        assert MINER_COMMODITY_MAP["SLV"] == "silver"
        assert MINER_COMMODITY_MAP["CCJ"] == "uranium"
        assert MINER_COMMODITY_MAP["FCX"] == "copper"

    def test_seasonal_strength_not_empty(self):
        for commodity, months in SEASONAL_STRENGTH.items():
            assert len(months) > 0, f"{commodity} has empty strong months"

    def test_seasonal_weakness_not_empty(self):
        for commodity, months in SEASONAL_WEAKNESS.items():
            assert len(months) > 0, f"{commodity} has empty weak months"

    def test_no_overlap_strong_and_weak(self):
        for commodity in SEASONAL_STRENGTH:
            strong = set(SEASONAL_STRENGTH.get(commodity, []))
            weak = set(SEASONAL_WEAKNESS.get(commodity, []))
            overlap = strong & weak
            assert not overlap, f"{commodity} has overlapping months: {overlap}"
