"""Tests for TierReader trade count floor."""

import json
import pytest
from unittest.mock import MagicMock, patch

from decision_engine.tier_reader import (
    TierReader,
    TierData,
    TRADE_COUNT_FLOOR,
    C_TIER_MAX_CONFIDENCE,
    C_TIER_MAX_POSITION_SIZE,
    DEFAULT_CONFIDENCE_MULTIPLIER,
    DEFAULT_POSITION_SIZE_MULTIPLIER,
)


class TestTradeCountFloor:
    """Tests for trade count floor enforcement in TierReader."""

    @staticmethod
    def _make_reader_with_data(tier_data: dict) -> TierReader:
        """Create a TierReader with mocked Redis returning given data."""
        reader = TierReader(host="localhost", port=6379, db=0)
        mock_redis = MagicMock()
        mock_redis.get.return_value = json.dumps(tier_data)
        reader._client = mock_redis
        return reader

    def test_trade_count_above_floor(self):
        """50 trades, S-tier → multipliers unchanged."""
        reader = self._make_reader_with_data({
            "symbol": "APH",
            "tier": "S",
            "composite_score": 85.0,
            "confidence_multiplier": 1.15,
            "position_size_multiplier": 1.15,
            "trade_count": 50,
        })

        td = reader.get_tier("APH")
        assert td is not None
        assert td.confidence_multiplier == 1.15
        assert td.position_size_multiplier == 1.15
        assert td.trade_count == 50

    def test_trade_count_below_floor(self):
        """8 trades, S-tier → capped at C-tier multipliers."""
        reader = self._make_reader_with_data({
            "symbol": "RTX",
            "tier": "S",
            "composite_score": 85.0,
            "confidence_multiplier": 1.15,
            "position_size_multiplier": 1.15,
            "trade_count": 8,
        })

        td = reader.get_tier("RTX")
        assert td is not None
        assert td.confidence_multiplier == C_TIER_MAX_CONFIDENCE  # 0.85
        assert td.position_size_multiplier == C_TIER_MAX_POSITION_SIZE  # 0.50
        assert td.trade_count == 8

    def test_trade_count_c_tier_unaffected(self):
        """8 trades, C-tier → already ≤ C, no change."""
        reader = self._make_reader_with_data({
            "symbol": "SLV",
            "tier": "C",
            "composite_score": 40.0,
            "confidence_multiplier": 0.85,
            "position_size_multiplier": 0.50,
            "trade_count": 8,
        })

        td = reader.get_tier("SLV")
        assert td is not None
        assert td.confidence_multiplier == 0.85
        assert td.position_size_multiplier == 0.50

    def test_trade_count_d_tier_unaffected(self):
        """5 trades, D-tier → already below C caps, no change."""
        reader = self._make_reader_with_data({
            "symbol": "XYZ",
            "tier": "D",
            "composite_score": 20.0,
            "confidence_multiplier": 0.65,
            "position_size_multiplier": 0.35,
            "trade_count": 5,
        })

        td = reader.get_tier("XYZ")
        assert td is not None
        assert td.confidence_multiplier == 0.65
        assert td.position_size_multiplier == 0.35

    def test_trade_count_zero_no_floor(self):
        """0 trades → no floor applied (0 = data unavailable)."""
        reader = self._make_reader_with_data({
            "symbol": "NEW",
            "tier": "A",
            "composite_score": 70.0,
            "confidence_multiplier": 1.05,
            "position_size_multiplier": 1.05,
            "trade_count": 0,
        })

        td = reader.get_tier("NEW")
        assert td is not None
        assert td.confidence_multiplier == 1.05
        assert td.position_size_multiplier == 1.05

    def test_trade_count_exactly_at_floor(self):
        """20 trades → no cap (not below floor)."""
        reader = self._make_reader_with_data({
            "symbol": "CCJ",
            "tier": "A",
            "composite_score": 72.0,
            "confidence_multiplier": 1.05,
            "position_size_multiplier": 1.05,
            "trade_count": 20,
        })

        td = reader.get_tier("CCJ")
        assert td is not None
        assert td.confidence_multiplier == 1.05
        assert td.position_size_multiplier == 1.05

    def test_trade_count_missing_from_redis(self):
        """trade_count not in Redis payload → defaults to 0, no floor."""
        reader = self._make_reader_with_data({
            "symbol": "OLD",
            "tier": "S",
            "composite_score": 90.0,
            "confidence_multiplier": 1.15,
            "position_size_multiplier": 1.15,
        })

        td = reader.get_tier("OLD")
        assert td is not None
        assert td.trade_count == 0
        assert td.confidence_multiplier == 1.15  # No floor applied

    def test_position_size_multiplier_floored(self):
        """get_position_size_multiplier respects trade count floor."""
        reader = self._make_reader_with_data({
            "symbol": "RTX",
            "tier": "S",
            "composite_score": 85.0,
            "confidence_multiplier": 1.15,
            "position_size_multiplier": 1.15,
            "trade_count": 11,
        })

        assert reader.get_position_size_multiplier("RTX") == C_TIER_MAX_POSITION_SIZE

    def test_confidence_multiplier_floored(self):
        """get_confidence_multiplier respects trade count floor."""
        reader = self._make_reader_with_data({
            "symbol": "ITA",
            "tier": "A",
            "composite_score": 72.0,
            "confidence_multiplier": 1.05,
            "position_size_multiplier": 1.05,
            "trade_count": 15,
        })

        assert reader.get_confidence_multiplier("ITA") == C_TIER_MAX_CONFIDENCE
