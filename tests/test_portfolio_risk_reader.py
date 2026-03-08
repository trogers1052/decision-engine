"""Tests for PortfolioRiskReader."""

import json
from unittest.mock import MagicMock, patch

import pytest

from decision_engine.portfolio_risk_reader import (
    PortfolioRiskReader,
    PortfolioRiskState,
    DAILY_STATE_KEY,
)


@pytest.fixture
def reader():
    """Reader with mocked Redis."""
    r = PortfolioRiskReader(host="localhost", port=6379, db=0)
    r._client = MagicMock()
    return r


class TestPortfolioRiskReader:
    def test_get_state_normal(self, reader):
        """Normal state read."""
        reader._client.get.return_value = json.dumps({
            "date": "2026-03-07",
            "stops_hit_today": 2,
            "stops_hit_symbols": ["CCJ", "URNM"],
            "daily_pnl_pct": -0.038,
            "actual_portfolio_heat": 0.062,
            "halted": False,
            "halt_reason": None,
            "open_position_count": 3,
            "gap_alerts": [],
        })
        state = reader.get_state()
        assert state is not None
        assert state.stops_hit_today == 2
        assert state.stops_hit_symbols == ["CCJ", "URNM"]
        assert abs(state.daily_pnl_pct - (-0.038)) < 0.0001
        assert abs(state.actual_portfolio_heat - 0.062) < 0.0001
        assert state.halted is False
        assert state.open_position_count == 3

    def test_get_state_halted(self, reader):
        """Halted state."""
        reader._client.get.return_value = json.dumps({
            "date": "2026-03-07",
            "stops_hit_today": 3,
            "stops_hit_symbols": ["CCJ", "URNM", "WPM"],
            "daily_pnl_pct": -0.07,
            "actual_portfolio_heat": 0.09,
            "halted": True,
            "halt_reason": "3 stops hit today (limit: 3)",
            "open_position_count": 2,
            "gap_alerts": [],
        })
        state = reader.get_state()
        assert state.halted is True
        assert "3 stops hit" in state.halt_reason

    def test_get_state_no_data(self, reader):
        """No data in Redis returns None (fail-open)."""
        reader._client.get.return_value = None
        assert reader.get_state() is None

    def test_get_state_no_client(self):
        """No Redis client returns None (fail-open)."""
        reader = PortfolioRiskReader(host="localhost", port=6379, db=0)
        reader._client = None
        assert reader.get_state() is None

    def test_get_state_redis_error(self, reader):
        """Redis error returns None (fail-open)."""
        import redis
        reader._client.get.side_effect = redis.RedisError("connection lost")
        assert reader.get_state() is None

    def test_get_state_invalid_json(self, reader):
        """Invalid JSON returns None."""
        reader._client.get.return_value = "not json"
        assert reader.get_state() is None

    def test_get_state_partial_data(self, reader):
        """Missing fields use defaults."""
        reader._client.get.return_value = json.dumps({
            "date": "2026-03-07",
        })
        state = reader.get_state()
        assert state is not None
        assert state.stops_hit_today == 0
        assert state.halted is False
        assert state.gap_alerts == []

    def test_connect_success(self):
        """Successful connection."""
        with patch("decision_engine.portfolio_risk_reader.redis.Redis") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            reader = PortfolioRiskReader(host="redis", port=6379, db=0)
            assert reader.connect() is True
            mock_client.ping.assert_called_once()

    def test_connect_failure(self):
        """Failed connection returns False."""
        import redis
        with patch("decision_engine.portfolio_risk_reader.redis.Redis") as mock_cls:
            mock_cls.return_value.ping.side_effect = redis.RedisError("fail")
            reader = PortfolioRiskReader(host="redis", port=6379, db=0)
            assert reader.connect() is False
            assert reader._client is None

    def test_close(self, reader):
        """Close cleans up."""
        client = reader._client
        reader.close()
        client.close.assert_called_once()
        assert reader._client is None

    def test_reads_correct_key(self, reader):
        """Reads from the expected Redis key."""
        reader._client.get.return_value = None
        reader.get_state()
        reader._client.get.assert_called_once_with(DAILY_STATE_KEY)
