"""
Tests for FeedbackAccuracyReader.

All tests use unittest.mock to avoid a real Redis connection.
"""

import json
import threading
import unittest
from unittest.mock import MagicMock, patch

from decision_engine.feedback_reader import FeedbackAccuracyReader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_reader(**kwargs) -> FeedbackAccuracyReader:
    """Return a reader with no Redis connection."""
    return FeedbackAccuracyReader(
        host="localhost", port=6379, db=0, **kwargs
    )


def _make_reader_with_data(data: dict) -> FeedbackAccuracyReader:
    """Return a reader pre-loaded with accuracy data."""
    reader = _make_reader()
    reader._data = data
    return reader


SAMPLE_DATA = {
    "MomentumReversal:BULL": {"trade_rate": 0.75, "multiplier": 1.25, "signal_count": 20},
    "MomentumReversal:ALL": {"trade_rate": 0.60, "multiplier": 1.10, "signal_count": 50},
    "RSIOversold:BEAR": {"trade_rate": 0.10, "multiplier": 0.60, "signal_count": 15},
    "RSIOversold:ALL": {"trade_rate": 0.30, "multiplier": 0.80, "signal_count": 40},
    "TrendAlignment:BULL": {"trade_rate": 1.00, "multiplier": 1.50, "signal_count": 12},
}


# ---------------------------------------------------------------------------
# get_multiplier — exact match, ALL fallback, and default
# ---------------------------------------------------------------------------

class TestGetMultiplier(unittest.TestCase):

    def setUp(self):
        self.reader = _make_reader_with_data(SAMPLE_DATA)

    def test_exact_match(self):
        """Exact rule:regime match returns the stored multiplier."""
        self.assertAlmostEqual(
            self.reader.get_multiplier("MomentumReversal", "BULL"), 1.25
        )

    def test_exact_match_bear(self):
        self.assertAlmostEqual(
            self.reader.get_multiplier("RSIOversold", "BEAR"), 0.60
        )

    def test_fallback_to_all(self):
        """When exact regime not found, falls back to rule:ALL."""
        self.assertAlmostEqual(
            self.reader.get_multiplier("MomentumReversal", "BEAR"), 1.10
        )

    def test_fallback_to_all_sideways(self):
        self.assertAlmostEqual(
            self.reader.get_multiplier("RSIOversold", "SIDEWAYS"), 0.80
        )

    def test_no_data_returns_1_0(self):
        """Unknown rule returns neutral multiplier 1.0."""
        self.assertAlmostEqual(
            self.reader.get_multiplier("UnknownRule", "BULL"), 1.0
        )

    def test_no_data_unknown_regime(self):
        self.assertAlmostEqual(
            self.reader.get_multiplier("UnknownRule", "UNKNOWN"), 1.0
        )

    def test_max_multiplier_is_1_5(self):
        """TrendAlignment:BULL has 100% trade rate → 1.50 multiplier."""
        self.assertAlmostEqual(
            self.reader.get_multiplier("TrendAlignment", "BULL"), 1.50
        )


# ---------------------------------------------------------------------------
# get_aggregate_multiplier — averaging across rules
# ---------------------------------------------------------------------------

class TestGetAggregateMultiplier(unittest.TestCase):

    def setUp(self):
        self.reader = _make_reader_with_data(SAMPLE_DATA)

    def test_empty_rules_returns_1_0(self):
        self.assertAlmostEqual(
            self.reader.get_aggregate_multiplier([], "BULL"), 1.0
        )

    def test_single_rule_with_data(self):
        result = self.reader.get_aggregate_multiplier(["MomentumReversal"], "BULL")
        self.assertAlmostEqual(result, 1.25)

    def test_two_rules_averaged(self):
        """MomentumReversal:BEAR→ALL=1.10, RSIOversold:BEAR=0.60 → avg=0.85"""
        result = self.reader.get_aggregate_multiplier(
            ["MomentumReversal", "RSIOversold"], "BEAR"
        )
        self.assertAlmostEqual(result, (1.10 + 0.60) / 2, places=2)

    def test_all_unknown_rules_returns_1_0(self):
        """When none of the rules have data, return neutral 1.0."""
        result = self.reader.get_aggregate_multiplier(
            ["NoData1", "NoData2", "NoData3"], "BULL"
        )
        self.assertAlmostEqual(result, 1.0)

    def test_mix_of_known_and_unknown_rules(self):
        """Only rules with data contribute to the average."""
        result = self.reader.get_aggregate_multiplier(
            ["MomentumReversal", "UnknownRule"], "BULL"
        )
        # MomentumReversal:BULL = 1.25, UnknownRule has no data → excluded
        self.assertAlmostEqual(result, 1.25)


# ---------------------------------------------------------------------------
# get_entry_count
# ---------------------------------------------------------------------------

class TestGetEntryCount(unittest.TestCase):

    def test_empty_reader(self):
        reader = _make_reader()
        self.assertEqual(reader.get_entry_count(), 0)

    def test_with_data(self):
        reader = _make_reader_with_data(SAMPLE_DATA)
        self.assertEqual(reader.get_entry_count(), len(SAMPLE_DATA))


# ---------------------------------------------------------------------------
# Default state
# ---------------------------------------------------------------------------

class TestDefaultState(unittest.TestCase):

    def setUp(self):
        self.reader = _make_reader()

    def test_default_multiplier_is_1_0(self):
        self.assertAlmostEqual(
            self.reader.get_multiplier("AnyRule", "BULL"), 1.0
        )

    def test_default_entry_count_is_zero(self):
        self.assertEqual(self.reader.get_entry_count(), 0)


# ---------------------------------------------------------------------------
# _refresh — parsing Redis payload
# ---------------------------------------------------------------------------

class TestRefresh(unittest.TestCase):

    def _refresh_with(self, payload) -> FeedbackAccuracyReader:
        """Create a reader, call _refresh with mocked Redis returning payload."""
        mock_redis = MagicMock()
        mock_redis.get.return_value = json.dumps(payload) if payload is not None else None
        reader = _make_reader()
        reader._client = mock_redis
        reader._refresh()
        return reader

    def test_refresh_loads_data(self):
        reader = self._refresh_with(SAMPLE_DATA)
        self.assertEqual(reader.get_entry_count(), len(SAMPLE_DATA))
        self.assertAlmostEqual(
            reader.get_multiplier("MomentumReversal", "BULL"), 1.25
        )

    def test_refresh_none_key_leaves_data_unchanged(self):
        """If Redis key absent, data stays at previous value."""
        reader = _make_reader_with_data(SAMPLE_DATA)
        mock_redis = MagicMock()
        mock_redis.get.return_value = None
        reader._client = mock_redis
        reader._refresh()
        self.assertEqual(reader.get_entry_count(), len(SAMPLE_DATA))

    def test_refresh_invalid_json_leaves_data_unchanged(self):
        reader = _make_reader_with_data(SAMPLE_DATA)
        mock_redis = MagicMock()
        mock_redis.get.return_value = "not-valid-json{"
        reader._client = mock_redis
        reader._refresh()
        self.assertEqual(reader.get_entry_count(), len(SAMPLE_DATA))

    def test_refresh_redis_error_leaves_data_unchanged(self):
        import redis as _redis
        reader = _make_reader_with_data(SAMPLE_DATA)
        mock_redis = MagicMock()
        mock_redis.get.side_effect = _redis.RedisError("connection lost")
        reader._client = mock_redis
        reader._refresh()
        self.assertEqual(reader.get_entry_count(), len(SAMPLE_DATA))

    def test_refresh_no_op_when_client_is_none(self):
        reader = _make_reader()
        reader._refresh()  # should not raise
        self.assertEqual(reader.get_entry_count(), 0)

    def test_refresh_replaces_all_data(self):
        """A new refresh completely replaces old data."""
        reader = _make_reader_with_data(SAMPLE_DATA)
        new_data = {"NewRule:BULL": {"trade_rate": 0.50, "multiplier": 1.00, "signal_count": 30}}
        mock_redis = MagicMock()
        mock_redis.get.return_value = json.dumps(new_data)
        reader._client = mock_redis
        reader._refresh()
        self.assertEqual(reader.get_entry_count(), 1)
        # Old data gone
        self.assertAlmostEqual(
            reader.get_multiplier("MomentumReversal", "BULL"), 1.0
        )


# ---------------------------------------------------------------------------
# start() — Redis connection failure is non-fatal
# ---------------------------------------------------------------------------

class TestStartConnectionFailure(unittest.TestCase):

    def test_start_with_unreachable_redis_sets_no_client(self):
        import redis as _redis
        with patch("decision_engine.feedback_reader.redis.Redis") as mock_cls:
            instance = MagicMock()
            instance.ping.side_effect = _redis.RedisError("refused")
            mock_cls.return_value = instance

            reader = _make_reader()
            reader.start()

            self.assertIsNone(reader._client)
            self.assertAlmostEqual(
                reader.get_multiplier("AnyRule", "BULL"), 1.0
            )


# ---------------------------------------------------------------------------
# stop() — graceful shutdown
# ---------------------------------------------------------------------------

class TestStop(unittest.TestCase):

    def test_stop_without_start_does_not_raise(self):
        reader = _make_reader()
        reader.stop()

    def test_stop_joins_background_thread(self):
        with patch("decision_engine.feedback_reader.redis.Redis") as mock_cls:
            instance = MagicMock()
            instance.ping.return_value = True
            instance.get.return_value = None
            mock_cls.return_value = instance

            reader = _make_reader(refresh_interval=60)
            reader.start()
            self.assertTrue(reader._thread.is_alive())

            reader.stop()
            self.assertFalse(reader._thread.is_alive())


# ---------------------------------------------------------------------------
# Thread safety — concurrent reads while _refresh runs
# ---------------------------------------------------------------------------

class TestThreadSafety(unittest.TestCase):

    def test_concurrent_reads_during_refresh(self):
        """Readers must never see torn state during refresh."""
        errors = []
        datasets = [
            {"Rule1:BULL": {"multiplier": 1.25}},
            {"Rule2:BEAR": {"multiplier": 0.60}},
            {"Rule1:BULL": {"multiplier": 1.50}},
        ]

        mock_redis = MagicMock()
        call_count = [0]

        def get_side_effect(key):
            d = datasets[call_count[0] % len(datasets)]
            call_count[0] += 1
            return json.dumps(d)

        mock_redis.get.side_effect = get_side_effect
        reader = _make_reader()
        reader._client = mock_redis

        def refresher():
            for _ in range(100):
                reader._refresh()

        def getter():
            for _ in range(500):
                m = reader.get_multiplier("Rule1", "BULL")
                if not isinstance(m, float):
                    errors.append(f"Non-float multiplier: {m!r}")

        threads = (
            [threading.Thread(target=refresher)]
            + [threading.Thread(target=getter) for _ in range(4)]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [], f"Thread safety violations: {errors}")


# ---------------------------------------------------------------------------
# Multiplier value range — clamp(0.5, 1.5)
# ---------------------------------------------------------------------------

class TestMultiplierRange(unittest.TestCase):
    """Stock-service computes multiplier as clamp(0.5, 1.5, 0.5 + trade_rate).
    The reader just reads and returns the stored value, but let's verify the
    expected range is correct for representative trade rates."""

    def test_zero_trade_rate_gives_0_5(self):
        reader = _make_reader_with_data({
            "NeverTraded:ALL": {"trade_rate": 0.0, "multiplier": 0.50, "signal_count": 20}
        })
        self.assertAlmostEqual(reader.get_multiplier("NeverTraded", "BULL"), 0.50)

    def test_half_trade_rate_gives_1_0(self):
        reader = _make_reader_with_data({
            "HalfTraded:ALL": {"trade_rate": 0.50, "multiplier": 1.00, "signal_count": 20}
        })
        self.assertAlmostEqual(reader.get_multiplier("HalfTraded", "BULL"), 1.00)

    def test_full_trade_rate_gives_1_5(self):
        reader = _make_reader_with_data({
            "AlwaysTraded:ALL": {"trade_rate": 1.00, "multiplier": 1.50, "signal_count": 20}
        })
        self.assertAlmostEqual(reader.get_multiplier("AlwaysTraded", "BULL"), 1.50)


if __name__ == "__main__":
    unittest.main()
