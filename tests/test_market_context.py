"""
Tests for MarketContextReader.

All tests use unittest.mock to avoid a real Redis connection.
"""

import json
import threading
import time
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

from decision_engine.market_context import MarketContextReader, REGIME_MULTIPLIERS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_reader(regime: str = "UNKNOWN", confidence: float = 0.0) -> MarketContextReader:
    """Return a reader with pre-seeded regime state (no Redis connection)."""
    reader = MarketContextReader(
        host="localhost", port=6379, db=0, key="market:context"
    )
    reader._regime = regime
    reader._regime_confidence = confidence
    return reader


def _make_connected_reader(redis_mock) -> MarketContextReader:
    """Return a reader whose _client is replaced with redis_mock."""
    reader = MarketContextReader(
        host="localhost", port=6379, db=0, key="market:context"
    )
    reader._client = redis_mock
    return reader


# ---------------------------------------------------------------------------
# get_multiplier — BUY signal regime scaling
# ---------------------------------------------------------------------------

class TestGetMultiplierBuySignals(unittest.TestCase):

    def test_bull_regime_returns_1_0(self):
        self.assertAlmostEqual(_make_reader("BULL").get_multiplier("BUY"), 1.0)

    def test_sideways_regime_returns_0_7(self):
        self.assertAlmostEqual(_make_reader("SIDEWAYS").get_multiplier("BUY"), 0.7)

    def test_bear_regime_returns_0_3(self):
        self.assertAlmostEqual(_make_reader("BEAR").get_multiplier("BUY"), 0.3)

    def test_unknown_regime_returns_1_0(self):
        self.assertAlmostEqual(_make_reader("UNKNOWN").get_multiplier("BUY"), 1.0)

    def test_unexpected_regime_falls_back_to_1_0(self):
        # An unrecognised regime string must not crash — fall back to 1.0.
        self.assertAlmostEqual(_make_reader("CRASH").get_multiplier("BUY"), 1.0)


class TestGetMultiplierNonBuySignals(unittest.TestCase):
    """SELL and WATCH signals must always return 1.0 regardless of regime."""

    def test_sell_in_bull_returns_1_0(self):
        self.assertAlmostEqual(_make_reader("BULL").get_multiplier("SELL"), 1.0)

    def test_sell_in_bear_returns_1_0(self):
        self.assertAlmostEqual(_make_reader("BEAR").get_multiplier("SELL"), 1.0)

    def test_watch_in_sideways_returns_1_0(self):
        self.assertAlmostEqual(_make_reader("SIDEWAYS").get_multiplier("WATCH"), 1.0)

    def test_watch_in_bear_returns_1_0(self):
        self.assertAlmostEqual(_make_reader("BEAR").get_multiplier("WATCH"), 1.0)


# ---------------------------------------------------------------------------
# REGIME_MULTIPLIERS constant
# ---------------------------------------------------------------------------

class TestRegimeMultipliersConstant(unittest.TestCase):

    def test_all_four_regimes_present(self):
        for regime in ("BULL", "SIDEWAYS", "BEAR", "UNKNOWN"):
            self.assertIn(regime, REGIME_MULTIPLIERS, f"{regime} missing from REGIME_MULTIPLIERS")

    def test_values_in_range(self):
        for regime, multiplier in REGIME_MULTIPLIERS.items():
            self.assertGreaterEqual(multiplier, 0.0, f"{regime} multiplier < 0")
            self.assertLessEqual(multiplier, 1.0, f"{regime} multiplier > 1")

    def test_bear_less_than_sideways_less_than_bull(self):
        self.assertLess(REGIME_MULTIPLIERS["BEAR"], REGIME_MULTIPLIERS["SIDEWAYS"])
        self.assertLessEqual(REGIME_MULTIPLIERS["SIDEWAYS"], REGIME_MULTIPLIERS["BULL"])


# ---------------------------------------------------------------------------
# Default state
# ---------------------------------------------------------------------------

class TestDefaultState(unittest.TestCase):

    def setUp(self):
        self.reader = MarketContextReader(
            host="localhost", port=6379, db=0, key="market:context"
        )

    def test_default_regime_is_unknown(self):
        self.assertEqual(self.reader.get_regime(), "UNKNOWN")

    def test_default_confidence_is_zero(self):
        self.assertAlmostEqual(self.reader.get_regime_confidence(), 0.0)

    def test_default_multiplier_buy_is_1_0(self):
        # UNKNOWN → 1.0; no penalty at startup before first publish
        self.assertAlmostEqual(self.reader.get_multiplier("BUY"), 1.0)


# ---------------------------------------------------------------------------
# _refresh — parsing Redis payload
# ---------------------------------------------------------------------------

class TestRefresh(unittest.TestCase):

    def _refresh_with(self, payload) -> MarketContextReader:
        """Create a reader, call _refresh with a mocked Redis returning payload."""
        mock_redis = MagicMock()
        mock_redis.get.return_value = json.dumps(payload) if payload is not None else None
        reader = _make_connected_reader(mock_redis)
        reader._refresh()
        return reader

    def test_refresh_bull_updates_regime(self):
        reader = self._refresh_with({"regime": "BULL", "regime_confidence": 0.85})
        self.assertEqual(reader.get_regime(), "BULL")
        self.assertAlmostEqual(reader.get_regime_confidence(), 0.85)

    def test_refresh_bear_updates_regime(self):
        reader = self._refresh_with({"regime": "BEAR", "regime_confidence": 0.60})
        self.assertEqual(reader.get_regime(), "BEAR")
        self.assertAlmostEqual(reader.get_regime_confidence(), 0.60)

    def test_refresh_sideways_updates_regime(self):
        reader = self._refresh_with({"regime": "SIDEWAYS", "regime_confidence": 0.72})
        self.assertEqual(reader.get_regime(), "SIDEWAYS")

    def test_refresh_none_key_leaves_regime_unchanged(self):
        """If the Redis key doesn't exist yet, regime must stay at its previous value."""
        mock_redis = MagicMock()
        mock_redis.get.return_value = None
        reader = _make_connected_reader(mock_redis)
        reader._regime = "BULL"
        reader._refresh()
        self.assertEqual(reader.get_regime(), "BULL")  # unchanged

    def test_refresh_invalid_json_leaves_regime_unchanged(self):
        """Corrupt Redis data must not crash or change state."""
        mock_redis = MagicMock()
        mock_redis.get.return_value = "not-valid-json{"
        reader = _make_connected_reader(mock_redis)
        reader._regime = "SIDEWAYS"
        reader._refresh()
        self.assertEqual(reader.get_regime(), "SIDEWAYS")  # unchanged

    def test_refresh_redis_error_leaves_regime_unchanged(self):
        """RedisError during GET must be caught and state left unchanged."""
        import redis as _redis
        mock_redis = MagicMock()
        mock_redis.get.side_effect = _redis.RedisError("connection refused")
        reader = _make_connected_reader(mock_redis)
        reader._regime = "BULL"
        reader._refresh()
        self.assertEqual(reader.get_regime(), "BULL")  # unchanged

    def test_refresh_lowercase_regime_normalised_to_upper(self):
        """context-service publishes uppercase, but be defensive about case."""
        reader = self._refresh_with({"regime": "bull", "regime_confidence": 0.9})
        self.assertEqual(reader.get_regime(), "BULL")

    def test_refresh_missing_confidence_defaults_to_zero(self):
        reader = self._refresh_with({"regime": "BULL"})
        self.assertAlmostEqual(reader.get_regime_confidence(), 0.0)

    def test_refresh_missing_regime_defaults_to_unknown(self):
        reader = self._refresh_with({"regime_confidence": 0.5})
        self.assertEqual(reader.get_regime(), "UNKNOWN")

    def test_refresh_no_op_when_client_is_none(self):
        """_refresh must not crash if called before Redis connected."""
        reader = MarketContextReader(
            host="localhost", port=6379, db=0, key="market:context"
        )
        # _client is None by default
        reader._refresh()  # should not raise
        self.assertEqual(reader.get_regime(), "UNKNOWN")


# ---------------------------------------------------------------------------
# start() — Redis connection failure is non-fatal
# ---------------------------------------------------------------------------

class TestStartConnectionFailure(unittest.TestCase):

    def test_start_with_unreachable_redis_sets_no_client(self):
        """start() must not raise even if Redis is unreachable at startup."""
        import redis as _redis
        with patch("decision_engine.market_context.redis.Redis") as mock_cls:
            instance = MagicMock()
            instance.ping.side_effect = _redis.RedisError("refused")
            mock_cls.return_value = instance

            reader = MarketContextReader(
                host="bad-host", port=6379, db=0, key="market:context"
            )
            reader.start()  # must not raise

            # Client should not be retained if ping failed
            self.assertIsNone(reader._client)
            # Regime stays at safe default
            self.assertEqual(reader.get_regime(), "UNKNOWN")


# ---------------------------------------------------------------------------
# stop() — graceful shutdown
# ---------------------------------------------------------------------------

class TestStop(unittest.TestCase):

    def test_stop_without_start_does_not_raise(self):
        reader = MarketContextReader(
            host="localhost", port=6379, db=0, key="market:context"
        )
        reader.stop()  # thread was never started — must not raise

    def test_stop_joins_background_thread(self):
        """After stop(), the background thread must no longer be alive."""
        import redis as _redis
        with patch("decision_engine.market_context.redis.Redis") as mock_cls:
            instance = MagicMock()
            instance.ping.return_value = True
            instance.get.return_value = None
            mock_cls.return_value = instance

            reader = MarketContextReader(
                host="localhost", port=6379, db=0, key="market:context",
                refresh_interval=60,  # long interval — thread just waits
            )
            reader.start()
            self.assertIsNotNone(reader._thread)
            self.assertTrue(reader._thread.is_alive())

            reader.stop()
            self.assertFalse(reader._thread.is_alive())


# ---------------------------------------------------------------------------
# Thread safety — concurrent reads while _refresh runs
# ---------------------------------------------------------------------------

class TestThreadSafety(unittest.TestCase):

    def test_concurrent_reads_during_refresh(self):
        """
        Multiple reader threads calling get_regime() while another thread
        calls _refresh() must never observe a torn state (partial update).
        """
        import redis as _redis

        regimes = ["BULL", "BEAR", "SIDEWAYS", "UNKNOWN"]
        errors = []
        valid_regimes = set(regimes)

        mock_redis = MagicMock()
        # Alternate between regimes on each Redis GET call
        call_count = [0]

        def get_side_effect(key):
            r = regimes[call_count[0] % len(regimes)]
            call_count[0] += 1
            return json.dumps({"regime": r, "regime_confidence": 0.8})

        mock_redis.get.side_effect = get_side_effect
        reader = _make_connected_reader(mock_redis)

        def refresher():
            for _ in range(100):
                reader._refresh()

        def getter():
            for _ in range(500):
                regime = reader.get_regime()
                confidence = reader.get_regime_confidence()
                if regime not in valid_regimes:
                    errors.append(f"Invalid regime: {regime!r}")
                if not (0.0 <= confidence <= 1.0):
                    errors.append(f"Invalid confidence: {confidence}")

        threads = (
            [threading.Thread(target=refresher)]
            + [threading.Thread(target=getter) for _ in range(4)]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [], f"Thread safety violations: {errors}")


if __name__ == "__main__":
    unittest.main()
