"""
Tests for DailyLossMonitor.

All tests use unittest.mock to avoid a real Redis connection.
"""

import json
import threading
import unittest
from unittest.mock import MagicMock, patch

from decision_engine.daily_loss_monitor import (
    DailyLossMonitor,
    BUYING_POWER_KEY,
    DAILY_EQUITY_OPEN_KEY,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_monitor(threshold_pct: float = 0.08) -> DailyLossMonitor:
    """Return a monitor with no Redis connection."""
    return DailyLossMonitor(
        host="localhost", port=6379, db=0, threshold_pct=threshold_pct,
    )


def _make_connected_monitor(
    redis_mock,
    threshold_pct: float = 0.08,
) -> DailyLossMonitor:
    """Return a monitor whose _client is replaced with redis_mock."""
    monitor = _make_monitor(threshold_pct=threshold_pct)
    monitor._client = redis_mock
    return monitor


def _mock_redis(opening_equity=None, current_equity=None, date_str="2026-03-06"):
    """Build a MagicMock Redis that returns the given equity values."""
    mock = MagicMock()

    def get_side_effect(key):
        if key == DAILY_EQUITY_OPEN_KEY and opening_equity is not None:
            return json.dumps({
                "equity": str(opening_equity),
                "date": date_str,
                "updated_at": "2026-03-06T13:30:00+00:00",
            })
        if key == BUYING_POWER_KEY and current_equity is not None:
            return json.dumps({
                "buying_power": "500.00",
                "cash": "500.00",
                "total_equity": str(current_equity),
                "updated_at": "2026-03-06T14:00:00+00:00",
            })
        return None

    mock.get.side_effect = get_side_effect
    return mock


# ---------------------------------------------------------------------------
# Default state
# ---------------------------------------------------------------------------

class TestDefaultState(unittest.TestCase):

    def test_default_not_halted(self):
        monitor = _make_monitor()
        self.assertFalse(monitor.is_halted())

    def test_default_pnl_is_zero(self):
        monitor = _make_monitor()
        self.assertAlmostEqual(monitor.get_daily_pnl_pct(), 0.0)

    def test_get_status_returns_dict(self):
        monitor = _make_monitor()
        status = monitor.get_status()
        self.assertIn("halted", status)
        self.assertIn("daily_pnl_pct", status)
        self.assertIn("threshold_pct", status)
        self.assertAlmostEqual(status["threshold_pct"], 0.08)


# ---------------------------------------------------------------------------
# _refresh — halt logic
# ---------------------------------------------------------------------------

class TestRefreshHaltLogic(unittest.TestCase):

    def test_no_halt_on_small_loss(self):
        """3% loss should NOT trip the 8% breaker."""
        mock = _mock_redis(opening_equity=1000.0, current_equity=970.0)
        monitor = _make_connected_monitor(mock)
        monitor._refresh()
        self.assertFalse(monitor.is_halted())
        self.assertAlmostEqual(monitor.get_daily_pnl_pct(), -0.03, places=4)

    def test_halt_on_large_loss(self):
        """9% loss SHOULD trip the 8% breaker."""
        mock = _mock_redis(opening_equity=1000.0, current_equity=910.0)
        monitor = _make_connected_monitor(mock)
        monitor._refresh()
        self.assertTrue(monitor.is_halted())
        self.assertAlmostEqual(monitor.get_daily_pnl_pct(), -0.09, places=4)

    def test_halt_at_exact_threshold(self):
        """Exactly 8% loss should trip the breaker (<=)."""
        mock = _mock_redis(opening_equity=1000.0, current_equity=920.0)
        monitor = _make_connected_monitor(mock)
        monitor._refresh()
        self.assertTrue(monitor.is_halted())
        self.assertAlmostEqual(monitor.get_daily_pnl_pct(), -0.08, places=4)

    def test_no_halt_on_gain(self):
        """2% gain should NOT trip the breaker."""
        mock = _mock_redis(opening_equity=1000.0, current_equity=1020.0)
        monitor = _make_connected_monitor(mock)
        monitor._refresh()
        self.assertFalse(monitor.is_halted())
        self.assertAlmostEqual(monitor.get_daily_pnl_pct(), 0.02, places=4)

    def test_no_halt_on_zero_change(self):
        """No change should NOT trip the breaker."""
        mock = _mock_redis(opening_equity=1000.0, current_equity=1000.0)
        monitor = _make_connected_monitor(mock)
        monitor._refresh()
        self.assertFalse(monitor.is_halted())

    def test_custom_threshold(self):
        """5% threshold should trip on 6% loss."""
        mock = _mock_redis(opening_equity=1000.0, current_equity=940.0)
        monitor = _make_connected_monitor(mock, threshold_pct=0.05)
        monitor._refresh()
        self.assertTrue(monitor.is_halted())

    def test_custom_threshold_not_tripped(self):
        """5% threshold should NOT trip on 3% loss."""
        mock = _mock_redis(opening_equity=1000.0, current_equity=970.0)
        monitor = _make_connected_monitor(mock, threshold_pct=0.05)
        monitor._refresh()
        self.assertFalse(monitor.is_halted())

    def test_recovery_clears_halt(self):
        """If equity recovers above threshold, halt should clear."""
        mock_down = _mock_redis(opening_equity=1000.0, current_equity=910.0)
        monitor = _make_connected_monitor(mock_down)
        monitor._refresh()
        self.assertTrue(monitor.is_halted())

        # Equity recovers
        mock_up = _mock_redis(opening_equity=1000.0, current_equity=950.0)
        monitor._client = mock_up
        monitor._refresh()
        self.assertFalse(monitor.is_halted())


# ---------------------------------------------------------------------------
# Fail-open scenarios
# ---------------------------------------------------------------------------

class TestFailOpen(unittest.TestCase):

    def test_fail_open_when_redis_unavailable(self):
        """No Redis client → not halted."""
        monitor = _make_monitor()
        # _client is None
        monitor._refresh()
        self.assertFalse(monitor.is_halted())

    def test_fail_open_when_opening_equity_missing(self):
        """No daily equity snapshot → not halted."""
        mock = _mock_redis(opening_equity=None, current_equity=910.0)
        monitor = _make_connected_monitor(mock)
        monitor._refresh()
        self.assertFalse(monitor.is_halted())

    def test_fail_open_when_buying_power_missing(self):
        """No current buying power → not halted."""
        mock = _mock_redis(opening_equity=1000.0, current_equity=None)
        monitor = _make_connected_monitor(mock)
        monitor._refresh()
        self.assertFalse(monitor.is_halted())

    def test_fail_open_on_redis_error(self):
        """RedisError during GET → not halted (state unchanged)."""
        import redis as _redis
        mock = MagicMock()
        mock.get.side_effect = _redis.RedisError("connection refused")
        monitor = _make_connected_monitor(mock)
        monitor._refresh()
        self.assertFalse(monitor.is_halted())

    def test_fail_open_on_corrupt_opening_equity(self):
        """Invalid JSON in daily equity key → not halted."""
        mock = MagicMock()
        mock.get.return_value = "not-valid-json{"
        monitor = _make_connected_monitor(mock)
        monitor._refresh()
        self.assertFalse(monitor.is_halted())

    def test_fail_open_on_zero_opening_equity(self):
        """Opening equity of 0 → not halted (avoid division by zero)."""
        mock = _mock_redis(opening_equity=0.0, current_equity=910.0)
        monitor = _make_connected_monitor(mock)
        monitor._refresh()
        self.assertFalse(monitor.is_halted())

    def test_fail_open_on_negative_opening_equity(self):
        """Negative opening equity → not halted."""
        mock = _mock_redis(opening_equity=-100.0, current_equity=910.0)
        monitor = _make_connected_monitor(mock)
        monitor._refresh()
        self.assertFalse(monitor.is_halted())


# ---------------------------------------------------------------------------
# New day reset
# ---------------------------------------------------------------------------

class TestNewDayReset(unittest.TestCase):

    def test_new_day_resets_halt_triggered_flag(self):
        """When snapshot date changes, halt_triggered_today resets."""
        # Day 1: trigger halt
        mock_day1 = _mock_redis(
            opening_equity=1000.0, current_equity=910.0, date_str="2026-03-06",
        )
        monitor = _make_connected_monitor(mock_day1)
        monitor._refresh()
        self.assertTrue(monitor.is_halted())
        self.assertTrue(monitor._halt_triggered_today)

        # Day 2: fresh start with new date
        mock_day2 = _mock_redis(
            opening_equity=910.0, current_equity=900.0, date_str="2026-03-07",
        )
        monitor._client = mock_day2
        monitor._refresh()
        # 1.1% loss from new opening — should NOT be halted
        self.assertFalse(monitor.is_halted())

    def test_new_day_can_trigger_halt_again(self):
        """After day reset, a new 8%+ loss should trigger halt again."""
        # Day 1: trigger halt
        mock_day1 = _mock_redis(
            opening_equity=1000.0, current_equity=910.0, date_str="2026-03-06",
        )
        monitor = _make_connected_monitor(mock_day1)
        monitor._refresh()
        self.assertTrue(monitor.is_halted())

        # Day 2: another bad day
        mock_day2 = _mock_redis(
            opening_equity=910.0, current_equity=830.0, date_str="2026-03-07",
        )
        monitor._client = mock_day2
        monitor._refresh()
        self.assertTrue(monitor.is_halted())


# ---------------------------------------------------------------------------
# get_status
# ---------------------------------------------------------------------------

class TestGetStatus(unittest.TestCase):

    def test_status_reflects_current_state(self):
        mock = _mock_redis(opening_equity=1000.0, current_equity=950.0)
        monitor = _make_connected_monitor(mock)
        monitor._refresh()
        status = monitor.get_status()
        self.assertFalse(status["halted"])
        self.assertAlmostEqual(status["daily_pnl_pct"], -0.05, places=4)
        self.assertAlmostEqual(status["opening_equity"], 1000.0)
        self.assertAlmostEqual(status["current_equity"], 950.0)
        self.assertAlmostEqual(status["threshold_pct"], 0.08)
        self.assertEqual(status["snapshot_date"], "2026-03-06")


# ---------------------------------------------------------------------------
# start() — Redis connection failure is non-fatal
# ---------------------------------------------------------------------------

class TestStartConnectionFailure(unittest.TestCase):

    def test_start_with_unreachable_redis_sets_no_client(self):
        """start() must not raise even if Redis is unreachable."""
        import redis as _redis
        with patch("decision_engine.daily_loss_monitor.redis.Redis") as mock_cls:
            instance = MagicMock()
            instance.ping.side_effect = _redis.RedisError("refused")
            mock_cls.return_value = instance

            monitor = DailyLossMonitor(
                host="bad-host", port=6379, db=0,
            )
            monitor.start()  # must not raise

            self.assertIsNone(monitor._client)
            self.assertFalse(monitor.is_halted())


# ---------------------------------------------------------------------------
# stop() — graceful shutdown
# ---------------------------------------------------------------------------

class TestStop(unittest.TestCase):

    def test_stop_without_start_does_not_raise(self):
        monitor = _make_monitor()
        monitor.stop()  # must not raise

    def test_stop_joins_background_thread(self):
        """After stop(), the background thread must no longer be alive."""
        with patch("decision_engine.daily_loss_monitor.redis.Redis") as mock_cls:
            instance = MagicMock()
            instance.ping.return_value = True
            instance.get.return_value = None
            mock_cls.return_value = instance

            monitor = DailyLossMonitor(
                host="localhost", port=6379, db=0,
                refresh_interval=60,
            )
            monitor.start()
            self.assertIsNotNone(monitor._thread)
            self.assertTrue(monitor._thread.is_alive())

            monitor.stop()
            self.assertFalse(monitor._thread.is_alive())


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety(unittest.TestCase):

    def test_concurrent_reads_during_refresh(self):
        """
        Multiple reader threads calling is_halted() while another thread
        calls _refresh() must never observe an inconsistent state.
        """
        errors = []

        # Alternate between loss and gain on each Redis GET cycle
        call_count = [0]
        equities = [
            (1000.0, 910.0),  # -9% → halted
            (1000.0, 980.0),  # -2% → not halted
        ]

        mock = MagicMock()

        def get_side_effect(key):
            idx = call_count[0] % len(equities)
            opening, current = equities[idx]
            if key == DAILY_EQUITY_OPEN_KEY:
                call_count[0] += 1
                return json.dumps({
                    "equity": str(opening),
                    "date": "2026-03-06",
                    "updated_at": "2026-03-06T13:30:00+00:00",
                })
            if key == BUYING_POWER_KEY:
                return json.dumps({
                    "buying_power": "500.00",
                    "cash": "500.00",
                    "total_equity": str(current),
                    "updated_at": "2026-03-06T14:00:00+00:00",
                })
            return None

        mock.get.side_effect = get_side_effect
        monitor = _make_connected_monitor(mock)

        def refresher():
            for _ in range(100):
                monitor._refresh()

        def reader():
            for _ in range(500):
                halted = monitor.is_halted()
                pnl = monitor.get_daily_pnl_pct()
                if not isinstance(halted, bool):
                    errors.append(f"is_halted returned non-bool: {halted!r}")
                if not isinstance(pnl, float):
                    errors.append(f"get_daily_pnl_pct returned non-float: {pnl!r}")

        threads = (
            [threading.Thread(target=refresher)]
            + [threading.Thread(target=reader) for _ in range(4)]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [], f"Thread safety violations: {errors}")


if __name__ == "__main__":
    unittest.main()
