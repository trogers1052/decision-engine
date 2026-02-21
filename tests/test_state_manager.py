"""
Tests for StateManager thread safety.

Verifies that concurrent reads and writes from multiple threads
cannot corrupt per-symbol state.
"""

import threading
import unittest
from datetime import datetime
from unittest.mock import MagicMock

from decision_engine.state_manager import StateManager, PositionInfo


class TestStateManagerThreadSafety(unittest.TestCase):

    def test_concurrent_get_state_creates_symbol_once(self):
        """Multiple threads racing to create the same symbol create exactly one entry."""
        sm = StateManager()
        results = []
        errors = []

        def create_state():
            try:
                state = sm.get_state("AAPL")
                results.append(id(state))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=create_state) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Errors during concurrent access: {errors}")
        self.assertEqual(len(results), 50)
        # All threads must have received the exact same object
        self.assertEqual(len(set(results)), 1, "get_state returned different objects for same symbol")

    def test_concurrent_update_indicators_no_corruption(self):
        """
        Multiple threads updating different symbols simultaneously must not
        corrupt each other's indicator data.
        """
        sm = StateManager()
        symbols = ["AAPL", "WPM", "CCJ", "SLV", "URNM"]
        iterations = 100
        errors = []

        def update_symbol(sym, value):
            try:
                for _ in range(iterations):
                    sm.update_indicators(
                        sym,
                        {"RSI_14": value, "close": value * 10},
                        datetime.utcnow(),
                    )
            except Exception as e:
                errors.append((sym, e))

        threads = [
            threading.Thread(target=update_symbol, args=(sym, i + 1.0))
            for i, sym in enumerate(symbols)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Errors during concurrent updates: {errors}")

        # Each symbol must end with its own value, not another symbol's value
        for i, sym in enumerate(symbols):
            state = sm.get_state(sym)
            self.assertIn("RSI_14", state.last_indicators)
            # The value must be the one assigned to THIS symbol (not cross-contaminated)
            self.assertAlmostEqual(
                state.last_indicators["RSI_14"],
                i + 1.0,
                places=5,
                msg=f"{sym} RSI_14 was corrupted by another thread",
            )

    def test_concurrent_open_and_close_positions(self):
        """Opening and closing positions from multiple threads doesn't leave phantom state."""
        sm = StateManager()
        errors = []

        def open_close(sym):
            try:
                sm.open_position(sym, price=100.0, shares=10.0)
                sm.close_position(sym)
            except Exception as e:
                errors.append((sym, e))

        symbols = [f"SYM{i}" for i in range(30)]
        threads = [threading.Thread(target=open_close, args=(sym,)) for sym in symbols]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Errors: {errors}")
        # All positions should be closed — no phantom open positions
        open_positions = sm.get_open_positions()
        self.assertEqual(len(open_positions), 0, f"Phantom open positions: {list(open_positions.keys())}")

    def test_get_all_symbols_safe_during_concurrent_inserts(self):
        """get_all_symbols must not raise RuntimeError due to dict size change during iteration."""
        sm = StateManager()
        errors = []
        symbols_seen = []

        def insert_symbols():
            for i in range(50):
                try:
                    sm.get_state(f"NEW{i}")
                except Exception as e:
                    errors.append(e)

        def read_symbols():
            for _ in range(50):
                try:
                    syms = sm.get_all_symbols()
                    symbols_seen.append(len(syms))
                except Exception as e:
                    errors.append(e)

        t1 = threading.Thread(target=insert_symbols)
        t2 = threading.Thread(target=read_symbols)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertEqual(len(errors), 0, f"Errors during concurrent insert+read: {errors}")


class TestStateManagerConsistency(unittest.TestCase):
    """Non-concurrent correctness tests for StateManager."""

    def test_get_state_creates_on_first_access(self):
        sm = StateManager()
        state = sm.get_state("AAPL")
        self.assertEqual(state.symbol, "AAPL")
        self.assertFalse(state.has_position)

    def test_get_state_returns_same_object(self):
        sm = StateManager()
        a = sm.get_state("AAPL")
        b = sm.get_state("AAPL")
        self.assertIs(a, b)

    def test_open_position_reflected_in_get_open_positions(self):
        sm = StateManager()
        sm.open_position("AAPL", price=150.0, shares=5.0)
        positions = sm.get_open_positions()
        self.assertIn("AAPL", positions)
        self.assertEqual(positions["AAPL"].entry_price, 150.0)

    def test_close_position_removes_from_open_positions(self):
        sm = StateManager()
        sm.open_position("AAPL", price=150.0, shares=5.0)
        sm.close_position("AAPL")
        self.assertNotIn("AAPL", sm.get_open_positions())

    def test_get_position_metadata_returns_none_when_no_position(self):
        sm = StateManager()
        self.assertIsNone(sm.get_position_metadata("AAPL"))


if __name__ == "__main__":
    unittest.main()
