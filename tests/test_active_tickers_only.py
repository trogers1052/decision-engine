"""
Tests for active_tickers_only config flag.

When enabled, the decision engine should only evaluate symbols listed in
active_tickers and silently skip context/indicator symbols like SPY, QQQ.
"""

import unittest
from unittest.mock import MagicMock, patch

from decision_engine.service import DecisionEngineService
from decision_engine.config import Settings


def _make_event(symbol: str) -> dict:
    """Build a minimal valid INDICATOR_UPDATE event."""
    return {
        "event_type": "INDICATOR_UPDATE",
        "data": {
            "symbol": symbol,
            "indicators": {"RSI_14": 45.0, "close": 50.0},
            "time": "2026-02-25T15:00:00Z",
        },
    }


class TestActiveTickersOnly(unittest.TestCase):
    """Verify the active_tickers_only whitelist gate."""

    def _build_service(self, active_only: bool, tickers: dict) -> DecisionEngineService:
        settings = MagicMock(spec=Settings)
        svc = DecisionEngineService(settings)
        svc._config = {
            "active_tickers_only": active_only,
            "active_tickers": tickers,
        }
        # Stub out downstream so we can detect whether evaluation proceeds
        svc.state_manager = MagicMock()
        svc._evaluate_rules = MagicMock(return_value=None)
        return svc

    def test_context_symbol_skipped_when_flag_enabled(self):
        """SPY is not in active_tickers — should be skipped entirely."""
        svc = self._build_service(
            active_only=True,
            tickers={"CCJ": {"rules": ["trend_continuation"]}},
        )
        svc.handle_indicator_event(_make_event("SPY"))

        svc.state_manager.update_indicators.assert_not_called()
        svc._evaluate_rules.assert_not_called()

    def test_trade_symbol_evaluated_when_flag_enabled(self):
        """CCJ is in active_tickers — should be evaluated normally."""
        svc = self._build_service(
            active_only=True,
            tickers={"CCJ": {"rules": ["trend_continuation"]}},
        )
        svc.handle_indicator_event(_make_event("CCJ"))

        svc.state_manager.update_indicators.assert_called_once()
        svc._evaluate_rules.assert_called_once()

    def test_all_symbols_evaluated_when_flag_disabled(self):
        """With flag off, even non-active symbols get evaluated."""
        svc = self._build_service(
            active_only=False,
            tickers={"CCJ": {"rules": ["trend_continuation"]}},
        )
        svc.handle_indicator_event(_make_event("SPY"))

        svc.state_manager.update_indicators.assert_called_once()
        svc._evaluate_rules.assert_called_once()

    def test_flag_defaults_to_false(self):
        """When active_tickers_only is absent from config, all symbols pass."""
        settings = MagicMock(spec=Settings)
        svc = DecisionEngineService(settings)
        svc._config = {}  # No flag set
        svc.state_manager = MagicMock()
        svc._evaluate_rules = MagicMock(return_value=None)

        svc.handle_indicator_event(_make_event("SPY"))

        svc.state_manager.update_indicators.assert_called_once()


if __name__ == "__main__":
    unittest.main()
