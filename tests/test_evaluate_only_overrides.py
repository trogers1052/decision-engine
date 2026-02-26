"""
Tests for evaluate_only_overrides config flag.

When enabled, the decision engine should only evaluate symbols listed in
symbol_overrides and silently skip context/indicator symbols like SPY, QQQ.
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


class TestEvaluateOnlyOverrides(unittest.TestCase):
    """Verify the evaluate_only_overrides whitelist gate."""

    def _build_service(self, evaluate_only: bool, overrides: dict) -> DecisionEngineService:
        settings = MagicMock(spec=Settings)
        svc = DecisionEngineService(settings)
        svc._config = {
            "evaluate_only_overrides": evaluate_only,
            "symbol_overrides": overrides,
        }
        # Stub out downstream so we can detect whether evaluation proceeds
        svc.state_manager = MagicMock()
        svc._evaluate_rules = MagicMock(return_value=None)
        return svc

    def test_context_symbol_skipped_when_flag_enabled(self):
        """SPY is not in symbol_overrides — should be skipped entirely."""
        svc = self._build_service(
            evaluate_only=True,
            overrides={"CCJ": {"rules": ["trend_continuation"]}},
        )
        svc.handle_indicator_event(_make_event("SPY"))

        svc.state_manager.update_indicators.assert_not_called()
        svc._evaluate_rules.assert_not_called()

    def test_trade_symbol_evaluated_when_flag_enabled(self):
        """CCJ is in symbol_overrides — should be evaluated normally."""
        svc = self._build_service(
            evaluate_only=True,
            overrides={"CCJ": {"rules": ["trend_continuation"]}},
        )
        svc.handle_indicator_event(_make_event("CCJ"))

        svc.state_manager.update_indicators.assert_called_once()
        svc._evaluate_rules.assert_called_once()

    def test_all_symbols_evaluated_when_flag_disabled(self):
        """With flag off, even non-override symbols get evaluated."""
        svc = self._build_service(
            evaluate_only=False,
            overrides={"CCJ": {"rules": ["trend_continuation"]}},
        )
        svc.handle_indicator_event(_make_event("SPY"))

        svc.state_manager.update_indicators.assert_called_once()
        svc._evaluate_rules.assert_called_once()

    def test_flag_defaults_to_false(self):
        """When evaluate_only_overrides is absent from config, all symbols pass."""
        settings = MagicMock(spec=Settings)
        svc = DecisionEngineService(settings)
        svc._config = {}  # No flag set
        svc.state_manager = MagicMock()
        svc._evaluate_rules = MagicMock(return_value=None)

        svc.handle_indicator_event(_make_event("SPY"))

        svc.state_manager.update_indicators.assert_called_once()


if __name__ == "__main__":
    unittest.main()
