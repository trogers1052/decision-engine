"""
State management for tracking per-symbol state.
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from .models.signals import AggregatedSignal, Signal
from .rules.base import SignalType

logger = logging.getLogger(__name__)


@dataclass
class SignalHistory:
    """History of signals for a single symbol."""
    symbol: str
    max_history: int = 50
    signals: deque = field(default_factory=lambda: deque(maxlen=50))

    def add_signal(self, signal: Signal):
        """Add a signal to history."""
        self.signals.append(signal)

    def get_last_signal(
        self,
        signal_type: Optional[SignalType] = None
    ) -> Optional[Signal]:
        """Get the most recent signal, optionally filtered by type."""
        for signal in reversed(self.signals):
            if signal_type is None or signal.signal_type == signal_type:
                return signal
        return None

    def get_recent_signals(self, count: int = 10) -> List[Signal]:
        """Get the N most recent signals."""
        return list(self.signals)[-count:]


@dataclass
class SymbolState:
    """Complete state for a single symbol."""
    symbol: str
    last_update: Optional[datetime] = None
    last_indicators: Dict[str, float] = field(default_factory=dict)
    current_signal: Optional[AggregatedSignal] = None
    signal_history: Optional[SignalHistory] = None

    # Position tracking (if integrated with portfolio)
    has_position: bool = False
    position_side: Optional[str] = None  # 'long' or 'short'

    def __post_init__(self):
        if self.signal_history is None:
            self.signal_history = SignalHistory(symbol=self.symbol)


class StateManager:
    """
    Manages per-symbol state across the decision engine.

    Responsibilities:
    - Track current signals per symbol
    - Maintain signal history
    - Provide context for rule evaluation
    """

    def __init__(self, redis_client=None):
        """
        Initialize the state manager.

        Args:
            redis_client: Optional Redis client for state persistence.
        """
        self._states: Dict[str, SymbolState] = {}
        self._redis = redis_client

    def get_state(self, symbol: str) -> SymbolState:
        """Get or create state for a symbol."""
        if symbol not in self._states:
            self._states[symbol] = SymbolState(symbol=symbol)
        return self._states[symbol]

    def update_indicators(
        self,
        symbol: str,
        indicators: Dict[str, float],
        timestamp: datetime,
    ):
        """Update the latest indicators for a symbol."""
        state = self.get_state(symbol)
        state.last_indicators = indicators
        state.last_update = timestamp

    def record_signal(self, symbol: str, signal: AggregatedSignal):
        """Record a new aggregated signal for a symbol."""
        state = self.get_state(symbol)
        state.current_signal = signal

        # Add individual signals to history
        for rule_signal in signal.contributing_signals:
            state.signal_history.add_signal(rule_signal)

        logger.debug(
            f"Recorded signal for {symbol}: {signal.signal_type.value} "
            f"(confidence: {signal.aggregate_confidence:.2f})"
        )

    def get_all_current_signals(self) -> Dict[str, AggregatedSignal]:
        """Get current signals for all symbols (for ranking)."""
        return {
            symbol: state.current_signal
            for symbol, state in self._states.items()
            if state.current_signal is not None
        }

    def get_symbols_with_signal(self, signal_type: SignalType) -> List[str]:
        """Get all symbols currently showing a specific signal type."""
        return [
            symbol
            for symbol, state in self._states.items()
            if state.current_signal and state.current_signal.signal_type == signal_type
        ]

    def get_all_symbols(self) -> List[str]:
        """Get all tracked symbols."""
        return list(self._states.keys())

    def clear_stale_signals(self, max_age_seconds: int = 300):
        """Clear signals older than max_age_seconds."""
        now = datetime.utcnow()
        cleared = 0

        for symbol, state in self._states.items():
            if state.current_signal:
                age = (now - state.current_signal.timestamp).total_seconds()
                if age > max_age_seconds:
                    state.current_signal = None
                    cleared += 1

        if cleared > 0:
            logger.info(f"Cleared {cleared} stale signals (older than {max_age_seconds}s)")

    def get_summary(self) -> Dict[str, int]:
        """Get a summary of current state."""
        buy_count = len(self.get_symbols_with_signal(SignalType.BUY))
        sell_count = len(self.get_symbols_with_signal(SignalType.SELL))
        watch_count = len(self.get_symbols_with_signal(SignalType.WATCH))

        return {
            "total_symbols": len(self._states),
            "buy_signals": buy_count,
            "sell_signals": sell_count,
            "watch_signals": watch_count,
        }
