"""
State management for tracking per-symbol state.
"""

import logging
import threading
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
class PositionInfo:
    """Detailed position information for a symbol."""
    entry_price: float  # Initial entry price
    avg_cost_basis: float  # Average cost after scale-ins
    total_shares: float  # Total shares held
    total_cost: float  # Total cost basis
    scale_in_count: int = 0  # Number of scale-ins
    entry_date: Optional[datetime] = None
    last_scale_in_date: Optional[datetime] = None

    def add_shares(self, price: float, shares: float) -> None:
        """Add shares to position (scale-in)."""
        self.total_shares += shares
        self.total_cost += price * shares
        self.avg_cost_basis = self.total_cost / self.total_shares
        self.scale_in_count += 1
        self.last_scale_in_date = datetime.utcnow()

    def to_dict(self) -> Dict:
        """Convert to dict for passing to rule context."""
        return {
            "entry_price": self.entry_price,
            "avg_cost_basis": self.avg_cost_basis,
            "total_shares": self.total_shares,
            "scale_in_count": self.scale_in_count,
        }


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
    position_info: Optional[PositionInfo] = None  # Detailed position for context

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

    Thread safety: all public methods are protected by an RLock so that
    multiple Kafka consumer threads can safely update and read state
    concurrently without corrupting per-symbol indicator or signal data.
    RLock (reentrant) is used so that methods that call other methods on
    this class (e.g. get_summary → get_open_positions) don't deadlock.
    """

    def __init__(self, redis_client=None):
        """
        Initialize the state manager.

        Args:
            redis_client: Optional Redis client for state persistence.
        """
        self._states: Dict[str, SymbolState] = {}
        self._redis = redis_client
        self._lock = threading.RLock()

    def get_state(self, symbol: str) -> SymbolState:
        """Get or create state for a symbol."""
        with self._lock:
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
        with self._lock:
            state = self.get_state(symbol)
            state.last_indicators = indicators
            state.last_update = timestamp

    def record_signal(self, symbol: str, signal: AggregatedSignal):
        """Record a new aggregated signal for a symbol."""
        with self._lock:
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
        with self._lock:
            return {
                symbol: state.current_signal
                for symbol, state in self._states.items()
                if state.current_signal is not None
            }

    def get_symbols_with_signal(self, signal_type: SignalType) -> List[str]:
        """Get all symbols currently showing a specific signal type."""
        with self._lock:
            return [
                symbol
                for symbol, state in self._states.items()
                if state.current_signal and state.current_signal.signal_type == signal_type
            ]

    def get_all_symbols(self) -> List[str]:
        """Get all tracked symbols."""
        with self._lock:
            return list(self._states.keys())

    def clear_stale_signals(self, max_age_seconds: int = 300):
        """Clear signals older than max_age_seconds."""
        with self._lock:
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
        with self._lock:
            buy_count = len(self.get_symbols_with_signal(SignalType.BUY))
            sell_count = len(self.get_symbols_with_signal(SignalType.SELL))
            watch_count = len(self.get_symbols_with_signal(SignalType.WATCH))
            positions_count = len(self.get_open_positions())

            return {
                "total_symbols": len(self._states),
                "buy_signals": buy_count,
                "sell_signals": sell_count,
                "watch_signals": watch_count,
                "open_positions": positions_count,
            }

    # =========================================================================
    # Position Tracking Methods
    # =========================================================================

    def open_position(
        self,
        symbol: str,
        price: float,
        shares: float,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Record opening a new position."""
        with self._lock:
            state = self.get_state(symbol)
            state.has_position = True
            state.position_side = "long"
            state.position_info = PositionInfo(
                entry_price=price,
                avg_cost_basis=price,
                total_shares=shares,
                total_cost=price * shares,
                scale_in_count=0,
                entry_date=timestamp or datetime.utcnow(),
            )
            logger.info(
                f"Opened position: {symbol} - {shares} shares @ ${price:.2f}"
            )

    def add_to_position(
        self,
        symbol: str,
        price: float,
        shares: float,
    ) -> Optional[PositionInfo]:
        """Add shares to existing position (scale-in)."""
        with self._lock:
            state = self.get_state(symbol)

            if not state.has_position or not state.position_info:
                logger.warning(f"Cannot scale into {symbol}: no existing position")
                return None

            old_avg = state.position_info.avg_cost_basis
            state.position_info.add_shares(price, shares)

            logger.info(
                f"Scale-in #{state.position_info.scale_in_count}: {symbol} - "
                f"added {shares} shares @ ${price:.2f}. "
                f"Avg cost: ${old_avg:.2f} -> ${state.position_info.avg_cost_basis:.2f}"
            )
            return state.position_info

    def close_position(self, symbol: str) -> Optional[PositionInfo]:
        """Close a position and return the final position info."""
        with self._lock:
            state = self.get_state(symbol)

            if not state.has_position:
                return None

            position_info = state.position_info
            state.has_position = False
            state.position_side = None
            state.position_info = None

            logger.info(f"Closed position: {symbol}")
            return position_info

    def get_position(self, symbol: str) -> Optional[PositionInfo]:
        """Get position info for a symbol."""
        with self._lock:
            state = self.get_state(symbol)
            return state.position_info if state.has_position else None

    def get_open_positions(self) -> Dict[str, PositionInfo]:
        """Get all open positions."""
        with self._lock:
            return {
                symbol: state.position_info
                for symbol, state in self._states.items()
                if state.has_position and state.position_info
            }

    def get_position_metadata(self, symbol: str) -> Optional[Dict]:
        """Get position metadata for rule context."""
        with self._lock:
            position = self.get_position(symbol)
            return position.to_dict() if position else None
