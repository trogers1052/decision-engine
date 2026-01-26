"""
Position Tracker - Listens to trading.orders to track open positions.

This enables the average_down rule to know when we have a position
and what our entry price is.
"""

import json
import logging
import threading
from datetime import datetime
from typing import Callable, List, Optional

from kafka import KafkaConsumer
from kafka.errors import KafkaError

logger = logging.getLogger(__name__)


class PositionTracker:
    """
    Tracks positions by consuming trading.orders Kafka topic.

    Order event format expected:
    {
        "event_type": "ORDER_FILLED",
        "data": {
            "symbol": "CCJ",
            "side": "buy" | "sell",
            "quantity": 100,
            "price": 52.30,
            "timestamp": "2026-01-25T10:30:00Z"
        }
    }
    """

    def __init__(
        self,
        brokers: List[str],
        topic: str = "trading.orders",
        consumer_group: str = "decision-engine-positions",
        on_position_open: Optional[Callable] = None,
        on_position_close: Optional[Callable] = None,
        on_scale_in: Optional[Callable] = None,
    ):
        self.brokers = brokers
        self.topic = topic
        self.consumer_group = consumer_group
        self.consumer: Optional[KafkaConsumer] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Callbacks
        self.on_position_open = on_position_open
        self.on_position_close = on_position_close
        self.on_scale_in = on_scale_in

        # Track positions internally
        self._positions: dict = {}  # symbol -> {"shares": x, "avg_cost": y}

    def connect(self) -> bool:
        """Connect to Kafka."""
        try:
            self.consumer = KafkaConsumer(
                self.topic,
                bootstrap_servers=self.brokers,
                group_id=self.consumer_group,
                auto_offset_reset="latest",
                enable_auto_commit=True,
                value_deserializer=lambda m: m,  # Keep as bytes, decode in handler
                consumer_timeout_ms=1000,  # 1 second poll timeout
            )
            logger.info(f"PositionTracker connected, subscribed to {self.topic}")
            return True
        except KafkaError as e:
            logger.error(f"Failed to connect PositionTracker: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to connect PositionTracker: {e}")
            return False

    def start(self):
        """Start consuming in background thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._consume_loop, daemon=True)
        self._thread.start()
        logger.info("PositionTracker started")

    def stop(self):
        """Stop consuming."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        if self.consumer:
            self.consumer.close()
        logger.info("PositionTracker stopped")

    def _consume_loop(self):
        """Main consume loop."""
        while self._running:
            try:
                # Poll for messages (with timeout from consumer_timeout_ms)
                for message in self.consumer:
                    if not self._running:
                        break
                    self._handle_message(message.value)
            except StopIteration:
                # No messages, continue polling
                continue
            except Exception as e:
                if self._running:
                    logger.error(f"Error in position tracker loop: {e}")

    def _handle_message(self, raw_msg: bytes):
        """Handle incoming order message."""
        try:
            event = json.loads(raw_msg.decode("utf-8"))

            event_type = event.get("event_type")
            if event_type != "ORDER_FILLED":
                return

            data = event.get("data", {})
            symbol = data.get("symbol")
            side = data.get("side", "").lower()
            quantity = float(data.get("quantity", 0))
            price = float(data.get("price", 0))
            timestamp_str = data.get("timestamp")

            if not symbol or not side or quantity <= 0 or price <= 0:
                logger.warning(f"Invalid order data: {data}")
                return

            # Parse timestamp
            timestamp = None
            if timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(
                        timestamp_str.replace("Z", "+00:00")
                    )
                except Exception:
                    timestamp = datetime.utcnow()

            # Process the order
            if side == "buy":
                self._handle_buy(symbol, price, quantity, timestamp)
            elif side == "sell":
                self._handle_sell(symbol, price, quantity)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse order message: {e}")
        except Exception as e:
            logger.error(f"Error handling order message: {e}")

    def _handle_buy(
        self,
        symbol: str,
        price: float,
        quantity: float,
        timestamp: Optional[datetime],
    ):
        """Handle a buy order."""
        if symbol in self._positions:
            # Scale-in to existing position
            pos = self._positions[symbol]
            old_shares = pos["shares"]
            old_cost = pos["avg_cost"] * old_shares
            new_shares = old_shares + quantity
            new_cost = old_cost + (price * quantity)
            new_avg = new_cost / new_shares

            pos["shares"] = new_shares
            pos["avg_cost"] = new_avg
            pos["scale_in_count"] = pos.get("scale_in_count", 0) + 1

            logger.info(
                f"Scale-in: {symbol} +{quantity} @ ${price:.2f}, "
                f"new avg: ${new_avg:.2f}, total: {new_shares} shares"
            )

            if self.on_scale_in:
                self.on_scale_in(symbol, price, quantity)

        else:
            # New position
            self._positions[symbol] = {
                "shares": quantity,
                "avg_cost": price,
                "entry_price": price,
                "scale_in_count": 0,
            }

            logger.info(f"New position: {symbol} {quantity} shares @ ${price:.2f}")

            if self.on_position_open:
                self.on_position_open(symbol, price, quantity, timestamp)

    def _handle_sell(self, symbol: str, price: float, quantity: float):
        """Handle a sell order."""
        if symbol not in self._positions:
            logger.warning(f"Sell for {symbol} but no position tracked")
            return

        pos = self._positions[symbol]
        pos["shares"] -= quantity

        if pos["shares"] <= 0:
            # Position closed
            del self._positions[symbol]
            logger.info(f"Position closed: {symbol} @ ${price:.2f}")

            if self.on_position_close:
                self.on_position_close(symbol)
        else:
            logger.info(
                f"Partial sell: {symbol} -{quantity} @ ${price:.2f}, "
                f"remaining: {pos['shares']} shares"
            )

    def has_position(self, symbol: str) -> bool:
        """Check if we have a position in a symbol."""
        return symbol in self._positions

    def get_position(self, symbol: str) -> Optional[dict]:
        """Get position info for a symbol."""
        return self._positions.get(symbol)

    def get_all_positions(self) -> dict:
        """Get all open positions."""
        return self._positions.copy()
