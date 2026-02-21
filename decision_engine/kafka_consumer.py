"""
Kafka consumer for indicator events.
"""

import json
import logging
from typing import Callable, Optional

from kafka import KafkaConsumer
from kafka.errors import KafkaError

logger = logging.getLogger(__name__)


class IndicatorConsumer:
    """Consumes indicator events from Kafka."""

    def __init__(
        self,
        brokers: list[str],
        topic: str,
        consumer_group: str,
        message_handler: Callable[[dict], None],
    ):
        """
        Initialize the Kafka consumer.

        Args:
            brokers: List of Kafka broker addresses.
            topic: Topic to consume from (stock.indicators).
            consumer_group: Consumer group ID.
            message_handler: Function to call for each message.
        """
        self.brokers = brokers
        self.topic = topic
        self.consumer_group = consumer_group
        self.message_handler = message_handler
        self._consumer: Optional[KafkaConsumer] = None

    def connect(self) -> bool:
        """
        Connect to Kafka brokers.

        Returns:
            True if connection successful, False otherwise.
        """
        try:
            logger.info(f"Connecting to Kafka brokers: {self.brokers}")
            logger.info(f"Topic: {self.topic}, Consumer Group: {self.consumer_group}")

            self._consumer = KafkaConsumer(
                self.topic,
                bootstrap_servers=self.brokers,
                group_id=self.consumer_group,
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                # Use 'earliest' to prevent data loss on restart
                auto_offset_reset="earliest",
                enable_auto_commit=False,
                request_timeout_ms=30000,
                session_timeout_ms=30000,
                heartbeat_interval_ms=10000,
                max_poll_interval_ms=300000,
            )

            logger.info("Successfully connected to Kafka consumer")
            return True

        except KafkaError as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            return False

    def start(self):
        """Start consuming messages."""
        if not self._consumer:
            raise RuntimeError("Consumer not connected. Call connect() first.")

        logger.info(f"Starting to consume messages from topic: {self.topic}")

        try:
            for message in self._consumer:
                try:
                    event = message.value
                    self.message_handler(event)
                    self._consumer.commit()
                except Exception as e:
                    logger.error(f"Error processing message: {e}", exc_info=True)
                    # Continue processing other messages

        except KeyboardInterrupt:
            logger.info("Consumer interrupted by user")
        except Exception as e:
            logger.error(f"Consumer error: {e}", exc_info=True)
            raise

    def close(self):
        """Close the Kafka consumer."""
        if self._consumer:
            try:
                self._consumer.close()
                logger.info("Kafka consumer closed")
            except Exception as e:
                logger.error(f"Error closing Kafka consumer: {e}")
