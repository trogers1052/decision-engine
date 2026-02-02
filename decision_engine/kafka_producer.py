"""
Kafka producer for decision and ranking events.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional

from kafka import KafkaProducer
from kafka.errors import KafkaError

from .models.signals import AggregatedSignal

logger = logging.getLogger(__name__)


class DecisionProducer:
    """Produces decision and ranking events to Kafka."""

    def __init__(
        self,
        brokers: list[str],
        decision_topic: str,
        ranking_topic: str,
    ):
        """
        Initialize the Kafka producer.

        Args:
            brokers: List of Kafka broker addresses.
            decision_topic: Topic for individual decision events.
            ranking_topic: Topic for ranking events.
        """
        self.brokers = brokers
        self.decision_topic = decision_topic
        self.ranking_topic = ranking_topic
        self._producer: Optional[KafkaProducer] = None

    def connect(self) -> bool:
        """
        Connect to Kafka brokers.

        Returns:
            True if connection successful, False otherwise.
        """
        try:
            logger.info(f"Connecting to Kafka brokers: {self.brokers}")

            self._producer = KafkaProducer(
                bootstrap_servers=self.brokers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8") if k else None,
                acks="all",
                retries=3,
                retry_backoff_ms=1000,
            )

            logger.info("Successfully connected to Kafka producer")
            return True

        except KafkaError as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            return False

    def publish_decision(
        self,
        signal: AggregatedSignal,
        indicators_snapshot: Dict[str, float],
        risk_result: Optional[Any] = None,
    ) -> bool:
        """
        Publish a decision event to Kafka.

        Args:
            signal: Aggregated signal for a symbol.
            indicators_snapshot: Current indicator values.
            risk_result: Optional RiskCheckResult from risk engine.

        Returns:
            True if published successfully, False otherwise.
        """
        if not self._producer:
            raise RuntimeError("Kafka producer not connected")

        try:
            event = {
                "event_type": "DECISION_UPDATE",
                "source": "decision-engine",
                "schema_version": "1.1",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "data": {
                    "symbol": signal.symbol,
                    "signal": signal.signal_type.value,
                    "confidence": round(signal.aggregate_confidence, 3),
                    "primary_reasoning": signal.primary_reasoning,
                    "rules_triggered": [
                        {
                            "rule_name": s.rule_name,
                            "confidence": round(s.confidence, 3),
                            "reasoning": s.reasoning,
                        }
                        for s in signal.contributing_signals
                    ],
                    "indicators_snapshot": {
                        k: round(v, 4) if isinstance(v, float) else v
                        for k, v in indicators_snapshot.items()
                    },
                    "metadata": {
                        "rules_evaluated": signal.rules_evaluated,
                        "rules_triggered": signal.rules_triggered,
                    },
                },
            }

            # Include risk assessment if available
            if risk_result is not None:
                event["data"]["risk_assessment"] = {
                    "passes": risk_result.passes,
                    "risk_score": round(risk_result.risk_score, 4),
                    "risk_level": risk_result.risk_level.value,
                    "recommended_shares": risk_result.recommended_shares,
                    "max_shares": risk_result.max_shares,
                    "recommended_dollar_amount": round(
                        risk_result.recommended_dollar_amount, 2
                    ),
                    "reason": risk_result.reason,
                    "risk_metrics": {
                        k: round(v, 4) if isinstance(v, float) else v
                        for k, v in risk_result.risk_metrics.items()
                    },
                    "warnings": risk_result.warnings,
                }

            # Use symbol as key for partitioning
            future = self._producer.send(
                self.decision_topic,
                key=signal.symbol,
                value=event,
            )

            # Wait for send to complete
            record_metadata = future.get(timeout=10)

            logger.info(
                f"Published DECISION for {signal.symbol}: {signal.signal_type.value} "
                f"(confidence: {signal.aggregate_confidence:.2f}) -> "
                f"{record_metadata.topic}:{record_metadata.partition}"
            )
            return True

        except KafkaError as e:
            logger.error(f"Failed to publish decision for {signal.symbol}: {e}")
            return False

    def publish_ranking(self, ranking_result) -> bool:
        """
        Publish a ranking event to Kafka.

        Args:
            ranking_result: RankingResult from the ranker.

        Returns:
            True if published successfully, False otherwise.
        """
        if not self._producer:
            raise RuntimeError("Kafka producer not connected")

        try:
            event = {
                "event_type": "RANKING_UPDATE",
                "source": "decision-engine",
                "schema_version": "1.0",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "data": ranking_result.to_dict(),
            }

            # No key for rankings - goes to any partition
            future = self._producer.send(
                self.ranking_topic,
                value=event,
            )

            record_metadata = future.get(timeout=10)

            top_symbols = [r.symbol for r in ranking_result.ranked_symbols[:3]]
            logger.info(
                f"Published RANKING for {ranking_result.signal_type.value}: "
                f"Top 3: {', '.join(top_symbols)} -> "
                f"{record_metadata.topic}:{record_metadata.partition}"
            )
            return True

        except KafkaError as e:
            logger.error(f"Failed to publish ranking: {e}")
            return False

    def close(self):
        """Close the Kafka producer connection."""
        if self._producer:
            try:
                self._producer.flush()
                self._producer.close()
                logger.info("Kafka producer closed")
            except Exception as e:
                logger.error(f"Error closing Kafka producer: {e}")
