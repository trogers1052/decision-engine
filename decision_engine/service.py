"""
Main decision engine service.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

from .config import Settings
from .kafka_consumer import IndicatorConsumer
from .kafka_producer import DecisionProducer
from .state_manager import StateManager
from .ranker import SymbolRanker, RankingCriteria
from .rules.base import Rule, SymbolContext, SignalType
from .rules.registry import RuleRegistry
from .models.signals import Signal, AggregatedSignal, ConfidenceAggregator

logger = logging.getLogger(__name__)


class DecisionEngineService:
    """
    Main decision engine service.

    Flow:
    1. Consume indicator events from Kafka (stock.indicators)
    2. Update state for the symbol
    3. Evaluate all enabled rules
    4. Aggregate signals with confidence
    5. Publish decision events (trading.decisions)
    6. Periodically publish rankings (trading.rankings)
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.consumer: Optional[IndicatorConsumer] = None
        self.producer: Optional[DecisionProducer] = None
        self.state_manager: StateManager = StateManager()
        self.ranker: SymbolRanker = SymbolRanker(RankingCriteria.COMPOSITE)

        # Rules loaded from config
        self.rules: List[Rule] = []
        self.rule_weights: Dict[str, float] = {}

        # Debouncing
        self._last_publish: Dict[str, datetime] = {}
        self._last_ranking_publish: Optional[datetime] = None

    def initialize(self) -> bool:
        """Initialize all connections and load rules."""
        try:
            # Load rules configuration
            config = self.settings.load_rules_config()
            self.rules, self.rule_weights = RuleRegistry.load_rules_from_config(config)

            if not self.rules:
                logger.warning("No rules loaded! Check your configuration.")

            # Initialize Kafka consumer
            self.consumer = IndicatorConsumer(
                brokers=self.settings.kafka_broker_list,
                topic=self.settings.kafka_input_topic,
                consumer_group=self.settings.kafka_consumer_group,
                message_handler=self.handle_indicator_event,
            )
            if not self.consumer.connect():
                logger.error("Failed to connect to Kafka consumer")
                return False

            # Initialize Kafka producer
            self.producer = DecisionProducer(
                brokers=self.settings.kafka_broker_list,
                decision_topic=self.settings.kafka_decision_topic,
                ranking_topic=self.settings.kafka_ranking_topic,
            )
            if not self.producer.connect():
                logger.error("Failed to connect to Kafka producer")
                return False

            logger.info(f"Decision engine initialized with {len(self.rules)} rules")
            for rule in self.rules:
                logger.info(f"  - {rule.name}: {rule.description}")

            return True

        except Exception as e:
            logger.error(f"Failed to initialize: {e}", exc_info=True)
            return False

    def handle_indicator_event(self, event: dict):
        """Handle incoming indicator event from Kafka."""
        try:
            if event.get("event_type") != "INDICATOR_UPDATE":
                logger.debug(f"Ignoring non-indicator event: {event.get('event_type')}")
                return

            data = event.get("data", {})
            symbol = data.get("symbol")
            indicators = data.get("indicators", {})
            data_quality = data.get("data_quality", {})

            if not symbol or not indicators:
                logger.debug("Missing symbol or indicators in event")
                return

            # Skip if data quality is not ready
            if data_quality and not data_quality.get("is_ready", True):
                logger.debug(f"Skipping {symbol}: data not ready")
                return

            # Parse timestamp
            time_str = data.get("time", datetime.utcnow().isoformat())
            try:
                timestamp = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
            except Exception:
                timestamp = datetime.utcnow()

            # Update state
            self.state_manager.update_indicators(symbol, indicators, timestamp)

            # Evaluate rules
            aggregated_signal = self._evaluate_rules(
                symbol, indicators, timestamp, data_quality
            )

            if aggregated_signal:
                # Record in state
                self.state_manager.record_signal(symbol, aggregated_signal)

                # Publish if above threshold and not debounced
                if self._should_publish(symbol, aggregated_signal):
                    self.producer.publish_decision(aggregated_signal, indicators)
                    self._last_publish[symbol] = datetime.utcnow()

            # Check if we should publish rankings
            self._maybe_publish_rankings()

        except Exception as e:
            logger.error(f"Error handling indicator event: {e}", exc_info=True)

    def _evaluate_rules(
        self,
        symbol: str,
        indicators: Dict[str, float],
        timestamp: datetime,
        data_quality: Optional[dict],
    ) -> Optional[AggregatedSignal]:
        """Evaluate all rules for a symbol."""

        # Build context
        state = self.state_manager.get_state(symbol)
        context = SymbolContext(
            symbol=symbol,
            indicators=indicators,
            timestamp=timestamp,
            data_quality=data_quality,
            previous_signals=state.signal_history.get_recent_signals(10),
            current_position=state.position_side,
        )

        # Evaluate each rule
        buy_signals: List[Signal] = []
        sell_signals: List[Signal] = []
        watch_signals: List[Signal] = []
        rules_evaluated = 0

        for rule in self.rules:
            if not rule.can_evaluate(context):
                logger.debug(f"Rule {rule.name} cannot evaluate (missing indicators)")
                continue

            rules_evaluated += 1
            result = rule.evaluate(context)

            if result.triggered:
                signal = Signal(
                    rule_name=rule.name,
                    rule_description=rule.description,
                    signal_type=result.signal,
                    confidence=result.confidence,
                    reasoning=result.reasoning,
                    contributing_factors=result.contributing_factors,
                    timestamp=timestamp,
                )

                if result.signal == SignalType.BUY:
                    buy_signals.append(signal)
                    logger.debug(f"  BUY signal from {rule.name}: {result.reasoning}")
                elif result.signal == SignalType.SELL:
                    sell_signals.append(signal)
                    logger.debug(f"  SELL signal from {rule.name}: {result.reasoning}")
                else:
                    watch_signals.append(signal)

        # Determine dominant signal type
        # Priority: BUY/SELL over WATCH, and if tied, the one with more signals
        if buy_signals and len(buy_signals) >= len(sell_signals):
            return self._aggregate_signals(
                symbol, SignalType.BUY, buy_signals, rules_evaluated, timestamp
            )
        elif sell_signals:
            return self._aggregate_signals(
                symbol, SignalType.SELL, sell_signals, rules_evaluated, timestamp
            )
        elif watch_signals:
            return self._aggregate_signals(
                symbol, SignalType.WATCH, watch_signals, rules_evaluated, timestamp
            )

        return None

    def _aggregate_signals(
        self,
        symbol: str,
        signal_type: SignalType,
        signals: List[Signal],
        rules_evaluated: int,
        timestamp: datetime,
    ) -> AggregatedSignal:
        """Aggregate multiple signals into one."""

        # Calculate aggregate confidence using consensus boost
        aggregate_confidence = ConfidenceAggregator.consensus_boost(signals)

        # Pick primary reasoning from highest confidence signal
        primary_signal = max(signals, key=lambda s: s.confidence)

        return AggregatedSignal(
            symbol=symbol,
            signal_type=signal_type,
            aggregate_confidence=aggregate_confidence,
            primary_reasoning=primary_signal.reasoning,
            contributing_signals=signals,
            timestamp=timestamp,
            rules_triggered=len(signals),
            rules_evaluated=rules_evaluated,
        )

    def _should_publish(self, symbol: str, signal: AggregatedSignal) -> bool:
        """Check if we should publish this signal."""
        # Check confidence threshold
        if signal.aggregate_confidence < self.settings.min_publish_confidence:
            return False

        # Check debounce
        last = self._last_publish.get(symbol)
        if last:
            elapsed = (datetime.utcnow() - last).total_seconds()
            if elapsed < self.settings.debounce_seconds:
                return False

        return True

    def _maybe_publish_rankings(self):
        """Publish rankings if interval has elapsed."""
        now = datetime.utcnow()

        if self._last_ranking_publish:
            elapsed = (now - self._last_ranking_publish).total_seconds()
            if elapsed < self.settings.ranking_interval_seconds:
                return

        # Get all current signals
        all_signals = self.state_manager.get_all_current_signals()

        if len(all_signals) < 2:
            return

        # Rank BUY signals
        buy_ranking = self.ranker.rank(all_signals, SignalType.BUY)
        if buy_ranking.ranked_symbols:
            self.producer.publish_ranking(buy_ranking)

        # Rank SELL signals (if any)
        sell_ranking = self.ranker.rank(all_signals, SignalType.SELL)
        if sell_ranking.ranked_symbols:
            self.producer.publish_ranking(sell_ranking)

        self._last_ranking_publish = now

        # Log summary
        summary = self.state_manager.get_summary()
        logger.info(
            f"State: {summary['buy_signals']} BUY, {summary['sell_signals']} SELL, "
            f"{summary['watch_signals']} WATCH across {summary['total_symbols']} symbols"
        )

    def start(self):
        """Start the service."""
        logger.info("=" * 60)
        logger.info("Starting Decision Engine")
        logger.info("=" * 60)
        logger.info(f"Consuming from: {self.settings.kafka_input_topic}")
        logger.info(f"Publishing decisions to: {self.settings.kafka_decision_topic}")
        logger.info(f"Publishing rankings to: {self.settings.kafka_ranking_topic}")
        logger.info(f"Min confidence to publish: {self.settings.min_publish_confidence}")
        logger.info(f"Ranking interval: {self.settings.ranking_interval_seconds}s")
        logger.info("=" * 60)

        self.consumer.start()

    def shutdown(self):
        """Graceful shutdown."""
        logger.info("Shutting down decision engine...")

        if self.consumer:
            self.consumer.close()
        if self.producer:
            self.producer.close()

        logger.info("Decision engine stopped")
