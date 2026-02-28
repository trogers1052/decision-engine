"""
Main decision engine service.
"""

import json
import logging
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import redis

from .config import Settings
from .kafka_consumer import IndicatorConsumer
from .kafka_producer import DecisionProducer
from .checklist import ChecklistEvaluator, ChecklistResult
from .market_context import MarketContextReader
from .state_manager import StateManager
from .position_tracker import PositionTracker
from .ranker import SymbolRanker, RankingCriteria
from .rules.base import Rule, SymbolContext, SignalType
from .rules.registry import RuleRegistry
from .models.signals import Signal, AggregatedSignal, ConfidenceAggregator
from .models.trade_plan import TradePlan
from .rules_cache import RulesCache
from .trade_planner import TradePlanEngine

# Risk engine integration (optional)
try:
    from risk_engine import RiskAdapter
    HAS_RISK_ENGINE = True
except ImportError:
    HAS_RISK_ENGINE = False
    RiskAdapter = None

logger = logging.getLogger(__name__)


class DecisionEngineService:
    """
    Main decision engine service.

    Flow:
    1. Consume indicator events from Kafka (stock.indicators)
    2. Update state for the symbol
    3. Evaluate all enabled rules (or symbol-specific rules if configured)
    4. Aggregate signals with confidence
    5. Publish decision events (trading.decisions)
    6. Periodically publish rankings (trading.rankings)
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.consumer: Optional[IndicatorConsumer] = None
        self.producer: Optional[DecisionProducer] = None
        self.state_manager: StateManager = StateManager()
        self.position_tracker: Optional[PositionTracker] = None
        self.ranker: SymbolRanker = SymbolRanker(RankingCriteria.COMPOSITE)

        # Rules loaded from config (default rules)
        self.rules: List[Rule] = []
        self.rule_weights: Dict[str, float] = {}

        # Symbol-specific rules cache
        self._symbol_rules: Dict[str, List[Rule]] = {}
        self._symbol_weights: Dict[str, Dict[str, float]] = {}
        self._symbol_exit_strategies: Dict[str, dict] = {}
        self._config: dict = {}  # Full config for symbol override lookups

        # Debouncing
        self._last_publish: Dict[str, datetime] = {}
        self._last_ranking_publish: Optional[datetime] = None

        # Risk engine integration
        self.risk_adapter = None
        self._risk_enabled = getattr(settings, 'risk_engine_enabled', True)

        # Whether the position tracker connected successfully; if False we skip
        # the SELL-signal suppression check because we can't trust the state.
        self._position_tracker_connected: bool = False

        # Trade plan engine (instantiated after config is loaded in initialize())
        self.trade_plan_engine: Optional[TradePlanEngine] = None

        # Rules cache for cross-service access
        self.rules_cache: Optional[RulesCache] = None

        # Market context reader (reads regime from context-service via Redis)
        self.market_context_reader: Optional[MarketContextReader] = None

        # Pre-trade checklist evaluator
        self.checklist_evaluator: Optional[ChecklistEvaluator] = None

    def initialize(self) -> bool:
        """Initialize all connections and load rules."""
        try:
            # Load rules configuration
            self._config = self.settings.load_rules_config()
            self.rules, self.rule_weights = RuleRegistry.load_rules_from_config(self._config)

            if not self.rules:
                logger.warning("No rules loaded! Check your configuration.")

            # Log symbol overrides
            active_tickers = RuleRegistry.get_active_tickers(self._config)
            if active_tickers:
                logger.info(f"Symbol-specific rules configured for: {list(active_tickers.keys())}")
                for symbol, rule_names in active_tickers.items():
                    logger.info(f"  {symbol}: {', '.join(rule_names)}")

            # Initialize trade plan engine (reads symbol exit strategies from config)
            symbol_exit_strategies = {
                sym: data.get("exit_strategy", {})
                for sym, data in self._config.get("active_tickers", {}).items()
            }
            tpe_enabled = self._config.get("trade_plan_engine", {}).get("enabled", True)
            if tpe_enabled:
                self.trade_plan_engine = TradePlanEngine.from_config(
                    config=self._config,
                    symbol_exit_strategies=symbol_exit_strategies,
                    redis_host=self.settings.redis_host,
                    redis_port=self.settings.redis_port,
                    redis_password=self.settings.redis_password,
                    redis_db=self.settings.redis_db,
                )
                logger.info("Trade plan engine initialized")
            else:
                logger.info("Trade plan engine disabled by configuration")

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

            # Initialize position tracker (tracks open positions for context enrichment)
            self.position_tracker = PositionTracker(
                brokers=self.settings.kafka_broker_list,
                topic=getattr(self.settings, 'kafka_orders_topic', 'trading.orders'),
                consumer_group=f"{self.settings.kafka_consumer_group}-positions",
                on_position_open=self._on_position_open,
                on_position_close=self._on_position_close,
                on_scale_in=self._on_scale_in,
            )
            if self.position_tracker.connect():
                logger.info("Position tracker connected to trading.orders")
                self._position_tracker_connected = True
            else:
                logger.warning(
                    "Position tracker failed to connect - position context unavailable; "
                    "SELL signal suppression is DISABLED (cannot verify position state)"
                )

            # Load existing positions from Redis (robinhood-sync stores them there)
            self._load_positions_from_redis()

            # Initialize rules cache and publish rules to Redis
            if getattr(self.settings, 'rules_cache_enabled', True):
                self.rules_cache = RulesCache(
                    host=self.settings.redis_host,
                    port=self.settings.redis_port,
                    db=self.settings.redis_db,
                    password=self.settings.redis_password,
                )
                if self.rules_cache.connect():
                    if self.rules_cache.publish_rules(self._config):
                        logger.info("Published rules to Redis cache for cross-service access")
                    else:
                        logger.warning("Failed to publish rules to Redis cache")
                else:
                    logger.warning("Failed to connect to Redis for rules cache")
                    self.rules_cache = None

            # Initialize market context reader (reads regime published by context-service)
            self.market_context_reader = MarketContextReader(
                host=self.settings.redis_host,
                port=self.settings.redis_port,
                db=self.settings.market_context_redis_db,
                key=self.settings.market_context_redis_key,
                password=self.settings.redis_password,
            )
            self.market_context_reader.start()

            # Initialize pre-trade checklist evaluator
            self.checklist_evaluator = ChecklistEvaluator(
                redis_host=self.settings.redis_host,
                redis_port=self.settings.redis_port,
                redis_db=self.settings.redis_db,
                redis_password=self.settings.redis_password or "",
            )
            self.checklist_evaluator.connect()  # best-effort; degrades gracefully

            # Initialize risk engine if available and enabled
            if HAS_RISK_ENGINE and self._risk_enabled:
                try:
                    self.risk_adapter = RiskAdapter(
                        config_path=getattr(
                            self.settings, 'risk_config_path',
                            'config/risk_config.yaml'
                        ),
                        kafka_brokers=self.settings.kafka_broker_list,
                    )
                    if self.risk_adapter.initialize():
                        logger.info("Risk engine initialized successfully")
                    else:
                        logger.warning("Risk engine failed to initialize")
                        self.risk_adapter = None
                except Exception as e:
                    logger.warning(f"Risk engine initialization error: {e}")
                    self.risk_adapter = None
            elif not HAS_RISK_ENGINE:
                logger.warning(
                    "RISK ENGINE NOT AVAILABLE — risk_engine package not installed. "
                    "All BUY signals will bypass portfolio risk checks!"
                )
            else:
                logger.info("Risk engine disabled by configuration")

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

            # Input validation at service boundary
            if not isinstance(symbol, str) or len(symbol) > 10:
                logger.warning(f"Invalid symbol in event: {symbol!r}")
                return
            if not isinstance(indicators, dict):
                logger.warning(f"Indicators not a dict for {symbol}")
                return
            # Reject non-finite indicator values early
            for ind_key, ind_val in indicators.items():
                if isinstance(ind_val, float) and (math.isnan(ind_val) or math.isinf(ind_val)):
                    logger.warning(
                        f"Non-finite indicator {ind_key}={ind_val} for {symbol} — skipping event"
                    )
                    return

            # Skip if data quality is not ready
            if data_quality and not data_quality.get("is_ready", True):
                logger.debug(f"Skipping {symbol}: data not ready")
                return

            # Skip symbols not in active_tickers
            if self._config.get("active_tickers_only", False):
                if symbol not in self._config.get("active_tickers", {}):
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

                # Generate trade plan for BUY signals before publishing
                trade_plan: Optional[TradePlan] = None
                checklist_result: Optional[ChecklistResult] = None
                if (
                    self.trade_plan_engine is not None
                    and aggregated_signal.signal_type == SignalType.BUY
                ):
                    try:
                        trade_plan = self.trade_plan_engine.generate(
                            aggregated_signal, indicators
                        )
                        if trade_plan.rr_warning:
                            logger.warning(
                                f"Trade plan R:R warning for {symbol}: {trade_plan.rr_warning}"
                            )
                        if trade_plan.warnings:
                            for w in trade_plan.warnings:
                                logger.warning(f"Trade plan warning for {symbol}: {w}")
                    except Exception as exc:
                        # Signal will publish WITHOUT a trade plan — the alert
                        # shows the signal but not entry/stop/target details.
                        # This is better than silently dropping the signal.
                        logger.error(
                            f"Trade plan generation failed for {symbol}: {exc} — "
                            f"signal will publish without trade plan",
                            exc_info=True,
                        )

                # Evaluate pre-trade checklist for all BUY signals.
                # trade_plan may be None (plan engine disabled or threw) — the
                # checklist still runs so the earnings hard gate always fires.
                if (
                    self.checklist_evaluator is not None
                    and aggregated_signal.signal_type == SignalType.BUY
                ):
                    try:
                        checklist_result = self.checklist_evaluator.evaluate(
                            trade_plan=trade_plan,
                            regime_id=aggregated_signal.regime_id,
                            symbol=symbol,
                        )
                    except Exception as exc:
                        logger.warning(
                            f"Checklist evaluation failed for {symbol}: {exc}"
                        )

                # Enforce BLOCKED checklist: suppress publication entirely.
                # A BLOCKED status means a hard gate failed (e.g. earnings
                # imminent, no stop loss defined) — the signal MUST NOT reach
                # the alert-service or the trader may act on it.
                if (
                    checklist_result is not None
                    and checklist_result.status == "BLOCKED"
                ):
                    logger.warning(
                        f"Checklist BLOCKED for {symbol} — suppressing "
                        f"{aggregated_signal.signal_type.value} signal "
                        f"(confidence={aggregated_signal.aggregate_confidence:.2f})"
                    )
                    return

                # Suppress BUY signals with R:R below minimum threshold.
                # This prevents low-quality signals from reaching Kafka/alerts.
                if (
                    trade_plan is not None
                    and aggregated_signal.signal_type == SignalType.BUY
                    and not trade_plan.plan_valid
                ):
                    min_rr = self._config.get("trade_plan_engine", {}).get("min_rr_ratio", 2.0)
                    logger.warning(
                        f"R:R gate: suppressing BUY for {symbol} — "
                        f"R:R {trade_plan.risk_reward_ratio:.1f}:1 below "
                        f"minimum {min_rr:.1f}:1"
                    )
                    return

                # Publish if above threshold and not debounced
                if self._should_publish(symbol, aggregated_signal):
                    # Check risk for BUY signals
                    risk_result = None
                    should_publish = True

                    if (
                        self.risk_adapter
                        and aggregated_signal.signal_type == SignalType.BUY
                    ):
                        try:
                            risk_result = self.risk_adapter.check_risk(
                                symbol=symbol,
                                signal_type="BUY",
                                confidence=aggregated_signal.aggregate_confidence,
                                indicators=indicators,
                            )
                            if not risk_result.passes:
                                logger.info(
                                    f"Risk check rejected {symbol}: {risk_result.reason}"
                                )
                                should_publish = False
                        except Exception as risk_exc:
                            # Fail-open: risk engine error should not silently kill signals.
                            # Publish the signal without risk data — trader makes the call.
                            logger.error(
                                f"Risk check error for {symbol}: {risk_exc} — "
                                f"publishing signal without risk assessment",
                                exc_info=True,
                            )
                            risk_result = None

                    if should_publish:
                        self.producer.publish_decision(
                            aggregated_signal,
                            indicators,
                            risk_result=risk_result,
                            trade_plan=trade_plan,
                            checklist_result=checklist_result,
                        )
                        self._last_publish[symbol] = datetime.utcnow()

            # Check if we should publish rankings
            self._maybe_publish_rankings()

        except Exception as e:
            logger.error(f"Error handling indicator event: {e}", exc_info=True)

    def _get_rules_for_symbol(self, symbol: str) -> tuple[List[Rule], Dict[str, float]]:
        """Get the appropriate rules for a symbol (override or default)."""
        # Check cache first
        if symbol in self._symbol_rules:
            return self._symbol_rules[symbol], self._symbol_weights.get(symbol, {})

        # Check for symbol-specific override
        rules, weights, exit_strategy = RuleRegistry.load_symbol_rules(self._config, symbol)

        if rules is not None:
            # Cache the symbol-specific rules
            self._symbol_rules[symbol] = rules
            self._symbol_weights[symbol] = weights
            self._symbol_exit_strategies[symbol] = exit_strategy
            logger.info(f"Using {len(rules)} override rules for {symbol}")
            return rules, weights

        # No override - use default rules
        return self.rules, self.rule_weights

    def get_exit_strategy(self, symbol: str) -> dict:
        """Get the exit strategy for a symbol."""
        if symbol in self._symbol_exit_strategies:
            return self._symbol_exit_strategies[symbol]
        return self._config.get("exit_strategy", {"profit_target": 0.07, "stop_loss": 0.05})

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

        # Get position metadata (current position state for context enrichment)
        position_metadata = self.state_manager.get_position_metadata(symbol)

        context = SymbolContext(
            symbol=symbol,
            indicators=indicators,
            timestamp=timestamp,
            data_quality=data_quality,
            previous_signals=state.signal_history.get_recent_signals(10),
            current_position=state.position_side,
            metadata=position_metadata,
        )

        # Get rules for this symbol (may be symbol-specific or default)
        rules, _ = self._get_rules_for_symbol(symbol)

        # Evaluate each rule
        buy_signals: List[Signal] = []
        sell_signals: List[Signal] = []
        watch_signals: List[Signal] = []
        rules_evaluated = 0

        for rule in rules:
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

        # Check consensus requirement
        aggregation_config = self._config.get("aggregation", {})
        require_consensus = aggregation_config.get("require_consensus", False)
        consensus_min_rules = aggregation_config.get("consensus_min_rules", 1)

        # Determine dominant signal type.
        # Require a strict majority: ties (equal buy and sell counts) produce no
        # actionable signal — a split vote is not a go signal.
        if buy_signals and len(buy_signals) > len(sell_signals):
            # Check if we have enough rules agreeing (consensus)
            if require_consensus and len(buy_signals) < consensus_min_rules:
                logger.info(
                    f"Consensus gate: skipping BUY for {symbol} — "
                    f"{len(buy_signals)} rule(s) triggered "
                    f"({', '.join(s.rule_name for s in buy_signals)}), "
                    f"need {consensus_min_rules}"
                )
                return None
            return self._aggregate_signals(
                symbol, SignalType.BUY, buy_signals, rules_evaluated, timestamp
            )
        elif sell_signals and len(sell_signals) > len(buy_signals):
            # Check if we have enough rules agreeing (consensus)
            if require_consensus and len(sell_signals) < consensus_min_rules:
                logger.info(
                    f"Consensus gate: skipping SELL for {symbol} — "
                    f"{len(sell_signals)} rule(s) triggered "
                    f"({', '.join(s.rule_name for s in sell_signals)}), "
                    f"need {consensus_min_rules}"
                )
                return None
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

        # Apply regime multiplier to BUY signals (SELL/WATCH unaffected)
        regime_id = "UNKNOWN"
        regime_confidence = 0.0
        if self.market_context_reader:
            regime_id = self.market_context_reader.get_regime()
            regime_confidence = self.market_context_reader.get_regime_confidence()
            multiplier = self.market_context_reader.get_multiplier(signal_type.value)
            if multiplier != 1.0:
                original = aggregate_confidence
                aggregate_confidence = min(aggregate_confidence * multiplier, 1.0)
                logger.debug(
                    f"{symbol}: regime={regime_id} multiplier={multiplier:.1f} "
                    f"confidence {original:.3f} → {aggregate_confidence:.3f}"
                )

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
            regime_id=regime_id,
            regime_confidence=regime_confidence,
        )

    def _should_publish(self, symbol: str, signal: AggregatedSignal) -> bool:
        """Check if we should publish this signal."""
        # Check confidence threshold
        if signal.aggregate_confidence < self.settings.min_publish_confidence:
            return False

        # Evict stale debounce entries to prevent unbounded growth on Pi
        now = datetime.utcnow()
        if len(self._last_publish) > 100:
            cutoff = now - timedelta(minutes=30)
            self._last_publish = {
                s: t for s, t in self._last_publish.items() if t > cutoff
            }

        # Check debounce
        last = self._last_publish.get(symbol)
        if last:
            elapsed = (now - last).total_seconds()
            if elapsed < self.settings.debounce_seconds:
                return False

        # Skip SELL signals if we don't have a position in this symbol.
        # Only applies when position tracking is healthy — if the tracker
        # failed to connect we can't trust the state, so we allow the signal
        # through rather than silently dropping legitimate SELL alerts.
        if signal.signal_type == SignalType.SELL and self._position_tracker_connected:
            if not self.state_manager.get_position(symbol):
                logger.warning(
                    f"Suppressing SELL for {symbol}: no tracked position "
                    f"(position tracker connected={self._position_tracker_connected})"
                )
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

        # Periodically evict stale symbol states to bound memory
        self.state_manager.evict_stale_states()

        # Log summary
        summary = self.state_manager.get_summary()
        logger.info(
            f"State: {summary['buy_signals']} BUY, {summary['sell_signals']} SELL, "
            f"{summary['watch_signals']} WATCH across {summary['total_symbols']} symbols"
        )

    # =========================================================================
    # Position Tracking Callbacks
    # =========================================================================

    def _on_position_open(
        self,
        symbol: str,
        price: float,
        shares: float,
        timestamp: Optional[datetime],
    ):
        """Called when a new position is opened."""
        self.state_manager.open_position(symbol, price, shares, timestamp)
        logger.info(f"Position tracking: opened {symbol} @ ${price:.2f}")

    def _on_position_close(self, symbol: str):
        """Called when a position is closed."""
        self.state_manager.close_position(symbol)
        logger.info(f"Position tracking: closed {symbol}")

    def _on_scale_in(self, symbol: str, price: float, shares: float):
        """Called when scaling into a position."""
        self.state_manager.add_to_position(symbol, price, shares)
        logger.info(f"Position tracking: scale-in {symbol} +{shares} @ ${price:.2f}")

    def _load_positions_from_redis(self):
        """Load existing positions from Redis on startup.

        robinhood-sync stores positions at 'robinhood:positions' as a hash
        with symbol -> JSON position data. This ensures we know about existing
        positions even after a service restart.
        """
        client = None
        try:
            client = redis.Redis(
                host=self.settings.redis_host,
                port=self.settings.redis_port,
                password=self.settings.redis_password or None,
                db=self.settings.redis_db,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
            )

            positions_data = client.hgetall("robinhood:positions")
            if not positions_data:
                logger.info("No existing positions found in Redis")
                return

            loaded_count = 0
            for symbol, pos_json in positions_data.items():
                try:
                    pos = json.loads(pos_json)
                    quantity = float(pos.get("quantity", 0))
                    avg_price = float(pos.get("average_buy_price", 0))

                    if quantity > 0:
                        self.state_manager.open_position(symbol, avg_price, quantity)
                        loaded_count += 1
                        logger.debug(f"Loaded position: {symbol} - {quantity} shares @ ${avg_price:.2f}")
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    logger.warning(f"Failed to parse position for {symbol}: {e}")

            if loaded_count > 0:
                logger.info(f"Loaded {loaded_count} existing positions from Redis")

        except redis.RedisError as e:
            logger.warning(f"Failed to load positions from Redis: {e}")
        finally:
            if client:
                try:
                    client.close()
                except Exception:
                    pass

    # =========================================================================
    # Service Lifecycle
    # =========================================================================

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
        logger.info(f"Position tracking: enabled (listening to trading.orders)")

        # Log symbol-specific configurations
        active_tickers = RuleRegistry.get_active_tickers(self._config)
        if active_tickers:
            logger.info("-" * 40)
            logger.info("Symbol-Specific Rule Configurations:")
            for symbol, rule_names in active_tickers.items():
                exit_strat = self._config.get("active_tickers", {}).get(symbol, {}).get("exit_strategy", {})
                pt = exit_strat.get("profit_target", 0.07) * 100
                sl = exit_strat.get("stop_loss", 0.05) * 100
                logger.info(f"  {symbol}: {len(rule_names)} rules, PT={pt:.0f}%, SL={sl:.0f}%")

        logger.info("=" * 60)

        # Start position tracker first
        if self.position_tracker:
            self.position_tracker.start()

        self.consumer.start()

    def shutdown(self):
        """Graceful shutdown."""
        logger.info("Shutting down decision engine...")

        if self.position_tracker:
            self.position_tracker.stop()
        if self.consumer:
            self.consumer.close()
        if self.producer:
            self.producer.close()
        if self.risk_adapter:
            self.risk_adapter.shutdown()
        if self.rules_cache:
            self.rules_cache.close()
        if self.market_context_reader:
            self.market_context_reader.stop()
        if self.checklist_evaluator:
            self.checklist_evaluator.close()

        logger.info("Decision engine stopped")
