"""Prometheus metrics for decision-engine."""

import logging
import os

logger = logging.getLogger(__name__)

_DEFAULT_PORT = 9093

try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server

    _HAS_PROMETHEUS = True
except ImportError:
    _HAS_PROMETHEUS = False

    # ------------------------------------------------------------------
    # Lightweight stubs so the rest of the codebase can call .inc(),
    # .labels(), .set(), .observe() unconditionally without guarding
    # every callsite.  These are zero-cost no-ops.
    # ------------------------------------------------------------------

    class _NoOpMetric:
        """No-op metric that silently absorbs any call."""

        def inc(self, amount=1):
            pass

        def set(self, value):
            pass

        def observe(self, value):
            pass

        def labels(self, **kwargs):
            return self

    class _NoOpFactory:
        """Creates _NoOpMetric instances, accepting the same args as prometheus_client."""

        def __call__(self, *args, **kwargs):
            return _NoOpMetric()

    Counter = _NoOpFactory()   # type: ignore[assignment,misc]
    Gauge = _NoOpFactory()     # type: ignore[assignment,misc]
    Histogram = _NoOpFactory() # type: ignore[assignment,misc]

    def start_http_server(port):  # type: ignore[assignment]
        pass


# ---------------------------------------------------------------------------
# Counters
# ---------------------------------------------------------------------------

INDICATORS_CONSUMED = Counter(
    "decision_indicators_consumed_total",
    "Indicator messages consumed from Kafka",
)

SIGNALS_GENERATED = Counter(
    "decision_signals_generated_total",
    "Raw signals produced by rules (before gates)",
    ["signal_type"],
)

SIGNALS_PUBLISHED = Counter(
    "decision_signals_published_total",
    "Signals that passed all gates and were published to Kafka",
    ["signal_type"],
)

SIGNALS_REJECTED = Counter(
    "decision_signals_rejected_total",
    "Signals blocked by a gate before publication",
    ["reason"],
)

RULE_EVALUATIONS = Counter(
    "decision_rule_evaluations_total",
    "Times each rule was evaluated",
    ["rule_name"],
)

RULE_FIRES = Counter(
    "decision_rule_fires_total",
    "Times each rule produced a signal",
    ["rule_name", "signal_type"],
)

CHECKLIST_RESULTS = Counter(
    "decision_checklist_results_total",
    "Pre-trade checklist result counts",
    ["status"],
)

TRADE_PLANS_GENERATED = Counter(
    "decision_trade_plans_generated_total",
    "Trade plans successfully created",
)

RISK_GATE_BLOCKS = Counter(
    "decision_risk_gate_blocks_total",
    "Signals blocked by the risk engine gate",
)

# ---------------------------------------------------------------------------
# Histogram
# ---------------------------------------------------------------------------

EVALUATION_DURATION = Histogram(
    "decision_evaluation_duration_seconds",
    "Time to evaluate all rules for one indicator update",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)

# ---------------------------------------------------------------------------
# Gauge
# ---------------------------------------------------------------------------

SYMBOLS_TRACKED = Gauge(
    "decision_symbols_tracked",
    "Number of symbols currently in the state manager",
)


def start_metrics_server() -> None:
    """Start Prometheus metrics HTTP server on METRICS_PORT (default 9093)."""
    if not _HAS_PROMETHEUS:
        logger.warning("prometheus_client not installed — metrics endpoint disabled")
        return
    port = int(os.environ.get("METRICS_PORT", str(_DEFAULT_PORT)))
    try:
        start_http_server(port)
    except Exception:
        logger.warning("prometheus_client metrics server failed to start")
        return
    logger.info(f"Metrics server listening on :{port}/metrics")
