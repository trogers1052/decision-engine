"""
Feedback accuracy reader — polls Redis for per-rule accuracy metrics
published by stock-service.

stock-service computes per-rule, per-regime accuracy from signal_feedback
and writes a JSON blob to the key `feedback:accuracy` (configurable) in
Redis db=0, refreshing every 15 minutes with a 30-min TTL.

This module reads that key on a background thread so the decision engine
can apply a Stage 4 confidence multiplier based on historical trade rates.

Multiplier mapping (computed by stock-service, stored in Redis):
  trade_rate=0%   → multiplier=0.50  (never traded — strong dampening)
  trade_rate=25%  → multiplier=0.75  (rarely traded — moderate dampening)
  trade_rate=50%  → multiplier=1.00  (neutral)
  trade_rate=75%  → multiplier=1.25  (often traded — modest boost)
  trade_rate=100% → multiplier=1.50  (always traded — strong boost)
"""

import json
import logging
import threading
from typing import Dict, List, Optional

import redis

logger = logging.getLogger(__name__)


class FeedbackAccuracyReader:
    """
    Reads per-rule accuracy data published by stock-service to Redis.

    Two data channels:
      - ``feedback:accuracy``  — trade-rate multiplier (did you trade it?)
      - ``feedback:outcome_quality`` — win-rate multiplier (did the signal hit its target?)

    Usage::

        reader = FeedbackAccuracyReader(host="redis", port=6379, db=0)
        reader.start()
        mult = reader.get_multiplier("MomentumReversal", "BULL")
        agg  = reader.get_aggregate_multiplier(["MomentumReversal", "RSIOversold"], "BULL")
        qual = reader.get_outcome_multiplier("MomentumReversal", "BULL")
        reader.stop()
    """

    def __init__(
        self,
        host: str,
        port: int,
        db: int,
        key: str = "feedback:accuracy",
        outcome_key: str = "feedback:outcome_quality",
        password: str = "",
        refresh_interval: int = 60,
    ):
        self._host = host
        self._port = port
        self._db = db
        self._key = key
        self._outcome_key = outcome_key
        self._password = password
        self._refresh_interval = refresh_interval

        self._client: Optional[redis.Redis] = None
        self._data: Dict[str, dict] = {}  # "rule_name:regime_id" -> {trade_rate, multiplier, signal_count}
        self._outcome_data: Dict[str, dict] = {}  # "rule_name:regime_id" -> {win_rate, multiplier, ...}
        self._lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Connect to Redis and start the background refresh thread."""
        try:
            self._client = redis.Redis(
                host=self._host,
                port=self._port,
                db=self._db,
                password=self._password or None,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
            )
            self._client.ping()
            logger.info(
                f"FeedbackAccuracyReader connected to Redis "
                f"{self._host}:{self._port}/db={self._db}"
            )
        except redis.RedisError as exc:
            logger.warning(
                f"FeedbackAccuracyReader could not connect to Redis: {exc}. "
                "Feedback multipliers will default to 1.0."
            )
            self._client = None
            return

        # Populate immediately before the first signal is evaluated.
        self._refresh()

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="feedback-accuracy-reader",
        )
        self._thread.start()

        with self._lock:
            entry_count = len(self._data)
        logger.info(
            f"FeedbackAccuracyReader started — refreshing every "
            f"{self._refresh_interval}s, {entry_count} rule-regime entries loaded"
        )

    def stop(self) -> None:
        """Stop the background thread and close the Redis connection."""
        self._stop_event.set()
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)
            if self._thread.is_alive():
                logger.warning("FeedbackAccuracyReader thread did not exit cleanly")
        logger.info("FeedbackAccuracyReader stopped")

    # ------------------------------------------------------------------
    # Public read API (thread-safe)
    # ------------------------------------------------------------------

    def get_multiplier(self, rule_name: str, regime_id: str) -> float:
        """
        Return the confidence multiplier for a specific rule in a regime.

        Lookup order:
          1. "{rule_name}:{regime_id}"  (exact match)
          2. "{rule_name}:ALL"          (regime-agnostic fallback)
          3. 1.0                        (no data — neutral)
        """
        with self._lock:
            # Try exact match first
            key = f"{rule_name}:{regime_id}"
            entry = self._data.get(key)
            if entry:
                return entry.get("multiplier", 1.0)

            # Fallback: rule with ALL regimes
            key_all = f"{rule_name}:ALL"
            entry_all = self._data.get(key_all)
            if entry_all:
                return entry_all.get("multiplier", 1.0)

            return 1.0

    def get_aggregate_multiplier(
        self, rule_names: List[str], regime_id: str
    ) -> float:
        """
        Return the average multiplier across multiple rules.

        If no rules have data, returns 1.0 (neutral).
        """
        if not rule_names:
            return 1.0

        total = 0.0
        count = 0
        for rule in rule_names:
            mult = self.get_multiplier(rule, regime_id)
            if mult != 1.0:
                total += mult
                count += 1
            else:
                # Check if there's actually data with multiplier=1.0 or no data
                with self._lock:
                    key = f"{rule}:{regime_id}"
                    key_all = f"{rule}:ALL"
                    if key in self._data or key_all in self._data:
                        total += mult
                        count += 1

        if count == 0:
            return 1.0

        return total / count

    def get_outcome_multiplier(self, rule_name: str, regime_id: str) -> float:
        """
        Return the outcome quality multiplier for a specific rule in a regime.

        Based on actual signal win rate (did the signal hit target or stop?).

        Lookup order:
          1. "{rule_name}:{regime_id}"  (exact match)
          2. "{rule_name}:ALL"          (regime-agnostic fallback)
          3. 1.0                        (no data — neutral)
        """
        with self._lock:
            key = f"{rule_name}:{regime_id}"
            entry = self._outcome_data.get(key)
            if entry:
                return entry.get("multiplier", 1.0)

            key_all = f"{rule_name}:ALL"
            entry_all = self._outcome_data.get(key_all)
            if entry_all:
                return entry_all.get("multiplier", 1.0)

            return 1.0

    def get_aggregate_outcome_multiplier(
        self, rule_names: List[str], regime_id: str
    ) -> float:
        """
        Return the average outcome quality multiplier across multiple rules.

        If no rules have outcome data, returns 1.0 (neutral).
        """
        if not rule_names:
            return 1.0

        total = 0.0
        count = 0
        for rule in rule_names:
            mult = self.get_outcome_multiplier(rule, regime_id)
            if mult != 1.0:
                total += mult
                count += 1
            else:
                with self._lock:
                    key = f"{rule}:{regime_id}"
                    key_all = f"{rule}:ALL"
                    if key in self._outcome_data or key_all in self._outcome_data:
                        total += mult
                        count += 1

        if count == 0:
            return 1.0

        return total / count

    def get_entry_count(self) -> int:
        """Return the number of rule-regime entries loaded."""
        with self._lock:
            return len(self._data)

    def get_outcome_entry_count(self) -> int:
        """Return the number of outcome quality entries loaded."""
        with self._lock:
            return len(self._outcome_data)

    # ------------------------------------------------------------------
    # Background thread
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop_event.wait(timeout=self._refresh_interval):
            self._refresh()

    def _refresh(self) -> None:
        if not self._client:
            return

        # Refresh trade-rate accuracy
        try:
            raw = self._client.get(self._key)
            if not raw:
                logger.debug(
                    "feedback:accuracy key absent in Redis — "
                    "stock-service may not have published yet"
                )
            else:
                data = json.loads(raw)
                with self._lock:
                    old_count = len(self._data)
                    self._data = data
                    new_count = len(self._data)
                    if new_count != old_count:
                        logger.info(
                            f"Feedback accuracy cache updated: "
                            f"{old_count} → {new_count} rule-regime entries"
                        )
        except (redis.RedisError, json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.warning(f"Failed to refresh feedback accuracy from Redis: {exc}")

        # Refresh outcome quality
        try:
            raw_outcome = self._client.get(self._outcome_key)
            if raw_outcome:
                outcome_data = json.loads(raw_outcome)
                with self._lock:
                    old_count = len(self._outcome_data)
                    self._outcome_data = outcome_data
                    new_count = len(self._outcome_data)
                    if new_count != old_count:
                        logger.info(
                            f"Outcome quality cache updated: "
                            f"{old_count} → {new_count} rule-regime entries"
                        )
        except (redis.RedisError, json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.warning(f"Failed to refresh outcome quality from Redis: {exc}")
