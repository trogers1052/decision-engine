"""
Market context reader — polls Redis for the regime published by context-service.

context-service writes a JSON blob to the key `market:context` (configurable)
in Redis db=0, refreshing every ≤5 minutes.  This module reads that key on a
background thread so the decision engine always has a fresh regime without
blocking the Kafka consumer loop.

Regime multipliers are applied to BUY signal confidence only:
  BULL     → 1.0  (no change — full confidence in a trending market)
  SIDEWAYS → 0.7  (reduce confidence — chop kills momentum strategies)
  BEAR     → 0.3  (strong reduction — most BUY signals are false positives)
  UNKNOWN  → 1.0  (no context yet — don't penalise at startup)
"""

import json
import logging
import threading
import time
from typing import Optional

import redis

logger = logging.getLogger(__name__)

# Confidence multipliers applied to BUY signals based on market regime.
REGIME_MULTIPLIERS: dict[str, float] = {
    "BULL": 1.0,
    "SIDEWAYS": 0.7,
    "BEAR": 0.3,
    "UNKNOWN": 1.0,
}


class MarketContextReader:
    """
    Reads market regime context published by context-service to Redis.

    Usage::

        reader = MarketContextReader(host="redis", port=6379, db=0, key="market:context")
        reader.start()                          # begins background refresh
        multiplier = reader.get_multiplier("BUY")
        regime     = reader.get_regime()
        reader.stop()                           # call on shutdown
    """

    def __init__(
        self,
        host: str,
        port: int,
        db: int,
        key: str,
        password: str = "",
        refresh_interval: int = 30,
    ):
        self._host = host
        self._port = port
        self._db = db
        self._key = key
        self._password = password
        self._refresh_interval = refresh_interval

        self._client: Optional[redis.Redis] = None
        self._regime: str = "UNKNOWN"
        self._regime_confidence: float = 0.0
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
                f"MarketContextReader connected to Redis {self._host}:{self._port}/db={self._db}"
            )
        except redis.RedisError as exc:
            logger.warning(
                f"MarketContextReader could not connect to Redis: {exc}. "
                "Regime will remain UNKNOWN until connectivity is restored."
            )
            self._client = None
            return

        # Populate immediately before the first Kafka message arrives.
        self._refresh()

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="market-context-reader",
        )
        self._thread.start()
        logger.info(
            f"MarketContextReader started — refreshing every {self._refresh_interval}s, "
            f"current regime: {self._regime}"
        )

    def stop(self) -> None:
        """Stop the background thread and close the Redis connection."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass
        logger.info("MarketContextReader stopped")

    # ------------------------------------------------------------------
    # Public read API (thread-safe)
    # ------------------------------------------------------------------

    def get_regime(self) -> str:
        """Return the current regime string (BULL / BEAR / SIDEWAYS / UNKNOWN)."""
        with self._lock:
            return self._regime

    def get_regime_confidence(self) -> float:
        """Return the regime confidence score (0.0–1.0)."""
        with self._lock:
            return self._regime_confidence

    def get_multiplier(self, signal_type: str) -> float:
        """
        Return the confidence multiplier for a given signal type.

        Only BUY signals are dampened in adverse regimes.  SELL and WATCH
        signals pass through unmodified (multiplier = 1.0).
        """
        if signal_type != "BUY":
            return 1.0
        with self._lock:
            return REGIME_MULTIPLIERS.get(self._regime, 1.0)

    # ------------------------------------------------------------------
    # Background thread
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop_event.wait(timeout=self._refresh_interval):
            self._refresh()

    def _refresh(self) -> None:
        if not self._client:
            return
        try:
            # Fetch from Redis WITHOUT holding the lock
            raw = self._client.get(self._key)
            if not raw:
                logger.debug(
                    "market:context key absent in Redis — "
                    "context-service may not have published yet"
                )
                return
            data = json.loads(raw)
            new_regime = str(data.get("regime", "UNKNOWN")).upper()
            new_confidence = float(data.get("regime_confidence", 0.0))

        except (redis.RedisError, json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.warning(f"Failed to refresh market context from Redis: {exc}")
            return

        # Only hold the lock for the assignment, not during I/O
        with self._lock:
            if new_regime != self._regime:
                logger.info(
                    f"Market regime: {self._regime} → {new_regime} "
                    f"(confidence={new_confidence:.2f})"
                )
            self._regime = new_regime
            self._regime_confidence = new_confidence
