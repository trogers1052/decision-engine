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

Staleness gate (added per portfolio risk audit):
  If context-service stops publishing, the last-known regime would persist
  silently forever.  We now track `updated_at` from the context payload and
  flag the context as stale when it exceeds STALE_THRESHOLD_SECONDS during
  market hours.  Decision-engine should suppress BUY signals when stale.
"""

import json
import logging
import threading
import time
from datetime import datetime, timezone
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

# Context is considered stale after 30 minutes without an update from
# context-service.  During market hours this means the service crashed or
# Kafka/Redis connectivity was lost.
STALE_THRESHOLD_SECONDS = 30 * 60  # 30 minutes


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
        self._context_updated_at: Optional[float] = None  # epoch seconds from context payload
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
        # Close Redis FIRST to interrupt any blocking I/O in _refresh(),
        # then join the thread. Otherwise the thread can block for up to
        # refresh_interval + socket_timeout before noticing the stop event.
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)
            if self._thread.is_alive():
                logger.warning("MarketContextReader thread did not exit cleanly")
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

    def is_stale(self) -> bool:
        """Return True if context data is older than STALE_THRESHOLD_SECONDS.

        Returns False (not stale) when:
          - No context has ever been received (startup grace period)
          - Context was updated recently
        """
        with self._lock:
            if self._context_updated_at is None:
                # Never received context — don't penalise during startup.
                # The regime is UNKNOWN which already gives neutral multiplier.
                return False
            age = time.time() - self._context_updated_at
            return age > STALE_THRESHOLD_SECONDS

    def get_staleness_seconds(self) -> Optional[float]:
        """Return how many seconds since the last context update, or None."""
        with self._lock:
            if self._context_updated_at is None:
                return None
            return time.time() - self._context_updated_at

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

            # Extract updated_at for staleness tracking.
            # context-service publishes this as an ISO timestamp string.
            new_updated_at: Optional[float] = None
            raw_ts = data.get("updated_at") or data.get("timestamp")
            if raw_ts:
                try:
                    dt = datetime.fromisoformat(str(raw_ts).replace("Z", "+00:00"))
                    new_updated_at = dt.timestamp()
                except (ValueError, TypeError):
                    # Fall back to "now" — we at least know the key was refreshed.
                    new_updated_at = time.time()
            else:
                # No timestamp in payload — use current time as best estimate.
                new_updated_at = time.time()

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
            self._context_updated_at = new_updated_at
