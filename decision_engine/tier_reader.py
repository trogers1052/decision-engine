"""
Tier reader — reads backtest tier data from Redis for confidence/position multipliers.

stock-service writes tier data to `stock:{SYMBOL}:tier` keys in Redis db=0.
This module reads those keys lazily (on first access per symbol) and caches
them in-memory with a configurable expiry.

Tier multipliers are applied to BUY signal confidence:
  S → 1.15  (elite — boosted confidence)
  A → 1.05  (strong — slight boost)
  B → 1.00  (good — neutral)
  C → 0.85  (average — dampened)
  D → 0.65  (weak — strong dampening)
  F → 0.00  (failed — suppressed entirely)

Graceful degradation: if Redis is unavailable, returns defaults
(1.0 confidence multiplier, 0.5 position size multiplier, not blacklisted).
"""

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, List

import redis

logger = logging.getLogger(__name__)

# Default multipliers when tier data is unavailable
DEFAULT_CONFIDENCE_MULTIPLIER = 1.0
DEFAULT_POSITION_SIZE_MULTIPLIER = 0.5  # conservative for unranked
DEFAULT_TIER_LABEL = ""


@dataclass
class TierData:
    """Cached tier data for a symbol."""
    symbol: str = ""
    tier: str = ""
    composite_score: float = 0.0
    confidence_multiplier: float = DEFAULT_CONFIDENCE_MULTIPLIER
    position_size_multiplier: float = DEFAULT_POSITION_SIZE_MULTIPLIER
    blacklisted: bool = False
    allowed_regimes: Optional[List[str]] = None  # None = unrestricted
    fetched_at: float = 0.0  # time.time() when fetched


class TierReader:
    """
    Reads tier ranking data from Redis for the decision engine.

    Usage::

        reader = TierReader(host="redis", port=6379, db=0)
        reader.start()
        mult = reader.get_confidence_multiplier("APH")  # 1.15 for S-tier
        reader.stop()
    """

    def __init__(
        self,
        host: str,
        port: int,
        db: int,
        password: str = "",
        cache_ttl_seconds: int = 21600,  # 6 hours
    ):
        self._host = host
        self._port = port
        self._db = db
        self._password = password
        self._cache_ttl = cache_ttl_seconds

        self._client: Optional[redis.Redis] = None
        self._cache: Dict[str, TierData] = {}
        self._lock = threading.RLock()

    def start(self) -> None:
        """Connect to Redis."""
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
                f"TierReader connected to Redis {self._host}:{self._port}/db={self._db}"
            )
        except redis.RedisError as exc:
            logger.warning(
                f"TierReader could not connect to Redis: {exc}. "
                "Tier data will use defaults until connectivity is restored."
            )
            self._client = None

    def stop(self) -> None:
        """Close the Redis connection."""
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass
        logger.info("TierReader stopped")

    def _fetch_tier(self, symbol: str) -> Optional[TierData]:
        """Fetch tier data from Redis for a symbol."""
        if not self._client:
            return None
        try:
            key = f"stock:{symbol}:tier"
            raw = self._client.get(key)
            if not raw:
                return None
            data = json.loads(raw)
            return TierData(
                symbol=data.get("symbol", symbol),
                tier=data.get("tier", ""),
                composite_score=float(data.get("composite_score", 0.0)),
                confidence_multiplier=float(data.get("confidence_multiplier", DEFAULT_CONFIDENCE_MULTIPLIER)),
                position_size_multiplier=float(data.get("position_size_multiplier", DEFAULT_POSITION_SIZE_MULTIPLIER)),
                blacklisted=bool(data.get("blacklisted", False)),
                allowed_regimes=data.get("allowed_regimes"),
                fetched_at=time.time(),
            )
        except (redis.RedisError, json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.warning(f"Failed to fetch tier data for {symbol}: {exc}")
            return None

    def _get_cached(self, symbol: str) -> Optional[TierData]:
        """Get tier data from cache, fetching from Redis if stale or missing."""
        with self._lock:
            cached = self._cache.get(symbol)
            if cached and (time.time() - cached.fetched_at) < self._cache_ttl:
                return cached

        # Fetch outside lock to avoid blocking other threads during I/O
        tier_data = self._fetch_tier(symbol)

        if tier_data:
            with self._lock:
                self._cache[symbol] = tier_data
            return tier_data

        return None

    def get_tier(self, symbol: str) -> Optional[TierData]:
        """Get full tier data for a symbol. Returns None if not ranked."""
        return self._get_cached(symbol)

    def get_confidence_multiplier(self, symbol: str) -> float:
        """Get tier confidence multiplier. Returns 1.0 if unranked."""
        td = self._get_cached(symbol)
        if td:
            return td.confidence_multiplier
        return DEFAULT_CONFIDENCE_MULTIPLIER

    def get_position_size_multiplier(self, symbol: str) -> float:
        """Get tier position size multiplier. Returns 0.5 if unranked."""
        td = self._get_cached(symbol)
        if td:
            return td.position_size_multiplier
        return DEFAULT_POSITION_SIZE_MULTIPLIER

    def get_tier_label(self, symbol: str) -> str:
        """Get tier letter (S/A/B/C/D/F). Returns '' if unranked."""
        td = self._get_cached(symbol)
        if td:
            return td.tier
        return DEFAULT_TIER_LABEL

    def is_blacklisted(self, symbol: str) -> bool:
        """Check if a symbol is blacklisted (F-tier, zero trades). Returns False if unranked."""
        td = self._get_cached(symbol)
        if td:
            return td.blacklisted
        return False

    def get_allowed_regimes(self, symbol: str) -> Optional[List[str]]:
        """
        Get allowed regimes for a symbol.

        Returns:
            None if unrestricted (passes any regime) or unranked.
            List of regime strings (e.g. ["BULL"]) if regime-conditional.
        """
        td = self._get_cached(symbol)
        if td:
            return td.allowed_regimes
        return None
