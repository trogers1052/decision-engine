"""
Redis rules cache for sharing rules across services.

Publishes the rules configuration to Redis so that:
- risk-engine can calculate stop losses using exit_strategy
- reporting-service can evaluate historical trades against rules
- alert-service can access rule parameters
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import redis

logger = logging.getLogger(__name__)

# Redis keys
RULES_CONFIG_KEY = "trading:rules:config"
RULES_UPDATED_KEY = "trading:rules:updated_at"
SYMBOL_RULES_PREFIX = "trading:rules:symbol:"
EXIT_STRATEGY_KEY = "trading:rules:exit_strategy"


class RulesCache:
    """
    Caches rules configuration in Redis for cross-service access.

    Used by decision-engine to publish rules on startup/reload.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 1,
        password: str = "",
    ):
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self._redis: Optional[redis.Redis] = None

    def connect(self) -> bool:
        """Connect to Redis."""
        try:
            self._redis = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password if self.password else None,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
            )
            self._redis.ping()
            logger.info(f"Rules cache connected to Redis at {self.host}:{self.port}")
            return True
        except redis.RedisError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            return False

    def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            try:
                self._redis.close()
            except Exception:
                pass
            self._redis = None

    def publish_rules(self, config: Dict[str, Any]) -> bool:
        """
        Publish full rules configuration to Redis.

        Args:
            config: The loaded rules.yaml configuration

        Returns:
            True if successful
        """
        if not self._redis:
            logger.error("Redis not connected")
            return False

        try:
            # Store full config
            self._redis.set(RULES_CONFIG_KEY, json.dumps(config))
            self._redis.set(RULES_UPDATED_KEY, datetime.utcnow().isoformat())

            # Store default exit strategy
            exit_strategy = config.get("exit_strategy", {
                "profit_target": 0.07,
                "stop_loss": 0.05,
            })
            self._redis.set(EXIT_STRATEGY_KEY, json.dumps(exit_strategy))

            # Store ticker-specific configs for quick lookup
            active_tickers = config.get("active_tickers", {})
            for symbol, override in active_tickers.items():
                key = f"{SYMBOL_RULES_PREFIX}{symbol}"
                self._redis.set(key, json.dumps(override))

            logger.info(
                f"Published rules config with {len(active_tickers)} active tickers"
            )
            return True

        except redis.RedisError as e:
            logger.error(f"Failed to publish rules: {e}")
            return False

    def get_symbol_exit_strategy(self, symbol: str) -> Dict[str, float]:
        """
        Get exit strategy for a specific symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Dict with profit_target and stop_loss percentages
        """
        if not self._redis:
            return {"profit_target": 0.07, "stop_loss": 0.05}

        try:
            # Check for symbol-specific override
            symbol_key = f"{SYMBOL_RULES_PREFIX}{symbol}"
            symbol_data = self._redis.get(symbol_key)

            if symbol_data:
                override = json.loads(symbol_data)
                if "exit_strategy" in override:
                    return override["exit_strategy"]

            # Fall back to default
            default_data = self._redis.get(EXIT_STRATEGY_KEY)
            if default_data:
                return json.loads(default_data)

            return {"profit_target": 0.07, "stop_loss": 0.05}

        except (redis.RedisError, json.JSONDecodeError) as e:
            logger.error(f"Error getting exit strategy: {e}")
            return {"profit_target": 0.07, "stop_loss": 0.05}


class RulesClient:
    """
    Client for reading rules from Redis cache.

    Used by risk-engine and reporting-service.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 1,
        password: str = "",
    ):
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self._redis: Optional[redis.Redis] = None
        self._config_cache: Optional[Dict] = None
        self._cache_time: Optional[datetime] = None
        self._cache_ttl_seconds = 60  # Refresh from Redis every 60 seconds

    def connect(self) -> bool:
        """Connect to Redis."""
        try:
            self._redis = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password if self.password else None,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
            )
            self._redis.ping()
            logger.info(f"Rules client connected to Redis at {self.host}:{self.port}")
            return True
        except redis.RedisError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            return False

    def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            try:
                self._redis.close()
            except Exception:
                pass
            self._redis = None

    def get_config(self, force_refresh: bool = False) -> Optional[Dict]:
        """
        Get the full rules configuration.

        Args:
            force_refresh: Force reload from Redis

        Returns:
            Full rules config dict or None
        """
        if not self._redis:
            return self._config_cache

        # Check cache
        now = datetime.utcnow()
        if (
            not force_refresh
            and self._config_cache
            and self._cache_time
            and (now - self._cache_time).total_seconds() < self._cache_ttl_seconds
        ):
            return self._config_cache

        try:
            data = self._redis.get(RULES_CONFIG_KEY)
            if data:
                self._config_cache = json.loads(data)
                self._cache_time = now
                return self._config_cache
        except (redis.RedisError, json.JSONDecodeError) as e:
            logger.error(f"Error loading rules config: {e}")

        return self._config_cache

    def get_exit_strategy(self, symbol: str) -> Dict[str, float]:
        """
        Get exit strategy for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Dict with profit_target and stop_loss
        """
        if not self._redis:
            return {"profit_target": 0.07, "stop_loss": 0.05}

        try:
            # Check symbol-specific first
            symbol_data = self._redis.get(f"{SYMBOL_RULES_PREFIX}{symbol}")
            if symbol_data:
                override = json.loads(symbol_data)
                if "exit_strategy" in override:
                    return override["exit_strategy"]

            # Default
            default_data = self._redis.get(EXIT_STRATEGY_KEY)
            if default_data:
                return json.loads(default_data)

        except (redis.RedisError, json.JSONDecodeError) as e:
            logger.error(f"Error getting exit strategy: {e}")

        return {"profit_target": 0.07, "stop_loss": 0.05}

    def get_symbol_config(self, symbol: str) -> Optional[Dict]:
        """
        Get symbol-specific configuration.

        Args:
            symbol: Stock symbol

        Returns:
            Symbol config or None
        """
        if not self._redis:
            return None

        try:
            data = self._redis.get(f"{SYMBOL_RULES_PREFIX}{symbol}")
            if data:
                return json.loads(data)
        except (redis.RedisError, json.JSONDecodeError) as e:
            logger.error(f"Error getting symbol config: {e}")

        return None

    def get_enabled_rules(self) -> List[str]:
        """Get list of enabled rule names."""
        config = self.get_config()
        if not config:
            return []

        rules = config.get("rules", {})
        return [
            name for name, settings in rules.items()
            if settings.get("enabled", False)
        ]

    def get_rule_settings(self, rule_name: str) -> Optional[Dict]:
        """Get settings for a specific rule."""
        config = self.get_config()
        if not config:
            return None

        return config.get("rules", {}).get(rule_name)

    def get_last_updated(self) -> Optional[datetime]:
        """Get when rules were last updated."""
        if not self._redis:
            return None

        try:
            data = self._redis.get(RULES_UPDATED_KEY)
            if data:
                return datetime.fromisoformat(data)
        except (redis.RedisError, ValueError):
            pass

        return None
