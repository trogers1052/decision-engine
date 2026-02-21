"""
Configuration management for Decision Engine.
"""

import yaml
from pathlib import Path
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Kafka configuration
    kafka_brokers: str = Field(
        "localhost:19092",
        description="Kafka broker addresses (comma-separated)"
    )
    kafka_consumer_group: str = Field(
        "decision-engine",
        description="Kafka consumer group"
    )
    kafka_input_topic: str = Field(
        "stock.indicators",
        description="Input topic for indicator events"
    )
    kafka_decision_topic: str = Field(
        "trading.decisions",
        description="Output topic for decision events"
    )
    kafka_ranking_topic: str = Field(
        "trading.rankings",
        description="Output topic for ranking events"
    )

    # Redis configuration (for state persistence)
    redis_host: str = Field("localhost", description="Redis host")
    redis_port: int = Field(6379, description="Redis port")
    redis_password: str = Field("", description="Redis password")
    redis_db: int = Field(1, description="Redis database number")

    # Market context (published by context-service, read from Redis db=0)
    market_context_redis_db: int = Field(
        0, description="Redis DB where context-service publishes market:context"
    )
    market_context_redis_key: str = Field(
        "market:context", description="Redis key for market regime context"
    )

    # Database (optional, for decision logging)
    db_host: str = Field("localhost", description="PostgreSQL host")
    db_port: int = Field(5432, description="PostgreSQL port")
    db_user: str = Field("trader", description="PostgreSQL user")
    db_password: str = Field(..., description="PostgreSQL password (required)")
    db_name: str = Field("trading_platform", description="Database name")

    # Rule configuration
    rules_config_path: str = Field(
        "config/rules.yaml",
        description="Path to rules configuration file"
    )

    # Processing settings
    min_publish_confidence: float = Field(
        0.5,
        description="Minimum confidence threshold to publish decisions"
    )
    ranking_interval_seconds: int = Field(
        60,
        description="How often to publish rankings (seconds)"
    )
    debounce_seconds: int = Field(
        5,
        description="Minimum time between publishes for same symbol"
    )

    # Rules cache settings
    rules_cache_enabled: bool = Field(
        True,
        description="Enable publishing rules to Redis for cross-service access"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @property
    def kafka_broker_list(self) -> List[str]:
        """Convert comma-separated brokers string to list."""
        return [b.strip() for b in self.kafka_brokers.split(",")]

    @property
    def database_url(self) -> str:
        """Get PostgreSQL connection URL."""
        return (
            f"postgresql://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    def load_rules_config(self) -> dict:
        """Load rules configuration from YAML file."""
        config_path = Path(self.rules_config_path)

        if not config_path.exists():
            # Return default config if file doesn't exist
            return self._default_rules_config()

        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _default_rules_config(self) -> dict:
        """Default rules configuration."""
        return {
            "rules": {
                "buy_dip_in_uptrend": {"enabled": True, "weight": 1.5},
                "strong_buy_signal": {"enabled": True, "weight": 2.0},
                "weekly_uptrend": {"enabled": True, "weight": 1.0},
                "rsi_oversold": {"enabled": True, "threshold": 30.0, "weight": 1.0},
                "rsi_overbought": {"enabled": True, "threshold": 70.0, "weight": 1.0},
                "macd_bullish_crossover": {"enabled": True, "weight": 1.0},
                "macd_bearish_crossover": {"enabled": True, "weight": 1.0},
                "trend_alignment": {"enabled": True, "weight": 1.2},
            },
            "aggregation": {
                "method": "consensus_boost",
                "min_confidence": 0.5,
            },
            "ranking": {
                "criteria": "composite",
                "publish_interval_seconds": 60,
                "min_symbols_for_ranking": 2,
            },
        }
