"""Portfolio risk state reader.

Reads the ``risk:daily_state`` key from Redis, written by the
stop-loss-guardian's portfolio monitor every 60 seconds.

Provides the decision-engine with real-time portfolio-level risk data:
- Number of stops hit today
- Actual portfolio heat (from real stop distances)
- Whether the guardian has halted new entries
- Gap risk alerts

Fail-open: if Redis is unavailable or the key is missing, returns None
and the decision-engine continues without portfolio gating.  Capital
protection still flows through the risk-engine's mandatory check.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import redis

logger = logging.getLogger(__name__)

DAILY_STATE_KEY = "risk:daily_state"


@dataclass
class PortfolioRiskState:
    """Parsed snapshot from the guardian's portfolio monitor."""
    trade_date: str
    stops_hit_today: int
    stops_hit_symbols: List[str]
    daily_pnl_pct: float
    actual_portfolio_heat: float
    halted: bool
    halt_reason: Optional[str]
    open_position_count: int
    gap_alerts: List[dict]
    sector_heat: Dict[str, float] = field(default_factory=dict)
    position_risks: Dict[str, dict] = field(default_factory=dict)


class PortfolioRiskReader:
    """Reads portfolio risk state from Redis (written by stop-loss-guardian)."""

    def __init__(self, host: str, port: int, db: int, password: str = ""):
        self._host = host
        self._port = port
        self._db = db
        self._password = password
        self._client: Optional[redis.Redis] = None

    def connect(self) -> bool:
        """Connect to Redis. Returns True on success."""
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
            logger.info("PortfolioRiskReader connected to Redis")
            return True
        except redis.RedisError as e:
            logger.warning(
                f"PortfolioRiskReader: Redis connection failed: {e}. "
                f"Portfolio risk gating will be inactive (fail-open)."
            )
            self._client = None
            return False

    def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None

    def get_state(self) -> Optional[PortfolioRiskState]:
        """Read current portfolio risk state.

        Returns:
            PortfolioRiskState if available, None otherwise (fail-open).
        """
        if not self._client:
            return None

        try:
            raw = self._client.get(DAILY_STATE_KEY)
        except redis.RedisError as e:
            logger.warning(f"PortfolioRiskReader: Redis read failed: {e}")
            return None

        if not raw:
            return None

        try:
            data = json.loads(raw)
            return PortfolioRiskState(
                trade_date=data.get("date", ""),
                stops_hit_today=data.get("stops_hit_today", 0),
                stops_hit_symbols=data.get("stops_hit_symbols", []),
                daily_pnl_pct=data.get("daily_pnl_pct", 0.0),
                actual_portfolio_heat=data.get("actual_portfolio_heat", 0.0),
                halted=data.get("halted", False),
                halt_reason=data.get("halt_reason"),
                open_position_count=data.get("open_position_count", 0),
                gap_alerts=data.get("gap_alerts", []),
                sector_heat=data.get("sector_heat", {}),
                position_risks=data.get("position_risks", {}),
            )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"PortfolioRiskReader: parse error: {e}")
            return None
