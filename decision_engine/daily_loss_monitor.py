"""
Daily loss circuit breaker — halts BUY signals when daily P&L exceeds threshold.

robinhood-sync writes two keys to Redis:
  - ``robinhood:buying_power``       → current total_equity (updated every ~60s)
  - ``trading:daily_equity_open``    → opening equity snapshot (first sync of each day)

This module reads both keys on a background thread, calculates the daily P&L %,
and sets an internal halt flag when the loss exceeds the configured threshold.

Fail-safe behaviour (fail-open):
  - Redis unavailable  → not halted (allow trading)
  - Missing opening equity key → not halted (no baseline to compare)
  - Missing buying power key → not halted (no current data)

Note: the risk engine itself is fail-closed (BUY signals blocked when unavailable).
The daily loss monitor is fail-open because Redis P&L data is supplementary.
"""

import json
import logging
import threading
import time
from typing import Optional

import redis

logger = logging.getLogger(__name__)

# Redis key written by robinhood-sync PositionStore
BUYING_POWER_KEY = "robinhood:buying_power"
# Redis key written by robinhood-sync store_daily_equity_open()
DAILY_EQUITY_OPEN_KEY = "trading:daily_equity_open"


class DailyLossMonitor:
    """
    Monitors daily P&L and halts BUY signals when loss exceeds threshold.

    Usage::

        monitor = DailyLossMonitor(host="redis", port=6379, db=0, threshold_pct=0.05)
        monitor.start()
        if monitor.is_halted():
            # suppress BUY signal
        monitor.stop()
    """

    def __init__(
        self,
        host: str,
        port: int,
        db: int,
        password: str = "",
        threshold_pct: float = 0.05,
        refresh_interval: int = 30,
    ):
        self._host = host
        self._port = port
        self._db = db
        self._password = password
        self._threshold_pct = threshold_pct
        self._refresh_interval = refresh_interval

        self._client: Optional[redis.Redis] = None
        self._halted: bool = False
        self._daily_pnl_pct: float = 0.0
        self._opening_equity: float = 0.0
        self._current_equity: float = 0.0
        self._halt_triggered_today: bool = False
        self._last_snapshot_date: str = ""
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
                f"DailyLossMonitor connected to Redis "
                f"{self._host}:{self._port}/db={self._db}"
            )
        except redis.RedisError as exc:
            logger.warning(
                f"DailyLossMonitor could not connect to Redis: {exc}. "
                "Daily loss circuit breaker will be inactive (fail-open)."
            )
            self._client = None
            return

        # Populate immediately.
        self._refresh()

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="daily-loss-monitor",
        )
        self._thread.start()
        logger.info(
            f"DailyLossMonitor started — threshold: {self._threshold_pct:.0%}, "
            f"refreshing every {self._refresh_interval}s"
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
                logger.warning("DailyLossMonitor thread did not exit cleanly")
        logger.info("DailyLossMonitor stopped")

    # ------------------------------------------------------------------
    # Public read API (thread-safe)
    # ------------------------------------------------------------------

    def is_halted(self) -> bool:
        """Return True if daily loss exceeds threshold and BUY signals should be suppressed."""
        with self._lock:
            return self._halted

    def get_daily_pnl_pct(self) -> float:
        """Return the current daily P&L as a fraction (e.g. -0.08 = -8%)."""
        with self._lock:
            return self._daily_pnl_pct

    def get_status(self) -> dict:
        """Return a diagnostic snapshot of the monitor state."""
        with self._lock:
            return {
                "halted": self._halted,
                "daily_pnl_pct": self._daily_pnl_pct,
                "opening_equity": self._opening_equity,
                "current_equity": self._current_equity,
                "threshold_pct": self._threshold_pct,
                "snapshot_date": self._last_snapshot_date,
            }

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
            # Read both keys WITHOUT holding the lock
            raw_open = self._client.get(DAILY_EQUITY_OPEN_KEY)
            raw_current = self._client.get(BUYING_POWER_KEY)
        except redis.RedisError as exc:
            logger.warning(f"DailyLossMonitor Redis read failed: {exc}")
            return  # fail-open: leave _halted unchanged on transient error

        # Parse opening equity
        if not raw_open:
            logger.debug(
                "DailyLossMonitor: no daily equity snapshot yet — "
                "circuit breaker inactive"
            )
            with self._lock:
                self._halted = False
            return

        try:
            open_data = json.loads(raw_open)
            opening_equity = float(open_data["equity"])
            snapshot_date = open_data.get("date", "")
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
            logger.warning(
                f"DailyLossMonitor: invalid daily equity data: {exc}"
            )
            with self._lock:
                self._halted = False
            return

        # Parse current equity
        if not raw_current:
            logger.debug(
                "DailyLossMonitor: no current buying power data — "
                "circuit breaker inactive"
            )
            with self._lock:
                self._halted = False
            return

        try:
            current_data = json.loads(raw_current)
            current_equity = float(current_data["total_equity"])
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
            logger.warning(
                f"DailyLossMonitor: invalid buying power data: {exc}"
            )
            with self._lock:
                self._halted = False
            return

        # Guard against zero/negative opening equity
        if opening_equity <= 0:
            logger.warning(
                f"DailyLossMonitor: invalid opening equity {opening_equity}"
            )
            with self._lock:
                self._halted = False
            return

        # Calculate daily P&L
        pnl_pct = (current_equity - opening_equity) / opening_equity

        # Detect new trading day (reset halt flag)
        new_day = snapshot_date != self._last_snapshot_date

        with self._lock:
            self._daily_pnl_pct = pnl_pct
            self._opening_equity = opening_equity
            self._current_equity = current_equity
            self._last_snapshot_date = snapshot_date

            if new_day:
                self._halt_triggered_today = False

            if pnl_pct <= -self._threshold_pct:
                self._halted = True
                if not self._halt_triggered_today:
                    logger.warning(
                        f"DAILY LOSS CIRCUIT BREAKER TRIPPED: {pnl_pct:.1%} loss "
                        f"(threshold: {self._threshold_pct:.0%}). "
                        f"Opening: ${opening_equity:.2f}, "
                        f"Current: ${current_equity:.2f}. "
                        f"All BUY signals suppressed until next trading day."
                    )
                    self._halt_triggered_today = True
            else:
                self._halted = False
