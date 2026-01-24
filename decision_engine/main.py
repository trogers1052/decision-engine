"""
Decision Engine - Entry point.

Evaluates trading rules against indicator data and produces
BUY/SELL/WATCH signals with cross-stock rankings.
"""

import logging
import signal
import sys

from .config import Settings
from .service import DecisionEngineService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)

# Global service instance for signal handling
_service: DecisionEngineService = None


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info(f"Received signal {signum}, shutting down...")
    if _service:
        _service.shutdown()
    sys.exit(0)


def main():
    """Main entry point."""
    global _service

    logger.info("Decision Engine starting...")

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Load configuration
        settings = Settings()
        logger.info(f"Loaded configuration from environment")

        # Create and initialize service
        _service = DecisionEngineService(settings)

        if not _service.initialize():
            logger.error("Failed to initialize service")
            sys.exit(1)

        # Start the service (blocks until shutdown)
        _service.start()

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        if _service:
            _service.shutdown()
        sys.exit(1)


if __name__ == "__main__":
    main()
