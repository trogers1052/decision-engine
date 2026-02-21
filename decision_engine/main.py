"""
Decision Engine - Entry point.

Evaluates trading rules against indicator data and produces
BUY/SELL/WATCH signals with cross-stock rankings.
"""

import logging
import os
import signal
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

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


def _start_health_server() -> None:
    """Start a minimal HTTP health server on a daemon thread."""
    port = int(os.environ.get("HEALTH_PORT", "8080"))

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok")
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, *args):
            pass  # suppress HTTP access logs

    server = HTTPServer(("", port), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info(f"Health server listening on :{port}/health")


def main():
    """Main entry point."""
    global _service

    _start_health_server()

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
