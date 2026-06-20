from __future__ import annotations

import logging


SERVER_LOGGER_NAME = "uvicorn.error"


def server_log(message: str, level: int = logging.INFO) -> None:
    """Emit operational logs through Uvicorn's server logger."""
    logging.getLogger(SERVER_LOGGER_NAME).log(level, message)
