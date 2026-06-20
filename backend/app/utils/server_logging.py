from __future__ import annotations

import logging


SERVER_LOGGER_NAME = "uvicorn.error"


def server_log(message: str, level: int = logging.INFO) -> None:
    """Emit operational logs through Uvicorn and stdout.

    Uvicorn's logger is not always visible when the app is launched through
    PowerShell jobs or test harnesses, so stdout is used as the reliable
    operator-facing sink.
    """
    logging.getLogger(SERVER_LOGGER_NAME).log(level, message)
    print(message, flush=True)
