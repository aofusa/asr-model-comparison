# API routers package
from __future__ import annotations

from .models import router as models_router
from .status import router as status_router
from .transcribe import router as transcribe_router

__all__ = ["models_router", "status_router", "transcribe_router"]
