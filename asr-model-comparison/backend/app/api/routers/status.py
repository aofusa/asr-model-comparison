"""
Lightweight status endpoint for real model usage monitoring.

Reports which model is currently resident in memory and rough memory usage.
Very useful when running on 24GB machines (ROG Ally X, M4 MacBook).
"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from app.services.model_manager import ModelManager
from app.api.routers.transcribe import get_model_manager

router = APIRouter(prefix="/api", tags=["status"])


@router.get("/status")
def get_system_status(
    model_manager: ModelManager = Depends(get_model_manager),
) -> dict:
    """
    Returns the current state of the model manager + basic memory info.

    This endpoint is intentionally lightweight and does not trigger any
    model loading or heavy computation.
    """
    status = model_manager.get_status()
    status["service"] = "amcp-backend"
    return status
