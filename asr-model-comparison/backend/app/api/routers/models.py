"""
Router for model discovery endpoint.
This endpoint returns static metadata and does NOT load any ASR models into memory.
"""
from __future__ import annotations

from fastapi import APIRouter

from app.models.available_models import AVAILABLE_MODELS
from app.models.schemas import ModelInfo

router = APIRouter(prefix="/api", tags=["models"])


@router.get("/models", response_model=list[ModelInfo])
async def get_available_models() -> list[ModelInfo]:
    """
    Return the list of all ASR models supported by this platform.

    This endpoint is lightweight and does not load any models into GPU/CPU memory.
    It is safe to call frequently for UI population.
    """
    return AVAILABLE_MODELS
