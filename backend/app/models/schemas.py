"""
Pydantic schemas for AMCP API requests and responses.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    """Metadata for a single available ASR model."""

    id: str = Field(..., description="Unique identifier for the model (e.g. 'whisper-tiny')")
    name: str = Field(..., description="Human readable name")
    family: Literal["whisper", "qwen3", "voxtral"] = Field(
        ..., description="Model family"
    )
    size: str = Field(..., description="Size label (tiny, 0.6B, Mini 4B, etc.)")
    parameters: str | None = Field(None, description="Approximate parameter count")
    description: str | None = Field(None, description="Short description of the model")
    recommended_vram_gb: float | None = Field(
        None, description="Recommended VRAM in GB for comfortable inference"
    )
    languages: list[str] = Field(
        default_factory=lambda: ["ja", "en", "zh"],
        description="Primary supported languages for this platform",
    )


class TranscriptionRequest(BaseModel):
    """Request body for transcription (when not using multipart form)."""

    model_id: str
    # Audio will usually come via UploadFile in FormData, not here.


class TranscriptionResponse(BaseModel):
    """Response for a successful transcription."""

    model_id: str
    text: str
    transcript_text: str | None = Field(
        None, description="Original ASR transcription before optional translation"
    )
    translated_text: str | None = Field(
        None, description="Translated text when translation is enabled"
    )
    processing_time_seconds: float = Field(..., description="Wall-clock time from request to result")
    language: str | None = None
    target_language: str | None = None
    # Future: confidence, segments, resource usage, etc.
