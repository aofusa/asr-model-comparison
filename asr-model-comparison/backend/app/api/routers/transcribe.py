"""
Router for the transcription endpoint.

POST /api/transcribe
- multipart/form-data
  - audio: file (wav, mp3, etc.)
  - model_id: string (one of the ids from /api/models)

Returns transcription + wall-clock processing time.
"""
from __future__ import annotations

import os
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.models.schemas import TranscriptionResponse
from app.services.model_manager import ModelManager
from app.services.asr_backends.factory import create_whisper_loader, whisper_unloader

router = APIRouter(prefix="/api", tags=["transcribe"])


def get_model_manager() -> ModelManager:
    """
    FastAPI dependency that returns a ModelManager.

    By default it uses cheap no-op loaders (great for unit tests and when
    you don't want to load real models).

    Set environment variable USE_REAL_WHISPER=1 to enable real Whisper models
    (CPU + int8 by default — suitable for ROG Ally X).
    """
    if os.getenv("USE_REAL_WHISPER", "0") in ("1", "true", "yes"):
        loader = create_whisper_loader(device="cpu", compute_type="int8")
        return ModelManager(model_loader=loader, model_unloader=whisper_unloader)
    return ModelManager()  # no-op (current default behavior)


# Note: In a real app we would use lifespan events, but for now a module-level
# instance is acceptable for the early stages of the project.


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    audio: Annotated[UploadFile, File(description="Audio file to transcribe")],
    model_id: Annotated[str, Form(description="Model ID from /api/models")],
    model_manager: ModelManager = Depends(get_model_manager),
) -> TranscriptionResponse:
    """
    Transcribe the uploaded audio using the selected model.

    The ModelManager guarantees that only one model is loaded at any time.
    Processing time is measured with high-resolution timer from request
    handling start until the result is ready.
    """
    if not audio.filename:
        raise HTTPException(status_code=400, detail="No audio file provided")

    # Read the entire file into memory (acceptable for short clips in this project)
    audio_bytes = await audio.read()

    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty")

    try:
        result = await model_manager.transcribe(
            audio=audio_bytes,
            model_id=model_id,
            filename=audio.filename,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        # In production we would log the full traceback here
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}") from exc

    # Ensure we always return the required shape
    text = result.get("text", "") if isinstance(result, dict) else str(result)
    proc_time = float(result.get("processing_time_seconds", 0.0)) if isinstance(result, dict) else 0.0

    return TranscriptionResponse(
        model_id=model_id,
        text=text,
        processing_time_seconds=proc_time,
        language=None,  # future extension
    )
