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

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
import json

from app.models.schemas import TranscriptionResponse
from app.models.available_models import AVAILABLE_MODELS
from app.services.model_manager import ModelManager
from app.services.asr_backends.factory import (
    create_qwen3_loader,
    create_voxtral_loader,
    create_whisper_loader,
    whisper_unloader,
)

router = APIRouter(prefix="/api", tags=["transcribe"])


def _get_real_loader_for_model(model_id: str):
    """Select the appropriate real loader based on model family."""
    model_info = next((m for m in AVAILABLE_MODELS if m.id == model_id), None)
    family = model_info.family if model_info else None

    if family == "whisper":
        return create_whisper_loader(device="cpu", compute_type="int8")
    elif family == "qwen3":
        return create_qwen3_loader(device="cpu")
    elif family == "voxtral":
        return create_voxtral_loader(device="cpu")
    else:
        return create_whisper_loader(device="cpu")


def get_model_manager(model_id: str | None = None) -> ModelManager:
    """
    FastAPI dependency that returns a ModelManager.

    - Default: cheap no-op (excellent for unit tests)
    - When USE_REAL_WHISPER=1 (or USE_REAL_MODELS=1): selects the correct
      real backend per family. Qwen3 and Voxtral currently raise clear
      "not yet implemented" errors with guidance.
    """
    use_real = os.getenv("USE_REAL_WHISPER", "0") in ("1", "true", "yes") or \
               os.getenv("USE_REAL_MODELS", "0") in ("1", "true", "yes")

    if not use_real:
        return ModelManager()

    # When real mode is on, we pick the loader based on the target model if known.
    # Note: For the actual transcribe call we get model_id from the form.
    # Here we default to whisper if no specific model is known yet.
    loader = create_whisper_loader(device="cpu", compute_type="int8")
    return ModelManager(model_loader=loader, model_unloader=whisper_unloader)


# Note: In a real app we would use lifespan events, but for now a module-level
# instance is acceptable for the early stages of the project.


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    audio: Annotated[UploadFile, File(description="Audio file to transcribe")],
    model_id: Annotated[str, Form(description="Model ID from /api/models")],
    language: Annotated[str | None, Form(description="Force language (e.g. 'ja', 'en', 'zh')")] = None,
    beam_size: Annotated[int | None, Form(description="Beam size for higher quality (slower, recommended for Japanese)")] = None,
    return_timestamps: Annotated[bool, Form(description="Return timestamps (important for real-time)")] = False,
    previous_text: Annotated[str | None, Form(description="Previous transcript chunk for real-time continuity")] = None,
) -> TranscriptionResponse:
    # Select the correct manager at request time based on the requested model
    # This allows proper support for Whisper, Qwen3, and Voxtral in real mode.
    use_real = os.getenv("USE_REAL_WHISPER", "0") in ("1", "true", "yes") or \
               os.getenv("USE_REAL_MODELS", "0") in ("1", "true", "yes")

    if use_real:
        loader = _get_real_loader_for_model(model_id)
        # Choose appropriate unloader
        if "qwen" in model_id.lower() or "voxtral" in model_id.lower():
            unloader = whisper_unloader  # reuse the strong cleanup one
        else:
            unloader = whisper_unloader
        model_manager = ModelManager(model_loader=loader, model_unloader=unloader)
    else:
        model_manager = ModelManager()

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
            language=language,
            beam_size=beam_size,
            return_timestamps=return_timestamps,
            previous_text=previous_text,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid input: {exc}") from exc
    except RuntimeError as exc:
        # Common for large model loading failures in real-time scenarios
        raise HTTPException(status_code=503, detail=f"Model temporarily unavailable: {exc}") from exc
    except Exception as exc:
        # Log the full error in production
        print(f"[Transcribe Error] model={model_id} error={exc}")
        raise HTTPException(status_code=500, detail="Transcription failed due to an internal error.") from exc

    # Ensure we always return the required shape
    text = result.get("text", "") if isinstance(result, dict) else str(result)
    proc_time = float(result.get("processing_time_seconds", 0.0)) if isinstance(result, dict) else 0.0

    return TranscriptionResponse(
        model_id=model_id,
        text=text,
        processing_time_seconds=proc_time,
        language=None,  # future extension
    )


# ============================================================
# Real-time WebSocket Streaming (for browser microphone)
# ============================================================

@router.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    """
    WebSocket endpoint for real-time audio streaming from the browser (practical version).

    Improved protocol for practicality:
    - First message from client should be a JSON config: {"type": "config", "model_id": "qwen3-asr-0.6b", "language": "ja", "beam_size": 6, "use_dedicated_class": true}
    - Server loads the model once and sends {"type": "ready"}
    - Client then sends binary audio chunks.
    - Server responds with JSON transcription results including accumulated context.

    Model is loaded only once per connection (critical for heavy models like Qwen3 1.7B / Voxtral 4B).
    """
    print("[WS] New connection attempt received", flush=True)
    try:
        await websocket.accept()
        print("[WS] accept() succeeded", flush=True)
    except Exception as e:
        print(f"[WS] accept() FAILED: {e}", flush=True)
        raise

    manager: ModelManager | None = None
    current_model_id = "qwen3-asr-0.6b"
    language = "ja"
    beam_size = 6
    use_dedicated_class = True
    previous_text = ""
    return_timestamps = True

    try:
        # Wait for initial config
        first_message = await websocket.receive()
        if "text" not in first_message:
            await websocket.send_json({"type": "error", "message": "First message must be config JSON"})
            await websocket.close()
            return

        try:
            config = json.loads(first_message["text"])
            if config.get("type") != "config":
                await websocket.send_json({"type": "error", "message": "First message must have type 'config'"})
                await websocket.close()
                return

            current_model_id = config.get("model_id", current_model_id)
            language = config.get("language", language)
            beam_size = config.get("beam_size", beam_size)
            use_dedicated_class = config.get("use_dedicated_class", use_dedicated_class)
            return_timestamps = config.get("return_timestamps", return_timestamps)
        except Exception:
            await websocket.send_json({"type": "error", "message": "Invalid config JSON"})
            await websocket.close()
            return

        # Load model once for this connection (key practical improvement)
        #
        # For lightweight E2E / protocol verification tests, it is strongly recommended
        # to use model_id="whisper-tiny" (runs fast on CPU with int8).
        # This allows running realistic WebSocket protocol tests without loading
        # heavy Qwen3/Voxtral models.
        if "qwen" in current_model_id.lower():
            loader = create_qwen3_loader(device="cpu", use_dedicated_class=use_dedicated_class)
        elif "voxtral" in current_model_id.lower():
            loader = create_voxtral_loader(device="cpu", use_dedicated_class=use_dedicated_class)
        else:
            # Default to Whisper (tiny is ideal for fast E2E protocol tests)
            loader = create_whisper_loader(device="cpu", compute_type="int8")

        manager = ModelManager(model_loader=loader)

        await websocket.send_json({
            "type": "ready",
            "model_id": current_model_id,
            "language": language,
            "return_timestamps": return_timestamps
        })

        # Main streaming loop
        while True:
            message = await websocket.receive()

            if "bytes" in message:
                audio_chunk = message["bytes"]
                if not audio_chunk:
                    continue

                try:
                    result = await manager.transcribe(
                        audio=audio_chunk,
                        model_id=current_model_id,
                        language=language,
                        beam_size=beam_size,
                        previous_text=previous_text,
                        return_timestamps=return_timestamps,
                        temperature=0.0,
                        repetition_penalty=1.15 if language == "ja" else 1.0,
                    )

                    new_text = result.get("text", "").strip()
                    if new_text:
                        previous_text = (previous_text + " " + new_text).strip()

                    await websocket.send_json({
                        "type": "transcription",
                        "model_id": current_model_id,
                        "text": new_text,
                        "is_final": False,
                        "chunks": result.get("chunks", []),
                        "language": result.get("language"),
                        "processing_time_seconds": result.get("processing_time_seconds", 0.0),
                    })

                except Exception as e:
                    await websocket.send_json({
                        "type": "error",
                        "code": "transcription_failed",
                        "message": str(e)
                    })

            elif "text" in message:
                try:
                    data = json.loads(message["text"])
                    msg_type = data.get("type")

                    if msg_type == "update":
                        beam_size = data.get("beam_size", beam_size)
                        language = data.get("language", language)
                        return_timestamps = data.get("return_timestamps", return_timestamps)

                    elif msg_type == "end":
                        await websocket.send_json({
                            "type": "final",
                            "model_id": current_model_id,
                            "text": previous_text,
                        })
                        break

                except Exception:
                    await websocket.send_json({
                        "type": "error",
                        "code": "invalid_message",
                        "message": "Could not parse control message"
                    })

    except WebSocketDisconnect:
        print("[WebSocket] Client disconnected")
    except Exception as e:
        print(f"[WebSocket] Unexpected error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "code": "unexpected_error",
                "message": str(e)
            })
        except:
            pass
    finally:
        if manager is not None:
            try:
                await manager.unload_current()
            except:
                pass
        try:
            await websocket.close()
        except:
            pass
