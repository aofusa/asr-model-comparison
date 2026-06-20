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
from app.utils.server_logging import server_log

# Realtime audio chunk normalization (addresses empty-text problem from browser mic blobs)
from app.utils.audio import normalize_to_wav_pcm_16k_mono

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
        server_log(
            f"[HTTP Transcribe] request model={model_id} filename={audio.filename} "
            f"bytes={len(audio_bytes)} language={language} beam_size={beam_size} "
            f"return_timestamps={return_timestamps} previous_text_len={len(previous_text or '')}"
        )
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
        server_log(f"[Transcribe Error] model={model_id} error={exc}")
        raise HTTPException(status_code=500, detail="Transcription failed due to an internal error.") from exc

    # Ensure we always return the required shape
    text = result.get("text", "") if isinstance(result, dict) else str(result)
    proc_time = float(result.get("processing_time_seconds", 0.0)) if isinstance(result, dict) else 0.0
    server_log(
        f"[HTTP Transcribe] complete model={model_id} text_len={len(text)} "
        f"processing_time={proc_time:.3f}s"
    )

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
    server_log("[WS] New connection attempt received")
    try:
        await websocket.accept()
        server_log("[WS] accept() succeeded")
    except Exception as e:
        server_log(f"[WS] accept() FAILED: {e}")
        raise

    manager: ModelManager | None = None
    current_model_id = "qwen3-asr-0.6b"
    language = "auto"
    target_language = "none"
    beam_size = 6
    temperature = 0.0
    repetition_penalty = 1.15
    use_dedicated_class = True
    previous_text = ""
    return_timestamps = True
    vad_filter = False
    chunk_index = 0

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
            target_language = config.get("target_language", target_language)
            beam_size = config.get("beam_size", beam_size)
            temperature = config.get("temperature", temperature)
            repetition_penalty = config.get("repetition_penalty", repetition_penalty)
            use_dedicated_class = config.get("use_dedicated_class", use_dedicated_class)
            return_timestamps = config.get("return_timestamps", return_timestamps)
            vad_filter = config.get("vad_filter", vad_filter)
            server_log(
                f"[WS Config] model={current_model_id} language={language} "
                f"target_language={target_language} beam_size={beam_size} "
                f"temperature={temperature} repetition_penalty={repetition_penalty} "
                f"use_dedicated_class={use_dedicated_class} "
                f"return_timestamps={return_timestamps} vad_filter={vad_filter}"
            )
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
        server_log(
            f"[WS Model] loader selected model={current_model_id} "
            f"use_dedicated_class={use_dedicated_class}"
        )

        await websocket.send_json({
            "type": "ready",
            "model_id": current_model_id,
            "language": language,
            "target_language": target_language,
            "return_timestamps": return_timestamps
        })

        # Main streaming loop
        while True:
            message = await websocket.receive()

            if "bytes" in message:
                raw_chunk = message["bytes"]
                if not raw_chunk:
                    continue
                chunk_index += 1

                # === Key fix for "話しかけても何も起きない" ===
                # Browser MediaRecorder sends webm/opus (or other) containers.
                # We normalize to proper 16kHz mono WAV so the ASR backend
                # (faster-whisper etc.) + its VAD can actually see speech.
                # Without this, short 2s chunks frequently decode to empty after VAD.
                audio_for_model = normalize_to_wav_pcm_16k_mono(raw_chunk)
                server_log(
                    f"[WS Audio] normalized model={current_model_id} chunk={chunk_index} "
                    f"raw_bytes={len(raw_chunk)} normalized_bytes={len(audio_for_model)}"
                )

                try:
                    asr_language = None if language in (None, "", "auto") else language
                    translation_target = None if target_language in (None, "", "none") else target_language
                    result = await manager.transcribe(
                        audio=audio_for_model,
                        model_id=current_model_id,
                        language=asr_language,
                        target_language=translation_target,
                        beam_size=beam_size,
                        previous_text=previous_text,
                        return_timestamps=return_timestamps,
                        temperature=temperature,
                        repetition_penalty=repetition_penalty,
                        vad_filter=vad_filter,
                    )

                    new_text = result.get("text", "").strip()
                    if new_text:
                        previous_text = (previous_text + " " + new_text).strip()

                    # had_speech heuristic: if the backend returned segments/chunks or we got text
                    had_speech = bool(new_text or result.get("chunks") or result.get("segments"))

                    # Diagnostic log (very useful when user reports "nothing happens")
                    server_log(
                        f"[WS Chunk] model={current_model_id} raw={len(raw_chunk)}B "
                        f"norm={len(audio_for_model)}B chunk={chunk_index} text_len={len(new_text)} "
                        f"had_speech={had_speech} proc={result.get('processing_time_seconds', 0):.3f}s"
                    )

                    await websocket.send_json({
                        "type": "transcription",
                        "model_id": current_model_id,
                        "text": new_text,
                        "is_final": False,
                        "chunks": result.get("chunks", []),
                        "language": result.get("language"),
                        "target_language": translation_target,
                        "processing_time_seconds": result.get("processing_time_seconds", 0.0),
                        "had_speech": had_speech,
                        "chunk_index": chunk_index,
                        "chunk_size_bytes": len(audio_for_model),
                    })

                except Exception as e:
                    server_log(
                        f"[WS Chunk Error] model={current_model_id} chunk={chunk_index} "
                        f"raw={len(raw_chunk)}B norm={len(audio_for_model)}B "
                        f"error={type(e).__name__}: {e}"
                    )
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
                        target_language = data.get("target_language", target_language)
                        return_timestamps = data.get("return_timestamps", return_timestamps)
                        vad_filter = data.get("vad_filter", vad_filter)

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
        server_log("[WebSocket] Client disconnected")
    except Exception as e:
        server_log(f"[WebSocket] Unexpected error: {e}")
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
