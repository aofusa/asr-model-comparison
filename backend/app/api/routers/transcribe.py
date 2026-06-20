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
from app.utils.audio import is_likely_speech, normalize_to_wav_pcm_16k_mono

router = APIRouter(prefix="/api", tags=["transcribe"])


def _merge_realtime_context(previous_text: str, new_text: str) -> str:
    """Append a chunk result without growing duplicate context."""
    previous_text = (previous_text or "").strip()
    new_text = (new_text or "").strip()

    if not new_text:
        return previous_text
    if not previous_text:
        return new_text
    if new_text == previous_text or previous_text.endswith(new_text):
        return previous_text
    if new_text.startswith(previous_text):
        return new_text

    return f"{previous_text} {new_text}".strip()


def _translation_enabled(target_language: str | None) -> bool:
    return target_language not in (None, "", "none", "auto")


def _normalize_transcription_result(
    result: dict | str,
    *,
    target_language: str | None = None,
) -> dict:
    """Return a stable original/translated text contract for HTTP and WS callers."""
    if not isinstance(result, dict):
        text = str(result).strip()
        return {
            "text": text,
            "transcript_text": text,
            "translated_text": None,
            "target_language": target_language,
            "language": None,
            "chunks": [],
            "processing_time_seconds": 0.0,
        }

    text = str(result.get("text", "") or "").strip()
    translated_text = result.get("translated_text")
    translated_text = str(translated_text).strip() if translated_text else None
    transcript_text = result.get("transcript_text")
    if transcript_text is None:
        transcript_text = text if not translated_text else str(result.get("original_text", "") or "").strip()
    transcript_text = str(transcript_text or "").strip()

    if not transcript_text and text and not translated_text:
        transcript_text = text
    display_text = translated_text or transcript_text or text

    return {
        **result,
        "text": display_text,
        "transcript_text": transcript_text,
        "translated_text": translated_text,
        "target_language": result.get("target_language", target_language),
        "language": result.get("language"),
        "chunks": result.get("chunks", []),
        "processing_time_seconds": float(result.get("processing_time_seconds", 0.0)),
    }


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
    target_language: Annotated[str | None, Form(description="Optional translation target language")] = None,
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
            f"target_language={target_language} return_timestamps={return_timestamps} "
            f"previous_text_len={len(previous_text or '')}"
        )
        result = await model_manager.transcribe(
            audio=audio_bytes,
            model_id=model_id,
            filename=audio.filename,
            language=language,
            beam_size=beam_size,
            return_timestamps=return_timestamps,
            previous_text=previous_text,
            target_language=None if target_language in (None, "", "none") else target_language,
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
    normalized = _normalize_transcription_result(result, target_language=target_language)
    text = normalized["text"]
    proc_time = normalized["processing_time_seconds"]
    server_log(
        f"[HTTP Transcribe] complete model={model_id} text_len={len(text)} "
        f"processing_time={proc_time:.3f}s"
    )

    return TranscriptionResponse(
        model_id=model_id,
        text=text,
        transcript_text=normalized["transcript_text"],
        translated_text=normalized["translated_text"],
        processing_time_seconds=proc_time,
        language=normalized["language"],
        target_language=normalized["target_language"],
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
    previous_translated_text = ""
    return_timestamps = True
    vad_filter = False
    audio_source = "microphone"
    chunk_index = 0

    def _is_closed_send_error(exc: Exception) -> bool:
        message = str(exc)
        return (
            'Cannot call "send" once a close message has been sent' in message
            or "Cannot call \"receive\" once a disconnect message has been received" in message
            or "WebSocket is not connected" in message
        )

    async def safe_send_json(payload: dict) -> bool:
        try:
            await websocket.send_json(payload)
            return True
        except WebSocketDisconnect:
            server_log("[WebSocket] Client disconnected before send")
            return False
        except RuntimeError as exc:
            if _is_closed_send_error(exc):
                server_log("[WebSocket] Client disconnected before send")
                return False
            raise

    try:
        # Wait for initial config
        first_message = await websocket.receive()
        if "text" not in first_message:
            await safe_send_json({"type": "error", "message": "First message must be config JSON"})
            await websocket.close()
            return

        try:
            config = json.loads(first_message["text"])
            if config.get("type") != "config":
                await safe_send_json({"type": "error", "message": "First message must have type 'config'"})
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
            requested_audio_source = str(config.get("audio_source", audio_source) or audio_source)
            if requested_audio_source in {"microphone", "system", "window"}:
                audio_source = requested_audio_source
            previous_text = str(config.get("previous_text", "") or "").strip()
            previous_translated_text = str(config.get("previous_translated_text", "") or "").strip()
            server_log(
                f"[WS Config] model={current_model_id} language={language} "
                f"target_language={target_language} beam_size={beam_size} "
                f"temperature={temperature} repetition_penalty={repetition_penalty} "
                f"use_dedicated_class={use_dedicated_class} "
                f"return_timestamps={return_timestamps} vad_filter={vad_filter} "
                f"audio_source={audio_source} previous_text_len={len(previous_text)} "
                f"previous_translated_text_len={len(previous_translated_text)}"
            )
        except Exception:
            await safe_send_json({"type": "error", "message": "Invalid config JSON"})
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

        async def send_model_progress(event: dict) -> None:
            await safe_send_json(event)

        manager = ModelManager(model_loader=loader, progress_callback=send_model_progress)
        server_log(
            f"[WS Model] loader selected model={current_model_id} "
            f"use_dedicated_class={use_dedicated_class}"
        )
        try:
            await manager.load_model(current_model_id)
        except Exception as exc:
            server_log(f"[WS Model] load failed model={current_model_id} error={exc}")
            await safe_send_json({
                "type": "error",
                "code": "model_load_failed",
                "message": str(exc),
                "model_id": current_model_id,
            })
            return
        server_log(f"[WS Ready] model loaded and ready model={current_model_id}")

        if not await safe_send_json({
            "type": "ready",
            "model_id": current_model_id,
            "language": language,
            "target_language": target_language,
            "return_timestamps": return_timestamps,
            "audio_source": audio_source,
        }):
            return

        # Main streaming loop
        while True:
            message = await websocket.receive()

            if "bytes" in message:
                raw_chunk = message["bytes"]
                if not raw_chunk:
                    if not await safe_send_json({
                        "type": "error",
                        "code": "empty_audio_chunk",
                        "message": "Received an empty audio chunk.",
                    }):
                        break
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

                if not is_likely_speech(audio_for_model):
                    server_log(
                        f"[WS Chunk] model={current_model_id} raw={len(raw_chunk)}B "
                        f"norm={len(audio_for_model)}B chunk={chunk_index} text_len=0 "
                        "had_speech=False proc=0.000s skipped=silence"
                    )
                    if not await safe_send_json({
                        "type": "transcription",
                        "model_id": current_model_id,
                        "text": "",
                        "transcript_text": "",
                        "translated_text": None,
                        "is_final": False,
                        "chunks": [],
                        "language": None if language in (None, "", "auto") else language,
                        "target_language": None if target_language in (None, "", "none") else target_language,
                        "processing_time_seconds": 0.0,
                        "had_speech": False,
                        "chunk_index": chunk_index,
                        "chunk_size_bytes": len(audio_for_model),
                        "accumulated_text": previous_text,
                        "accumulated_transcript_text": previous_text,
                        "accumulated_translated_text": previous_translated_text,
                    }):
                        break
                    continue

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

                    normalized_result = _normalize_transcription_result(
                        result,
                        target_language=translation_target,
                    )
                    new_text = normalized_result["text"]
                    new_transcript_text = normalized_result["transcript_text"]
                    new_translated_text = normalized_result["translated_text"]
                    if new_transcript_text:
                        previous_text = _merge_realtime_context(previous_text, new_transcript_text)
                    if new_translated_text:
                        previous_translated_text = _merge_realtime_context(
                            previous_translated_text,
                            new_translated_text,
                        )

                    # had_speech heuristic: if the backend returned segments/chunks or we got text
                    had_speech = bool(
                        new_transcript_text
                        or new_translated_text
                        or normalized_result.get("chunks")
                        or normalized_result.get("segments")
                    )

                    # Diagnostic log (very useful when user reports "nothing happens")
                    server_log(
                        f"[WS Chunk] model={current_model_id} raw={len(raw_chunk)}B "
                        f"norm={len(audio_for_model)}B chunk={chunk_index} text_len={len(new_text)} "
                        f"had_speech={had_speech} proc={result.get('processing_time_seconds', 0):.3f}s"
                    )

                    if not await safe_send_json({
                        "type": "transcription",
                        "model_id": current_model_id,
                        "text": new_text,
                        "transcript_text": new_transcript_text,
                        "translated_text": new_translated_text,
                        "is_final": False,
                        "chunks": normalized_result.get("chunks", []),
                        "language": normalized_result.get("language"),
                        "target_language": translation_target,
                        "processing_time_seconds": normalized_result.get("processing_time_seconds", 0.0),
                        "had_speech": had_speech,
                        "chunk_index": chunk_index,
                        "chunk_size_bytes": len(audio_for_model),
                        "accumulated_text": previous_text,
                        "accumulated_transcript_text": previous_text,
                        "accumulated_translated_text": previous_translated_text,
                    }):
                        break

                except Exception as e:
                    server_log(
                        f"[WS Chunk Error] model={current_model_id} chunk={chunk_index} "
                        f"raw={len(raw_chunk)}B norm={len(audio_for_model)}B "
                        f"error={type(e).__name__}: {e}"
                    )
                    if not await safe_send_json({
                        "type": "error",
                        "code": "transcription_failed",
                        "message": str(e)
                    }):
                        break

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
                        await safe_send_json({
                            "type": "final",
                            "model_id": current_model_id,
                            "text": previous_translated_text or previous_text,
                            "transcript_text": previous_text,
                            "translated_text": previous_translated_text or None,
                            "target_language": None if target_language in (None, "", "none") else target_language,
                        })
                        break

                except Exception:
                    if not await safe_send_json({
                        "type": "error",
                        "code": "invalid_message",
                        "message": "Could not parse control message"
                    }):
                        break

    except WebSocketDisconnect:
        server_log("[WebSocket] Client disconnected")
    except Exception as e:
        if _is_closed_send_error(e):
            server_log("[WebSocket] Client disconnected")
        else:
            server_log(f"[WebSocket] Unexpected error: {e}")
            await safe_send_json({
                "type": "error",
                "code": "unexpected_error",
                "message": str(e)
            })
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
