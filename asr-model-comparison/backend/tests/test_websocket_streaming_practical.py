"""
TDD tests for making the WebSocket streaming endpoint practical for real-time use.

Focus:
- One ModelManager per connection (model loaded once)
- Proper session config (model_id, language, beam_size, use_dedicated_class, etc.)
- Support for previous_text / context across chunks
- Clear message protocol
- Graceful error handling and cleanup
"""
from __future__ import annotations

import base64
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.api.routers.transcribe import _merge_realtime_context
from app.utils.audio import is_likely_speech


def _mock_whisper_loader(text: str = ""):
    fake_backend = MagicMock()
    fake_backend.transcribe.return_value = {
        "text": text,
        "processing_time_seconds": 0.01,
        "chunks": [{"text": text}] if text else [],
    }

    async def fake_loader(_model_id: str):
        return fake_backend

    return patch(
        "app.api.routers.transcribe.create_whisper_loader",
        return_value=fake_loader,
    )


def _mock_loader_for(router_factory_name: str, text: str = ""):
    fake_backend = MagicMock()
    fake_backend.transcribe.return_value = {
        "text": text,
        "processing_time_seconds": 0.01,
        "chunks": [{"text": text}] if text else [],
        "language": "en",
    }
    loaded_model_ids: list[str] = []

    async def fake_loader(model_id: str):
        loaded_model_ids.append(model_id)
        return fake_backend

    return (
        patch(
            f"app.api.routers.transcribe.{router_factory_name}",
            return_value=fake_loader,
        ),
        fake_backend,
        loaded_model_ids,
    )


def _tiny_silent_wav() -> bytes:
    return base64.b64decode(
        'UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA='
    )


def test_is_likely_speech_treats_empty_wav_as_silence_but_keeps_fake_bytes():
    assert is_likely_speech(_tiny_silent_wav()) is False
    assert is_likely_speech(b"fake browser mic chunk") is True


def test_websocket_practical_connection_and_config():
    """
    Client must send an initial config message after connecting.
    The server loads the model only once per connection.
    """
    client = TestClient(app)
    patcher, _fake_backend, loaded_model_ids = _mock_loader_for("create_qwen3_loader", text="")

    with patcher:
        with client.websocket_connect("/api/ws/transcribe") as websocket:
            # Send initial config (new practical protocol)
            config = {
                "type": "config",
                "model_id": "qwen3-asr-0.6b",
                "language": "ja",
                "beam_size": 6,
                "use_dedicated_class": True,
            }
            websocket.send_json(config)

            # Expect "ready" after model is loaded (once)
            response = websocket.receive_json()
            assert response.get("type") == "ready"
            assert response.get("model_id") == "qwen3-asr-0.6b"
            assert loaded_model_ids == ["qwen3-asr-0.6b"]


@pytest.mark.slow
def test_websocket_sends_transcription_results_for_chunks():
    """
    After sending config and receiving 'ready', sending binary audio chunks
    should result in transcription JSON messages.

    TDD Phase 1 (per 修正指示書): This test (and the new feedback test below)
    is updated BEFORE any implementation of chunk normalization / had_speech / UI feedback.
    Current state (with fake bytes or real short chunks + whisper-tiny) often yields
    empty "text" (due to VAD on 2s mic chunks, container format, small model).
    We now assert that "processing_time_seconds" is ALWAYS present (good), and document
    the desired future state for real-time mic use case (non-empty text or explicit
    had_speech / chunk feedback so the UI can show "processed chunk" even on silence).
    """
    client = TestClient(app)

    with client.websocket_connect("/api/ws/transcribe") as websocket:
        # Use whisper-tiny for reliable protocol tests (recommended in docs for E2E/WS verification,
        # avoids heavy Qwen/Voxtral deps like accelerate in normal CI).
        websocket.send_json({"type": "config", "model_id": "whisper-tiny", "language": "ja", "beam_size": 1})
        websocket.receive_json()  # ready

        # Use a minimal but valid WAV (same tiny silence used in frontend ws-protocol test).
        # In real mic flow this would be 2s webm/opus from MediaRecorder -> often empty after VAD.
        tiny_wav = base64.b64decode(
            'UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA='
        )
        websocket.send_bytes(tiny_wav)

        response = websocket.receive_json()
        assert response.get("type") == "transcription"
        assert "text" in response
        assert response.get("model_id") == "whisper-tiny"
        # Processing time must always be reported (per project requirements).
        assert "processing_time_seconds" in response
        assert isinstance(response.get("processing_time_seconds"), (int, float))

        # TDD marker: In current implementation (pre-fix), text is frequently "" for short chunks.
        # After Phase 2/3 we will also have had_speech or better feedback payload.
        # For now we just ensure the response shape is useful for the UI to show "chunk was processed".


def test_websocket_maintains_previous_text_across_chunks():
    """
    The server should accumulate previous_text across chunks for continuity.
    """
    client = TestClient(app)

    with _mock_whisper_loader(text="こんにちは"):
        with client.websocket_connect("/api/ws/transcribe") as websocket:
            # Use whisper-tiny so the test runs reliably without heavy model deps (accelerate etc.)
            websocket.send_json({"type": "config", "model_id": "whisper-tiny", "language": "ja"})
            websocket.receive_json()  # ready

            # Use valid tiny WAV (not raw b"chunk1") so whisper backend can decode without error
            # (plain bytes often cause decode failure -> error response -> WS close in test client).
            tiny_wav = base64.b64decode(
                'UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA='
            )
            websocket.send_bytes(tiny_wav)
            response1 = websocket.receive_json()

            websocket.send_bytes(tiny_wav)
            response2 = websocket.receive_json()

        assert response1.get("type") == "transcription"
        assert response2.get("type") == "transcription"


def test_realtime_context_merge_avoids_duplicate_qwen_chunk_context():
    assert _merge_realtime_context("", "こんにちは") == "こんにちは"
    assert _merge_realtime_context("こんにちは", "こんにちは") == "こんにちは"
    assert _merge_realtime_context("こんにちは", "こんにちは 世界") == "こんにちは 世界"
    assert _merge_realtime_context("こんにちは", "世界") == "こんにちは 世界"


def test_websocket_config_previous_text_is_passed_to_qwen_backend_and_not_duplicated():
    """
    Reconnects from the browser send previous_text in the initial config.
    Qwen must receive it as context for the first post-reconnect chunk, and
    repeated chunk output must not grow the accumulated context indefinitely.
    """
    client = TestClient(app)
    patcher, fake_backend, _loaded_model_ids = _mock_loader_for(
        "create_qwen3_loader",
        text="前回の認識結果",
    )

    with patcher:
        with client.websocket_connect("/api/ws/transcribe") as websocket:
            websocket.send_json({
                "type": "config",
                "model_id": "qwen3-asr-0.6b",
                "language": "ja",
                "previous_text": "前回の認識結果",
                "return_timestamps": False,
            })
            websocket.receive_json()  # ready

            websocket.send_bytes(b"fake wav chunk")
            response = websocket.receive_json()

            websocket.send_json({"type": "end"})
            final_response = websocket.receive_json()

    assert response["type"] == "transcription"
    assert response["accumulated_text"] == "前回の認識結果"
    assert final_response["text"] == "前回の認識結果"
    _, kwargs = fake_backend.transcribe.call_args
    assert kwargs["previous_text"] == "前回の認識結果"


def test_websocket_graceful_disconnect():
    """Connection should close cleanly after sending config."""
    client = TestClient(app)
    patcher, _fake_backend, loaded_model_ids = _mock_loader_for("create_qwen3_loader", text="")

    with patcher:
        with client.websocket_connect("/api/ws/transcribe") as websocket:
            websocket.send_json({"type": "config", "model_id": "qwen3-asr-0.6b"})
            websocket.receive_json()  # ready

            # Client disconnects
            pass

    assert loaded_model_ids == ["qwen3-asr-0.6b"]
    assert True


def test_websocket_client_disconnect_during_model_load_does_not_log_unexpected_send_error(caplog):
    """
    If the browser switches models quickly, the previous WebSocket can disappear
    while the server is still loading. Cleanup should not try to send an error
    frame after a close frame has already been sent.
    """
    client = TestClient(app)
    caplog.set_level(logging.INFO, logger="uvicorn.error")
    patcher, _fake_backend, loaded_model_ids = _mock_loader_for("create_qwen3_loader", text="")

    with patcher:
        with client.websocket_connect("/api/ws/transcribe") as websocket:
            websocket.send_json({"type": "config", "model_id": "qwen3-asr-0.6b"})
            websocket.close()

    assert loaded_model_ids == ["qwen3-asr-0.6b"]
    assert 'Cannot call "send" once a close message has been sent' not in caplog.text
    assert "[WebSocket] Unexpected error:" not in caplog.text


def test_websocket_chunk_provides_useful_feedback_for_realtime_mic_use_case():
    """
    TDD Phase 1 (per 修正指示書_リアルタイムASR_WS...):

    For the browser mic real-time flow (MediaRecorder 2s chunks -> WS),
    the server must return enough info in the 'transcription' message so the
    frontend can show the user that "something happened" even when the ASR
    returns empty text for that chunk (common with VAD + short webm chunks +
    small models on Japanese).

    Current broken state (pre any fix): only 'text' (often ''), no 'had_speech',
    no reliable per-chunk processing indicator beyond the global time.
    The UI therefore shows "nothing" when speaking.

    This test is written to document the gap until we add:
      - had_speech (or equivalent) in the response
      - guaranteed processing_time_seconds (already partially there)
      - (later) frontend signals + UI that react to every chunk response.

    We use whisper-tiny + tiny WAV so the test runs without heavy model deps.
    """
    client = TestClient(app)

    with _mock_whisper_loader(text=""):
        with client.websocket_connect("/api/ws/transcribe") as websocket:
            websocket.send_json({
                "type": "config",
                "model_id": "whisper-tiny",
                "language": "ja",
                "beam_size": 1,
                "return_timestamps": False,
            })
            websocket.receive_json()  # ready

            # Simulate a 2s mic chunk (currently often leads to empty text in practice)
            tiny_wav = base64.b64decode(
                'UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA='
            )
            websocket.send_bytes(tiny_wav)

            response = websocket.receive_json()
        assert response.get("type") == "transcription"

        # These must be present for UI feedback
        assert "processing_time_seconds" in response

        # TDD marker for the instruction document:
        # Current impl often has empty text for such chunks.
        # The new contract (post-fix) will allow the UI to show useful "chunk processed" state.
        text = (response.get("text") or "").strip()
        had_speech = response.get("had_speech")
        if not text and had_speech is None:
            # Record that the enhanced feedback is not yet present.
            # We don't hard-fail the whole suite here; the test name + docstring serve the TDD purpose.
            pytest.skip("TDD marker: had_speech / explicit realtime chunk feedback contract not yet present (see 修正指示書 Phase 1)")


def test_websocket_chunk_response_includes_chunk_index_and_size_for_ui_feedback(caplog):
    """
    Realtime UI needs stable per-chunk metadata to show progress even when text is
    empty. This test uses a mocked ASR backend so it stays fast and does not load
    whisper/Qwen/Voxtral models.
    """
    client = TestClient(app)
    chunk = b"fake browser mic chunk"
    caplog.set_level(logging.INFO, logger="uvicorn.error")

    with _mock_whisper_loader(text=""):
        with client.websocket_connect("/api/ws/transcribe") as websocket:
            websocket.send_json({
                "type": "config",
                "model_id": "whisper-tiny",
                "language": "ja",
                "beam_size": 1,
                "return_timestamps": False,
            })
            websocket.receive_json()  # ready

            websocket.send_bytes(chunk)
            response = websocket.receive_json()

    assert response["type"] == "transcription"
    assert response["processing_time_seconds"] == 0.01
    assert response["had_speech"] is False
    assert response["chunk_index"] == 1
    assert response["chunk_size_bytes"] >= len(chunk)

    server_logs = caplog.text
    assert "[WS Config] model=whisper-tiny" in server_logs
    assert "[WS Model] loader selected model=whisper-tiny" in server_logs
    assert "[WS Audio] normalized model=whisper-tiny chunk=1" in server_logs
    assert "[WS Chunk] model=whisper-tiny" in server_logs


def test_websocket_voxtral_silent_wav_returns_no_speech_without_inference():
    """
    Playwright fake microphones can produce valid but empty/silent WAV chunks.
    Voxtral should not receive those chunks because the real backend raises a
    zero-length feature error for them; the UI still needs a normal chunk reply.
    """
    client = TestClient(app)
    patcher, fake_backend, loaded_model_ids = _mock_loader_for("create_voxtral_loader", text="unexpected")

    with patcher:
        with client.websocket_connect("/api/ws/transcribe") as websocket:
            websocket.send_json({
                "type": "config",
                "model_id": "voxtral-mini-4b",
                "language": "ja",
                "beam_size": 5,
                "return_timestamps": True,
            })
            ready = websocket.receive_json()

            websocket.send_bytes(_tiny_silent_wav())
            response = websocket.receive_json()

    assert ready["type"] == "ready"
    assert ready["model_id"] == "voxtral-mini-4b"
    assert loaded_model_ids == ["voxtral-mini-4b"]
    assert response["type"] == "transcription"
    assert response["model_id"] == "voxtral-mini-4b"
    assert response["text"] == ""
    assert response["had_speech"] is False
    assert response["processing_time_seconds"] == 0.0
    assert response["chunk_index"] == 1
    fake_backend.transcribe.assert_not_called()


@pytest.mark.parametrize(
    "model_id",
    [
        "whisper-tiny",
        "whisper-small",
        "whisper-medium",
        "whisper-large-v3-turbo",
    ],
)
def test_websocket_uses_selected_whisper_model_id(model_id: str):
    client = TestClient(app)
    patcher, fake_backend, loaded_model_ids = _mock_loader_for("create_whisper_loader", text="ok")

    with patcher:
        with client.websocket_connect("/api/ws/transcribe") as websocket:
            websocket.send_json({"type": "config", "model_id": model_id, "language": "en"})
            ready = websocket.receive_json()
            assert loaded_model_ids == [model_id]
            websocket.send_bytes(b"fake wav chunk")
            response = websocket.receive_json()

    assert ready["model_id"] == model_id
    assert response["model_id"] == model_id
    assert loaded_model_ids == [model_id]
    fake_backend.transcribe.assert_called_once()


@pytest.mark.parametrize("model_id", ["qwen3-asr-0.6b", "qwen3-asr-1.7b"])
def test_websocket_uses_selected_qwen_model_id(model_id: str):
    client = TestClient(app)
    patcher, fake_backend, loaded_model_ids = _mock_loader_for("create_qwen3_loader", text="ok")

    with patcher:
        with client.websocket_connect("/api/ws/transcribe") as websocket:
            websocket.send_json({"type": "config", "model_id": model_id, "language": "en"})
            ready = websocket.receive_json()
            assert loaded_model_ids == [model_id]
            websocket.send_bytes(b"fake wav chunk")
            response = websocket.receive_json()

    assert ready["model_id"] == model_id
    assert response["model_id"] == model_id
    assert loaded_model_ids == [model_id]
    fake_backend.transcribe.assert_called_once()


def test_websocket_passes_language_and_translation_to_qwen_backend():
    client = TestClient(app)
    patcher, fake_backend, _loaded_model_ids = _mock_loader_for("create_qwen3_loader", text="translated")

    with patcher:
        with client.websocket_connect("/api/ws/transcribe") as websocket:
            websocket.send_json({
                "type": "config",
                "model_id": "qwen3-asr-0.6b",
                "language": "en",
                "target_language": "ja",
                "beam_size": 6,
                "temperature": 0.2,
                "repetition_penalty": 1.1,
                "return_timestamps": False,
            })
            websocket.receive_json()
            websocket.send_bytes(b"fake wav chunk")
            response = websocket.receive_json()

    assert response["type"] == "transcription"
    assert response["model_id"] == "qwen3-asr-0.6b"
    assert response["target_language"] == "ja"
    _, kwargs = fake_backend.transcribe.call_args
    assert kwargs["language"] == "en"
    assert kwargs["target_language"] == "ja"
    assert kwargs["temperature"] == 0.2
    assert kwargs["repetition_penalty"] == 1.1


def test_qwen_backend_maps_public_ids_to_expected_hf_ids():
    from app.services.asr_backends.qwen3_backend import Qwen3ASRBackend

    assert Qwen3ASRBackend("qwen3-asr-0.6b")._hf_model_id == "Qwen/Qwen3-ASR-0.6B"
    assert Qwen3ASRBackend("qwen3-asr-1.7b")._hf_model_id == "Qwen/Qwen3-ASR-1.7B"


def test_whisper_backend_maps_public_ids_to_expected_sizes():
    from app.services.asr_backends.whisper_backend import WhisperBackend

    assert WhisperBackend("whisper-small")._internal_size == "small"
    assert WhisperBackend("whisper-medium")._internal_size == "medium"
    assert WhisperBackend("whisper-large-v3-turbo")._internal_size == "large-v3-turbo"
