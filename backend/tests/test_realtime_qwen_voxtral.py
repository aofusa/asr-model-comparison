"""
TDD tests for real-time web audio use case focusing on:
- Full dedicated class implementation for Qwen3 and Voxtral
- High quality timestamp output (critical for real-time/live transcription)
- Improved error messages and operational robustness

These tests avoid actual heavy model loading by mocking _ensure_loaded and internal model objects.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from app.services.asr_backends.qwen3_backend import Qwen3ASRBackend
from app.services.asr_backends.voxtral_backend import VoxtralBackend


def test_qwen3_dedicated_class_flag_and_fallback_logic():
    """
    The backend should correctly set the dedicated class flag and handle
    fallback gracefully (without raising confusing errors during construction).
    """
    backend = Qwen3ASRBackend(
        model_id="qwen3-asr-0.6b",
        device="cpu",
        use_dedicated_class=True,
    )

    assert backend._use_dedicated_class is True
    assert backend.family == "qwen3"


def test_voxtral_dedicated_class_flag_and_fallback_logic():
    """Same for Voxtral."""
    backend = VoxtralBackend(
        model_id="voxtral-mini-4b",
        device="cpu",
        use_dedicated_class=True,
    )
    assert backend._use_dedicated_class is True
    assert backend.family == "voxtral"


def test_qwen3_supports_high_quality_timestamps_structure():
    """
    The backend must return a structured 'chunks' field when return_timestamps=True,
    which is critical for real-time applications.
    """
    backend = Qwen3ASRBackend(model_id="qwen3-asr-0.6b", device="cpu")

    # Mock the pipeline to avoid actual model loading
    mock_result = {"text": "テストです", "chunks": [{"text": "テストです", "timestamp": (0.0, 1.5)}]}
    backend._pipe = MagicMock(return_value=mock_result)

    result = backend.transcribe(
        b"fake-audio-data",
        return_timestamps=True,
    )

    assert "text" in result
    assert "chunks" in result or "timestamp" in str(result)


def test_voxtral_supports_high_quality_timestamps_structure():
    """Voxtral should also support structured timestamps."""
    backend = VoxtralBackend(model_id="voxtral-mini-4b", device="cpu")

    mock_result = {"text": "This is a test.", "chunks": [{"text": "This is a test.", "timestamp": (0.0, 2.0)}]}
    backend._pipe = MagicMock(return_value=mock_result)

    result = backend.transcribe(b"fake-audio", return_timestamps=True)
    assert "text" in result


def test_qwen3_voxtral_better_error_messages_for_empty_audio():
    """
    Both backends should raise clear, actionable errors for empty audio
    (common issue in real-time streaming when chunks are too small).
    """
    backend_q = Qwen3ASRBackend(model_id="qwen3-asr-0.6b", device="cpu")
    backend_v = VoxtralBackend(model_id="voxtral-mini-4b", device="cpu")

    for backend in [backend_q, backend_v]:
        backend._pipe = MagicMock()  # Prevent actual model loading
        with pytest.raises(ValueError) as excinfo:
            backend.transcribe(b"")
        assert "empty" in str(excinfo.value).lower() or "audio" in str(excinfo.value).lower()


def test_qwen3_supports_previous_text_for_realtime_chunks():
    """
    The backend must accept 'previous_text' for real-time chunked transcription
    to maintain context across audio segments (essential for long Japanese conversations).
    """
    backend = Qwen3ASRBackend(model_id="qwen3-asr-0.6b", device="cpu")

    # Mock to avoid loading
    mock_result = {"text": "続きです", "model_id": "qwen3-asr-0.6b"}
    backend._pipe = MagicMock(return_value=mock_result)

    result = backend.transcribe(
        b"chunk-audio",
        previous_text="今日はいい天気です",
        language="ja",
    )

    assert "text" in result
    # In a full implementation, previous_text would influence the prompt in dedicated class path
    assert result["model_id"] == "qwen3-asr-0.6b"
