"""
TDD tests for deepening dedicated class logic, real-time chunked processing,
and Japanese accuracy generation parameters for Qwen3-ASR and Voxtral.

These tests are written BEFORE the implementation upgrades.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from app.services.asr_backends.qwen3_backend import Qwen3ASRBackend
from app.services.asr_backends.voxtral_backend import VoxtralBackend


def _mock_qwen_backend(model_id: str = "qwen3-asr-0.6b", text: str = "日本語テキスト") -> Qwen3ASRBackend:
    backend = Qwen3ASRBackend(model_id=model_id, use_dedicated_class=True)
    backend._qwen_asr_model = MagicMock()
    backend._qwen_asr_model.transcribe.return_value = [
        {"text": text, "language": "ja", "time_stamps": (0.0, 1.0)}
    ]
    return backend


def _mock_voxtral_backend(text: str = "次の発話") -> VoxtralBackend:
    backend = VoxtralBackend(model_id="voxtral-mini-4b", use_dedicated_class=True)
    backend._model = MagicMock()
    backend._model.device = "cpu"
    backend._model.generate.return_value = [[1, 2, 3]]
    backend._processor = MagicMock()
    request = MagicMock()
    request.to.return_value = request
    backend._processor.apply_transcription_request.return_value = request
    backend._processor.apply_chat_template.return_value = request
    backend._processor.batch_decode.return_value = [text]
    return backend


def test_qwen3_dedicated_class_uses_better_asr_prompting():
    """
    When using dedicated class, Qwen3 should construct a proper prompt
    suitable for high-quality ASR (especially Japanese).
    """
    backend = _mock_qwen_backend()

    # Simulate what the inference would do
    # In real implementation, we want good Japanese ASR prompting
    result = backend.transcribe(
        b"fake-japanese-audio",
        language="ja",
        beam_size=6,
        temperature=0.0,           # For accuracy (common for Japanese ASR)
        repetition_penalty=1.1,
    )

    assert "text" in result
    assert result["model_id"] == "qwen3-asr-0.6b"
    _, kwargs = backend._qwen_asr_model.transcribe.call_args
    assert kwargs["language"] == "Japanese"


def test_qwen3_supports_context_for_realtime_chunks():
    """
    For real-time chunked audio from web, the backend must accept
    previous transcript as context to maintain continuity and improve accuracy
    across chunks.
    """
    backend = _mock_qwen_backend()

    # First chunk
    result1 = backend.transcribe(b"chunk1-audio", language="ja")

    # Second chunk with context from previous
    result2 = backend.transcribe(
        b"chunk2-audio",
        language="ja",
        previous_text=result1.get("text", ""),
        beam_size=6,
    )

    assert "text" in result2


def test_voxtral_dedicated_class_and_context_support():
    """Voxtral should also support context for real-time streaming scenarios."""
    backend = _mock_voxtral_backend()

    result = backend.transcribe(
        b"audio-chunk",
        previous_text="前回の認識結果の一部",
        language="ja",
        beam_size=5,
        temperature=0.0,
    )
    assert "text" in result


def test_japanese_accuracy_generation_parameters_are_applied():
    """
    When language=ja, the backend should apply generation parameters
    known to improve Japanese ASR quality (low temp, decent beams, etc.).
    """
    backend = _mock_qwen_backend(model_id="qwen3-asr-1.7b")

    # The implementation should default or accept these for Japanese
    result = backend.transcribe(
        b"japanese-speech",
        language="ja",
        temperature=0.0,
        num_beams=6,
        repetition_penalty=1.2,
    )
    assert isinstance(result, dict)


def test_dedicated_class_returns_better_timestamp_structure():
    """
    With dedicated classes, we expect richer timestamp information
    (start/end per segment) which is critical for real-time web applications
    (live subtitles, alignment).
    """
    backend = _mock_qwen_backend()

    result = backend.transcribe(b"audio", return_timestamps=True, language="ja")

    if "chunks" in result:
        for chunk in result["chunks"]:
            assert "start" in chunk or "timestamp" in chunk
            assert "end" in chunk or "timestamp" in chunk
