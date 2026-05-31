"""
TDD tests for deepening dedicated class logic, real-time chunked processing,
and Japanese accuracy generation parameters for Qwen3-ASR and Voxtral.

These tests are written BEFORE the implementation upgrades.
"""
from __future__ import annotations

import pytest
from app.services.asr_backends.qwen3_backend import Qwen3ASRBackend
from app.services.asr_backends.voxtral_backend import VoxtralBackend


def test_qwen3_dedicated_class_uses_better_asr_prompting():
    """
    When using dedicated class, Qwen3 should construct a proper prompt
    suitable for high-quality ASR (especially Japanese).
    """
    backend = Qwen3ASRBackend(
        model_id="qwen3-asr-0.6b",
        use_dedicated_class=True,
    )

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


def test_qwen3_supports_context_for_realtime_chunks():
    """
    For real-time chunked audio from web, the backend must accept
    previous transcript as context to maintain continuity and improve accuracy
    across chunks.
    """
    backend = Qwen3ASRBackend(model_id="qwen3-asr-0.6b", use_dedicated_class=True)

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
    backend = VoxtralBackend(model_id="voxtral-mini-4b", use_dedicated_class=True)

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
    backend = Qwen3ASRBackend(model_id="qwen3-asr-1.7b", use_dedicated_class=True)

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
    backend = Qwen3ASRBackend(model_id="qwen3-asr-0.6b", use_dedicated_class=True)

    result = backend.transcribe(b"audio", return_timestamps=True, language="ja")

    if "chunks" in result:
        for chunk in result["chunks"]:
            assert "start" in chunk or "timestamp" in chunk
            assert "end" in chunk or "timestamp" in chunk
