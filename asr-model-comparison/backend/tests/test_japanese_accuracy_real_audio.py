"""
Slow integration tests for Japanese accuracy verification using REAL audio files.

These tests are intended for Qwen3-ASR and Voxtral as the main models in this project.

Requirements to run:
- Set environment variable USE_REAL_MODELS=1
- Place a real Japanese audio file at: backend/tests/audio_samples/ja_01.wav (or configure via JA_TEST_AUDIO env var)
- The audio should be short (5-30 seconds) spoken Japanese.

The goal is not unit testing, but practical accuracy verification on real Japanese speech.

Run with:
    pytest -m slow -s tests/test_japanese_accuracy_real_audio.py
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

# Audio samples are now located next to the tests for better organization
AUDIO_SAMPLES_DIR = Path(__file__).parent / "audio_samples"
DEFAULT_JA_AUDIO = AUDIO_SAMPLES_DIR / "ja_01.wav"


def _find_japanese_test_audio() -> Path | None:
    """Allow override via environment variable for flexibility."""
    env_path = os.getenv("JA_TEST_AUDIO")
    if env_path:
        p = Path(env_path)
        return p if p.exists() else None

    if DEFAULT_JA_AUDIO.exists():
        return DEFAULT_JA_AUDIO

    return None


@pytest.mark.slow
@pytest.mark.skipif(
    os.getenv("USE_REAL_MODELS") != "1" and os.getenv("USE_REAL_WHISPER") != "1",
    reason="Requires USE_REAL_MODELS=1 (or USE_REAL_WHISPER=1) to run real model tests",
)
def test_japanese_accuracy_qwen3_with_real_audio():
    """
    Verify that Qwen3-ASR can produce reasonable Japanese output on real speech audio.
    """
    audio_path = _find_japanese_test_audio()
    if audio_path is None:
        pytest.skip(
            "No Japanese test audio found. "
            "Place a file at backend/tests/audio_samples/ja_01.wav or set JA_TEST_AUDIO environment variable."
        )

    from app.services.asr_backends.qwen3_backend import Qwen3ASRBackend

    # Prefer the 0.6B model for faster iteration during accuracy verification
    backend = Qwen3ASRBackend(
        model_id="qwen3-asr-0.6b",
        device="cpu",
        use_dedicated_class=True,   # Use dedicated class for better Japanese quality
    )

    result = backend.transcribe(
        str(audio_path),
        language="ja",
        beam_size=6,
        temperature=0.0,
        repetition_penalty=1.15,
        return_timestamps=True,
    )

    text = result.get("text", "").strip()

    assert text, "Transcription result should not be empty"
    # Basic sanity check for Japanese output
    assert any(
        "\u3040" <= ch <= "\u309F" or  # Hiragana
        "\u30A0" <= ch <= "\u30FF" or  # Katakana
        "\u4E00" <= ch <= "\u9FFF"     # Kanji
        for ch in text
    ), f"Expected Japanese characters in output, got: {text[:100]}..."

    print(f"\n[Qwen3 0.6B] Japanese real-audio transcription:\n{text}\n")


@pytest.mark.slow
@pytest.mark.skipif(
    os.getenv("USE_REAL_MODELS") != "1",
    reason="Requires USE_REAL_MODELS=1 for Voxtral",
)
def test_japanese_accuracy_voxtral_with_real_audio():
    """
    Verify that Voxtral can produce reasonable Japanese output on real speech audio.
    """
    audio_path = _find_japanese_test_audio()
    if audio_path is None:
        pytest.skip("No Japanese test audio found.")

    from app.services.asr_backends.voxtral_backend import VoxtralBackend

    backend = VoxtralBackend(
        model_id="voxtral-mini-4b",
        device="cpu",
        use_dedicated_class=True,
    )

    result = backend.transcribe(
        str(audio_path),
        language="ja",
        beam_size=5,
        temperature=0.0,
        repetition_penalty=1.1,
        return_timestamps=True,
    )

    text = result.get("text", "").strip()

    assert text, "Transcription result should not be empty"
    assert any(
        "\u3040" <= ch <= "\u309F" or
        "\u30A0" <= ch <= "\u30FF" or
        "\u4E00" <= ch <= "\u9FFF"
        for ch in text
    ), f"Expected Japanese characters in output, got: {text[:100]}..."

    print(f"\n[Voxtral] Japanese real-audio transcription:\n{text}\n")


@pytest.mark.slow
def test_japanese_audio_file_exists_or_skip():
    """
    Helper test that clearly tells the developer what audio file is expected.
    """
    audio_path = _find_japanese_test_audio()
    if audio_path is None:
        pytest.skip(
            "No Japanese test audio present.\n"
            "To run real Japanese accuracy verification tests:\n"
            "1. Record or obtain a short Japanese speech clip (5-30s).\n"
            "2. Save it as backend/tests/audio_samples/ja_01.wav (16kHz mono WAV recommended).\n"
            "3. Run with: USE_REAL_MODELS=1 pytest -m slow -s tests/test_japanese_accuracy_real_audio.py"
        )

    assert audio_path.exists()
    print(f"Found Japanese test audio: {audio_path}")