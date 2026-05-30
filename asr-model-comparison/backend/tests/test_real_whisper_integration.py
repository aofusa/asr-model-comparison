"""
Integration tests with REAL faster-whisper model.

These tests are written BEFORE implementing any real model loading code
(following strict TDD).

They are marked as "slow" because:
- They require faster-whisper + torch CPU to be installed
- They download the 'tiny' model on first run (~150MB)
- They actually run inference (even tiny model takes several seconds on CPU)

Run with: pytest -m "slow" -s

Without the marker, these tests are automatically skipped.
"""
from __future__ import annotations

import io
import math
import wave
from typing import Any

import pytest

# These imports will fail until faster-whisper is installed — that's expected
try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None  # type: ignore

from app.services.model_manager import ModelManager


def _generate_test_wav_bytes(duration_sec: float = 4.0, sample_rate: int = 16000) -> bytes:
    """
    Generate a minimal synthetic 'speech-like' WAV in memory.
    This avoids needing external audio files for the basic integration test.
    """
    num_samples = int(sample_rate * duration_sec)
    audio = bytearray()

    # Simple sine wave with frequency modulation (simulates voiced speech)
    for i in range(num_samples):
        t = i / sample_rate
        # Modulate between ~120Hz and ~180Hz
        freq = 140 + 30 * math.sin(2 * math.pi * 0.8 * t)
        value = int(16000 * math.sin(2 * math.pi * freq * t) * (0.6 + 0.4 * math.sin(2 * math.pi * 3 * t)))
        # Add some noise
        value += int(800 * (hash(i) % 100 - 50) / 100)
        audio.extend(value.to_bytes(2, "little", signed=True))

    # Build WAV header + data
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio)

    buffer.seek(0)
    return buffer.read()


@pytest.mark.slow
@pytest.mark.skipif(WhisperModel is None, reason="faster-whisper not installed")
def test_real_whisper_tiny_can_be_loaded_and_run():
    """
    Smoke test: Can we actually load whisper-tiny via faster-whisper and
    get a transcription result from synthetic audio?
    """
    model = WhisperModel("tiny", device="cpu", compute_type="int8")

    wav_bytes = _generate_test_wav_bytes(duration_sec=3.5)

    # faster-whisper accepts file-like or path; we write to a temp file for simplicity
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(wav_bytes)
        tmp_path = tmp.name

    try:
        segments, info = model.transcribe(tmp_path, beam_size=1, language="en")
        text = " ".join(segment.text for segment in segments).strip()
        # We don't assert specific words because it's synthetic audio,
        # but we must get *some* output without crashing.
        assert isinstance(text, str)
        assert info.language is not None
    finally:
        os.unlink(tmp_path)


@pytest.mark.slow
@pytest.mark.skipif(WhisperModel is None, reason="faster-whisper not installed")
@pytest.mark.asyncio
async def test_model_manager_can_use_real_whisper_loader():
    """
    Critical integration test:
    The ModelManager (already proven with mocks) must be able to accept
    a real faster-whisper loader and still obey the single-model rule.
    """
    async def real_whisper_loader(internal_name: str):
        # internal_name will be "tiny" for our mapping
        return WhisperModel(internal_name, device="cpu", compute_type="int8")

    async def real_unloader(instance: Any):
        # faster-whisper models don't need explicit cleanup beyond GC + cache clear
        del instance

    manager = ModelManager(
        model_loader=real_whisper_loader,
        model_unloader=real_unloader,
    )

    # This should actually download + load the tiny model
    await manager.load_model("whisper-tiny")
    assert manager.current_model_id == "whisper-tiny"
    assert manager.is_model_loaded is True

    # Now switch to another whisper variant - must unload the first
    await manager.load_model("whisper-small")
    assert manager.current_model_id == "whisper-small"

    await manager.unload_current()
    assert manager.current_model_id is None
